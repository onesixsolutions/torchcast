import copy
import math
import sys
import types

from typing import List, Optional, Sequence, Dict, Union, Collection
from warnings import warn

import torch
from torch import Tensor, nn, jit
from torch.nn import functional as F

from torchcast.process.utils import Identity
from torchcast.internals.utils import is_near_zero, validate_gt_shape
from torchcast.process.process import Process

DEFAULT_MCOV_MULTI = 1.0
DEFAULT_PCOV_MULTI = 0.1  # less than measure-cov by default
DEFAULT_ICOV_MULTI = 0.5  # somewhere in between
# ``Covariance.from_processes(cov_type='initial')`` uses method='low_rank' by default from this many elements on:
LOW_RANK_MIN_INITIAL = 15

# the parameters for each ``method``, in the order ``set_matrix_()`` computes them:
_METHOD_PARAMS = {
    'log_cholesky': ('cholesky_log_diag', 'cholesky_off_diag'),
    'sd_corr': ('log_std_devs', 'corr_unconstrained'),
    'low_rank': ('lr_mat', 'log_std_devs'),
}


class Covariance(nn.Module):
    """
    The :class:`.Covariance` can be used when you'd like more control over the covariance specification of a state-
    space model. For example, if you're training on diverse time-serieses that vary in scale/behavior, you could use an
    :class:`torch.nn.Embedding` to predict the variance of each series, with the group-ids as predictors:

    .. code-block:: python3

        kf = KalmanFilter(
            measures=measures,
            processes=processes,
            measure_covariance=Covariance.from_measures(
                measures,
                predict_variance=torch.nn.Sequential(
                    torch.nn.Embedding(len(group_ids), len(measures), padding_idx=0),
                    torch.nn.Softplus()
                ),
                expected_kwargs=['group_ids']
            ),
            process_covariance=Covariance.from_processes(
                processes,
                predict_variance=torch.nn.Sequential(
                    torch.nn.Embedding(len(group_ids), Covariance.from_processes(processes).param_rank, padding_idx=0),
                    torch.nn.Softplus()
                ),
                expected_kwargs=['group_ids']
            )
        )

    """

    # Construction -----------------------------------------------------------------------------------------------------

    @classmethod
    def from_processes(cls,
                       processes: Sequence[Process],
                       cov_type: str = 'process',
                       predict_variance: Union[bool, nn.Module] = None,
                       corr_groups: Optional[Sequence[Sequence[str]]] = None,
                       stationary: Union[Sequence[str], Dict[str, bool], None] = None,
                       method: Optional[str] = None,
                       **kwargs) -> 'Covariance':
        """
        :param processes: The ``processes`` used in your :class:`.StateSpaceModel`.
        :param cov_type: The type of covariance, either 'process' or 'initial' (default: 'process').
        :param predict_variance: Will the variance be predicted upon calling ``forward()``? This is implemented as a
         multiplier on the base variance given from the 'method'. You can either pass ``True`` in which case it is
         expected you will pass multipliers as 'process_var_multi' when ``forward()`` is called; *or* you can pass a
         :class:`torch.nn.Module` that will predict the multipliers, in which case you'll pass input(s) to this
         module at forward. Either way please note these should output strictly positive values with shape
         ``(num_groups, num_times, self.param_rank)``.
        :param corr_groups: Optionally, groups of process ids whose processes are uncorrelated with those in other
         groups (within a group, and within each process, they stay correlated). Processes not in any group form one
         more group together. E.g. with processes 'level_a', 'transient_a', 'level_b', 'transient_b':
         ``corr_groups=[['transient_a', 'transient_b']]`` makes the transients correlated with each other, and the
         levels correlated with each other, but no transient correlated with any level. Note that "uncorrelated" has
         to be transitive this way: you can't make 'transient_a' uncorrelated with 'level_a' only, while both stay
         correlated with 'level_b' (masking can't guarantee a valid covariance for that pattern). For other masks, see
         ``corr_mask`` in :func:`Covariance.__init__`.
        :param stationary: Only for ``cov_type='initial'``: ids of decaying processes (e.g. ``LocalLevel`` with
         ``decay=(.5, .95)``) whose initial covariance is their stationary covariance, rather than free parameters.
         For elements ``i`` and ``j`` of these processes it's ``Q[i, j] / (1 - decay_i * decay_j)``, where ``Q`` is the
         (base) process-covariance and ``decay`` each element's transition to itself (the :class:`.KalmanFilter`
         computes this); between these and other elements it's zero (there's no stationary covariance with, e.g., a
         random walk). These elements get no parameters here. With a learned decay this is recommended: otherwise the
         decay trades off against the initial variance (a faster decay with a larger initial variance fits the start
         of a series about as well as a slower decay with a smaller one), so the decay is pinned down by the first few
         timesteps rather than the whole series; the :class:`.KalmanFilter` warns about this. Each element of these
         processes needs process-noise, a decay that's less than 1 (a learned decay may be bounded at 1, e.g.
         ``decay=(.5, 1.0)``: the stationary variance grows without bound as it nears 1, so the likelihood keeps it
         below, and an error is raised if it reaches 1), and no transitions into it from other elements.
         Either a list of process ids (all stationary) or a dict of process ids to ``True``/``False``, where ``False``
         keeps free parameters on purpose (silencing the warning): e.g. for a decaying level that captures how far
         series start from their long-run level after a common start event (sign-up, launch), whose initial spread
         is larger than its stationary one. Note that listing processes here changes this covariance's parameters and
         its ``mask`` buffer (both shrink), so a state-dict saved without it won't load into one with it: to warm-start,
         drop all of the saved ``initial_covariance.*`` entries, and set the free block with
         :func:`Covariance.set_matrix_` if needed.
        :param method: The parameterization, see :func:`Covariance.__init__`. If not given: "low_rank" for an initial
         covariance with 15 or more elements with parameters (not counting ``stationary`` ones) and no
         ``corr_groups``/``corr_mask``, since it's weakly informed (mainly by the start of each series) and grows
         quadratically with the state; otherwise "log_cholesky".
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.StateSpaceModel`.
        """

        assert cov_type in {'process', 'initial'}
        if isinstance(stationary, str) or any(not isinstance(pid, str) for pid in (stationary or ())):
            raise TypeError("``stationary`` should be a list of process ids, or a dict of process ids to booleans.")
        if not isinstance(stationary, dict):
            stationary = {pid: True for pid in (stationary or ())}
        if any(not isinstance(v, bool) for v in stationary.values()):
            raise TypeError("``stationary``'s values should be ``True`` or ``False``.")
        if stationary and cov_type != 'initial':
            raise ValueError("``stationary`` is only for ``cov_type='initial'``.")
        unknown = set(stationary) - {p.id for p in processes}
        if unknown:
            raise ValueError(f"``stationary`` has unknown process ids: {sorted(unknown)}.")
        if corr_groups is not None:
            if 'corr_mask' in kwargs:
                raise TypeError("Pass ``corr_groups`` or ``corr_mask``, not both.")
            kwargs['corr_mask'] = cls._corr_groups_to_mask(
                element_ids=[p.id for p in processes for _ in p.state_elements],
                corr_groups=corr_groups,
                kind='process id'
            )
        state_rank = 0
        no_cov_idx = []
        stationary_idx = []
        nonstationary_idx = []
        for p in processes:
            no_cov_elements = [nm for nm, se in p.state_elements.items() if not getattr(se, f'has_{cov_type}_variance')]
            for i, nm in enumerate(p.state_elements):
                if nm in no_cov_elements:
                    no_cov_idx.append(state_rank + i)
                elif stationary.get(p.id) is True:
                    stationary_idx.append(state_rank + i)
                elif stationary.get(p.id) is False:
                    nonstationary_idx.append(state_rank + i)
            state_rank += len(p.state_elements)

        if method is None:
            # a large initial covariance is weakly informed (mainly by the start of each series) and grows quadratically
            # with the state, so by default it's a factor model:
            param_rank = state_rank - len(no_cov_idx) - len(stationary_idx)
            use_low_rank = cov_type == 'initial' and 'corr_mask' not in kwargs and param_rank >= LOW_RANK_MIN_INITIAL
            method = 'low_rank' if use_low_rank else 'log_cholesky'

        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = DEFAULT_PCOV_MULTI if cov_type == 'process' else DEFAULT_ICOV_MULTI

        if cov_type not in {'initial', 'process'}:
            raise ValueError(f"Unrecognized cov_type {cov_type}, expected 'initial' or 'process'.")

        if predict_variance is True:
            predict_variance = Identity()
            if 'expected_kwargs' not in kwargs:
                kwargs['expected_kwargs'] = [f'{cov_type}_var_multi']

        if stationary_idx:
            kwargs['stationary_idx'] = stationary_idx
        if nonstationary_idx:
            kwargs['nonstationary_idx'] = nonstationary_idx
        return cls(
            rank=state_rank,
            empty_idx=no_cov_idx,
            id=f'{cov_type}_covariance',
            predict_variance=predict_variance,
            method=method,
            **kwargs
        )

    @classmethod
    def from_measures(cls,
                      measures: Sequence[str],
                      predict_variance: Union[bool, nn.Module] = None,
                      corr_groups: Optional[Sequence[Sequence[str]]] = None,
                      method: Optional[str] = None,
                      **kwargs) -> 'Covariance':
        """
        :param measures: The ``measures`` used in your :class:`.StateSpaceModel`.
        :param predict_variance: Will the variance be predicted upon calling ``forward()``? This is implemented as a
         multiplier on the base variance given from the 'method'. You can either pass ``True`` in which case it is
         expected you will pass multipliers as 'measure_var_multi' when ``forward()`` is called; *or* you can pass a
         :class:`torch.nn.Module` that will predict the multipliers, in which case you'll pass input(s) to this
         module at forward. Either way please note these should output strictly positive values with shape
         ``(num_groups, num_times, len(measures))``.
        :param corr_groups: Optionally, groups of measures that are uncorrelated with those in other groups; measures
         not in any group form one more group together. See :func:`Covariance.from_processes`.
        :param method: The parameterization, see :func:`Covariance.__init__`. If not given, "log_cholesky".
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.StateSpaceModel`.
        """
        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        elif not isinstance(measures[0], str):
            # not good duck-typing, but too easy to accidentally pass dataset.measures instead of dataset.measures[0]
            raise RuntimeError(f"`measures[0]` is {type(measures[0])}, expected str")
        if corr_groups is not None:
            if 'corr_mask' in kwargs:
                raise TypeError("Pass ``corr_groups`` or ``corr_mask``, not both.")
            kwargs['corr_mask'] = cls._corr_groups_to_mask(
                element_ids=list(measures), corr_groups=corr_groups, kind='measure'
            )
        if method is None:
            method = 'log_cholesky'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = DEFAULT_MCOV_MULTI

        if predict_variance is True:
            predict_variance = Identity()
            if 'expected_kwargs' not in kwargs:
                kwargs['expected_kwargs'] = [f'measure_var_multi']

        return cls(
            rank=len(measures), id='measure_covariance', predict_variance=predict_variance, method=method, **kwargs
        )

    @staticmethod
    def _corr_groups_to_mask(element_ids: Sequence[str], corr_groups: Sequence[Sequence[str]], kind: str) -> Tensor:
        """
        A ``corr_mask`` from ``corr_groups`` (see :func:`Covariance.from_processes`).

        :param element_ids: For each element (row/column) of the covariance, the id it's grouped by (e.g. its process's
         id); can repeat.
        :param corr_groups: Groups of ids. Ids not in any group form one more group together.
        :param kind: What the ids are, for error messages.
        """
        if isinstance(corr_groups, str) or any(isinstance(g, str) for g in corr_groups):
            raise TypeError(f"``corr_groups`` should be a list of lists of {kind}s, e.g. ``[['a', 'b']]``.")
        known = list(dict.fromkeys(element_ids))
        group_of = {}
        for g, ids in enumerate(corr_groups):
            for id_ in ids:
                if id_ not in known:
                    raise ValueError(f"``corr_groups`` has unknown {kind} '{id_}'; expected one of {known}.")
                if id_ in group_of:
                    raise ValueError(f"``corr_groups`` has {kind} '{id_}' in more than one group.")
                group_of[id_] = g
        rest = len(corr_groups)
        groups = torch.tensor([group_of.get(id_, rest) for id_ in element_ids])
        return (groups.unsqueeze(-1) == groups.unsqueeze(-2)).to(torch.get_default_dtype())

    def __init__(self,
                 rank: int,
                 init_diag_multi: float,
                 method: str = 'log_cholesky',
                 empty_idx: List[int] = (),
                 predict_variance: Optional[nn.Module] = None,
                 expected_kwargs: Optional[Sequence[str]] = None,
                 id: Optional[str] = None,
                 corr_mask: Optional[Tensor] = None,
                 stationary_idx: Sequence[int] = (),
                 nonstationary_idx: Sequence[int] = ()):
        """
        You should rarely call this directly. Instead, call :func:`Covariance.from_measures` and
        :func:`Covariance.from_processes`.

        :param rank: The number of elements along the diagonal.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
         This can be useful to provide intelligent starting-values to speed up optimization.
        :param method: The parameterization for the covariance. The default, "log_cholesky", parameterizes the
         covariance using the cholesky factorization (which is itself split into two tensors: the log-transformed
         diagonal elements and the off-diagonal). The other currently supported option is "low_rank", a factor
         model: the covariance is ``V @ V.t() + D``, with ``V`` (``lr_mat``) a G*K matrix of factor-loadings (G the
         number of elements with variance, K the number of factors) and ``D`` a diagonal matrix of
         ``exp(log_std_devs) ** 2``. This suits a large covariance that's weakly informed by the data, e.g. one
         dominated by a few common factors; it's the default for a large initial covariance (see
         :func:`Covariance.from_processes`). K defaults to ``int(sqrt(G))`` (an arbitrary rule of thumb: pass e.g.
         "low_rank3" for 3 factors). Rotations of ``V`` leave ``V @ V.t()`` unchanged; that's harmless for fitting,
         but makes the hessian singular, so for :func:`StateSpaceModel.get_laplace_mvnorm`,
         :func:`Covariance.prepare_for_hessian_` rotates ``V`` to a canonical form and leaves out the entries that pin
         the rotation.
         Finally, "sd_corr" separates scale from correlation: ``diag(std) @ R @ diag(std)``, with parameters
         ``log_std_devs`` and ``corr_unconstrained``. ``R`` is a correlation matrix built from canonical partial
         correlations ``tanh(corr_unconstrained)`` (as in Stan's ``cholesky_factor_corr``), one per off-diagonal
         element. Unlike "log_cholesky" (where off-diagonal entries are in absolute units, and also change the
         variances), each parameter is dimensionless and the variances are set by ``log_std_devs`` alone; this is
         better conditioned for optimization when elements have very different scales, and lets you freeze variances
         and correlations separately (see :func:`Covariance.get_param_idx` and
         :func:`Covariance.get_off_diag_idx`). Note that a zero ``corr_unconstrained`` is a zero *partial* correlation
         (given the earlier elements): element ``i`` is uncorrelated with all others when every entry in its row *and*
         column is zero. Existing modules can be converted with :func:`Covariance.to_method`.
        :param empty_idx: In some cases (e.g. process-covariance) we will have some elements with no variance.
        :param predict_variance: For predicting variance, see :func:`Covariance.from_measures`.
        :param expected_kwargs: If ``predict_variance`` is set, this allows you to set the keyword that will be passed
         at ``forward()``.
        :param id: Identifier for this covariance. Typically left ``None`` and set when passed to the
         :class:`.StateSpaceModel`.
        :param corr_mask: Optionally, a ``(rank, rank)`` tensor that the covariance is multiplied by elementwise, e.g.
         with zeros to force some pairs of elements to be uncorrelated. It must be symmetric, with ones on the
         diagonal, and positive semi-definite, which guarantees the masked covariance is too (Schur product theorem).
         For the typical case, groups of elements that are uncorrelated with each other, it's easier to pass
         ``corr_groups`` to :func:`Covariance.from_processes` or :func:`Covariance.from_measures`, which builds
         this. For a mask of zeros and ones (which, being positive semi-definite, is always such a grouping, up to the
         order of elements), "log_cholesky" and "sd_corr" apply it to the cholesky factor too (to the correlation's
         cholesky factor for "sd_corr"): the factor's entries for masked-out pairs are zero, and aren't parameters
         (``cholesky_off_diag``/``corr_unconstrained`` have one entry per pair that isn't masked out; see
         :func:`Covariance.get_off_diag_idx`). This loses nothing: a covariance that's zero across groups has a
         cholesky factor that's zero there too. With other masks (e.g. entries of 0.5), and with "low_rank", the mask
         only applies to the covariance, so the parameters behind masked-out entries are redundant (they don't change
         the masked covariance's zeros, but do change its other entries).
        :param stationary_idx: For an initial covariance: elements (of ``0..rank-1``) whose covariance is computed by
         the :class:`.KalmanFilter` (the stationary covariance), so they get no parameters here (like ``empty_idx``).
         Typically set with ``stationary`` in :func:`Covariance.from_processes`. ``corr_mask`` doesn't apply to them.
        :param nonstationary_idx: For an initial covariance: elements (of ``0..rank-1``) that keep free parameters on
         purpose, though they could be stationary (``False`` in ``stationary``); this only silences the
         :class:`.KalmanFilter`'s warning.
        """

        super().__init__()

        self.id = id
        self.rank = rank

        self.stationary_idx = sorted(int(i) for i in stationary_idx)
        if not all(0 <= i < rank for i in self.stationary_idx):
            raise ValueError(f"``stationary_idx`` out of range for rank {rank}: {self.stationary_idx}.")
        self.nonstationary_idx = sorted(int(i) for i in nonstationary_idx)
        if set(self.nonstationary_idx) & set(self.stationary_idx):
            raise ValueError("``stationary_idx`` and ``nonstationary_idx`` overlap.")
        self.empty_idx = set(empty_idx) | set(self.stationary_idx)
        assert all(isinstance(x, int) for x in self.empty_idx)
        self.param_rank = self.rank - len(self.empty_idx)
        mask = self._get_mini_cov_mask(rank=self.rank, empty_idx=self.empty_idx)
        self.register_buffer('mask', mask)

        if corr_mask is not None:
            self.register_buffer('corr_mask', self._validate_corr_mask(corr_mask, rank=self.rank))

        self._set_params(method, init_diag_multi)

        self.var_predict_module = predict_variance

        if self.var_predict_module and expected_kwargs is None:
            raise ValueError("Please explicitly specify ``expected_kwargs`` if passing ``predict_variance``.")
        if isinstance(expected_kwargs, str):
            expected_kwargs = [expected_kwargs]
        self.expected_kwargs: Optional[List[str]] = None if expected_kwargs is None else list(expected_kwargs)

    @staticmethod
    def _validate_corr_mask(corr_mask: Tensor, rank: int) -> Tensor:
        corr_mask = torch.as_tensor(corr_mask, dtype=torch.get_default_dtype()).detach().clone()
        if corr_mask.shape != (rank, rank):
            raise ValueError(f"Expected ``corr_mask`` to have shape {(rank, rank)}, got {tuple(corr_mask.shape)}.")
        if not torch.equal(corr_mask, corr_mask.t()):
            raise ValueError("``corr_mask`` must be symmetric.")
        if not (corr_mask.diagonal() == 1).all():
            raise ValueError("``corr_mask`` must have ones on the diagonal.")
        if torch.linalg.eigvalsh(corr_mask.to(torch.float64)).min() < -1e-8:
            raise ValueError(
                "``corr_mask`` must be positive semi-definite (otherwise the masked covariance may not be), e.g. a "
                "block-diagonal matrix of ones; see ``corr_groups`` in ``Covariance.from_processes()``."
            )
        return corr_mask

    @staticmethod
    def _get_mini_cov_mask(rank: int, empty_idx: Collection[int], **kwargs) -> Tensor:
        """
        A ``(rank, param_rank)`` matrix that embeds the covariance of the elements with parameters (the "mini-cov")
        into the full ``(rank, rank)`` covariance, with zero rows/columns for ``empty_idx``.
        """
        param_rank = rank - len(empty_idx)
        mask = torch.zeros((rank, param_rank), **kwargs)
        c = 0
        for r in range(rank):
            if r not in empty_idx:
                mask[r, c] = 1.
                c += 1
        return mask

    def _set_params(self, method: str, init_diag_multi: float):
        self.cholesky_log_diag: Optional[nn.Parameter] = None
        self.cholesky_off_diag: Optional[nn.Parameter] = None
        self.lr_mat: Optional[nn.Parameter] = None
        self.log_std_devs: Optional[nn.Parameter] = None
        self.corr_unconstrained: Optional[nn.Parameter] = None
        fkw = {'device': self.mask.device, 'dtype': self.mask.dtype}
        if method == 'log_cholesky':
            self.method = method
            self.cholesky_log_diag = nn.Parameter(.1 * torch.randn(self.param_rank, **fkw) + math.log(init_diag_multi))
            self.cholesky_off_diag = nn.Parameter(.1 * torch.randn(self._get_num_off_diag_params(), **fkw))
        elif method == 'sd_corr':
            self.method = method
            self.log_std_devs = nn.Parameter(.1 * torch.randn(self.param_rank, **fkw) + math.log(init_diag_multi))
            self.corr_unconstrained = nn.Parameter(.1 * torch.randn(self._get_num_off_diag_params(), **fkw))
        elif method.startswith('low_rank'):
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                low_rank = max(int(math.sqrt(self.param_rank)), 1)
            if not 1 <= low_rank <= self.param_rank:
                raise ValueError(f"``method='{method}'``: the number of factors must be in 1..{self.param_rank}.")
            # half of each element's initial variance (``init_diag_multi ** 2``, as for the other methods) from the
            # factors, half from the diagonal (factor-loadings near zero would start at a saddle-point):
            self.lr_mat = nn.Parameter(
                torch.randn(self.param_rank, low_rank, **fkw) * init_diag_multi * math.sqrt(.5 / low_rank)
            )
            self.log_std_devs = nn.Parameter(
                data=.1 * torch.randn(self.param_rank, **fkw) + math.log(init_diag_multi * math.sqrt(.5))
            )
        else:
            raise NotImplementedError(method)

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @property
    def non_empty_idx(self) -> List[int]:
        return [i for i in range(self.rank) if i not in self.empty_idx]

    # The covariance matrix --------------------------------------------------------------------------------------------

    def forward(self,
                inputs: Dict[str, Tensor],
                num_groups: int,
                num_times: int,
                _ignore_input: bool = False) -> Tensor:
        mini_cov = self._get_mini_cov()
        mini_cov = validate_gt_shape(
            mini_cov, num_groups=num_groups, num_times=num_times, trailing_dim=[self.param_rank, self.param_rank]
        )

        if self.var_predict_module is not None and not _ignore_input:
            pred = self.var_predict_module(*[inputs[x] for x in self.expected_kwargs])
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise RuntimeError(f"{self.id}'s `predict_variance` produced nans/infs")
            if (pred < 0).any():
                raise RuntimeError(f"{self.id}'s `predict_variance` produced values <0; needs exp/softplus layer.")
            pred = validate_gt_shape(pred, num_groups=num_groups, num_times=num_times, trailing_dim=[self.param_rank])
            mini_cov = mini_cov * pred.unsqueeze(-2) * pred.unsqueeze(-1)

        mask = self.mask.unsqueeze(0).unsqueeze(0)

        return mask @ mini_cov @ mask.transpose(-1, -2)

    def _get_mini_cov(self) -> Tensor:
        if self.method == 'log_cholesky':
            assert self.cholesky_log_diag is not None
            L = log_chol_to_chol(self.cholesky_log_diag, self._get_off_diag())
            mini_cov = L @ L.t()
        elif self.method == 'sd_corr':
            assert self.log_std_devs is not None
            L = self.log_std_devs.exp().unsqueeze(-1) * corr_cholesky_from_unconstrained(
                self._get_off_diag(), rank=self.param_rank
            )
            mini_cov = L @ L.t()
        elif self.method == 'low_rank':
            assert self.lr_mat is not None
            assert self.log_std_devs is not None
            mini_cov = self.lr_mat @ self.lr_mat.t() + torch.diag_embed(self.log_std_devs.exp() ** 2)
        else:
            raise NotImplementedError(self.method)

        # (``getattr``: modules pickled before ``corr_mask`` existed don't have the buffer). With a masked factor, this
        # is redundant but makes the zeros exact:
        corr_mask = getattr(self, 'corr_mask', None)
        if corr_mask is not None:
            idx = self.non_empty_idx
            mini_cov = mini_cov * corr_mask[idx][:, idx]

        if is_near_zero(mini_cov.diagonal(dim1=-2, dim2=-1), atol=1e-12).any():
            warn(
                f"`{self.id}` has near-zero along the diagonal. Will add 1e-12 to the diagonal. "
                f"Values:\n{mini_cov.diag()}"
            )
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1], device=mini_cov.device, dtype=mini_cov.dtype) * 1e-12
        return mini_cov

    def _get_off_diag(self) -> Tensor:
        """
        The off-diagonal entries for the cholesky factor (the correlation's, for "sd_corr"), in ``tril_indices``
        order: ``cholesky_off_diag``/``corr_unconstrained`` itself or, with a binary ``corr_mask``, its entries
        scattered into place, with zeros for masked-out pairs.
        """
        off_diag = self.cholesky_off_diag if self.method == 'log_cholesky' else self.corr_unconstrained
        assert off_diag is not None
        keep = self._get_off_diag_keep()
        if keep is None:
            return off_diag
        free = keep.nonzero().squeeze(-1).to(off_diag.device)
        return off_diag.new_zeros(keep.shape).index_put((free,), off_diag)

    def _get_binary_mini_mask(self) -> Optional[Tensor]:
        """
        A binary (0/1) ``corr_mask`` restricted to the elements with parameters (``(param_rank, param_rank)``), or
        ``None`` if there's no ``corr_mask`` or it isn't binary.
        """
        # (``getattr``: modules pickled before ``corr_mask`` existed don't have the buffer)
        corr_mask = getattr(self, 'corr_mask', None)
        if corr_mask is None or not ((corr_mask == 0) | (corr_mask == 1)).all():
            return None
        return corr_mask[self.non_empty_idx][:, self.non_empty_idx]

    def _get_off_diag_keep(self) -> Optional[Tensor]:
        """
        For a binary ``corr_mask`` and a cholesky-based ``method``: the mask for the off-diagonal parameters
        (``cholesky_off_diag`` or ``corr_unconstrained``), in their ``tril_indices`` order. ``None`` otherwise.
        """
        mini_mask = self._get_binary_mini_mask()
        if mini_mask is None or self.method not in ('log_cholesky', 'sd_corr'):
            return None
        rows, cols = torch.tril_indices(self.param_rank, self.param_rank, offset=-1).to(mini_mask.device)
        return mini_mask[rows, cols]

    def _get_num_off_diag_params(self) -> int:
        keep = self._get_off_diag_keep()
        return num_off_diag(self.param_rank) if keep is None else int(keep.sum())

    def get_corr_cholesky(self) -> Tensor:
        """
        The cholesky factor of the correlation matrix (of the ``param_rank`` block, i.e. excluding ``empty_idx``, and
        before any ``predict_variance``). Not used by torchcast internally, but can be useful if (e.g.) you want to set
        up an LKJ prior for your model:
        ``torch.distributions.LKJCholesky(cov.param_rank, eta).log_prob(cov.get_corr_cholesky())``.
        """
        if self.method == 'sd_corr':
            return corr_cholesky_from_unconstrained(self._get_off_diag(), rank=self.param_rank)
        mini_cov = self._get_mini_cov()
        std = mini_cov.diagonal(dim1=-2, dim2=-1).sqrt()
        return torch.linalg.cholesky(mini_cov / (std.unsqueeze(-1) * std.unsqueeze(-2)))

    # Setting parameters, and converting between methods ---------------------------------------------------------------

    @torch.no_grad()
    def set_matrix_(self, cov: Tensor) -> 'Covariance':
        """
        Set this module's parameters (in-place) so that its covariance (before any ``predict_variance``) equals
        ``cov``. For "low_rank", which can't represent an arbitrary matrix, this fits the factor model to ``cov``
        (by principal-axis factoring: alternating the top-K eigendecomposition of ``cov - D`` with the residual
        diagonal ``D``, kept positive), which is exact if ``cov`` has that structure, and an approximation otherwise.

        :param cov: A ``(rank, rank)`` covariance matrix, whose rows/columns in ``empty_idx`` are zero (as are its
         entries masked out by a binary ``corr_mask``); or just the ``(param_rank, param_rank)`` block for the
         non-empty elements.
        :return: This module.
        """
        cov = torch.as_tensor(cov).detach()
        if cov.shape == (self.param_rank, self.param_rank):
            mini_cov = cov
        elif cov.shape == (self.rank, self.rank):
            empty = sorted(self.empty_idx)
            if empty:
                tol = 1e-8 * cov.diagonal().abs().max()
                if (cov[empty].abs() > tol).any() or (cov[:, empty].abs() > tol).any():
                    raise ValueError(f"``cov`` has non-zero entries in the rows/columns of ``empty_idx`` ({empty}).")
            mini_cov = cov[self.non_empty_idx][:, self.non_empty_idx]
        else:
            raise ValueError(
                f"Expected ``cov`` to have shape {(self.rank, self.rank)} or {(self.param_rank, self.param_rank)}, "
                f"got {tuple(cov.shape)}."
            )
        self._validate_masked_entries(mini_cov)
        for name, value in self._get_params_for_matrix(mini_cov).items():
            getattr(self, name).copy_(value)
        return self

    def _validate_masked_entries(self, mini_cov: Tensor):
        """
        Raise if a ``(param_rank, param_rank)`` covariance has non-zero entries for pairs masked out by a binary
        ``corr_mask``: they can't be represented (and for a cholesky-based method, dropping the factor's entries
        behind them would also change the entries within groups).
        """
        mini_mask = self._get_binary_mini_mask()
        if mini_mask is None:
            return
        tol = 1e-8 * mini_cov.diagonal().abs().max()
        if ((mini_cov * (1 - mini_mask.to(mini_cov))).abs() > tol).any():
            raise ValueError("``cov`` has non-zero entries for pairs of elements masked out by ``corr_mask``.")

    def _get_params_for_matrix(self, mini_cov: Tensor) -> Dict[str, Tensor]:
        """The parameters' values (by name, as float64 on cpu) for a ``(param_rank, param_rank)`` covariance."""
        # converting in float64 so that the round-trip is exact up to the module's dtype (on cpu: no float64 on mps;
        # the caller moves the results back):
        mini_cov = mini_cov.detach().cpu().to(torch.float64)
        mini_cov = (mini_cov + mini_cov.t()) / 2
        if self.method == 'log_cholesky':
            L = _cholesky(mini_cov)
            values = L.diagonal().log(), L[tuple(torch.tril_indices(self.param_rank, self.param_rank, offset=-1))]
        elif self.method == 'sd_corr':
            std = mini_cov.diagonal().sqrt()
            L_corr = _cholesky(mini_cov / (std.unsqueeze(-1) * std.unsqueeze(-2)))
            values = std.log(), corr_cholesky_to_unconstrained(L_corr)
        elif self.method == 'low_rank':
            lr_mat, std = _fit_factor_model(mini_cov, num_factors=self.lr_mat.shape[-1])
            values = lr_mat, std.log()
        else:
            raise NotImplementedError(f"``set_matrix_()`` isn't supported for method='{self.method}'.")
        keep = self._get_off_diag_keep()
        if keep is not None:
            # (only the pairs that aren't masked out are parameters; with ``cov`` zero across groups, as
            # ``set_matrix_()`` checks, the dropped entries are zero up to floating-point error)
            values = values[0], values[1][keep.cpu().bool()]
        return dict(zip(_METHOD_PARAMS[self.method], values))

    @classmethod
    def from_matrix(cls,
                    cov: Tensor,
                    method: str = 'sd_corr',
                    empty_idx: Collection[int] = (),
                    **kwargs) -> 'Covariance':
        """
        Create a :class:`.Covariance` whose covariance (before any ``predict_variance``) equals ``cov``.

        :param cov: A ``(rank, rank)`` covariance matrix, whose rows/columns in ``empty_idx`` are zero.
        :param method: The parameterization, see :func:`Covariance.__init__`.
        :param empty_idx: Elements with no variance.
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        """
        kwargs.setdefault('init_diag_multi', 1.0)
        out = cls(rank=cov.shape[-1], method=method, empty_idx=list(empty_idx), **kwargs)
        return out.set_matrix_(cov)

    def to_method(self, method: str) -> 'Covariance':
        """
        A copy of this module with a different parameterization (``method``) but the same covariance (exactly, up to
        floating-point error), e.g. to convert a fitted model:
        ``model.initial_covariance = model.initial_covariance.to_method('sd_corr')``. Other attributes (``id``,
        ``empty_idx``, ``predict_variance``) are carried over. Note that a model's state-dict saved after converting
        can only be loaded into a model created with the new ``method``.
        """
        out = copy.deepcopy(self)
        out._set_params(method, init_diag_multi=1.0)
        return out.set_matrix_(self._get_mini_cov().detach())

    def _load_from_state_dict(self,
                              state_dict,
                              prefix,
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs):
        # a state-dict saved with a different ``method`` would otherwise fail with a generic missing/unexpected-keys
        # error (or, for ``log_std_devs``, which 'low_rank' and 'sd_corr' share, partially load):
        present = {k[len(prefix):] for k in state_dict if k.startswith(prefix)}
        expected = set(_METHOD_PARAMS.get(self.method, ()))
        if not expected <= present:
            saved_with = [m for m, nms in _METHOD_PARAMS.items() if m != self.method and set(nms) <= present]
            if saved_with:
                saved = saved_with[0]
                if saved == 'low_rank':
                    saved = f"low_rank{state_dict[prefix + 'lr_mat'].shape[-1]}"
                error_msgs.append(
                    f"`{self.id or type(self).__name__}` uses method='{self.method}', but the state-dict was saved "
                    f"with method='{saved}'. To load it, create this covariance with ``method='{saved}'`` (e.g. "
                    f"``Covariance.from_processes(..., method='{saved}')`` or ``Covariance.from_measures(..., "
                    f"method='{saved}')``), then, if you like, convert it with ``.to_method('{self.method}')``. "
                    f"(torchcast 2.0 changed the default methods: \"low_rank\" for an initial covariance with "
                    f"{LOW_RANK_MIN_INITIAL} or more elements with parameters, and \"log_cholesky\" for every "
                    f"measure-covariance.)"
                )
                return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # Indexing the parameters ------------------------------------------------------------------------------------------

    def get_param_idx(self, i: int) -> int:
        """
        The index of element ``i`` (of ``0..rank-1``) in the per-element parameters (``log_std_devs`` for
        "sd_corr"/"low_rank", ``cholesky_log_diag`` for "log_cholesky"), accounting for ``empty_idx``. E.g., to
        freeze some elements' std-devs and correlations (the frozen entries get exactly zero gradients):

        .. code-block:: python3

            cov = Covariance.from_processes(processes, cov_type='initial', method='sd_corr')
            keep_std = torch.ones_like(cov.log_std_devs)
            keep_std[cov.get_param_idx(3)] = 0
            cov.log_std_devs.register_hook(lambda g: g * keep_std)
            keep_corr = torch.ones_like(cov.corr_unconstrained)
            keep_corr[cov.get_off_diag_idx(3, 0)] = 0
            cov.corr_unconstrained.register_hook(lambda g: g * keep_corr)

        For "sd_corr", these entries are interpretable on their own; for "log_cholesky" they're entries of the
        cholesky factor, which mix scale and correlation.
        """
        if i in self.empty_idx:
            raise ValueError(f"Element {i} is in ``empty_idx``, so it has no parameters.")
        if not 0 <= i < self.rank:
            raise IndexError(f"Element {i} out of range for rank {self.rank}.")
        return self.non_empty_idx.index(i)

    def get_off_diag_idx(self, i: int, j: int) -> int:
        """
        The index of the pair of elements ``(i, j)`` (in either order, of ``0..rank-1``) in the off-diagonal
        parameters (``corr_unconstrained`` for "sd_corr", ``cholesky_off_diag`` for "log_cholesky"), accounting for
        ``empty_idx``, and for pairs masked out by a binary ``corr_mask`` (which have no parameter). See
        :func:`Covariance.get_param_idx`.
        """
        row, col = sorted((self.get_param_idx(i), self.get_param_idx(j)), reverse=True)
        if row == col:
            raise ValueError("``i`` and ``j`` must be different elements.")
        idx = row * (row - 1) // 2 + col
        keep = self._get_off_diag_keep()
        if keep is None:
            return idx
        if not keep[idx]:
            raise ValueError(f"Elements {i} and {j} are masked out by ``corr_mask``, so they have no parameter.")
        return int(keep[:idx].sum())

    # Low-rank rotation, for the hessian -------------------------------------------------------------------------------

    def prepare_for_hessian_(self) -> Dict[str, Tensor]:
        """
        Called by :func:`StateSpaceModel.get_laplace_mvnorm` (on any submodule that has this method) before computing
        the hessian: put the parameters (in-place, without changing the model) into a canonical form, and return, by
        parameter name, boolean masks of the entries to leave out of the hessian, because the model only defines
        them up to a symmetry along which the likelihood is flat (so the hessian is singular there).

        For "low_rank", rotations of the factor-loadings ``V`` leave the covariance unchanged: this rotates ``V`` to
        lower-trapezoidal form (see ``Covariance._rotate_low_rank_()``) and leaves out the entries above the
        diagonal. (With a binary ``corr_mask``, each group of elements can be rotated separately; only one rotation
        is pinned, so the hessian stays singular, with a warning.) Other methods have no such symmetry.
        """
        if self.method != 'low_rank':
            return {}
        mini_mask = self._get_binary_mini_mask()
        if mini_mask is not None and (mini_mask == 0).any():
            warn(
                f"`{self.id}` has method='low_rank' and a ``corr_mask``, so each group's factor-loadings can be "
                f"rotated separately; only one rotation is pinned for the hessian, which stays singular."
            )
        order = self._get_lr_order()
        self._rotate_low_rank_(order)
        return {'lr_mat': self._get_lr_redundant(order)}

    @torch.no_grad()
    def _rotate_low_rank_(self, order: Sequence[int]) -> 'Covariance':
        """
        For "low_rank": rotate the factor-loadings ``V`` (in-place) to lower-trapezoidal form (``V[i, j] = 0`` for
        ``j > i``, with the elements in ``order``, e.g. from ``_get_lr_order()``), by an LQ decomposition, with a
        non-negative diagonal. This leaves the covariance unchanged (exactly, up to floating-point error) and pins down
        the rotation, which is otherwise arbitrary: equivalent loadings give the same result. See
        :func:`Covariance.prepare_for_hessian_`.
        """
        if self.method != 'low_rank':
            raise TypeError(f"``_rotate_low_rank_()`` is for method='low_rank', not '{self.method}'.")
        rotated = _to_lower_trapezoidal(self.lr_mat.detach().cpu().to(torch.float64), order)
        self.lr_mat.copy_(rotated.masked_fill(self._get_lr_redundant(order).cpu(), 0.))
        return self

    def _get_lr_order(self) -> List[int]:
        """
        For "low_rank": an order of the (``param_rank``) elements that leads with the K whose factor-loadings are the
        most linearly independent (greedily, as in a pivoted QR decomposition), followed by the rest in their usual
        order. Rotating ``V`` to lower-trapezoidal form in this order pins the rotation on rows that are well away from
        singular. (It depends only on ``V @ V.t()``, so a rotation doesn't change it.)
        """
        assert self.lr_mat is not None
        return _get_pivot_order(self.lr_mat.detach())

    def _get_lr_redundant(self, order: Optional[Sequence[int]] = None) -> Tensor:
        """
        For "low_rank": a ``(param_rank, K)`` boolean mask of the entries of ``lr_mat`` above the diagonal, in
        ``order`` (default: :func:`Covariance._get_lr_order`). After ``Covariance._rotate_low_rank_()`` they're zero,
        and they span the rotations.
        """
        assert self.lr_mat is not None
        order = self._get_lr_order() if order is None else list(order)
        position = torch.empty(self.param_rank, dtype=torch.long)
        position[order] = torch.arange(self.param_rank)
        return (position.unsqueeze(-1) < torch.arange(self.lr_mat.shape[-1])).to(self.lr_mat.device)

    # Deprecated -------------------------------------------------------------------------------------------------------

    @staticmethod
    def log_chol_to_chol(log_diag: Tensor, off_diag: Tensor) -> Tensor:
        """Deprecated: use :func:`torchcast.covariance.log_chol_to_chol`."""
        warn(
            "``Covariance.log_chol_to_chol()`` is deprecated, use ``torchcast.covariance.log_chol_to_chol()``.",
            FutureWarning
        )
        return log_chol_to_chol(log_diag, off_diag)


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


def _log_cosh(x: Tensor) -> Tensor:
    # stable for large |x|, and smooth (exact second derivative at 0, unlike formulas using abs(x))
    return x + F.softplus(-2. * x) - math.log(2.)


def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    """
    A lower-triangular cholesky factor from its log-diagonal and its off-diagonal entries (filled into the strict lower
    triangle in the order of ``torch.tril_indices(rank, rank, -1)``), as parameterized by
    ``Covariance(method='log_cholesky')``.

    :param log_diag: Tensor whose last dim is ``rank``.
    :param off_diag: Tensor whose last dim is ``rank * (rank - 1) / 2``.
    :return: Tensor with trailing dims ``(rank, rank)``.
    """
    assert log_diag.shape[:-1] == off_diag.shape[:-1]

    rank = log_diag.shape[-1]
    L1 = torch.diag_embed(torch.exp(log_diag))

    L2 = torch.zeros_like(L1)
    mask = torch.tril_indices(rank, rank, offset=-1)
    L2[mask[0], mask[1]] = off_diag
    return L1 + L2


def corr_cholesky_from_unconstrained(u: Tensor, rank: int) -> Tensor:
    """
    The cholesky factor of a correlation matrix, from unconstrained parameters: ``tanh(u)`` are canonical partial
    correlations, filled into the strict lower triangle in the order of ``torch.tril_indices(rank, rank, -1)``.
    Row ``i`` is ``L[i, j] = z_ij * sqrt(1 - sum_{k<j} L[i, k]^2)``, ``L[i, i] = sqrt(1 - sum_{k<i} L[i, k]^2)``, so
    each row has unit norm.

    :param u: Tensor whose last dim is ``rank * (rank - 1) / 2``.
    :param rank: Size of the correlation matrix.
    :return: Tensor with trailing dims ``(rank, rank)``.
    """
    U = u.new_zeros(u.shape[:-1] + (rank, rank))
    rows, cols = torch.tril_indices(rank, rank, offset=-1).to(u.device)  # (not implemented on mps)
    U[..., rows, cols] = u
    # the remaining squared row-norm before column j is prod_{k<j} (1 - tanh(u_ik)^2), whose sqrt is
    # exp(-sum_{k<j} log(cosh(u_ik))); computed this way it's stable (no 1 - tanh^2 underflow) for large |u|:
    lc = _log_cosh(U)
    remain = torch.exp(lc - lc.cumsum(-1))
    return torch.tanh(U) * remain + torch.diag_embed(remain.diagonal(dim1=-2, dim2=-1))


def corr_cholesky_to_unconstrained(L: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Inverse of :func:`corr_cholesky_from_unconstrained`. Best done in float64 (``eps`` clamps the partial
    correlations away from +/-1, for near-singular correlation matrices).
    """
    rank = L.shape[-1]
    sq = L ** 2
    remain = (1 - (sq.cumsum(-1) - sq)).clamp_min(eps)
    z = (L / remain.sqrt()).clamp(-1 + eps, 1 - eps)
    rows, cols = torch.tril_indices(rank, rank, offset=-1).to(L.device)
    return torch.atanh(z[..., rows, cols])


def _cholesky(cov: Tensor) -> Tensor:
    L, info = torch.linalg.cholesky_ex(cov)
    if info.any():
        raise ValueError("Covariance matrix is not positive-definite.")
    return L


def cov2corr(cov: Tensor) -> Tensor:
    """The correlation matrix (or matrices, over leading dims) of a covariance matrix."""
    std_ = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return cov / (std_.unsqueeze(-1) * std_.unsqueeze(-2))


def _get_pivot_order(V: Tensor) -> List[int]:
    """
    An order of ``V``'s rows: first, greedily, the ``V.shape[-1]`` rows that are the most linearly independent (each
    the one with the largest norm after projecting out those already chosen, as in a pivoted QR decomposition of
    ``V.t()``), then the rest in their usual order. Depends only on ``V @ V.t()``.
    """
    residual = V.cpu().to(torch.float64).clone()
    lead = []
    for _ in range(min(V.shape)):
        norms = (residual ** 2).sum(-1)
        norms[lead] = -1.
        i = int(norms.argmax())
        lead.append(i)
        if norms[i] <= 0:
            continue
        q = residual[i] / norms[i].sqrt()
        residual = residual - (residual @ q).unsqueeze(-1) * q
    return lead + [i for i in range(V.shape[0]) if i not in lead]


def _to_lower_trapezoidal(V: Tensor, order: Sequence[int]) -> Tensor:
    """
    Rotate ``V`` (``(rank, K)``) to the lower-trapezoidal form with the rows in ``order`` (``V[order[i], j] = 0`` for
    ``j > i``), by an LQ decomposition, which leaves ``V @ V.t()`` unchanged. The diagonal (``V[order[j], j]``) is made
    non-negative, which makes this form unique: equivalent ``V``s (differing by a rotation) give the same result.
    """
    order = list(order)
    _, R = torch.linalg.qr(V[order].t())
    signs = torch.where(R.diagonal() < 0, -1., 1.).to(R)
    out = torch.zeros_like(V)
    out[order] = R.t() * signs
    return out


def _fit_factor_model(cov: Tensor, num_factors: int, num_paf_iter: int = 2_000) -> tuple[Tensor, Tensor]:
    """
    Fit ``cov ~= V @ V.t() + diag(std ** 2)``, with ``V`` lower-trapezoidal (see ``_get_pivot_order``): principal-axis
    factoring (alternating the top eigenvectors of ``cov - D`` with the residual diagonal ``D``) for a start, then
    polished with L-BFGS on the squared error (principal-axis factoring alone can converge very slowly).

    :return: ``V`` (``(rank, num_factors)``) and ``std`` (``(rank,)``).
    """
    diag = cov.diagonal()
    min_var = diag * 1e-8
    # start the uniqueness at the residual variance given all other elements (a standard start):
    uniq = (1 / torch.linalg.inv(cov).diagonal()).clamp(min=min_var)
    for _ in range(num_paf_iter):
        evals, evecs = torch.linalg.eigh(cov - torch.diag(uniq))
        V = evecs[:, -num_factors:] * evals[-num_factors:].clamp(min=0).sqrt()
        new_uniq = (diag - (V ** 2).sum(-1)).clamp(min=min_var)
        converged = (new_uniq - uniq).abs().max() <= 1e-12 * diag.max()
        uniq = new_uniq
        if converged:
            break

    if not converged:
        V = V.clone(memory_format=torch.contiguous_format).requires_grad_()
        log_uniq = uniq.log().requires_grad_()
        opt = torch.optim.LBFGS(
            [V, log_uniq], max_iter=1_000, tolerance_grad=1e-15, tolerance_change=1e-20, line_search_fn='strong_wolfe'
        )
        scale = diag.max()

        def closure() -> Tensor:
            opt.zero_grad()
            loss = ((V @ V.t() + torch.diag(log_uniq.exp()) - cov) / scale).pow(2).sum()
            loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(closure)
        V, uniq = V.detach(), log_uniq.detach().exp()

    return _to_lower_trapezoidal(V, _get_pivot_order(V)), uniq.sqrt()


# backwards compat shim
class _DeprecatedBase(types.ModuleType):
    def __getattr__(self, name):
        if name not in globals():
            # (an ``AttributeError``, as ``getattr(module, name, default)`` and tools probing modules expect)
            raise AttributeError(f"module 'torchcast.covariance' has no attribute '{name}'")
        # (no warning for dunders such as ``__file__``, which tools like sphinx look up)
        if not (name.startswith('__') and name.endswith('__')):
            warn(
                f"`torchcast.covariance.base.{name}` is deprecated, instead just import from "
                f"`torchcast.covariance.{name}`.",
                FutureWarning,
                stacklevel=2
            )
        return globals()[name]


_base = _DeprecatedBase('covariance.base')
sys.modules['torchcast.covariance.base'] = _base
