import copy
import difflib
import inspect
import math
import sys
import types

from typing import Any, List, Mapping, Optional, Sequence, Set, Dict, Tuple, Union, Collection
import warnings
from warnings import warn

import torch
from torch import Tensor, nn, jit
from torch.nn import functional as F

from torchcast.process.utils import Identity, ProcessKwarg
from torchcast.internals.utils import is_near_zero
from torchcast.process.process import Process

DEFAULT_MCOV_MULTI = 1.0
DEFAULT_PCOV_MULTI = 0.1  # less than measure-cov by default
DEFAULT_ICOV_MULTI = 0.5  # somewhere in between
# a covariance's possible ``role`` in a model, and how to create one for it:
_HOW_TO_CREATE = {
    'process': "Covariance.from_processes(processes, cov_type='process', ...)",
    'initial': "Covariance.from_processes(processes, cov_type='initial', ...)",
    'measure': "Covariance.from_measures(measures, ...)",
}

# the parameters for each ``method``, in the order ``set_matrix_()`` computes them:
_METHOD_PARAMS = {
    'log_cholesky': ('cholesky_log_diag', 'cholesky_off_diag'),
    'sd_corr': ('log_std_devs', 'corr_unconstrained'),
    'low_rank': ('lr_mat', 'log_std_devs'),
}


class Covariance(nn.Module):
    """
    The :class:`.Covariance` can be used when you'd like more control over the covariance specification of a state-
    space model. The covariance can be *predicted*, varying by group and/or time:

    - ``predict_std``: a module whose output (strictly positive, e.g. via a softplus) multiplies each element's
      std-deviation. E.g. if you're training on diverse time-serieses that vary in scale, you could use an
      :class:`torch.nn.Embedding` with the group-ids as predictors.
    - ``predict_corr``: a module whose output is *added* to the unconstrained correlation-parameters
      (``corr_unconstrained``, see ``method='sd_corr'``), so correlations can vary too. Its output has one column per
      entry of ``corr_unconstrained``, in the order given by :func:`Covariance.get_off_diag_idx`. Initializing it to
      output zeros starts it at the base correlations.

    The modules are called with the ``predictor_kwargs``, which you pass to the model's ``forward()`` (and ``fit()``).
    Their outputs have shape ``(num_groups, num_times, ...)``, or ``(num_groups, ...)`` if they only vary by group.
    For example, a measurement-covariance whose std-devs vary by group and correlations vary by season:

    .. code-block:: python3

        kf = KalmanFilter(
            measures=measures,
            processes=processes,
            measure_covariance=Covariance.from_measures(
                measures,
                predict_std=torch.nn.Sequential(
                    torch.nn.Embedding(num_groups, len(measures)),
                    torch.nn.Softplus()
                ),
                predict_corr=SeasonalCorrelation(...),  # e.g. a linear model of fourier features
                predictor_kwargs=['group_ids', 'season_X']
            )
        )
        kf.fit(y, group_ids=group_ids, season_X=season_X)

    Here both modules receive both kwargs (``module(group_ids, season_X)``): wrap a module to ignore what it doesn't
    need. A kwarg named ``{role}_covariance__{name}`` (e.g. ``measure_covariance__season_X``) overrides ``name`` for
    this covariance.

    A covariance knows its ``role`` in a model ("process", "initial" or "measure") and the names of its elements
    (``element_names``: ``'{process_id}.{state_element}'`` for the state, or the measures), which
    :func:`Covariance.from_processes` and :func:`Covariance.from_measures` set. The model checks both, so e.g. a
    process-covariance passed as the ``initial_covariance``, or a covariance created from other processes (or in
    another order), raises an error rather than fitting the wrong model.
    """

    # Construction -----------------------------------------------------------------------------------------------------

    @classmethod
    def from_processes(cls,
                       processes: Sequence[Process],
                       cov_type: str = 'process',
                       predict_variance: Union[bool, nn.Module, None] = None,
                       corr_groups: Optional[Sequence[Sequence[str]]] = None,
                       stationary: Union[Sequence[str], Dict[str, bool], None] = None,
                       method: Optional[str] = None,
                       **kwargs) -> 'Covariance':
        """
        :param processes: The ``processes`` used in your :class:`.StateSpaceModel`.
        :param cov_type: The type of covariance, either 'process' or 'initial' (default: 'process').
        :param predict_variance: Deprecated, use ``predict_std``.
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
        :param method: The parameterization, see :func:`Covariance.__init__`. If not given, "sd_corr" if
         ``predict_corr`` is passed, otherwise "log_cholesky".
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`, e.g. ``predict_std``,
         ``predict_corr``, ``predictor_kwargs``. Note that the outputs of a ``predict_std`` module have
         ``param_rank`` columns, i.e. one for each state-element with parameters (not ``empty_idx`` or
         ``stationary``).
        :return: A :class:`.Covariance` object that can be used in your :class:`.StateSpaceModel`.
        """

        if cov_type not in {'initial', 'process'}:
            raise ValueError(f"Unrecognized cov_type {cov_type}, expected 'initial' or 'process'.")
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

        if method is None and kwargs.get('predict_corr') is not None:
            method = 'sd_corr'
        if method is None:
            method = 'log_cholesky'

        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = DEFAULT_PCOV_MULTI if cov_type == 'process' else DEFAULT_ICOV_MULTI
        _deprecated_predict_variance(predict_variance, kwargs, default_kwarg=f'{cov_type}_var_multi')

        if stationary_idx:
            kwargs['stationary_idx'] = stationary_idx
        if nonstationary_idx:
            kwargs['nonstationary_idx'] = nonstationary_idx
        return cls(
            rank=state_rank,
            empty_idx=no_cov_idx,
            role=cov_type,
            element_names=get_state_element_names(processes),
            method=method,
            **kwargs
        )

    @classmethod
    def from_measures(cls,
                      measures: Sequence[str],
                      predict_variance: Union[bool, nn.Module, None] = None,
                      corr_groups: Optional[Sequence[Sequence[str]]] = None,
                      method: Optional[str] = None,
                      **kwargs) -> 'Covariance':
        """
        :param measures: The ``measures`` used in your :class:`.StateSpaceModel`.
        :param predict_variance: Deprecated, use ``predict_std``.
        :param corr_groups: Optionally, groups of measures that are uncorrelated with those in other groups; measures
         not in any group form one more group together. See :func:`Covariance.from_processes`.
        :param method: The parameterization, see :func:`Covariance.__init__`. If not given, "sd_corr" if
         ``predict_corr`` is passed, otherwise "log_cholesky".
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`, e.g. ``predict_std``,
         ``predict_corr``, ``predictor_kwargs``.
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
            method = 'sd_corr' if kwargs.get('predict_corr') is not None else 'log_cholesky'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = DEFAULT_MCOV_MULTI
        _deprecated_predict_variance(predict_variance, kwargs, default_kwarg='measure_var_multi')

        return cls(rank=len(measures), role='measure', element_names=list(measures), method=method, **kwargs)

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
                 method: Optional[str] = None,
                 empty_idx: List[int] = (),
                 predict_std: Optional[nn.Module] = None,
                 predict_corr: Optional[nn.Module] = None,
                 predictor_kwargs: Sequence[Union[str, ProcessKwarg]] = (),
                 role: Optional[str] = None,
                 element_names: Optional[Sequence[str]] = None,
                 id: Optional[str] = None,
                 predict_variance: Optional[nn.Module] = None,
                 expected_kwargs: Optional[Sequence[str]] = None,
                 corr_mask: Optional[Tensor] = None,
                 stationary_idx: Sequence[int] = (),
                 nonstationary_idx: Sequence[int] = ()):
        """
        You should rarely call this directly. Instead, call :func:`Covariance.from_measures` and
        :func:`Covariance.from_processes`.

        :param rank: The number of elements along the diagonal.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
         This can be useful to provide intelligent starting-values to speed up optimization.
        :param method: The parameterization for the covariance. The default, "log_cholesky" (or "sd_corr" if
         ``predict_corr`` is passed), parameterizes the covariance using the cholesky factorization (which is itself
         split into two tensors: the log-transformed diagonal elements and the off-diagonal). The other option is
         "sd_corr", which separates scale from correlation: ``diag(std) @ R @ diag(std)``, with parameters
         ``log_std_devs`` and ``corr_unconstrained``. ``R`` is a correlation matrix built from canonical partial
         correlations ``tanh(corr_unconstrained)`` (as in Stan's ``cholesky_factor_corr``), one per off-diagonal
         element. Unlike "log_cholesky" (where off-diagonal entries are in absolute units, and also change the
         variances), each parameter is dimensionless and the variances are set by ``log_std_devs`` alone; this is
         better conditioned for optimization when elements have very different scales, and lets you freeze variances
         and correlations separately (see :func:`Covariance.get_param_idx` and
         :func:`Covariance.get_off_diag_idx`). Note that a zero ``corr_unconstrained`` is a zero *partial* correlation
         (given the earlier elements): element ``i`` is uncorrelated with all others when every entry in its row *and*
         column is zero. Existing modules can be converted with :func:`Covariance.to_method`. ("low_rank" is no
         longer available for new modules; a pickled "low_rank" covariance still loads, with a warning, and converts
         exactly with :func:`Covariance.to_method`.)
        :param empty_idx: In some cases (e.g. process-covariance) we will have some elements with no variance.
        :param predict_std: A module predicting (strictly positive) multipliers on the std-devs, with ``param_rank``
         columns. See the class docstring.
        :param predict_corr: A module predicting additive adjustments to ``corr_unconstrained``, with one column per
         entry (``param_rank * (param_rank - 1) / 2``, or fewer with a binary ``corr_mask``). Requires
         ``method='sd_corr'``. See the class docstring.
        :param predictor_kwargs: The names of the keyword-arguments passed to the model's ``forward()`` that are
         inputs to ``predict_std``/``predict_corr``. A string is a tensor whose first dimension is the group (e.g.
         ``group_ids``, or a ``(num_groups, num_times, ...)`` tensor); for inputs that aren't group-indexed, pass
         ``ProcessKwarg(name, is_group_time_tensor=False)``.
        :param role: The covariance's role in a model: "process", "initial" or "measure". Set by
         :func:`Covariance.from_processes` / :func:`Covariance.from_measures`; if ``None``, the model sets it. A model
         raises if a covariance with a different role is passed.
        :param element_names: The names of the ``rank`` elements: ``'{process_id}.{state_element}'`` for a process or
         initial covariance (see :func:`get_state_element_names`), the measures for a measure covariance. Set by
         :func:`Covariance.from_processes` / :func:`Covariance.from_measures`; if ``None``, the model sets it. A model
         raises if they don't match its own (e.g. a covariance created from other processes, or in another order).
        :param id: Deprecated: the id is now derived from ``role`` (e.g. "process_covariance").
        :param predict_variance: Deprecated, use ``predict_std``.
        :param expected_kwargs: Deprecated, use ``predictor_kwargs``.
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
         cholesky factor that's zero there too. With other masks (e.g. entries of 0.5), the mask only applies to the
         covariance, so the parameters behind masked-out entries are redundant (they don't change the masked
         covariance's zeros, but do change its other entries). It also applies to ``predict_corr``'s
         adjusted covariances.
        :param stationary_idx: For an initial covariance: elements (of ``0..rank-1``) whose covariance is computed by
         the :class:`.KalmanFilter` (the stationary covariance), so they get no parameters here (like ``empty_idx``).
         Typically set with ``stationary`` in :func:`Covariance.from_processes`. ``corr_mask`` doesn't apply to them.
        :param nonstationary_idx: For an initial covariance: elements (of ``0..rank-1``) that keep free parameters on
         purpose, though they could be stationary (``False`` in ``stationary``); this only silences the
         :class:`.KalmanFilter`'s warning.
        """

        super().__init__()

        if predict_variance is not None:
            warn("``predict_variance`` is deprecated, use ``predict_std``.", FutureWarning, stacklevel=2)
            if predict_std is not None:
                raise TypeError("Pass ``predict_std`` or ``predict_variance`` (deprecated), not both.")
            predict_std = predict_variance
        if expected_kwargs is not None:
            warn("``expected_kwargs`` is deprecated, use ``predictor_kwargs``.", FutureWarning, stacklevel=2)
            if predictor_kwargs:
                raise TypeError("Pass ``predictor_kwargs`` or ``expected_kwargs`` (deprecated), not both.")
            predictor_kwargs = expected_kwargs

        if method is None:
            method = 'sd_corr' if predict_corr is not None else 'log_cholesky'
        if predict_corr is not None and method != 'sd_corr':
            raise ValueError(f"``predict_corr`` requires ``method='sd_corr'`` (got '{method}').")

        if id is not None:
            warn(
                "``Covariance(id=...)`` is deprecated: a covariance's ``role`` is set by ``from_processes()`` / "
                "``from_measures()``, or by the model it's passed to.",
                FutureWarning,
                stacklevel=2
            )
            if role is None and id.endswith('_covariance') and id[:-len('_covariance')] in _HOW_TO_CREATE:
                role = id[:-len('_covariance')]
        if role is not None and role not in _HOW_TO_CREATE:
            raise ValueError(f"``role`` should be one of {list(_HOW_TO_CREATE)}, got '{role}'.")
        self.role = role
        self.rank = rank
        self.element_names: Optional[List[str]] = None
        if element_names is not None:
            element_names = list(element_names)
            if len(element_names) != rank:
                raise ValueError(f"``element_names`` has {len(element_names)} names, expected {rank} (``rank``).")
            if len(set(element_names)) != rank:
                raise ValueError(f"``element_names`` aren't unique: {element_names}.")
            self.element_names = element_names

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

        self.predict_std = predict_std
        self.predict_corr = predict_corr
        if isinstance(predictor_kwargs, (str, ProcessKwarg)):
            predictor_kwargs = [predictor_kwargs]
        self.predictor_kwargs: List[ProcessKwarg] = [
            k if isinstance(k, ProcessKwarg) else ProcessKwarg(name=k, is_group_time_tensor=True)
            for k in predictor_kwargs
        ]
        if (predict_std is not None or predict_corr is not None) and not self.predictor_kwargs:
            raise ValueError("Please specify ``predictor_kwargs`` if passing ``predict_std`` or ``predict_corr``.")
        if self.predictor_kwargs and predict_std is None and predict_corr is None:
            raise ValueError("``predictor_kwargs`` were passed, but no ``predict_std`` or ``predict_corr``.")

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
            raise ValueError(
                "``method='low_rank'`` is no longer available for new models: as a default for large covariances it "
                "could fit as well as \"log_cholesky\" but forecast worse (its free diagonal can collapse, leaving a "
                "near-singular prior). Use \"log_cholesky\" or \"sd_corr\". (A pickled \"low_rank\" covariance still "
                "loads, and converts exactly with ``.to_method('log_cholesky')``.)"
            )
        else:
            raise NotImplementedError(method)

    @property
    def id(self) -> Optional[str]:
        """
        ``'{role}_covariance'`` (e.g. "measure_covariance"), or ``None`` if the role isn't set yet. Used in messages,
        and as the prefix of keyword-arguments that override ``predictor_kwargs`` for this covariance.
        """
        return None if self.role is None else f'{self.role}_covariance'

    @property
    def _label(self) -> str:
        return self.id or type(self).__name__

    def _bind(self, role: str, element_names: Sequence[str]) -> 'Covariance':
        """
        Called by a model on the covariance it's given for ``role``, with the names of the elements the model expects
        (see ``element_names`` in :func:`Covariance.__init__`): raises if they don't match the covariance's, and sets
        them if they weren't set.
        """
        assert role in _HOW_TO_CREATE
        how = _HOW_TO_CREATE[role]
        if self.role is not None and self.role != role:
            # e.g. ``Covariance.from_processes(processes)`` (a process-covariance) passed as an initial covariance:
            raise ValueError(
                f"This covariance was created as a ``{self.role}`` covariance, but it's being used as the ``{role}`` "
                f"covariance. Create it with ``{how}``."
            )
        if role == 'initial' and self.is_predicted:
            raise ValueError(
                "The ``initial`` covariance has ``predict_std`` / ``predict_corr``, but it applies before any data, so "
                "it isn't predicted (those would have no effect)."
            )
        element_names = list(element_names)
        kind, source = ('measures', 'measures') if role == 'measure' else ('state-elements', 'processes')
        if self.element_names is None:
            if self.rank != len(element_names):
                raise ValueError(
                    f"The ``{role}`` covariance has rank {self.rank}, but the model has {len(element_names)} {kind}: "
                    f"{element_names}. Create it with ``{how}``."
                )
        elif self.element_names != element_names:
            raise ValueError(
                f"The ``{role}`` covariance was created for the {kind} {self.element_names}, but the model's are "
                f"{element_names}. Create it with ``{how}``, with the model's {source}, in the same order."
            )
        self.role = role
        self.element_names = element_names
        return self

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        """Deprecated: a covariance's ``role`` is set by ``from_processes()`` / ``from_measures()``, or by the model."""
        warn("``Covariance.set_id()`` is deprecated (the id is derived from ``role``).", FutureWarning, stacklevel=2)
        role = id[:-len('_covariance')] if id.endswith('_covariance') else None
        if role not in _HOW_TO_CREATE:
            raise ValueError(f"Expected one of {[f'{r}_covariance' for r in _HOW_TO_CREATE]}, got '{id}'.")
        if self.role is not None and self.role != role:
            raise ValueError(f"This covariance was created as a ``{self.role}`` covariance, not ``{role}``.")
        self.role = role
        return self

    @property
    def non_empty_idx(self) -> List[int]:
        return [i for i in range(self.rank) if i not in self.empty_idx]

    @property
    def is_predicted(self) -> bool:
        return self.predict_std is not None or self.predict_corr is not None

    @property
    def expected_kwargs(self) -> Optional[List[str]]:
        warn("``expected_kwargs`` is deprecated, use ``predictor_kwargs``.", FutureWarning, stacklevel=2)
        return [k.name for k in self.predictor_kwargs] or None


    # The covariance matrix --------------------------------------------------------------------------------------------

    def forward(self,
                inputs: Optional[Dict[str, Tensor]] = None,
                num_groups: int = 1,
                num_times: int = 1,
                _ignore_input: bool = False) -> Tensor:
        """
        :param inputs: A dictionary with the ``predictor_kwargs`` (if any).
        :param num_groups: Number of groups.
        :param num_times: Number of timesteps.
        :return: A ``(num_groups, num_times, rank, rank)`` tensor. (The models use :func:`Covariance.batch_cov`,
         which doesn't expand dimensions that don't vary.)
        """
        if self.is_predicted and not _ignore_input:
            inputs, _ = self.get_predictor_inputs(inputs or {})
            cov = self._get_predicted_cov(inputs, num_groups=num_groups, num_times=num_times)
        else:
            cov = self.base_cov()[None, None]
        return cov.expand(num_groups, num_times, -1, -1)

    def base_cov(self) -> Tensor:
        """
        The ``(rank, rank)`` covariance, before any ``predict_std`` / ``predict_corr``.
        """
        return self._embed(self._get_mini_cov())

    def batch_cov(self, kwargs: Mapping[str, Any], num_groups: int, num_times: int) -> Tuple[Tensor, Set[str]]:
        """
        The covariance for a batch, given a model's ``forward()`` keyword-arguments (from which the
        ``predictor_kwargs`` are picked out).

        :return: A tensor with shape ``(G, T, rank, rank)``, where ``G`` is ``num_groups`` -- or 1, if the covariance
         doesn't vary by group -- and ``T`` is ``num_times`` or 1 (see ``as_timestep_list()``); and the set of
         keyword-arguments that were used.
        """
        if not self.is_predicted:
            return self.base_cov()[None, None], set()
        inputs, used = self.get_predictor_inputs(kwargs)
        return self._get_predicted_cov(inputs, num_groups=num_groups, num_times=num_times), used

    def get_predictor_inputs(self, kwargs: Mapping[str, Any]) -> Tuple[List[Tensor], Set[str]]:
        """
        Pick out the inputs to ``predict_std`` / ``predict_corr`` from a model's ``forward()`` keyword-arguments.

        :return: The inputs (in the order of ``predictor_kwargs``), and the set of keys that were used.
        """
        inputs, used = [], set()
        for pkwarg in self.predictor_kwargs:
            key = f'{self.id}__{pkwarg.name}'
            if self.id is None or key not in kwargs:
                key = pkwarg.name
            if key not in kwargs:
                raise TypeError(
                    f"`{self._label}` expected a `{pkwarg.name}` (or `{self.id}__{pkwarg.name}`) keyword-argument."
                )
            inputs.append(kwargs[key])
            used.add(key)
        return inputs, used

    def _get_predicted_cov(self, inputs: Sequence[Tensor], num_groups: int, num_times: int) -> Tensor:
        std_multi = corr_delta = None
        if self.predict_std is not None:
            std_multi = self._validate_prediction(
                self.predict_std(*inputs), 'predict_std', self.param_rank, num_groups, num_times
            )
            if (std_multi < 0).any():
                raise RuntimeError(f"`{self._label}`'s `predict_std` produced values <0; needs exp/softplus layer.")
        if self.predict_corr is not None:
            corr_delta = self._validate_prediction(
                self.predict_corr(*inputs), 'predict_corr', self._get_num_off_diag_params(), num_groups, num_times
            )

        if corr_delta is None:
            mini_cov = self._get_mini_cov() * std_multi.unsqueeze(-1) * std_multi.unsqueeze(-2)
        else:
            std = self.log_std_devs.exp()
            if std_multi is not None:
                std = std * std_multi
            off_diag = self.corr_unconstrained + corr_delta
            keep = self._get_off_diag_keep()
            if keep is not None:
                off_diag = _unpack_off_diag(off_diag, keep)
            L = std.unsqueeze(-1) * corr_cholesky_from_unconstrained(off_diag, rank=self.param_rank)
            mini_cov = self._apply_corr_mask(L @ L.transpose(-1, -2))
        return self._embed(mini_cov)

    def _validate_prediction(self, pred: Tensor, name: str, num_cols: int, num_groups: int, num_times: int) -> Tensor:
        """
        Standardize a prediction to ``(G, T, num_cols)``, where ``G`` is ``num_groups`` or 1 and ``T`` is
        ``num_times`` or 1.
        """
        if pred.shape[-1] != num_cols:
            if pred.shape[-1] == 1:
                pred = pred.expand(*pred.shape[:-1], num_cols)
            else:
                raise ValueError(
                    f"`{self._label}`'s `{name}` should output {num_cols} columns, got shape {pred.shape}."
                )
        if pred.ndim == 1:
            pred = pred[None, None]
        elif pred.ndim == 2:
            pred = pred.unsqueeze(1)
        if pred.ndim != 3 or pred.shape[0] not in (1, num_groups) or pred.shape[1] not in (1, num_times):
            raise ValueError(
                f"`{self._label}`'s `{name}` output should have shape (num_groups, num_times, {num_cols}) or "
                f"(num_groups, {num_cols}), i.e. ({num_groups}, {num_times}, {num_cols}); got {tuple(pred.shape)}."
            )
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise RuntimeError(f"`{self._label}`'s `{name}` produced nans/infs")
        return pred

    def _embed(self, mini_cov: Tensor) -> Tensor:
        if self.param_rank == self.rank:
            return mini_cov
        return self.mask @ mini_cov @ self.mask.t()

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

        # with a masked factor, this is redundant but makes the zeros exact:
        mini_cov = self._apply_corr_mask(mini_cov)

        if is_near_zero(mini_cov.diagonal(dim1=-2, dim2=-1), atol=1e-12).any():
            warn(
                f"`{self._label}` has near-zero along the diagonal. Will add 1e-12 to the diagonal. "
                f"Values:\n{mini_cov.diag()}"
            )
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1], device=mini_cov.device, dtype=mini_cov.dtype) * 1e-12
        return mini_cov

    def _apply_corr_mask(self, mini_cov: Tensor) -> Tensor:
        # (``getattr``: modules pickled before ``corr_mask`` existed don't have the buffer)
        corr_mask = getattr(self, 'corr_mask', None)
        if corr_mask is None:
            return mini_cov
        idx = self.non_empty_idx
        return mini_cov * corr_mask[idx][:, idx]

    def _get_off_diag(self) -> Tensor:
        """
        The off-diagonal entries for the cholesky factor (the correlation's, for "sd_corr"), in ``tril_indices``
        order: ``cholesky_off_diag``/``corr_unconstrained`` itself or, with a binary ``corr_mask``, its entries
        scattered into place, with zeros for masked-out pairs.
        """
        off_diag = self.cholesky_off_diag if self.method == 'log_cholesky' else self.corr_unconstrained
        assert off_diag is not None
        keep = self._get_off_diag_keep()
        return off_diag if keep is None else _unpack_off_diag(off_diag, keep)

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
        before any ``predict_std``/``predict_corr``). Not used by torchcast internally, but can be useful if (e.g.) you
        want to set up an LKJ prior for your model:
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
        Set this module's parameters (in-place) so that its covariance (before any ``predict_std``/``predict_corr``)
        equals ``cov``.

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
        Create a :class:`.Covariance` whose covariance (before any ``predict_std``/``predict_corr``) equals ``cov``.

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
        ``model.initial_covariance = model.initial_covariance.to_method('sd_corr')``. Other attributes (``role``,
        ``element_names``, ``empty_idx``, ``predict_std``, ``predict_corr``) are carried over. Note that a model's
        state-dict saved after converting can only be loaded into a model created with the new ``method``.
        """
        if self.predict_corr is not None and method != 'sd_corr':
            raise ValueError(f"``predict_corr`` requires ``method='sd_corr'`` (got '{method}').")
        with warnings.catch_warnings():
            # (deep-copying unpickles, which warns for a 'low_rank' covariance -- the one you'd be converting away from)
            warnings.filterwarnings('ignore', message=".*method='low_rank'", category=FutureWarning)
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
        # (older versions called ``predict_std`` ``var_predict_module``:)
        for k in [k for k in state_dict if k.startswith(f'{prefix}var_predict_module.')]:
            state_dict[k.replace(f'{prefix}var_predict_module.', f'{prefix}predict_std.', 1)] = state_dict.pop(k)
        # a state-dict saved with a different ``method`` would otherwise fail with a generic missing/unexpected-keys
        # error (or, for ``log_std_devs``, which 'low_rank' and 'sd_corr' share, partially load):
        present = {k[len(prefix):] for k in state_dict if k.startswith(prefix)}
        expected = set(_METHOD_PARAMS.get(self.method, ()))
        if not expected <= present:
            saved_with = [m for m, nms in _METHOD_PARAMS.items() if m != self.method and set(nms) <= present]
            if saved_with:
                saved = saved_with[0]
                label = self._label
                if saved == 'low_rank':
                    msg = (
                        f"`{label}`: the state-dict was saved with method='low_rank', which torchcast 2.0 no longer "
                        f"offers for new models (it was the default for a measure-covariance with more than 5 "
                        f"measures). To keep the fitted covariance, unpickle the saved model if you have it (pickled "
                        f"'low_rank' covariances still load), or load the state-dict with an older torchcast and "
                        f"convert it with ``.to_method('{self.method}')``."
                    )
                else:
                    msg = (
                        f"`{label}` uses method='{self.method}', but the state-dict was saved with method='{saved}'. "
                        f"To load it, create this covariance with ``method='{saved}'`` (e.g. "
                        f"``Covariance.from_processes(..., method='{saved}')`` or ``Covariance.from_measures(..., "
                        f"method='{saved}')``), then, if you like, convert it with ``.to_method('{self.method}')``."
                    )
                error_msgs.append(msg)
                return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __setstate__(self, state: dict):
        # unpickling a module saved by an older version (``var_predict_module``, ``expected_kwargs``):
        modules = state.get('_modules', {})
        if 'var_predict_module' in modules:
            modules['predict_std'] = modules.pop('var_predict_module')
        modules.setdefault('predict_std', None)
        modules.setdefault('predict_corr', None)
        if 'predictor_kwargs' not in state:
            state['predictor_kwargs'] = [
                ProcessKwarg(name=k, is_group_time_tensor=True) for k in (state.pop('expected_kwargs', None) or [])
            ]
        # (``id`` is now derived from ``role``)
        if 'role' not in state:
            old_id = state.pop('id', None) or ''
            role = old_id[:-len('_covariance')] if old_id.endswith('_covariance') else None
            state['role'] = role if role in _HOW_TO_CREATE else None
        state.setdefault('element_names', None)
        super().__setstate__(state)
        if self.method == 'low_rank':
            warn(
                f"`{self._label}` uses method='low_rank', which is deprecated and no longer available for new models; "
                f"convert it (exactly) with ``.to_method('log_cholesky')``.",
                FutureWarning
            )

    # Indexing the parameters ------------------------------------------------------------------------------------------

    def get_param_idx(self, i: int) -> int:
        """
        The index of element ``i`` (of ``0..rank-1``) in the per-element parameters (``log_std_devs`` for
        "sd_corr", ``cholesky_log_diag`` for "log_cholesky"), accounting for ``empty_idx``. E.g., to
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

    # Deprecated -------------------------------------------------------------------------------------------------------

    @staticmethod
    def log_chol_to_chol(log_diag: Tensor, off_diag: Tensor) -> Tensor:
        """Deprecated: use :func:`torchcast.covariance.log_chol_to_chol`."""
        warn(
            "``Covariance.log_chol_to_chol()`` is deprecated, use ``torchcast.covariance.log_chol_to_chol()``.",
            FutureWarning
        )
        return log_chol_to_chol(log_diag, off_diag)


def _make_covariance(spec: Union[Covariance, dict, None],
                    role: str,
                    processes: Optional[Sequence[Process]] = None,
                    measures: Optional[Sequence[str]] = None,
                    **kwargs) -> Covariance:
    """
    How models create a covariance from their ``{role}_covariance`` argument: a :class:`.Covariance` is returned
    as-is (the model then checks its ``role`` and ``element_names``); a dict of options (or ``None``, for the
    defaults) is passed to :func:`Covariance.from_measures` (for ``role='measure'``) or
    :func:`Covariance.from_processes` (with ``cov_type=role``), with the model's ``measures`` / ``processes``.

    :param spec: A :class:`.Covariance`, a dict of options, or ``None``.
    :param role: "process", "initial" or "measure".
    :param processes: The model's processes (for "process" and "initial").
    :param measures: The model's measures (for "measure").
    :param kwargs: Options the model sets (e.g. ``BinomialFilter``'s ``empty_idx`` for its binary measures).
    """
    if isinstance(spec, Covariance):
        return spec
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(f"``{role}_covariance`` should be a ``Covariance``, a dict of options, or None; got {spec!r}.")
    _validate_options(spec, role=role, model_sets=set(kwargs))
    if role == 'measure':
        return Covariance.from_measures(list(measures), **spec, **kwargs)
    return Covariance.from_processes(list(processes), cov_type=role, **spec, **kwargs)


# options a model's covariance-dict can't include: the model sets them, or they're internal to ``from_processes()``
_MODEL_SETS = {'processes', 'measures', 'cov_type', 'rank', 'role', 'element_names', 'id'}
_INTERNAL = {'self', 'stationary_idx', 'nonstationary_idx'}
# the initial covariance is used before any data, as ``base_cov()``, so predicting it has no effect:
_NOT_FOR_INITIAL = {'predict_std', 'predict_corr', 'predictor_kwargs', 'predict_variance', 'expected_kwargs'}
# accepted (with a deprecation warning), but not suggested:
_DEPRECATED_OPTIONS = {'predict_variance', 'expected_kwargs'}


def _get_options(role: str) -> List[str]:
    """The options a model's ``{role}_covariance`` dict can include, in the order of the factory's signature."""
    factory = Covariance.from_measures if role == 'measure' else Covariance.from_processes
    names = []
    for fun in (factory, Covariance.__init__):
        for nm, param in inspect.signature(fun).parameters.items():
            if param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL) and nm not in names:
                names.append(nm)
    exclude = _MODEL_SETS | _INTERNAL
    if role != 'measure':
        exclude = exclude | {'empty_idx'}  # (from the processes' state-elements)
    if role != 'initial':
        exclude = exclude | {'stationary'}
    if role == 'initial':
        exclude = exclude | _NOT_FOR_INITIAL
    return [nm for nm in names if nm not in exclude]


def _validate_options(spec: dict, role: str, model_sets: Set[str] = frozenset()):
    """Raise for keys in a model's ``{role}_covariance`` dict that aren't options, with what they could be."""
    arg = f"``{role}_covariance``"
    options = [nm for nm in _get_options(role) if nm not in model_sets]
    unknown = [k for k in spec if k not in options]
    options = [nm for nm in options if nm not in _DEPRECATED_OPTIONS]
    if not unknown:
        return
    key = unknown[0]
    if key in _MODEL_SETS or key in model_sets or (key == 'empty_idx' and role != 'measure'):
        raise TypeError(f"{arg} can't include ``{key}``: the model sets it.")
    if role == 'initial' and key in _NOT_FOR_INITIAL:
        raise TypeError(
            f"{arg} can't include ``{key}``: the initial covariance applies before any data, so it isn't predicted."
        )
    if key == 'stationary':
        raise TypeError(f"{arg} can't include ``stationary``: it's only for the ``initial_covariance``.")
    close = difflib.get_close_matches(str(key), options, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    raise TypeError(f"{arg} got an unknown option '{key}'.{hint} Options: {', '.join(options)}.")


def get_state_element_names(processes: Sequence[Process]) -> List[str]:
    """
    The names of the state-elements, in the order of the state vector: ``'{process_id}.{state_element}'``. These are
    the ``element_names`` of a process or initial :class:`.Covariance`.
    """
    return [f'{p.id}.{nm}' for p in processes for nm in p.state_elements]


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


def _unpack_off_diag(packed: Tensor, keep: Tensor) -> Tensor:
    """Scatter ``packed`` (along its last dim) into the positions where ``keep`` is non-zero, with zeros elsewhere."""
    free = keep.nonzero().squeeze(-1).to(packed.device)
    return packed.new_zeros((*packed.shape[:-1], keep.shape[-1])).index_copy(-1, free, packed)


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


def _deprecated_predict_variance(predict_variance: Union[bool, nn.Module, None], kwargs: dict, default_kwarg: str):
    if predict_variance is None:
        return
    warn("``predict_variance`` is deprecated, use ``predict_std``.", FutureWarning, stacklevel=3)
    if 'predict_std' in kwargs:
        raise TypeError("Pass ``predict_std`` or ``predict_variance`` (deprecated), not both.")
    if predict_variance is True:
        predict_variance = Identity()
        if 'expected_kwargs' not in kwargs and 'predictor_kwargs' not in kwargs:
            kwargs['predictor_kwargs'] = [default_kwarg]
    kwargs['predict_std'] = predict_variance


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
