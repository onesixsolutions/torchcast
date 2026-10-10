import copy
import math
import sys
import types

from typing import Any, List, Mapping, Optional, Sequence, Set, Dict, Tuple, Union, Collection
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
      pair of elements, in the order given by :func:`Covariance.off_diag_idx`. Initializing it to output zeros starts
      it at the base correlations.

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
    need. A kwarg named ``{id}__{name}`` (e.g. ``measure_covariance__season_X``) overrides ``name`` for this
    covariance.
    """

    @classmethod
    def from_processes(cls,
                       processes: Sequence[Process],
                       cov_type: str = 'process',
                       predict_variance: Union[bool, nn.Module, None] = None,
                       **kwargs) -> 'Covariance':
        """
        :param processes: The ``processes`` used in your :class:`.StateSpaceModel`.
        :param cov_type: The type of covariance, either 'process' or 'initial' (default: 'process').
        :param predict_variance: Deprecated, use ``predict_std``.
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`, e.g. ``predict_std``,
         ``predict_corr``, ``predictor_kwargs``. Note that the outputs of a ``predict_std`` module have
         ``param_rank`` columns, i.e. one for each state-element with a variance (not ``empty_idx``).
        :return: A :class:`.Covariance` object that can be used in your :class:`.StateSpaceModel`.
        """
        if cov_type not in {'initial', 'process'}:
            raise ValueError(f"Unrecognized cov_type {cov_type}, expected 'initial' or 'process'.")
        state_rank = 0
        no_cov_idx = []
        for p in processes:
            no_cov_elements = [nm for nm, se in p.state_elements.items() if not getattr(se, f'has_{cov_type}_variance')]
            for i, se in enumerate(p.state_elements):
                if se in no_cov_elements:
                    no_cov_idx.append(state_rank + i)
            state_rank += len(p.state_elements)

        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = DEFAULT_PCOV_MULTI if cov_type == 'process' else DEFAULT_ICOV_MULTI
        _deprecated_predict_variance(predict_variance, kwargs, default_kwarg=f'{cov_type}_var_multi')

        return cls(rank=state_rank, empty_idx=no_cov_idx, id=f'{cov_type}_covariance', **kwargs)

    @classmethod
    def from_measures(cls,
                      measures: Sequence[str],
                      predict_variance: Union[bool, nn.Module, None] = None,
                      **kwargs) -> 'Covariance':
        """
        :param measures: The ``measures`` used in your :class:`.StateSpaceModel`.
        :param predict_variance: Deprecated, use ``predict_std``.
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
        if 'method' not in kwargs:
            if kwargs.get('predict_corr') is not None:
                kwargs['method'] = 'sd_corr'
            elif len(measures) > 5:
                kwargs['method'] = 'low_rank'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = DEFAULT_MCOV_MULTI
        _deprecated_predict_variance(predict_variance, kwargs, default_kwarg='measure_var_multi')

        return cls(rank=len(measures), id='measure_covariance', **kwargs)

    def __init__(self,
                 rank: int,
                 init_diag_multi: float,
                 method: Optional[str] = None,
                 empty_idx: List[int] = (),
                 predict_std: Optional[nn.Module] = None,
                 predict_corr: Optional[nn.Module] = None,
                 predictor_kwargs: Sequence[Union[str, ProcessKwarg]] = (),
                 id: Optional[str] = None,
                 predict_variance: Optional[nn.Module] = None,
                 expected_kwargs: Optional[Sequence[str]] = None):
        """
        You should rarely call this directly. Instead, call :func:`Covariance.from_measures` and
        :func:`Covariance.from_processes`.

        :param rank: The number of elements along the diagonal.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
         This can be useful to provide intelligent starting-values to speed up optimization.
        :param method: The parameterization for the covariance. The default, "log_cholesky" (or "sd_corr" if
         ``predict_corr`` is passed), parameterizes the
         covariance using the cholesky factorization (which is itself split into two tensors: the log-transformed
         diagonal elements and the off-diagonal). The other currently supported option is "low_rank", which
         parameterizes the covariance with two tensors: (a) the log-transformed std-deviations, and (b) a 'low rank'
         G*K tensor where G is the number of random-effects and K is int(sqrt(G)). Then the covariance is
         ``D + V @ V.t()`` where D is a diagonal-matrix with the std-deviations**2, and V is the low-rank tensor.
         Finally, "sd_corr" separates scale from correlation: ``diag(std) @ R @ diag(std)``, with parameters
         ``log_std_devs`` and ``corr_unconstrained``. ``R`` is a correlation matrix built from canonical partial
         correlations ``tanh(corr_unconstrained)`` (as in Stan's ``cholesky_factor_corr``), one per off-diagonal
         element. Unlike "log_cholesky" (where off-diagonal entries are in absolute units, and also change the
         variances), each parameter is dimensionless and the variances are set by ``log_std_devs`` alone; this is
         better conditioned for optimization when elements have very different scales, and lets you freeze variances
         and correlations separately (see :func:`Covariance.param_idx` and :func:`Covariance.off_diag_idx`). Note
         that a zero ``corr_unconstrained`` is a zero *partial* correlation (given the earlier elements): element
         ``i`` is uncorrelated with all others when every entry in its row *and* column is zero. Existing modules
         can be converted with :func:`Covariance.to_method`.
        :param empty_idx: In some cases (e.g. process-covariance) we will have some elements with no variance.
        :param predict_std: A module predicting (strictly positive) multipliers on the std-devs, with ``param_rank``
         columns. See the class docstring.
        :param predict_corr: A module predicting additive adjustments to ``corr_unconstrained``, with
         ``param_rank * (param_rank - 1) / 2`` columns. Requires ``method='sd_corr'``. See the class docstring.
        :param predictor_kwargs: The names of the keyword-arguments passed to the model's ``forward()`` that are
         inputs to ``predict_std``/``predict_corr``. A string is a tensor whose first dimension is the group (e.g.
         ``group_ids``, or a ``(num_groups, num_times, ...)`` tensor), which ``fit(chunk_size=...)`` splits into
         chunks; for inputs that aren't group-indexed, pass ``ProcessKwarg(name, is_group_time_tensor=False)``.
        :param id: Identifier for this covariance. Typically left ``None`` and set when passed to the
         :class:`.StateSpaceModel`.
        :param predict_variance: Deprecated, use ``predict_std``.
        :param expected_kwargs: Deprecated, use ``predictor_kwargs``.
        """

        super().__init__()

        if predict_variance is not None:
            warn("``predict_variance`` is deprecated, use ``predict_std``.", DeprecationWarning, stacklevel=2)
            if predict_std is not None:
                raise TypeError("Pass ``predict_std`` or ``predict_variance`` (deprecated), not both.")
            predict_std = predict_variance
        if expected_kwargs is not None:
            warn("``expected_kwargs`` is deprecated, use ``predictor_kwargs``.", DeprecationWarning, stacklevel=2)
            if predictor_kwargs:
                raise TypeError("Pass ``predictor_kwargs`` or ``expected_kwargs`` (deprecated), not both.")
            predictor_kwargs = expected_kwargs

        if method is None:
            method = 'sd_corr' if predict_corr is not None else 'log_cholesky'
        if predict_corr is not None and method != 'sd_corr':
            raise ValueError(f"``predict_corr`` requires ``method='sd_corr'`` (got '{method}').")

        self.id = id
        self.rank = rank

        self.empty_idx = set(empty_idx)
        assert all(isinstance(x, int) for x in self.empty_idx)
        self.param_rank = self.rank - len(self.empty_idx)
        mask = mini_cov_mask(rank=self.rank, empty_idx=self.empty_idx)
        self.register_buffer('mask', mask)

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

    @property
    def expected_kwargs(self) -> Optional[List[str]]:
        warn("``expected_kwargs`` is deprecated, use ``predictor_kwargs``.", DeprecationWarning, stacklevel=2)
        return [k.name for k in self.predictor_kwargs] or None

    @property
    def is_predicted(self) -> bool:
        return self.predict_std is not None or self.predict_corr is not None

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
        super().__setstate__(state)

    @property
    def non_empty_idx(self) -> List[int]:
        return [i for i in range(self.rank) if i not in self.empty_idx]

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
            self.cholesky_off_diag = nn.Parameter(.1 * torch.randn(num_off_diag(self.param_rank), **fkw))
        elif method == 'sd_corr':
            self.method = method
            self.log_std_devs = nn.Parameter(.1 * torch.randn(self.param_rank, **fkw) + math.log(init_diag_multi))
            self.corr_unconstrained = nn.Parameter(.1 * torch.randn(num_off_diag(self.param_rank), **fkw))
        elif method.startswith('low_rank'):
            warn("``method='low_rank'`` is experimental")
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                low_rank = int(math.sqrt(self.param_rank))
            self.lr_mat = nn.Parameter(data=.01 * torch.randn(self.param_rank, low_rank, **fkw))
            self.log_std_devs = nn.Parameter(
                data=.1 * torch.randn(self.param_rank, **fkw) + math.log(init_diag_multi)
            )
        else:
            raise NotImplementedError(method)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # a state-dict saved with a different ``method`` would otherwise fail with a generic missing/unexpected-keys
        # error (or, for ``log_std_devs``, which 'low_rank' and 'sd_corr' share, partially load):
        # (older versions called ``predict_std`` ``var_predict_module``:)
        for k in [k for k in state_dict if k.startswith(f'{prefix}var_predict_module.')]:
            state_dict[k.replace(f'{prefix}var_predict_module.', f'{prefix}predict_std.', 1)] = state_dict.pop(k)
        present = {k[len(prefix):] for k in state_dict if k.startswith(prefix)}
        expected = set(_METHOD_PARAMS.get(self.method, ()))
        if not expected <= present:
            saved_with = [m for m, nms in _METHOD_PARAMS.items() if m != self.method and set(nms) <= present]
            if saved_with:
                error_msgs.append(
                    f"`{self.id or type(self).__name__}` uses method='{self.method}', but the state-dict was saved "
                    f"with method='{saved_with[0]}'. Load it into a module with method='{saved_with[0]}', then "
                    f"convert with ``.to_method('{self.method}')``."
                )
                return
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @staticmethod
    def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
        assert log_diag.shape[:-1] == off_diag.shape[:-1]

        rank = log_diag.shape[-1]
        L1 = torch.diag_embed(torch.exp(log_diag))

        L2 = torch.zeros_like(L1)
        mask = torch.tril_indices(rank, rank, offset=-1)
        L2[mask[0], mask[1]] = off_diag
        return L1 + L2

    def _get_mini_cov(self) -> Tensor:
        if self.method == 'log_cholesky':
            assert self.cholesky_log_diag is not None
            assert self.cholesky_off_diag is not None
            L = self.log_chol_to_chol(self.cholesky_log_diag, self.cholesky_off_diag)
            mini_cov = L @ L.t()
        elif self.method == 'sd_corr':
            assert self.log_std_devs is not None
            assert self.corr_unconstrained is not None
            L = self.log_std_devs.exp().unsqueeze(-1) * corr_cholesky_from_unconstrained(
                self.corr_unconstrained, rank=self.param_rank
            )
            mini_cov = L @ L.t()
        elif self.method == 'low_rank':
            assert self.lr_mat is not None
            assert self.log_std_devs is not None
            mini_cov = (
                    self.lr_mat @ self.lr_mat.t() +
                    torch.diag_embed(self.log_std_devs.exp() ** 2)
            )
        else:
            raise NotImplementedError(self.method)

        if is_near_zero(mini_cov.diagonal(dim1=-2, dim2=-1), atol=1e-12).any():
            warn(
                f"`{self.id}` has near-zero along the diagonal. Will add 1e-12 to the diagonal. "
                f"Values:\n{mini_cov.diag()}"
            )
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1], device=mini_cov.device, dtype=mini_cov.dtype) * 1e-12
        return mini_cov

    def corr_cholesky(self) -> Tensor:
        """
        The cholesky factor of the correlation matrix (of the ``param_rank`` block, i.e. excluding ``empty_idx``, and
        before any ``predict_variance``). E.g. for an LKJ prior:
        ``torch.distributions.LKJCholesky(cov.param_rank, eta).log_prob(cov.corr_cholesky())``.
        """
        if self.method == 'sd_corr':
            return corr_cholesky_from_unconstrained(self.corr_unconstrained, rank=self.param_rank)
        mini_cov = self._get_mini_cov()
        std = mini_cov.diagonal(dim1=-2, dim2=-1).sqrt()
        return torch.linalg.cholesky(mini_cov / (std.unsqueeze(-1) * std.unsqueeze(-2)))

    @torch.no_grad()
    def set_matrix_(self, cov: Tensor) -> 'Covariance':
        """
        Set this module's parameters (in-place) so that its covariance (before any ``predict_variance``) equals
        ``cov``. Supported for "log_cholesky" and "sd_corr" (a "low_rank" covariance can't represent an arbitrary
        matrix).

        :param cov: A ``(rank, rank)`` covariance matrix, whose rows/columns in ``empty_idx`` are zero; or just the
         ``(param_rank, param_rank)`` block for the non-empty elements.
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
        # converting in float64 so that the round-trip is exact up to the module's dtype (on cpu: no float64 on mps;
        # ``copy_`` below moves the results back):
        mini_cov = mini_cov.cpu().to(torch.float64)
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
        for name, value in zip(_METHOD_PARAMS[self.method], values):
            getattr(self, name).copy_(value)
        return self

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
        if self.predict_corr is not None and method != 'sd_corr':
            raise ValueError(f"``predict_corr`` requires ``method='sd_corr'`` (got '{method}').")
        out = copy.deepcopy(self)
        out._set_params(method, init_diag_multi=1.0)
        return out.set_matrix_(self._get_mini_cov().detach())

    def param_idx(self, i: int) -> int:
        """
        The index of element ``i`` (of ``0..rank-1``) in the per-element parameters (``log_std_devs`` for
        "sd_corr"/"low_rank", ``cholesky_log_diag`` for "log_cholesky"), accounting for ``empty_idx``. E.g., to
        freeze some elements' std-devs and correlations (the frozen entries get exactly zero gradients):

        .. code-block:: python3

            cov = Covariance.from_processes(processes, cov_type='initial', method='sd_corr')
            keep_std = torch.ones_like(cov.log_std_devs)
            keep_std[cov.param_idx(3)] = 0
            cov.log_std_devs.register_hook(lambda g: g * keep_std)
            keep_corr = torch.ones_like(cov.corr_unconstrained)
            keep_corr[cov.off_diag_idx(3, 0)] = 0
            cov.corr_unconstrained.register_hook(lambda g: g * keep_corr)

        For "sd_corr", these entries are interpretable on their own; for "log_cholesky" they're entries of the
        cholesky factor, which mix scale and correlation.
        """
        if i in self.empty_idx:
            raise ValueError(f"Element {i} is in ``empty_idx``, so it has no parameters.")
        if not 0 <= i < self.rank:
            raise IndexError(f"Element {i} out of range for rank {self.rank}.")
        return self.non_empty_idx.index(i)

    def off_diag_idx(self, i: int, j: int) -> int:
        """
        The index of the pair of elements ``(i, j)`` (in either order, of ``0..rank-1``) in the off-diagonal
        parameters (``corr_unconstrained`` for "sd_corr", ``cholesky_off_diag`` for "log_cholesky"), accounting for
        ``empty_idx``. See :func:`Covariance.param_idx`.
        """
        row, col = sorted((self.param_idx(i), self.param_idx(j)), reverse=True)
        if row == col:
            raise ValueError("``i`` and ``j`` must be different elements.")
        return row * (row - 1) // 2 + col

    def base_cov(self) -> Tensor:
        """
        The ``(rank, rank)`` covariance, before any ``predict_std`` / ``predict_corr``.
        """
        return self._embed(self._get_mini_cov())

    def _embed(self, mini_cov: Tensor) -> Tensor:
        if self.param_rank == self.rank:
            return mini_cov
        return self.mask @ mini_cov @ self.mask.t()

    def get_predictor_inputs(self, kwargs: Mapping[str, Any]) -> Tuple[List[Tensor], Set[str]]:
        """
        Pick out the inputs to ``predict_std`` / ``predict_corr`` from a model's ``forward()`` keyword-arguments.

        :return: The inputs (in the order of ``predictor_kwargs``), and the set of keys that were used.
        """
        inputs, used = [], set()
        for pkwarg in self.predictor_kwargs:
            key = f'{self.id}__{pkwarg.name}'
            if key not in kwargs:
                key = pkwarg.name
            if key not in kwargs:
                raise TypeError(
                    f"`{self.id}` expected a `{pkwarg.name}` (or `{self.id}__{pkwarg.name}`) keyword-argument."
                )
            inputs.append(kwargs[key])
            used.add(key)
        return inputs, used

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
        return self._predicted_cov(inputs, num_groups=num_groups, num_times=num_times), used

    def _predicted_cov(self, inputs: Sequence[Tensor], num_groups: int, num_times: int) -> Tensor:
        std_multi = corr_delta = None
        if self.predict_std is not None:
            std_multi = self._validate_prediction(
                self.predict_std(*inputs), 'predict_std', self.param_rank, num_groups, num_times
            )
            if (std_multi < 0).any():
                raise RuntimeError(f"{self.id}'s `predict_std` produced values <0; needs exp/softplus layer.")
        if self.predict_corr is not None:
            corr_delta = self._validate_prediction(
                self.predict_corr(*inputs), 'predict_corr', num_off_diag(self.param_rank), num_groups, num_times
            )

        if corr_delta is None:
            mini_cov = self._get_mini_cov() * std_multi.unsqueeze(-1) * std_multi.unsqueeze(-2)
        else:
            std = self.log_std_devs.exp()
            if std_multi is not None:
                std = std * std_multi
            L = std.unsqueeze(-1) * corr_cholesky_from_unconstrained(
                self.corr_unconstrained + corr_delta, rank=self.param_rank
            )
            mini_cov = L @ L.transpose(-1, -2)
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
                raise ValueError(f"`{self.id}`'s `{name}` should output {num_cols} columns, got shape {pred.shape}.")
        if pred.ndim == 1:
            pred = pred[None, None]
        elif pred.ndim == 2:
            pred = pred.unsqueeze(1)
        if pred.ndim != 3 or pred.shape[0] not in (1, num_groups) or pred.shape[1] not in (1, num_times):
            raise ValueError(
                f"`{self.id}`'s `{name}` output should have shape (num_groups, num_times, {num_cols}) or "
                f"(num_groups, {num_cols}), i.e. ({num_groups}, {num_times}, {num_cols}); got {tuple(pred.shape)}."
            )
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise RuntimeError(f"`{self.id}`'s `{name}` produced nans/infs")
        return pred

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
            cov = self._predicted_cov(inputs, num_groups=num_groups, num_times=num_times)
        else:
            cov = self.base_cov()[None, None]
        return cov.expand(num_groups, num_times, -1, -1)


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


def _log_cosh(x: Tensor) -> Tensor:
    # stable for large |x|, and smooth (exact second derivative at 0, unlike formulas using abs(x))
    return x + F.softplus(-2. * x) - math.log(2.)


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
    std_ = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    # TODO: cov / std_.unsqueeze(-1) / std_.unsqueeze(-2)
    return cov / (std_.unsqueeze(-1) @ std_.unsqueeze(-2))


def mini_cov_mask(rank: int, empty_idx: Collection[int], **kwargs) -> Tensor:
    param_rank = rank - len(empty_idx)
    mask = torch.zeros((rank, param_rank), **kwargs)
    c = 0
    for r in range(rank):
        if r not in empty_idx:
            mask[r, c] = 1.
            c += 1
    return mask


def _deprecated_predict_variance(predict_variance: Union[bool, nn.Module, None], kwargs: dict, default_kwarg: str):
    if predict_variance is None:
        return
    warn("``predict_variance`` is deprecated, use ``predict_std``.", DeprecationWarning, stacklevel=3)
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
        if name != '__file__':
            warn(
                f"`torchcast.covariance.base.{name}` is deprecated, instead just import from "
                f"`torchcast.covariance.{name}`.",
                DeprecationWarning,
                stacklevel=2
            )
        return globals()[name]


_base = _DeprecatedBase('covariance.base')
sys.modules['torchcast.covariance.base'] = _base
