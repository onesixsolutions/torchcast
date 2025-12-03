import math

from typing import List, Optional, Sequence, Dict, Union
from warnings import warn

import torch

from torch import Tensor, nn, jit

from torchcast.process.utils import Identity
from torchcast.covariance.util import num_off_diag, mini_cov_mask
from torchcast.internals.utils import is_near_zero, validate_gt_shape
from torchcast.process.process import Process


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

    @classmethod
    def from_processes(cls,
                       processes: Sequence[Process],
                       cov_type: str = 'process',
                       predict_variance: Union[bool, nn.Module] = None,
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
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.StateSpaceModel`.
        """

        assert cov_type in {'process', 'initial'}
        state_rank = 0
        no_cov_idx = []
        for p in processes:
            no_cov_elements = [nm for nm, se in p.state_elements.items() if not getattr(se, f'has_{cov_type}_variance')]
            for i, se in enumerate(p.state_elements):
                if se in no_cov_elements:
                    no_cov_idx.append(state_rank + i)
            state_rank += len(p.state_elements)

        if cov_type == 'process':
            # by default, assume process cov is less than measure cov:
            if 'init_diag_multi' not in kwargs:
                kwargs['init_diag_multi'] = .05
        elif cov_type != 'initial':
            raise ValueError(f"Unrecognized cov_type {cov_type}, expected 'initial' or 'process'.")

        if predict_variance is True:
            predict_variance = Identity()
            if 'expected_kwargs' not in kwargs:
                kwargs['expected_kwargs'] = [f'{cov_type}_var_multi']

        return cls(
            rank=state_rank,
            empty_idx=no_cov_idx,
            id=f'{cov_type}_covariance',
            predict_variance=predict_variance,
            **kwargs
        )

    @classmethod
    def from_measures(cls,
                      measures: Sequence[str],
                      predict_variance: Union[bool, nn.Module] = None,
                      **kwargs) -> 'Covariance':
        """
        :param measures: The ``measures`` used in your :class:`.StateSpaceModel`.
        :param predict_variance: Will the variance be predicted upon calling ``forward()``? This is implemented as a
         multiplier on the base variance given from the 'method'. You can either pass ``True`` in which case it is
         expected you will pass multipliers as 'measure_var_multi' when ``forward()`` is called; *or* you can pass a
         :class:`torch.nn.Module` that will predict the multipliers, in which case you'll pass input(s) to this
         module at forward. Either way please note these should output strictly positive values with shape
         ``(num_groups, num_times, len(measures))``.
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.StateSpaceModel`.
        """
        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        elif not isinstance(measures[0], str):
            # not good duck-typing, but too easy to accidentally pass dataset.measures instead of dataset.measures[0]
            raise RuntimeError(f"`measures[0]` is {type(measures[0])}, expected str")
        if 'method' not in kwargs and len(measures) > 5:
            kwargs['method'] = 'low_rank'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = 1.0

        if predict_variance is True:
            predict_variance = Identity()
            if 'expected_kwargs' not in kwargs:
                kwargs['expected_kwargs'] = [f'measure_var_multi']

        return cls(rank=len(measures), id='measure_covariance', predict_variance=predict_variance, **kwargs)

    def __init__(self,
                 rank: int,
                 method: str = 'log_cholesky',
                 empty_idx: List[int] = (),
                 predict_variance: Optional[nn.Module] = None,
                 expected_kwargs: Optional[Sequence[str]] = None,
                 id: Optional[str] = None,
                 init_diag_multi: float = 0.1):
        """
        You should rarely call this directly. Instead, call :func:`Covariance.from_measures` and
        :func:`Covariance.from_processes`.

        :param rank: The number of elements along the diagonal.
        :param method: The parameterization for the covariance. The default, "log_cholesky", parameterizes the
         covariance using the cholesky factorization (which is itself split into two tensors: the log-transformed
         diagonal elements and the off-diagonal). The other currently supported option is "low_rank", which
         parameterizes the covariance with two tensors: (a) the log-transformed std-deviations, and (b) a 'low rank'
         G*K tensor where G is the number of random-effects and K is int(sqrt(G)). Then the covariance is
         ``D + V @ V.t()`` where D is a diagonal-matrix with the std-deviations**2, and V is the low-rank tensor.
        :param empty_idx: In some cases (e.g. process-covariance) we will have some elements with no variance.
        :param predict_variance: For predicting variance, see :func:`Covariance.from_measures`.
        :param expected_kwargs: If ``predict_variance`` is set, this allows you to set the keyword that will be passed
         at ``forward()``.
        :param id: Identifier for this covariance. Typically left ``None`` and set when passed to the
         :class:`.StateSpaceModel`.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
         This can be useful to provide intelligent starting-values to speed up optimization.
        """

        super().__init__()

        self.id = id
        self.rank = rank

        self.empty_idx = set(empty_idx)
        assert all(isinstance(x, int) for x in self.empty_idx)
        self.param_rank = self.rank - len(self.empty_idx)
        mask = mini_cov_mask(rank=self.rank, empty_idx=self.empty_idx)
        self.register_buffer('mask', mask)

        self._set_params(method, init_diag_multi)

        self.var_predict_module = predict_variance

        if self.var_predict_module and expected_kwargs is None:
            raise ValueError("Please explicitly specify ``expected_kwargs`` if passing ``predict_variance``.")
        if isinstance(expected_kwargs, str):
            expected_kwargs = [expected_kwargs]
        self.expected_kwargs: Optional[List[str]] = None if expected_kwargs is None else list(expected_kwargs)

    @property
    def non_empty_idx(self) -> List[int]:
        return [i for i in range(self.rank) if i not in self.empty_idx]

    def _set_params(self, method: str, init_diag_multi: float):
        self.cholesky_log_diag: Optional[nn.Parameter] = None
        self.cholesky_off_diag: Optional[nn.Parameter] = None
        self.lr_mat: Optional[nn.Parameter] = None
        self.log_std_devs: Optional[nn.Parameter] = None
        if method == 'log_cholesky':
            self.method = method
            self.cholesky_log_diag = nn.Parameter(.1 * torch.randn(self.param_rank) + math.log(init_diag_multi))
            self.cholesky_off_diag = nn.Parameter(.1 * torch.randn(num_off_diag(self.param_rank)))
        elif method.startswith('low_rank'):
            warn("``method='low_rank'`` is experimental")
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                low_rank = int(math.sqrt(self.param_rank))
            self.lr_mat = nn.Parameter(data=.01 * torch.randn(self.param_rank, low_rank))
            self.log_std_devs = nn.Parameter(data=.1 * torch.randn(self.param_rank) + math.log(init_diag_multi))
        else:
            raise NotImplementedError(method)

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
