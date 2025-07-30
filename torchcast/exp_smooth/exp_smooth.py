"""
The :class:`.ExpSmoother` is a :class:`torch.nn.Module` which generates forecasts using exponential smoothing.

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.
"""
from typing import Sequence, Optional

import torch
from torch import Tensor

from torchcast.exp_smooth.smoothing_matrix import SmoothingMatrix
from torchcast.covariance import Covariance
from torchcast.internals.utils import update_tensor
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel


class ExpSmoother(StateSpaceModel):
    def __init__(self,
                 processes: Sequence[Process],
                 measures: Sequence[str],
                 measure_covariance: Optional[Covariance] = None,
                 smoothing_matrix: Optional[SmoothingMatrix] = None):

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance
        )
        if smoothing_matrix is None:
            smoothing_matrix = SmoothingMatrix.from_measures_and_processes(measures=measures, processes=processes)
        self.smoothing_matrix = smoothing_matrix.set_id('smoothing_matrix')

    def initial_covariance(self, inputs: dict, num_groups: int, num_times: int, _ignore_input: bool = False) -> Tensor:
        # initial covariance is always zero. this will be replaced by the 1-step-ahead covariance in the first call to
        # predict
        ms = self._get_measure_scaling()
        return torch.zeros((num_groups, num_times, self.state_rank, self.state_rank), dtype=ms.dtype, device=ms.device)

    @property
    def _update_mat_dims(self) -> dict:
        return super()._update_mat_dims | {'K': (-2,)}

    def _parse_kwargs(self,
                      num_groups: int,
                      num_timesteps: int,
                      measure_covs: Sequence[torch.Tensor],
                      **kwargs) -> tuple[dict[str, Sequence], dict[str, Sequence], set]:
        predict_kwargs, update_kwargs, used_keys = super()._parse_kwargs(
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            measure_covs=measure_covs,
            **kwargs
        )

        # process-variance:
        smat_kwargs = {}
        if self.smoothing_matrix.expected_kwargs:
            smat_kwargs = {k: kwargs[k] for k in self.smoothing_matrix.expected_kwargs}
        used_keys |= set(smat_kwargs)
        Ks = self.smoothing_matrix(smat_kwargs, num_groups=num_groups, num_times=num_timesteps)
        update_kwargs['K'] = Ks.unbind(1)

        if self.smoothing_matrix.expected_kwargs or self.measure_covariance.expected_kwargs:
            predict_kwargs['cov1step'] = Ks @ torch.stack(measure_covs, 1) @ Ks.transpose(-1, -2)
        else:
            K1 = update_kwargs['K'][0]
            measure_cov = measure_covs[0]
            cov1step = K1 @ measure_cov @ K1.transpose(-1, -2)
            predict_kwargs['cov1step'] = [cov1step] * num_timesteps

        return predict_kwargs, update_kwargs, used_keys
    def _update_step(self,
                     input: torch.Tensor,
                     mean: torch.Tensor,
                     cov: torch.Tensor,
                     measured_mean: torch.Tensor,
                     measure_mat: torch.Tensor,
                     measure_cov: torch.Tensor,
                     K: torch.Tensor,
                     **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if kwargs:
            raise TypeError(f"`{type(self).__name__}._update_step()` received unexpected kwargs: {list(kwargs)}")
        resid = input - measured_mean
        new_mean = self._mean_update(mean=mean, K=K, resid=resid)
        # this method doesn't waste compute creating new_cov; then in predict below, cov will be replaced by cov1step
        new_cov = torch.tensor(0.0, dtype=mean.dtype, device=mean.device)
        return new_mean, new_cov

    def _predict_cov(self,
                     cov: torch.Tensor,
                     transition_mat: torch.Tensor,
                     cov1step: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # new_cov will at least be cov1step (see note above in _update_step)
        new_cov = cov1step

        # fastpath: if the call to update returned the zero-dim tensor (see _update above) then we are done
        if len(cov.shape):
            if mask is None or mask.all():
                mask = slice(None)
            F = transition_mat[mask]
            # we'll hit this under two conditions:
            # - this is a >1 step ahead forecast, so we didn't just call update(), but instead of a real cov from a
            #   previous call to predict (and that cov will be at least `cov1step`)
            # - we did just call update(), but some of the cov elements were excluded because `input` was nan. in that
            #   case:
            #   - the excluded elements will have cov!=0, which means the op below will cause uncertainty to increase,
            #     which is what we want (for those group*measures, this is a >1 step ahead forecast).
            #   - the included elements will have cov=0, which means the op below is just new_cov=new_cov, which means
            #     we will use cov1step for those group*measures that were just updated -- which is again what we want.
            new_cov = update_tensor(
                orig=new_cov,
                new=new_cov[mask] + F @ cov[mask] @ F.permute(0, 2, 1),
                mask=mask
            )
        return new_cov
