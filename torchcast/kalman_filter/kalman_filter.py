"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts using the full kalman-filtering
algorithm.

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.
"""
from typing import Sequence

from torchcast.covariance import Covariance
from torchcast.internals.utils import update_tensor
from torchcast.process import Process
from torchcast.state_space.state_space import StateSpaceModel

from typing import Optional

import torch


class KalmanFilter(StateSpaceModel):
    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None):

        if initial_covariance is None:
            initial_covariance = Covariance.from_processes(processes, cov_type='initial')

        if process_covariance is None:
            process_covariance = Covariance.from_processes(processes, cov_type='process')

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance,
        )
        self.process_covariance = process_covariance.set_id('process_covariance')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')

    def _predict_cov(self,
                     cov: torch.Tensor,
                     transition_mat: torch.Tensor,
                     Q: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None or mask.all():
            mask = slice(None)
        F = transition_mat[mask]
        Q = Q[mask]

        new_cov = update_tensor(cov, new=(F @ cov[mask] @ F.permute(0, 2, 1) + Q), mask=mask)
        return new_cov

    def _update_step(self,
                     input: torch.Tensor,
                     mean: torch.Tensor,
                     cov: torch.Tensor,
                     measured_mean: torch.Tensor,
                     measure_mat: torch.Tensor,
                     measure_cov: torch.Tensor,
                     **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if kwargs:
            raise TypeError(f"`{type(self).__name__}._update_step()` received unexpected kwargs: {list(kwargs)}")
        resid = input - measured_mean
        K = self._kalman_gain(cov=cov, H=measure_mat, R=measure_cov)
        new_mean = self._mean_update(mean=mean, K=K, resid=resid)
        new_cov = self._covariance_update(cov=cov, K=K, H=measure_mat, R=measure_cov)
        return new_mean, new_cov

    @staticmethod
    def _covariance_update(cov: torch.Tensor, K: torch.Tensor, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
        ikh = I - K @ H
        return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)

    @staticmethod
    def _kalman_gain(cov: torch.Tensor, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        measured_cov = cov @ H.permute(0, 2, 1)
        system_cov = H @ measured_cov + R
        A = system_cov.permute(0, 2, 1)
        B = measured_cov.permute(0, 2, 1)
        Kt = torch.linalg.solve(A, B)
        K = Kt.permute(0, 2, 1)
        return K

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
        pcov_kwargs = {}
        if self.process_covariance.expected_kwargs:
            pcov_kwargs = {k: kwargs[k] for k in self.process_covariance.expected_kwargs}
        used_keys |= set(pcov_kwargs)
        pcov_raw = self.process_covariance(pcov_kwargs, num_groups=num_groups, num_times=num_timesteps)
        measure_scaling = torch.diag_embed(self._get_measure_scaling().unsqueeze(0).unsqueeze(0))
        Qs = measure_scaling @ pcov_raw @ measure_scaling
        predict_kwargs['Q'] = Qs.unbind(1)

        return predict_kwargs, update_kwargs, used_keys
