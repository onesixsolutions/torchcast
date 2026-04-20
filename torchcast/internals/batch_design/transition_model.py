from typing import Sequence, Optional
from functools import cached_property

import torch
from torchcast.internals.utils import update_tensor

from .design_model import DesignModel


class TransitionModel(DesignModel):
    def __init__(self,
                 processes: torch.nn.ModuleDict,
                 measures: Sequence[str],
                 num_groups: int,
                 num_timesteps: int):
        super().__init__(
            processes=processes,
            num_groups=num_groups,
            num_timesteps=num_timesteps
        )
        self.measures = measures

        is_time_varying = False # could be supported in the future
        n_times = self.num_timesteps if is_time_varying else 1

        F = torch.zeros(
            (self.num_groups, n_times, self.state_rank, self.state_rank),
            device=self.device,
            dtype=self.dtype
        )
        for pid, process in self.processes.items():
            if process.linear_transition:
                pidx = self.process2slice[pid]
                F[:, :, pidx, pidx] = process.get_transition_matrix()
            else:
                raise NotImplementedError

        if is_time_varying:
            self.transition_mats = F.unbind(1)
        else:
            # much faster for backward-step:
            F0 = F.squeeze(1)
            self.transition_mats = [F0] * self.num_timesteps

    def __call__(self,
                 mean: torch.Tensor,
                 time: int,
                 mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert time >= 0
        if mask is None or mask.all():
            mask = slice(None)
        F = self.transition_mats[time]
        new_mean = update_tensor(mean, new=(F[mask] @ mean[mask].unsqueeze(-1)).squeeze(-1), mask=mask)
        return new_mean, F
