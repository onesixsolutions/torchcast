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

        zeros = torch.zeros(
            (self.num_groups, self.num_timesteps, self.state_rank, self.state_rank),
            device=self.device,
            dtype=self.dtype
        )
        F = []
        for pid, process in self.processes.items():
            if process.linear_transition:
                pidx = self.process2slice[pid]
                # note: as in other parts, assuming autograd makes it more efficient to create clones then sum vs.
                # repeated masks on the same tensor. should verify that
                thisF = zeros.clone()
                thisF[:, :, pidx, pidx] = process.get_transition_matrix()
                F.append(thisF)
            else:
                raise NotImplementedError
        self._transition_mats = torch.stack(F, dim=0).sum(0)

    @cached_property
    def transition_mats(self) -> Sequence[torch.Tensor]:
        return self._transition_mats.to(device=self.device, dtype=self.dtype).unbind(1)

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
