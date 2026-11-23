from typing import Sequence, Optional, Union
from functools import cached_property

import torch
from torchcast.internals.utils import update_tensor, stack_per_time

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
                 time: Union[int, Sequence[int]],
                 mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param mean: A ``(num_groups, state_rank)`` tensor; or, if ``time`` is a sequence, a ``(num_groups,
         len(time), state_rank)`` tensor.
        :param time: The timestep; or a sequence of timesteps, one for each row of ``mean``'s second dimension.
        :param mask: The groups to update (the others keep their ``mean``): None for all of them, otherwise an
         index-tensor.
        :return: The new mean, and the transition-matrix (for a sequence of ``time``, with a time-dimension, see
         ``stack_per_time``).
        """
        if mask is None:
            mask = slice(None)
        if isinstance(time, int):
            assert time >= 0
            F = self.transition_mats[time]
        else:
            F = stack_per_time(self.transition_mats, time)
        new_mean = update_tensor(mean, new=(F[mask] @ mean[mask].unsqueeze(-1)).squeeze(-1), mask=mask)
        return new_mean, F
