from functools import cached_property

from typing import TYPE_CHECKING

import torch

from torchcast.process.utils import process2slice

if TYPE_CHECKING:
    from torchcast.process import Process


class DesignModel:
    def __init__(self,
                 processes: torch.nn.ModuleDict,
                 num_groups: int,
                 num_timesteps: int):
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.processes: dict[str, 'Process'] = processes

    @property
    def device(self) -> torch.device:
        device = None
        for param in self.processes.parameters():
            if device is None:
                device = param.device
            elif device != param.device:
                raise RuntimeError("Multiple devices!")
        return device

    @property
    def dtype(self) -> torch.dtype:
        dtype = None
        for param in self.processes.parameters():
            if dtype is None:
                dtype = param.dtype
            elif dtype != param.dtype:
                raise RuntimeError("Multiple dtypes!")
        return dtype

    def __call__(self,
                 mean: torch.Tensor,
                 time: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @cached_property
    def state_rank(self) -> int:
        return sum(p.rank for p in self.processes.values())

    @cached_property
    def process2slice(self) -> dict[str, slice]:
        """
        Returns a mapping from process id to the slice of the state vector that contains its state elements.
        """
        return process2slice(self.processes)

