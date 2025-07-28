from dataclasses import dataclass

from typing import Optional, Union
from warnings import warn

import torch
from torch import Tensor, nn


def standardize_decay(decay: Optional[Union[torch.nn.Module, tuple[float, float]]],
                      lower: float = .95) -> Union[float, torch.nn.Module]:
    if decay:
        if isinstance(decay, bool):
            decay = (lower, 1.00)
        if isinstance(decay, tuple):
            decay = Bounded(*decay)
    else:
        decay = 1.0

    if isinstance(decay, (float, int)):
        assert 0 < decay <= 1.0
        decay = float(decay)
    return decay


@dataclass
class ProcessKwarg:
    name: str
    is_group_time_tensor: bool


class FixedValue(torch.nn.Module):
    """
    Helper class for cases where the user can either input a fixed value or a callable.

    (note: needs to be not just a callable but a module so that it can be stored in a ModuleDict)
    """

    def __init__(self, value: float):
        super().__init__()
        self.value = torch.as_tensor(value)

    def forward(self) -> torch.Tensor:
        return self.value


class StateElement(torch.nn.Module):
    def __init__(self,
                 name: str,
                 measure_multi: Union[float, torch.nn.Module, None],
                 has_process_variance: bool,
                 has_initial_variance: bool = True):
        super().__init__()
        self.name = name
        if not callable(measure_multi) and measure_multi is not None:
            measure_multi = FixedValue(measure_multi)
        self._measure_multi = measure_multi
        self.has_process_variance = has_process_variance
        self.has_initial_variance = has_initial_variance
        self.transitions_to = torch.nn.ModuleDict()
        # self-transitions by default:
        self.set_transition_to(self, multi=1.0)

    @property
    def measure_multi(self) -> torch.nn.Module:
        if self._measure_multi is None:
            # should not hit this because you'd only set measure_multi to None if
            # the parent process has a custom method to construct the measurement-matrix (e.g. linearModel)
            raise RuntimeError(f"StateElement {self.name} has no measure_multi set")
        return self._measure_multi

    def set_transition_to(self, state_element: 'StateElement', multi: Union[float, torch.nn.Module]):
        if not callable(multi):
            multi = FixedValue(multi)
        self.transitions_to[state_element.name] = multi
        return self


class Identity(nn.Module):
    """
    Identity function
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


class NoInputSequential(torch.nn.Sequential):
    """
    Sequential but the first module takes no arguments
    """

    def forward(self) -> torch.Tensor:
        input = None
        for module in self:
            input = module() if input is None else module(input)
        return input


class Bounded(nn.Module):
    """
    Transforms input to fall within `value`, a tuple of (lower, upper)
    """

    def __init__(self, lower: float, upper: float):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.randn(1) * 0.1)
        self.lower = lower
        self.upper = upper

    def forward(self) -> Tensor:
        return torch.sigmoid(self.raw) * (self.upper - self.lower) + self.lower


class Multi(nn.Module):
    """
    Multiplies input by `value`
    """

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input * self.value
