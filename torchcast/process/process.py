import numpy as np
from typing import Sequence, Optional

import torch

from .utils import StateElement, ProcessKwarg


class Process(torch.nn.Module):
    """
    This is the base class. The process is defined by the state-elements it generates predictions for. It generates
    two kinds of predictions: (1) the measurement matrix, (2) the transition matrix.

    :param id: Unique identifier for the process
    :param state_elements: List of ``StateElement``s.
    :param measure: The name of the measure for this process.
    """

    linear_measurement = True
    linear_transition = True

    def __init__(self,
                 id: str,
                 state_elements: Sequence[StateElement],
                 measure: Optional[str] = None):
        super().__init__()

        self.id = id
        self.state_elements = torch.nn.ModuleDict()
        for state_element in state_elements:
            if state_element.name in self.state_elements:
                raise ValueError(f"duplicate state-element name: {state_element.name} in process {id}")
            self.state_elements[state_element.name] = state_element
        self._measure = measure
        self._device = None

        self.initial_mean = torch.nn.Parameter(torch.randn(self.rank) * 0.1)

    def get_initial_mean(self, start_offsets: Optional[Sequence]) -> torch.Tensor:
        return self.initial_mean

    @property
    def dt_unit(self) -> Optional[np.timedelta64]:
        return None

    @property
    def intercept_state_element(self) -> Optional[str]:
        return None

    @torch.no_grad()
    def update_intercept(self, value: torch.Tensor):
        if self.intercept_state_element:
            idx = next(i for i, s in enumerate(self.state_elements.values()) if s.name == self.intercept_state_element)
            self.initial_mean[idx] = value
        else:
            raise ValueError(f"{self.id} has no intercept state element to update")

    @property
    def has_measure(self) -> bool:
        return self._measure is not None

    @property
    def measure(self) -> str:
        if self._measure is None:
            raise ValueError(f"tried to access `measure` before it was set (process: {self.id})")
        return self._measure

    @measure.setter
    def measure(self, value: str):
        if self._measure is not None and self._measure != value:
            raise ValueError(f"tried to set `measure` on process {self.id} after it was already set")
        self._measure = value

    @property
    def measurement_kwargs(self) -> Sequence[ProcessKwarg]:
        """
        List of `ProcessKwarg`s specifying required keyword-arguments for ``get_measurement_matrix()`` (or
        ``prepare_measurement_cache()`` if non-linear).
        """
        return []

    @property
    def rank(self) -> int:
        return len(self.state_elements)

    # measurement ----
    def get_measurement_matrix(self, **kwargs) -> torch.Tensor:
        if kwargs:
            raise ValueError(f"{self.id} received unexpected kwargs: {set(kwargs)}")
        if not self.linear_measurement:
            raise TypeError(f"This method should never be called because {self.id} has nonlinear measure-fun.")
        out = [state_element.measure_multi() for state_element in self.state_elements.values()]
        return torch.stack(out, dim=0)

    def prepare_measurement_cache(self, **kwargs) -> dict:
        """
        For processes with a linear measurement function, we can compute the measurement matrix for all groups*times
        at the outset, which is more efficient than doing it for each timestep. However, for processes with non-linear
        measure-funs, we need to (a) apply our measurement-fun to the state-mean at each timestep separately, and
        (b) compute the jacobian of the measurement-fun at each timestep separately (since it depends on the mean).

        Nevertheless, there might be some computations that can be done for all groups*times at the outset. For example,
        for a saturated linear model, we still want to validate the model-matrix and "unbind" it into a list for each
        timestep, so we can efficiently index into that. This method allows us to do any of those pre-computations then
        store that dict, to be passed later to ``get_measured_mean()`` and ``get_measurement_jacobian()``.
        """
        raise NotImplementedError

    def get_measured_mean(self, mean: torch.Tensor, time: Optional[int], cache: dict) -> torch.Tensor:
        """
        Take the state-mean and apply the measurement function to it.

        :param mean: The state-mean at the given time.
        :param time: The current time-step. This is used to index into the cache. Must support time=None for the case
         when `mean` isn't a single timepoint but a group*time*etc tensor.
        :param cache: A dictionary returned by ``prepare_measurement_cache()``.
        """
        raise NotImplementedError

    def get_measurement_jacobian(self, mean: torch.Tensor, time: Optional[int], cache: dict) -> torch.Tensor:
        """
        Take the state-mean and compute the jacobian of the measurement function at that mean.

        :param mean: The state-mean at the given time.
        :param time: The current time-step. This is used to index into the cache. Must support time=None for the case
         when `mean` isn't a single timepoint but a group*time*etc tensor.
        :param cache: A dictionary returned by ``prepare_measurement_cache()``.
        """
        raise NotImplementedError

    # transitions ----
    def get_transition_matrix(self) -> torch.Tensor:
        if not self.linear_transition:
            raise TypeError(f"This method should never be called because {self.id} has nonlinear transition-fun.")

        name_to_idx = {name: i for i, name in enumerate(self.state_elements)}

        # with autograd we can't modify in-place, so first need a data-structure to hold possible multiple values as
        # a list...
        transition_sums = {}
        for state_element in self.state_elements.values():
            from_idx = name_to_idx[state_element.name]
            for to_name, multi in state_element.transitions_to.items():
                to_idx = name_to_idx[to_name]
                key = (to_idx, from_idx)
                if key not in transition_sums:
                    transition_sums[key] = []
                transition_sums[key].append(multi())

        # ...then on the second loop, we sum each list for a single element:
        out = torch.zeros((self.rank, self.rank), device=self.initial_mean.device)
        for (from_idx, to_idx), multis in transition_sums.items():
            out[from_idx, to_idx] = sum(multis)
        return out

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id={self.id}, measure={self._measure})'
