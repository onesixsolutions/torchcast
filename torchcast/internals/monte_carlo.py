from functools import lru_cache

from copy import deepcopy

from typing import Optional

import numpy as np
import torch


class FixedWhiteNoise:
    _random_state = None

    def __init__(self,
                 num_samples: int = 500,
                 random_state: Optional[np.random.RandomState] = None):
        self._num_samples = num_samples
        self.reset(random_state)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value
        self.__call__.cache_clear()

    def reset(self, random_state: Optional[np.random.RandomState] = None):
        random_state = random_state or np.random.RandomState()
        self._random_state = random_state.get_state()
        return self

    @lru_cache(maxsize=1)
    def __call__(self, num_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.num_samples:
            return torch.zeros((1, num_dim), dtype=dtype, device=device)
        random_state = np.random.RandomState()
        random_state.set_state(self._random_state)
        return torch.as_tensor(random_state.randn(self.num_samples, num_dim), dtype=dtype, device=device)
