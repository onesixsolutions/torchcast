from functools import lru_cache

from typing import Optional

import numpy as np
import torch


class FixedWhiteNoise:
    """
    Fixed draws of standard-normal noise (e.g. a model's ``mc_sampling``, for its monte-carlo log-prob), so that a
    monte-carlo loss is a deterministic function of the parameters.

    Without a ``random_state``, the draws are seeded from ``torch.initial_seed()`` -- the seed last set by
    ``torch.manual_seed()`` (or a random one fixed for the process) -- so ``torch.manual_seed()`` before creating the
    model makes them (and so a fit) reproducible.
    """
    _random_state = None

    def __init__(self,
                 num_samples: int = 500,
                 random_state: Optional[np.random.RandomState] = None):
        self._num_samples = num_samples
        self.reset(random_state)

    def __repr__(self) -> str:
        return "FixedWhiteNoise(num_samples={})".format(self.num_samples)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value
        self.__call__.cache_clear()

    def reset(self, random_state: Optional[np.random.RandomState] = None):
        if random_state is None:
            random_state = np.random.RandomState(torch.initial_seed() % 2 ** 32)
        self._random_state = random_state.get_state()
        return self

    @lru_cache(maxsize=1)
    def __call__(self, num_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.num_samples:
            return torch.zeros((1, num_dim), dtype=dtype, device=device)
        random_state = np.random.RandomState()
        random_state.set_state(self._random_state)
        return torch.as_tensor(random_state.randn(self.num_samples, num_dim), dtype=dtype, device=device)
