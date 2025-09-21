"""
Adaptive measure variance modules for StateSpaceModel.

This module provides implementations for adaptively updating measurement covariance
based on prediction residuals.
"""
import warnings

import torch
import torch.nn as nn
from torch.nn.init import normal_
from typing import Optional, Sequence

from torchcast.process.utils import Bounded


class AdaptiveScaler(nn.Module):
    def initialize(self, num_timesteps: int):
        """
        If relevant, use num_timesteps to initialize parameters
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset internal state (e.g., running statistics).
        """
        raise NotImplementedError

    def forward(self, residuals: torch.Tensor, skip_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EWMAdaptiveScaler(AdaptiveScaler):
    """
    Exponentially Weighted Moving Average (EWM) based adaptive scaling.
    """

    def __init__(self,
                 num_measures: int,
                 eps: float = 1e-3):
        super().__init__()

        # initial alpha:
        self._rhos = torch.nn.ModuleList(
            [Bounded(0.0, 1.0) for _ in range(num_measures)]
        )

        # decay speed:
        self._taus = torch.nn.Parameter(torch.randn(num_measures) * .1)

        # coef from log-std to multiplier:
        # initialize with small positive value
        self.weight = nn.Parameter(torch.randn(num_measures).abs() * .01)

        # prevent scaling from going to zero:
        self.eps = eps

        self._running = None
        self._time = None
        self._called_initialize = None

    @torch.no_grad()
    def initialize(self, num_timesteps: int):
        # tau is halflife, by default we'll set halflife to 10% of num_timesteps
        normal_(self._taus, std=.1)
        self._taus += torch.log(torch.tensor(.10 * num_timesteps, dtype=self._taus.dtype))
        self._called_initialize = True

    @property
    def alpha(self) -> torch.Tensor:
        alphas = []
        for i, rho in enumerate(self._rhos):
            tau = torch.exp(self._taus[i])
            alpha = tau * rho() / (self._time[:, i] + tau)
            alphas.append(alpha)
        return torch.stack(alphas, -1)

    def reset(self):
        self._running = None
        self._time = None
        if self._called_initialize is None:
            warnings.warn("Consider calling adaptive scaler's `initialize()` method before use.")
            self._called_initialize = False  # only warn once

    def forward(self, residuals: torch.Tensor, skip_mask: torch.Tensor) -> torch.Tensor:
        if self._running is None:
            self._running = torch.zeros_like(residuals)
            self._time = torch.zeros_like(residuals)
        self._time += (~skip_mask).int()

        sq_resids = residuals ** 2
        alpha = torch.zeros_like(sq_resids)
        alpha[~skip_mask] = self.alpha[~skip_mask]
        ewma = (1 - alpha) * self._running + alpha * sq_resids
        self._running = ewma.clamp(self.eps)
        log_running_std = torch.log(self._running ** .5)
        return torch.exp(torch.clamp(log_running_std * self.weight, max=8))
