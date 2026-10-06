"""
Adaptive variance modules for StateSpaceModel.

This module provides implementations for adaptively updating covariance-scaling
based on prediction residuals.

This is useful when training on multiple time-serieses that differ by orders of magnitude; or if the variance for a
single time-series is heterogeneous wrt time.
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

    def forward(self,
                residuals: torch.Tensor,
                skip_mask: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param residuals: A ``(num_groups, num_measures)`` tensor of residuals.
        :param skip_mask: A boolean tensor, same shape as ``residuals``, indicating residuals to skip (e.g. nans).
        :param weights: Optional tensor, same shape as ``residuals``, with values in [0, 1] indicating how much each
         residual should count. Only passed for models with mixture components, where it's the probability that the
         observation came from the standard regime (as opposed to e.g. an outlier-regime that already explains it).
        :return: A ``(num_groups, num_measures)`` tensor of multipliers for the standard-deviations.
        """
        raise NotImplementedError


class EWMAdaptiveScaler(AdaptiveScaler):
    """
    Exponentially Weighted Moving Average (EWM) based adaptive scaling.

    Tracks a running mean of squared residuals per group and measure, and returns ``running_std ** weight`` as a
    multiplier on the standard-deviations. The running mean starts at 1 (multiplier 1, i.e. no adjustment), so groups
    with no observations get no adjustment, and groups with few observations are shrunk towards no adjustment.
    """
    # The value the running mean of squared residuals starts at. Instances created by older versions of torchcast
    # started at 0 (so groups with few/no observations were shrunk towards zero variance). This class-level default
    # keeps that behavior for instances unpickled from those versions (unpickling doesn't call ``__init__``), while
    # ``__init__`` sets the new default. For ``load_state_dict()``, see ``_load_from_state_dict``.
    _running_init: float = 0.0

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

        self._running_init = 1.0

        self._running = None
        self._time = None
        self._called_initialize = None

    def get_extra_state(self) -> dict:
        return {'running_init': self._running_init}

    def set_extra_state(self, state: dict):
        self._running_init = state['running_init']

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # state-dicts saved by older versions have no extra-state; those params were fit with a running-init of 0:
        extra_state_key = prefix + '_extra_state'
        if extra_state_key not in state_dict:
            state_dict = {**state_dict, extra_state_key: {'running_init': 0.0}}
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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

    def forward(self,
                residuals: torch.Tensor,
                skip_mask: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._running is None:
            self._running = torch.full_like(residuals, self._running_init)
            self._time = torch.zeros_like(residuals)
        if weights is None:
            self._time += (~skip_mask).int()
        else:
            # a partially-weighted observation only partially counts towards the elapsed time:
            self._time = self._time + weights * (~skip_mask)

        sq_resids = residuals ** 2
        alpha = torch.zeros_like(sq_resids)
        alpha[~skip_mask] = self.alpha[~skip_mask]
        if weights is not None:
            # an observation with weight w moves the running average w-as-much. (note this is different from
            # down-weighting the residual itself, which would imply the observation is small, rather than that it
            # (partially) doesn't count.)
            alpha = alpha * weights
        ewma = (1 - alpha) * self._running + alpha * sq_resids
        self._running = ewma.clamp(self.eps)
        log_running_std = torch.log(self._running ** .5)
        return torch.exp(torch.clamp(log_running_std * self.weight, max=8))
