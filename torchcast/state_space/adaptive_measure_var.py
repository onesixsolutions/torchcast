"""
Adaptive measure variance modules for StateSpaceModel.

This module provides implementations for adaptively updating measurement covariance
based on prediction residuals.
"""
import torch
import torch.nn as nn
from torch.nn.init import normal_
from typing import Optional, Sequence

from torchcast.process.utils import Bounded


class AdaptiveMeasureVar(nn.Module):
    def reset(self):
        raise NotImplementedError

    def forward(self, residuals: torch.Tensor, skip_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EWMAdaptiveMeasureVar(AdaptiveMeasureVar):
    """
    Exponentially Weighted Moving Average (EWM) based adaptive measure variance.
    """

    def __init__(self,
                 num_measures: int,
                 bounds: Optional[Sequence[tuple[float, float]]] = None,
                 eps: float = 1e-3):
        super().__init__()

        if bounds is None:
            bounds = [(0.0, 1.0)] * num_measures
        else:
            assert len(bounds) == num_measures

        self._alphas = torch.nn.ModuleList()
        for lower, upper in bounds:
            self._alphas.append(Bounded(lower, upper))
            normal_(self._alphas[-1].raw, -3, .1)  # initialize with lower alpha for more smoothing by default
        self._running = None

        self.weight = nn.Parameter(torch.randn(num_measures).abs() * .01)  # initialize with small positive value

        self.eps = eps

    @property
    def alpha(self) -> torch.Tensor:
        return torch.stack([alpha() for alpha in self._alphas], -1)

    def reset(self):
        self._running = None

    def forward(self, residuals: torch.Tensor, skip_mask: torch.Tensor) -> torch.Tensor:
        sq_resids = residuals ** 2
        new = torch.zeros_like(sq_resids)
        if self._running is None:
            self._running = torch.zeros_like(sq_resids)
        new[skip_mask] = self._running[skip_mask]
        new[~skip_mask] = (1 - self.alpha) * self._running[~skip_mask] + self.alpha * sq_resids[~skip_mask]
        self._running = new.clamp(self.eps)
        log_running_std = torch.log(self._running ** .5)
        return torch.exp(log_running_std * self.weight)
