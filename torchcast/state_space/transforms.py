"""
Transforms for mapping predictions from the scale a measure is modeled on (e.g. log) back to its original scale.
"""
import math
from typing import Optional

import numpy as np
import torch


class Transform(torch.nn.Module):
    """
    Maps a measure from the scale it's modeled on back to its original scale, e.g. for
    ``Predictions.to_dataframe(transform=...)``.

    Subclasses implement :func:`inverse`, which must be elementwise and monotonically increasing (so quantiles can be
    back-transformed directly). The back-transformed mean, :func:`inverse_mean`, is computed by gauss-hermite
    quadrature by default; subclasses can override it with a closed form.

    :param bias_adjust: How much bias-adjustment to apply when back-transforming the mean, between 0 and 1. The
     back-transformed mean is computed with the variance scaled by this: 1 (the default, ``None``) gives the mean of
     the back-transformed distribution; 0 gives no bias-adjustment (i.e. the back-transformed median). Doesn't affect
     intervals. Ignored (with a warning) for monte-carlo predictions, where the mean is of back-transformed samples.
    """
    num_nodes: int = 32

    def __init__(self, bias_adjust: Optional[float] = None):
        super().__init__()
        if bias_adjust is not None and not 0 <= bias_adjust <= 1:
            raise ValueError("`bias_adjust` must be between 0 and 1.")
        self.bias_adjust = bias_adjust

    @property
    def _var_multi(self) -> float:
        return 1. if self.bias_adjust is None else self.bias_adjust

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Values on the modeled (transformed) scale.
        :return: Values on the original scale.
        """
        raise NotImplementedError

    def inverse_mean(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        The mean on the original scale, ``E[inverse(Y)]``, for gaussian ``Y`` on the modeled scale.

        :param mean: The mean of ``Y``.
        :param var: The variance of ``Y``, same shape as ``mean``.
        :return: A tensor with the same shape as ``mean``.
        """
        var = var * self._var_multi
        # E[f(Y)], Y ~ N(mean, var)  ~=  sum_i w_i / sqrt(pi) * f(mean + sqrt(2 * var) * x_i)
        x, w = np.polynomial.hermite.hermgauss(self.num_nodes)
        x = torch.as_tensor(x, dtype=mean.dtype, device=mean.device)
        w = torch.as_tensor(w / math.sqrt(math.pi), dtype=mean.dtype, device=mean.device)
        nodes = mean.unsqueeze(-1) + (2 * var).sqrt().unsqueeze(-1) * x
        return (w * self.inverse(nodes)).sum(-1)


class LogTransform(Transform):
    """
    For a measure that was log-transformed before modeling.

    :param base: The base of the logarithm; defaults to e.
    :param bias_adjust: See :class:`Transform`.
    """

    def __init__(self, base: float = math.e, bias_adjust: Optional[float] = None):
        super().__init__(bias_adjust=bias_adjust)
        assert base > 0
        self.base = base

    @property
    def _log_base(self) -> float:
        return math.log(self.base)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * self._log_base)

    def inverse_mean(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # closed-form (lognormal)
        return torch.exp(mean * self._log_base + self._var_multi * var * self._log_base ** 2 / 2)


class BoxCoxTransform(Transform):
    """
    For a measure that was Box-Cox transformed before modeling: ``(y ** lmbda - 1) / lmbda`` (``log(y)`` if
    ``lmbda == 0``).

    :param lmbda: The Box-Cox parameter. Must be non-negative: for negative values, the back-transformed mean of a
     gaussian does not exist (it's infinite).
    :param bias_adjust: See :class:`Transform`.
    """

    def __init__(self, lmbda: float, bias_adjust: Optional[float] = None):
        super().__init__(bias_adjust=bias_adjust)
        if lmbda < 0:
            raise ValueError("`lmbda` must be non-negative.")
        self.lmbda = lmbda

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.lmbda == 0:
            return torch.exp(x)
        # the inverse only exists where (lmbda * x + 1) > 0; clamp the (far-tail) region where it doesn't:
        return (self.lmbda * x + 1).clamp_min(0) ** (1 / self.lmbda)

    def inverse_mean(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        if self.lmbda == 0:
            return torch.exp(mean + self._var_multi * var / 2)
        return super().inverse_mean(mean, var)
