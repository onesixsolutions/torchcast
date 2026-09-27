"""
Transforms for mapping predictions from the scale a measure is modeled on (e.g. log) back to its original scale.
"""
import math

import numpy as np
import torch


class Transform(torch.nn.Module):
    """
    Maps a measure from the scale it's modeled on back to its original scale, e.g. for
    ``Predictions.to_dataframe(transform=...)``.

    Subclasses implement :func:`inverse`, which must be elementwise and monotonically increasing (so quantiles can be
    back-transformed directly). The back-transformed mean, :func:`inverse_mean`, is computed by gauss-hermite
    quadrature by default; subclasses can override it with a closed form.
    """
    num_nodes: int = 32

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
    """

    def __init__(self, base: float = math.e):
        super().__init__()
        assert base > 0
        self.base = base

    @property
    def _log_base(self) -> float:
        return math.log(self.base)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * self._log_base)

    def inverse_mean(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # closed-form (lognormal)
        return torch.exp(mean * self._log_base + var * self._log_base ** 2 / 2)


class BoxCoxTransform(Transform):
    """
    For a measure that was Box-Cox transformed before modeling: ``(y ** lmbda - 1) / lmbda`` (``log(y)`` if
    ``lmbda == 0``).

    :param lmbda: The Box-Cox parameter. Must be non-negative: for negative values, the back-transformed mean of a
     gaussian does not exist (it's infinite).
    """

    def __init__(self, lmbda: float):
        super().__init__()
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
            return torch.exp(mean + var / 2)
        return super().inverse_mean(mean, var)
