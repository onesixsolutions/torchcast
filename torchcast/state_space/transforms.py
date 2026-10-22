"""
Transforms for mapping predictions from the scale a measure is modeled on (e.g. log) back to its original scale.
"""
import math
from typing import Optional, TYPE_CHECKING, Mapping

import numpy as np
import torch

if TYPE_CHECKING:
    from .predictions import Predictions


class Transform(torch.nn.Module):
    """
    Maps a measure from its original to the scale it's modeled on (and vice versa), e.g. for
    ``Predictions.to_dataframe(transform=...)``.

    Subclasses implement :func:`transform` (the transformation applied before modeling -- e.g. by
    ``TimeSeriesDataset.from_dataframe(transform=...)``) and its inverse, :func:`inverse`, which must be elementwise and
    monotonically increasing (so quantiles can be back-transformed directly). The back-transformed mean,
    :func:`inverse_mean`, is ``E[inverse(Y)]`` where ``Y = mean + sqrt(var) * Z``, and ``Z`` is standard normal. This
    is computed by gauss-hermite quadrature by default; subclasses can override ``_expected_inverse`` with a closed
    form, or ``_noise_nodes`` to use a different distribution for ``Z`` (see :class:`SmearingTransform`).

    :param bias_adjust: How much bias-adjustment to apply when back-transforming the mean, between 0 and 1. The
     back-transformed mean is computed with the variance scaled by this: 1 (the default, ``None``) gives the mean of
     the back-transformed distribution; 0 gives no bias-adjustment (i.e. the back-transformed median). Doesn't affect
     intervals. For monte-carlo means (e.g. ``use_map=False`` for nonlinear measures), the variance of the sampled state
     is scaled too. Ignored (with a warning) for ``Predictions.to_dataframe(derived=...)``, which back-transforms
     samples.
    """
    num_nodes: int = 32
    # the quadrature's intermediates have (number of elements) x (number of nodes) values: it's computed in chunks of
    # elements, so they stay below about this many (e.g. for many monte-carlo samples x rows):
    max_quadrature_size: int = 2 ** 24

    def __init__(self, bias_adjust: Optional[float] = None):
        super().__init__()
        if bias_adjust is not None and not 0 <= bias_adjust <= 1:
            raise ValueError("`bias_adjust` must be between 0 and 1.")
        self.bias_adjust = bias_adjust

    @property
    def var_multi(self) -> float:
        return 1. if self.bias_adjust is None else self.bias_adjust

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Values on the original scale.
        :return: Values on the modeled (transformed) scale.
        """
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Values on the modeled (transformed) scale.
        :return: Values on the original scale.
        """
        raise NotImplementedError

    def inverse_mean(self,
                     mean: torch.Tensor,
                     var: torch.Tensor,
                     adjust_bias: bool = True,
                     regime: Optional[str] = None) -> torch.Tensor:
        """
        The mean on the original scale, ``E[inverse(Y)]``, for ``Y`` on the modeled scale with the given mean and
        variance -- with the variance scaled by ``bias_adjust``, unless ``adjust_bias=False``.

        :param mean: The mean of ``Y``.
        :param var: The variance of ``Y``, same shape as ``mean`` (or broadcastable to it).
        :param adjust_bias: Whether to apply ``bias_adjust`` (False: the mean of the back-transformed distribution,
         whatever ``bias_adjust`` is).
        :param regime: For a measure with mixture components, the regime that ``mean`` and ``var`` are for: None for
         the standard regime, else the component's ``id``. Only matters for a transform whose noise-distribution
         differs by regime (a :class:`SmearingTransform` from :func:`SmearingTransform.from_predictions`).
        :return: A tensor with the same shape as ``mean``.
        """
        return self._expected_inverse(mean, var * (self.var_multi if adjust_bias else 1.), regime=regime)

    def _expected_inverse(self, mean: torch.Tensor, var: torch.Tensor, regime: Optional[str] = None) -> torch.Tensor:
        """
        ``E[inverse(mean + sqrt(var) * Z)]``, with ``Z`` from ``_noise_nodes``: by quadrature (in chunks, see
        ``max_quadrature_size``). The hook for subclasses to override with a closed form (``inverse_mean`` applies
        ``bias_adjust``, so they needn't).
        """
        z, w = self._noise_nodes(dtype=mean.dtype, device=mean.device, regime=regime)
        mean, var = torch.broadcast_tensors(mean, var)
        if not mean.numel():
            return mean.clone()
        chunk_size = max(1, self.max_quadrature_size // len(z))
        out = [
            (w * self.inverse(mean_chunk.unsqueeze(-1) + var_chunk.sqrt().unsqueeze(-1) * z)).sum(-1)
            for mean_chunk, var_chunk in zip(mean.reshape(-1).split(chunk_size), var.reshape(-1).split(chunk_size))
        ]
        return torch.cat(out).view(mean.shape)

    def _noise_nodes(self,
                     dtype: torch.dtype,
                     device: torch.device,
                     regime: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The distribution of the standardized noise ``Z`` (see ``_expected_inverse``), as nodes and weights: by
        default, gauss-hermite quadrature for a standard normal (for every regime).

        :return: A tuple of 1-D tensors: the nodes, and their weights (which sum to 1).
        """
        # E[f(Z)], Z ~ N(0, 1)  ~=  sum_i w_i / sqrt(pi) * f(sqrt(2) * x_i)
        x, w = np.polynomial.hermite.hermgauss(self.num_nodes)
        return (
            torch.as_tensor(math.sqrt(2) * x, dtype=dtype, device=device),
            torch.as_tensor(w / math.sqrt(math.pi), dtype=dtype, device=device),
        )


class _IdentityTransform(Transform):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _expected_inverse(self, mean: torch.Tensor, var: torch.Tensor, regime: Optional[str] = None) -> torch.Tensor:
        return mean


class LogTransform(Transform):
    """
    For a measure that was log-transformed before modeling.

    :param offset: A value to be added before transforming (e.g. to prevent passing zero to log) and removed after
     back-transforming.
    :param base: The base of the logarithm; defaults to e.
    :param bias_adjust: See :class:`Transform`.
    """

    def __init__(self, offset: float = 0.0, base: float = math.e, bias_adjust: Optional[float] = None):
        super().__init__(bias_adjust=bias_adjust)
        assert base > 0
        self._log_base = math.log(base)
        assert offset >= 0
        self.offset = offset

    def _remove_offset(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x - self.offset, min=0.0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.offset) / self._log_base

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self._remove_offset(torch.exp(x * self._log_base))

    def _expected_inverse(self, mean: torch.Tensor, var: torch.Tensor, regime: Optional[str] = None) -> torch.Tensor:
        # closed-form (lognormal)
        return self._remove_offset(torch.exp(mean * self._log_base + var * self._log_base ** 2 / 2))


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

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.lmbda == 0:
            return torch.log(x)
        return (x ** self.lmbda - 1) / self.lmbda

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.lmbda == 0:
            return torch.exp(x)
        # the inverse only exists where (lmbda * x + 1) > 0; clamp the (far-tail) region where it doesn't:
        return (self.lmbda * x + 1).clamp_min(0) ** (1 / self.lmbda)

    def _expected_inverse(self, mean: torch.Tensor, var: torch.Tensor, regime: Optional[str] = None) -> torch.Tensor:
        if self.lmbda == 0:
            return torch.exp(mean + var / 2)
        return super()._expected_inverse(mean, var, regime=regime)


class SmearingTransform(Transform):
    """
    Wraps a base transform; back-transformed means use the empirical distribution of standardized residuals
    instead of assuming the noise is gaussian (Duan's "smearing" estimator). Useful when the residuals on the modeled
    scale have heavier (or lighter) tails, or skew, than a gaussian -- which the back-transformed mean is sensitive
    to. Intervals are unaffected (they're still the model's gaussian quantiles, back-transformed).

    The mean is ``sum_i w_i * inverse(mean + sqrt(var) * z_i)``, where ``z_i`` are standardized residuals: residuals
    divided by the predicted standard-deviation, e.g. from the model's (1-step-ahead) predictions on its training
    data. They're *not* re-standardized, so if the predicted variance is too small (large) on average, the smearing
    corrects for that too. See :func:`from_predictions`.

    For a measure with mixture components, each regime has its own mean and variance (a component's regime is the
    standard one, offset, with extra variance); ``residuals`` are used for every regime -- except components with their
    own, in ``component_residuals`` (see :func:`from_predictions`, which smears each regime with its own residuals).

    :param base: The transform that was applied before modeling, e.g. :class:`LogTransform`. Its ``bias_adjust``
     (which scales ``var`` above) is used.
    :param residuals: A 1-D tensor of standardized residuals. Non-finite values are dropped.
    :param weights: Optional weights for the residuals (e.g. the probability that each came from the standard
     regime of a mixture measure).
    :param num_nodes: The residuals' (weighted) empirical distribution is summarized by this many quantiles (so the
     cost doesn't grow with the number of residuals). If there are fewer residuals than this, they're used directly.
    :param component_residuals: Optionally, for mixture components with their own standardized residuals:
     ``{component_id: (residuals, weights)}`` (``weights`` can be None), used for that component's regime.
    """

    def __init__(self,
                 base: Transform,
                 residuals: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 num_nodes: int = 100,
                 component_residuals: Optional[Mapping[str, tuple[torch.Tensor, Optional[torch.Tensor]]]] = None):
        if isinstance(base, SmearingTransform):
            raise ValueError("`base` can't itself be a `SmearingTransform`.")
        super().__init__(bias_adjust=base.bias_adjust)
        self.base = base

        residuals = torch.as_tensor(residuals, dtype=torch.float64).reshape(-1)
        if weights is None:
            weights = torch.ones_like(residuals)
        weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        if weights.shape != residuals.shape:
            raise ValueError("`weights` must have the same number of elements as `residuals`.")
        if (weights < 0).any():
            raise ValueError("`weights` must be non-negative.")
        keep = torch.isfinite(residuals) & torch.isfinite(weights) & (weights > 0)
        residuals, weights = residuals[keep], weights[keep]
        if not len(residuals):
            raise ValueError("No (finite, positively weighted) residuals.")
        weights = weights / weights.sum()

        if len(residuals) > num_nodes:
            # summarize by quantiles (at the midpoints of `num_nodes` equal-probability bins):
            order = residuals.argsort()
            residuals, weights = residuals[order], weights[order]
            cdf = weights.cumsum(0) - weights / 2
            probs = (np.arange(num_nodes) + .5) / num_nodes
            residuals = torch.as_tensor(
                np.interp(probs, cdf.cpu().numpy(), residuals.cpu().numpy()), device=residuals.device
            )
            weights = torch.full((num_nodes,), 1 / num_nodes, dtype=torch.float64, device=residuals.device)
        self.register_buffer('residuals', residuals.to(torch.get_default_dtype()))
        self.register_buffer('weights', weights.to(torch.get_default_dtype()))
        # (each a SmearingTransform, for its residuals; so they're buffers, e.g. moved by `.to()`)
        self.component_residuals = torch.nn.ModuleDict({
            component_id: SmearingTransform(base, residuals=r, weights=w, num_nodes=num_nodes)
            for component_id, (r, w) in (component_residuals or {}).items()
        })

    @classmethod
    def from_predictions(cls,
                         base: Transform,
                         predictions: 'Predictions',
                         y: torch.Tensor,
                         measure: str,
                         num_nodes: int = 100) -> 'SmearingTransform':
        """
        Create from the standardized residuals of predictions for ``measure`` -- typically 1-step-ahead
        predictions on the training data.

        For a measure with mixture components, each regime is smeared with its own residuals: each observation's
        residual standardized by that regime's predicted mean and variance, and weighted by the probability (given the
        observation) that it came from that regime. So a component's back-transformed mean uses the residuals
        attributed to it -- rather than e.g. a lognormal mean, which is sensitive to a large component variance.
        (With few observations attributed to a component, this is correspondingly noisy; a component with none uses
        the standard regime's.)

        :param base: See :class:`SmearingTransform`.
        :param predictions: A :class:`.Predictions` object.
        :param y: The observations, laid out like the input to the model (``(num_groups, num_timesteps,
         num_measures)``); can have fewer timesteps than ``predictions``.
        :param measure: The measure.
        :param num_nodes: See :class:`SmearingTransform`.
        :return: A :class:`SmearingTransform`.
        """
        measures = list(predictions.measurement_model.measures)
        j = measures.index(measure)
        if measure in predictions._nonlinear_measures:
            raise NotImplementedError(
                f"`from_predictions()` isn't yet supported for '{measure}', which has a nonlinear measured-mean."
            )
        with torch.no_grad():
            obs = y[..., j]
            num_timesteps = obs.shape[1]
            mixture = predictions.mixture
            if mixture is None or measure not in mixture.mixture_measures:
                measured_mean, system_cov = predictions._measured_moments_flat()
                batch_shape = predictions.state_means.shape[0:2]
                mean = measured_mean[:, j].view(*batch_shape)[:, :num_timesteps]
                var = system_cov[:, j, j].view(*batch_shape)[:, :num_timesteps]
                return cls(base, residuals=(obs - mean) / var.sqrt(), num_nodes=num_nodes)

            mix = predictions.get_mixture(measure)
            probs, means, vars_ = (x[:, :num_timesteps] for x in (mix.probs, mix.means, mix.vars))
            # P(regime | observation). (missing observations give nan weights/residuals, which are dropped -- so skip
            # validation, which rejects nans.)
            normal = torch.distributions.Normal(means, vars_.sqrt(), validate_args=False)
            log_liks = normal.log_prob(obs.unsqueeze(-1))
            weights = torch.softmax(probs.clamp_min(1e-30).log() + log_liks, -1)
            residuals = (obs.unsqueeze(-1) - means) / vars_.sqrt()
            component_residuals = {
                label: (residuals[..., k], weights[..., k])
                for k, label in enumerate(mix.labels)
                if k > 0 and (weights[..., k].nan_to_num() > 0).any()
            }
            return cls(
                base,
                residuals=residuals[..., 0],
                weights=weights[..., 0],
                num_nodes=num_nodes,
                component_residuals=component_residuals,
            )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.transform(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.inverse(x)

    def _noise_nodes(self,
                     dtype: torch.dtype,
                     device: torch.device,
                     regime: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if regime is not None and regime in self.component_residuals:
            return self.component_residuals[regime]._noise_nodes(dtype=dtype, device=device)
        return self.residuals.to(dtype=dtype, device=device), self.weights.to(dtype=dtype, device=device)
