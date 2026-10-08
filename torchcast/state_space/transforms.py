"""
Transforms for mapping predictions from the scale a measure is modeled on (e.g. log) back to its original scale.
"""
import math
from typing import Optional, TYPE_CHECKING, Mapping, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from .predictions import Predictions


class Transform(torch.nn.Module):
    """
    Maps a measure from the scale it's modeled on back to its original scale, e.g. for
    ``Predictions.to_dataframe(transform=...)``.

    Subclasses implement :func:`inverse`, which must be elementwise and monotonically increasing (so quantiles can be
    back-transformed directly). The back-transformed mean, :func:`inverse_mean`, is ``E[inverse(Y)]`` where ``Y =
    mean + sqrt(var) * Z``, and ``Z`` is standard normal. This is computed by gauss-hermite quadrature by default;
    subclasses can override :func:`expected_inverse` with a closed form, or :func:`noise_nodes` to use a different
    distribution for ``Z`` (see :class:`SmearingTransform`).

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
        The mean on the original scale, ``E[inverse(Y)]``, for ``Y`` on the modeled scale with the given mean and
        variance -- with the variance scaled by ``bias_adjust``.

        :param mean: The mean of ``Y``.
        :param var: The variance of ``Y``, same shape as ``mean``.
        :return: A tensor with the same shape as ``mean``.
        """
        return self.expected_inverse(mean, var * self._var_multi)

    def expected_inverse(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        ``E[inverse(mean + sqrt(var) * Z)]``, with ``Z`` from :func:`noise_nodes`. Like :func:`inverse_mean`, but
        without ``bias_adjust``. Subclasses can override this with a closed form.

        :param mean: A tensor.
        :param var: A tensor with the same shape as ``mean``.
        :return: A tensor with the same shape as ``mean``.
        """
        z, w = self.noise_nodes(dtype=mean.dtype, device=mean.device)
        nodes = mean.unsqueeze(-1) + var.sqrt().unsqueeze(-1) * z
        return (w * self.inverse(nodes)).sum(-1)

    def noise_nodes(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The distribution of the standardized noise ``Z`` (see :func:`expected_inverse`), as nodes and weights: by
        default, gauss-hermite quadrature for a standard normal.

        :return: A tuple of 1-D tensors: the nodes, and their weights (which sum to 1).
        """
        # E[f(Z)], Z ~ N(0, 1)  ~=  sum_i w_i / sqrt(pi) * f(sqrt(2) * x_i)
        x, w = np.polynomial.hermite.hermgauss(self.num_nodes)
        return (
            torch.as_tensor(math.sqrt(2) * x, dtype=dtype, device=device),
            torch.as_tensor(w / math.sqrt(math.pi), dtype=dtype, device=device),
        )

    def for_regime(self, component_id: str) -> 'Transform':
        """
        The transform to use for the back-transformed mean of a mixture component's regime (for measures with
        mixture components). By default, this same transform -- with gaussian noise, since a component is gaussian by
        definition (e.g. :class:`SmearingTransform` returns its base). See :class:`RegimeTransform` to choose.

        :param component_id: The ``id`` of the :class:`.MixtureComponent`.
        """
        return self


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

    def expected_inverse(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # closed-form (lognormal)
        return torch.exp(mean * self._log_base + var * self._log_base ** 2 / 2)


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

    def expected_inverse(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        if self.lmbda == 0:
            return torch.exp(mean + var / 2)
        return super().expected_inverse(mean, var)


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

    For a measure with mixture components, this applies to the standard regime only: by default, the components'
    regimes use ``base`` (gaussian). To smear each regime with its own residuals, see :func:`from_predictions`; to
    choose per regime, :class:`RegimeTransform`.

    :param base: The transform that was applied before modeling, e.g. :class:`LogTransform`. Its ``bias_adjust``
     (which scales ``var`` above) is used.
    :param residuals: A 1-D tensor of standardized residuals. Non-finite values are dropped.
    :param weights: Optional weights for the residuals (e.g. the probability that each came from the standard
     regime of a mixture measure).
    :param num_nodes: The residuals' (weighted) empirical distribution is summarized by this many quantiles (so the
     cost doesn't grow with the number of residuals). If there are fewer residuals than this, they're used directly.
    """

    def __init__(self,
                 base: Transform,
                 residuals: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 num_nodes: int = 100):
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
            residuals = torch.as_tensor(np.interp(probs, cdf.numpy(), residuals.numpy()))
            weights = torch.full((num_nodes,), 1 / num_nodes, dtype=torch.float64)
        self.register_buffer('residuals', residuals.to(torch.get_default_dtype()))
        self.register_buffer('weights', weights.to(torch.get_default_dtype()))

    @classmethod
    def from_predictions(cls,
                         base: Transform,
                         predictions: 'Predictions',
                         y: torch.Tensor,
                         measure: str,
                         num_nodes: int = 100) -> Union['SmearingTransform', 'RegimeTransform']:
        """
        Create from the standardized residuals of predictions for ``measure`` -- typically 1-step-ahead
        predictions on the training data.

        For a measure with mixture components, each regime is smeared with its own residuals: standardized by that
        regime's predicted mean and variance, and weighted by the probability (given the observation) that it came
        from that regime. This returns a :class:`RegimeTransform`. So a component's back-transformed mean uses the
        residuals attributed to it -- rather than e.g. a lognormal mean, which is sensitive to a large component
        variance. (With few observations attributed to a component, this is correspondingly noisy. To use a
        different transform for some regimes, e.g. ``LogTransform(bias_adjust=0)``, build a :class:`RegimeTransform`
        from this one's ``standard``.)

        :param base: See :class:`SmearingTransform`.
        :param predictions: A :class:`.Predictions` object.
        :param y: The observations, laid out like the input to the model (``(num_groups, num_timesteps,
         num_measures)``); can have fewer timesteps than ``predictions``.
        :param measure: The measure.
        :param num_nodes: See :class:`SmearingTransform`.
        :return: A :class:`SmearingTransform`; or for a measure with mixture components, a :class:`RegimeTransform`
         of them.
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
            by_regime = {}
            for k, label in enumerate(mix.labels):
                if not (weights[..., k].nan_to_num() > 0).any():
                    by_regime[label] = base  # (no observations attributed to this regime)
                    continue
                by_regime[label] = cls(base, residuals=residuals[..., k], weights=weights[..., k], num_nodes=num_nodes)
        standard = by_regime.pop(mix.labels[0])
        return RegimeTransform(standard=standard, components=by_regime)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.inverse(x)

    def noise_nodes(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.residuals.to(dtype=dtype, device=device), self.weights.to(dtype=dtype, device=device)

    def for_regime(self, component_id: str) -> Transform:
        return self.base


class RegimeTransform(Transform):
    """
    For a measure with mixture components: a different back-transformed mean for each regime. E.g.::

        RegimeTransform(
            standard=SmearingTransform(LogTransform(), residuals),
            components={'quick': LogTransform(bias_adjust=0)},
        )

    All the transforms must have the same :func:`inverse` (here: exp) -- intervals are back-transformed with it --
    and only differ in how each regime's mean is back-transformed.

    :param standard: The transform for the standard regime. It's also used for everything that isn't
     regime-specific: intervals, actuals, and measures without mixture components.
    :param components: A dictionary of ``{component_id: Transform}``. Components that aren't listed use
     ``standard.for_regime(component_id)``.
    """

    def __init__(self, standard: Transform, components: Optional[Mapping[str, Transform]] = None):
        if isinstance(standard, RegimeTransform):
            raise ValueError("`standard` can't itself be a `RegimeTransform`.")
        super().__init__(bias_adjust=standard.bias_adjust)
        self.standard = standard
        self.components = torch.nn.ModuleDict(dict(components or {}))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.standard.inverse(x)

    def inverse_mean(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return self.standard.inverse_mean(mean, var)

    def expected_inverse(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return self.standard.expected_inverse(mean, var)

    def noise_nodes(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.standard.noise_nodes(dtype=dtype, device=device)

    def for_regime(self, component_id: str) -> Transform:
        if component_id in self.components:
            return self.components[component_id]
        return self.standard.for_regime(component_id)
