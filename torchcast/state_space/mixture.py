"""
Mixture-of-regimes support for state-space models.

A measure can have one or more :class:`MixtureComponent` alternatives to its standard measurement model. Each
component is an *offset* from the standard regime: in the component's regime, an observation is the usual
(state-dependent) measured-mean plus the component's learned offset, with its learned variance added to the
measurement-noise. So e.g. a "quick visit" component means "spend well below *this* player's usual level", rather
than below some global level. Observations in a component's regime still update the state (accounting for the
offset and the extra noise).

The :class:`MixtureModel` owns the components and enumerates a fixed table of regime "combos" -- one entry per
combination of regimes across the mixture measures (the standard regime, or one of that measure's components). Regime
probabilities are tracked jointly over this table.
"""
import itertools
from dataclasses import dataclass
from typing import Optional, Sequence, Mapping

import torch


class MixtureComponent(torch.nn.Module):
    """
    An alternative ('weird') regime for a single measure: an offset from the standard regime's (state-dependent)
    measured-mean, with a learnable mean (the offset), variance (added to the measurement-noise variance), and
    base-rate.

    :param measure: The measure this component applies to.
    :param mean_init: Initial value for the component's offset, relative to the standard regime's measured-mean.
    :param prob_init: Initial value for the component's base-rate, i.e. the long-run probability that an observation
     for ``measure`` comes from this component (exact when it's the only component for this measure). With
     ``predictors``, this is the base-rate when the predictors are zero.
    :param id: A unique identifier (within the measure).
    :param predictors: Optional names of predictors of the component's base-rate: each has a learned coefficient,
     added to the component's logit (vs. the standard regime). E.g. a treatment that makes this regime more common.
     The predictors are passed to the model's ``forward()`` as a ``(num_groups, num_timesteps, len(predictors))``
     tensor ``X`` -- or, to use a different tensor than other consumers of ``X`` (e.g. a ``LinearModel``), as
     ``{id}__X``. Like the predictors of a ``LinearModel``, they need to cover the forecast horizon too: the
     regime-prior for each timestep uses that timestep's predictors.
    """

    def __init__(self,
                 measure: str,
                 mean_init: float,
                 prob_init: float,
                 id: str,
                 predictors: Optional[Sequence[str]] = None):
        super().__init__()
        self.measure = measure
        self.id = id
        if isinstance(predictors, str):
            raise ValueError("`predictors` should be a list of strings, not a string.")
        self.predictors = list(predictors or [])

        mean_init = torch.as_tensor(mean_init, dtype=torch.get_default_dtype())
        self.mean = torch.nn.Parameter(mean_init)
        self._log_std = torch.nn.Parameter(torch.zeros_like(mean_init))

        assert 0 < prob_init < 1
        prob_init = torch.as_tensor(prob_init, dtype=torch.get_default_dtype())
        # logit relative to the standard regime, whose logit is fixed at 0
        self.logit = torch.nn.Parameter(torch.log(prob_init) - torch.log1p(-prob_init))
        self.coefs = None
        if self.predictors:
            self.coefs = torch.nn.Parameter(torch.zeros(len(self.predictors)))

    kwarg_name = 'X'  # the default name of the ``forward()`` keyword-argument with the predictors

    def get_logit(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param X: If the component has predictors, a ``(..., len(predictors))`` tensor.
        :return: The logit vs. the standard regime: a scalar, or ``X.shape[:-1]`` with predictors.
        """
        if not self.predictors:
            return self.logit
        if X is None:
            raise ValueError(f"MixtureComponent '{self.id}' has predictors, so needs `X` for its base-rate.")
        return self.logit + X @ self.coefs

    @property
    def var(self) -> torch.Tensor:
        return self._log_std.exp() ** 2

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id={repr(self.id)}, measure={repr(self.measure)})'


class RegimeTransition(torch.nn.Module):
    """
    Base-class for how regime-probabilities evolve over time. Subclasses implement :func:`initial` (the regime-prior
    at the first timestep) and :func:`forward` (the regime-prior at the next timestep, given the regime-posterior at
    the current timestep).

    Both receive ``base_probs``, the long-run probability of each combo implied by the :class:`MixtureComponent`
    base-rates; implementations may use or ignore it. It's a ``(num_combos,)`` tensor, or -- if any components have
    predictors -- ``(num_groups, num_combos)``, for the timestep that the regime-prior is for.
    """

    def __init__(self, num_combos: int):
        super().__init__()
        self.num_combos = num_combos

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        :param base_probs: A ``(num_combos,)`` or ``(num_groups, num_combos)`` tensor.
        :param num_groups: The number of groups.
        :return: A ``(num_groups, num_combos)`` tensor of regime-probabilities.
        """
        raise NotImplementedError

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        """
        :param posterior: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the current timestep.
        :param base_probs: A ``(num_combos,)`` or ``(num_groups, num_combos)`` tensor, for the next timestep.
        :return: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the next timestep.
        """
        raise NotImplementedError


class StickyTransition(RegimeTransition):
    """
    Each combo ``i`` persists with probability ``stay_i``; otherwise the next combo is drawn from a shared distribution
    ``b``. That is, ``T[i, j] = stay_i * (i == j) + (1 - stay_i) * b_j``.

    Rather than learning ``b`` directly, it's derived so that the stationary distribution of ``T`` is ``base_probs``
    (``b ∝ base_probs * (1 - stay)``). So ``base_probs`` keeps its interpretation as the long-run probability of each
    combo, and is also the initial distribution. With ``stay = 0`` this reduces to a static mixture where the
    regime-prior is always ``base_probs``. (If the base-rates vary over time -- components with predictors -- then
    ``base_probs`` is the stationary distribution for the current predictors.)

    :param num_combos: The number of regime-combos.
    :param stay_init: Initial value for the probability of persisting in the current combo.
    """

    def __init__(self, num_combos: int, stay_init: float = 0.1):
        super().__init__(num_combos=num_combos)
        assert 0 < stay_init < 1
        stay_init = torch.as_tensor(stay_init, dtype=torch.get_default_dtype())
        self._stay_logit = torch.nn.Parameter(
            (torch.log(stay_init) - torch.log1p(-stay_init)).expand(num_combos).clone()
        )

    @property
    def stay(self) -> torch.Tensor:
        return torch.sigmoid(self._stay_logit)

    def _jump_probs(self, base_probs: torch.Tensor) -> torch.Tensor:
        b = base_probs * (1 - self.stay)
        return b / b.sum(-1, keepdim=True)

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        stay = self.stay
        # equivalent to `posterior @ self.matrix(base_probs)`, without materializing the matrix:
        leave = (posterior * (1 - stay)).sum(-1, keepdim=True)
        return posterior * stay + leave * self._jump_probs(base_probs)

    def matrix(self, base_probs: torch.Tensor) -> torch.Tensor:
        """
        :param base_probs: A ``(num_combos,)`` tensor.
        :return: The ``(num_combos, num_combos)`` transition-matrix.
        """
        stay = self.stay
        return torch.diag(stay) + (1 - stay).unsqueeze(-1) * self._jump_probs(base_probs).unsqueeze(0)


class MixtureModel(torch.nn.Module):
    """
    Experimental. Mixture components for a state-space model: pass as the ``mixture`` argument of
    :class:`.KalmanFilter` (or :class:`.BinomialFilter`). Owns the :class:`MixtureComponent` objects, the
    :class:`RegimeTransition`, and the fixed table of regime-combos.

    A combo is a tuple with one entry per mixture measure (in the order of ``self.mixture_measures``): either ``None``
    (the standard regime) or one of that measure's components. Measures without components are always in the standard
    regime, and do not appear in the table. The first combo is always all-standard.

    :param components: The mixture components. Mixture measures are ordered by their first appearance here.
    :param transition: A :class:`RegimeTransition`, controlling how regime-probabilities evolve over time. Defaults to
     :class:`StickyTransition`.
    :param univariate_prob: If True, the per-timestep regime-probabilities are computed using only the likelihood of
     the mixture measures, rather than of all observed measures. This is an approximation (exact if the other
     measures' residuals are uncorrelated with the mixture measures'), but can be cheaper. (Non-gaussian measures,
     e.g. binary, never influence the regime-probabilities.)
    :param joseph_form: Whether the update-step for the *non-standard* regime-combos (those with any measure in a
     component's regime) uses the Joseph form of the covariance update. (The standard regime's update follows the
     model's ``joseph_form``.) With mixtures, the update runs once for each regime-combo, and the Joseph form's
     intermediate results (kept for the backward pass) can dominate memory-use during training; ``False`` uses the
     simpler ``P - K @ H @ P`` instead -- less memory (and compute), but less numerically robust. The non-standard
     combos are the safer place for this: a component's extra variance makes their update better-conditioned, and
     their covariances only matter in proportion to their posterior probability. Default True.
    """

    def __init__(self,
                 components: Sequence[MixtureComponent],
                 transition: Optional[RegimeTransition] = None,
                 univariate_prob: bool = False,
                 joseph_form: bool = True):
        super().__init__()
        if not components:
            raise ValueError("`components` cannot be empty.")

        by_measure = {}
        for component in components:
            by_measure.setdefault(component.measure, []).append(component)
        for measure, comps in by_measure.items():
            ids = [c.id for c in comps]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Mixture components must have unique ids within a measure, but '{measure}' got {ids}")

        self.mixture_measures = list(by_measure)
        self.components = torch.nn.ModuleList([c for m in self.mixture_measures for c in by_measure[m]])
        self._by_measure = by_measure
        self.univariate_prob = univariate_prob
        self.joseph_form = joseph_form

        self.combos: list[tuple[Optional[MixtureComponent], ...]] = list(
            itertools.product(*[[None] + self._by_measure[m] for m in self.mixture_measures])
        )

        if transition is None:
            transition = StickyTransition(num_combos=self.num_combos)
        elif transition.num_combos != self.num_combos:
            raise ValueError(f"`transition` has {transition.num_combos} combos, but expected {self.num_combos}")
        self.transition = transition

    def validate(self, measures: Sequence[str], non_gaussian_measures: Sequence[str] = ()) -> None:
        """
        Called by the :class:`.StateSpaceModel`. Mixture measures must be measures of the model, with a gaussian
        likelihood. (A nonlinear measured-mean -- from a measure-function or nonlinear processes -- is fine.)

        :param measures: The model's measures.
        :param non_gaussian_measures: Measures whose likelihood isn't gaussian (e.g. the binary measures of a
         :class:`.BinomialFilter`).
        """
        for component in self.components:
            if component.measure not in measures:
                raise ValueError(
                    f"MixtureComponent '{component.id}' has measure '{component.measure}' not in `measures`"
                )
        for measure in self.mixture_measures:
            if measure in non_gaussian_measures:
                raise ValueError(
                    f"Mixture components are not yet supported for '{measure}', which has a non-gaussian likelihood."
                )

    @property
    def num_combos(self) -> int:
        return len(self.combos)

    @property
    def has_predictors(self) -> bool:
        return any(c.predictors for c in self.components)

    def get_component_X(self, kwargs: Mapping) -> tuple[dict[int, torch.Tensor], set[str]]:
        """
        Pick out the predictors of components' base-rates from the model's ``forward()`` kwargs.

        :param kwargs: The keyword-arguments.
        :return: A tuple of (1) a dictionary mapping the index (in ``self.components``) of each component with
         predictors to its ``(num_groups, num_timesteps, num_predictors)`` tensor, and (2) the keys used.
        """
        out, used = {}, set()
        for i, component in enumerate(self.components):
            if not component.predictors:
                continue
            key = f'{component.id}__{component.kwarg_name}'
            if key not in kwargs:
                key = component.kwarg_name
            if key not in kwargs:
                raise TypeError(
                    f"MixtureComponent '{component.id}' has predictors, so expected a `{component.kwarg_name}` (or "
                    f"`{component.id}__{component.kwarg_name}`) keyword-argument."
                )
            X = kwargs[key]
            if X.ndim != 3 or X.shape[-1] != len(component.predictors):
                raise ValueError(
                    f"Expected `{key}` to have shape (num_groups, num_timesteps, {len(component.predictors)}) for "
                    f"MixtureComponent '{component.id}', got {tuple(X.shape)}."
                )
            out[i] = X
            used.add(key)
        return out, used

    def log_base_probs(self, component_X: Optional[Mapping[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        :param component_X: Required if any components have predictors: the output of :func:`get_component_X`, or
         slices of it (e.g. for one timestep), all with the same leading dims.
        :return: The log long-run probability of each combo: a ``(num_combos,)`` tensor, or with predictors
         ``(*leading_dims, num_combos)``. Regimes are independent across measures; within a measure, the standard
         regime has logit 0 and each component has its own logit.
        """
        component_X = component_X or {}
        logits = [c.get_logit(component_X.get(i)) for i, c in enumerate(self.components)]
        batch_shape = torch.broadcast_shapes(*(lg.shape for lg in logits))
        logits = {id(c): lg.expand(batch_shape) for c, lg in zip(self.components, logits)}

        per_measure = {}
        for measure, comps in self._by_measure.items():
            standard = torch.zeros(batch_shape, dtype=comps[0].logit.dtype, device=comps[0].logit.device)
            lps = torch.log_softmax(torch.stack([standard] + [logits[id(c)] for c in comps], -1), -1).unbind(-1)
            per_measure[measure] = {None: lps[0], **{id(c): lp for c, lp in zip(comps, lps[1:])}}
        return torch.stack([
            sum(per_measure[m][None if c is None else id(c)] for m, c in zip(self.mixture_measures, combo))
            for combo in self.combos
        ], -1)

    def base_probs(self, component_X: Optional[Mapping[int, torch.Tensor]] = None) -> torch.Tensor:
        return self.log_base_probs(component_X).exp()

    def effective_combos(self, measures: Sequence[str]) -> tuple[list[tuple], torch.Tensor]:
        """
        When only a subset of measures is observed, combos that differ only in unobserved measures are
        indistinguishable (they imply the same likelihood and the same state-update).

        :param measures: The measures observed.
        :return: A tuple of (1) the list of unique 'effective' combos -- each a tuple of ``(measure_idx, component)``
         pairs for the observed measures that are in a non-standard regime (``measure_idx`` indexes into
         ``measures``); the first is always the empty tuple (all-standard); and (2) a ``(num_combos,)`` long tensor
         mapping each combo to its effective combo.
        """
        midx = {m: measures.index(m) for m in self.mixture_measures if m in measures}
        effective = [()]
        mapping = []
        for combo in self.combos:
            eff = tuple(
                (midx[m], c) for m, c in zip(self.mixture_measures, combo) if c is not None and m in midx
            )
            if eff not in effective:
                effective.append(eff)
            mapping.append(effective.index(eff))
        return effective, torch.as_tensor(mapping, dtype=torch.long)

    @staticmethod
    def effective_offsets(effective_combo: tuple, num_measures: int, like: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param effective_combo: An entry from :func:`effective_combos`: ``(measure_idx, component)`` pairs.
        :param num_measures: The number of (observed) measures that ``measure_idx`` indexes.
        :param like: A tensor whose dtype/device to use.
        :return: Two ``(num_measures,)`` tensors: the offset to the measured-mean, and the variance to add to the
         measurement-noise, for each measure.
        """
        zero = torch.zeros((), dtype=like.dtype, device=like.device)
        shift, extra_var = [zero] * num_measures, [zero] * num_measures
        for i, component in effective_combo:
            shift[i], extra_var[i] = component.mean, component.var
        return torch.stack(shift), torch.stack(extra_var)

    def combo_offsets(self, measures: Sequence[str], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param measures: The measures.
        :param like: A tensor whose dtype/device to use.
        :return: Two ``(num_combos, len(measures))`` tensors: for each combo, the offset to each measure's
         measured-mean, and the variance to add to its measurement-noise (zero for measures in the standard regime).
        """
        measures = list(measures)
        offsets = [
            self.effective_offsets(
                tuple(
                    (measures.index(m), c) for m, c in zip(self.mixture_measures, combo)
                    if c is not None and m in measures
                ),
                num_measures=len(measures),
                like=like,
            )
            for combo in self.combos
        ]
        return torch.stack([o[0] for o in offsets]), torch.stack([o[1] for o in offsets])

    def standard_probs(self, regime_probs: torch.Tensor, measures: Sequence[str]) -> torch.Tensor:
        """
        :param regime_probs: A ``(num_groups, num_combos)`` tensor of regime-probabilities.
        :param measures: The measures to return probabilities for.
        :return: A ``(num_groups, len(measures))`` tensor with the probability that each measure is in its standard
         regime (always 1 for measures without mixture components).
        """
        mask = torch.ones((self.num_combos, len(measures)), dtype=regime_probs.dtype, device=regime_probs.device)
        for j, measure in enumerate(measures):
            if measure in self.mixture_measures:
                k = self.mixture_measures.index(measure)
                mask[:, j] = torch.as_tensor([float(combo[k] is None) for combo in self.combos])
        return regime_probs @ mask


@dataclass
class MixtureOfNormals:
    """
    A batch of univariate mixtures of normals: the predictive distribution of a single measure with mixture components.
    The last dimension of each tensor indexes the mixture's components; the first is the standard regime. (Each
    component's mean and variance are the standard regime's plus that component's offset and extra variance.)

    For example, to get the mean on the original scale of a log-transformed measure, back-transform each component and
    then mix: ``(mix.probs * torch.exp(mix.means + mix.vars / 2)).sum(-1)``.

    :param labels: A name for each component (``'standard'``, then the :class:`MixtureComponent` ids).
    :param probs: The probability of each component.
    :param means: The mean of each component.
    :param vars: The variance of each component.
    """
    labels: list
    probs: torch.Tensor
    means: torch.Tensor
    vars: torch.Tensor

    def mean(self) -> torch.Tensor:
        return (self.probs * self.means).sum(-1)

    def var(self) -> torch.Tensor:
        return (self.probs * (self.vars + self.means ** 2)).sum(-1) - self.mean() ** 2

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        z = (value.unsqueeze(-1) - self.means) / self.vars.sqrt()
        return (self.probs * torch.special.ndtr(z)).sum(-1)

    def quantile(self, q: float, num_iter: int = 60) -> torch.Tensor:
        """
        The ``q``-th quantile, found by bisection on the cdf.
        """
        assert 0 < q < 1
        sd = self.vars.sqrt()
        lower = (self.means - 10 * sd).min(-1).values
        upper = (self.means + 10 * sd).max(-1).values
        for _ in range(num_iter):
            mid = (lower + upper) / 2
            below = self.cdf(mid) < q
            lower = torch.where(below, mid, lower)
            upper = torch.where(below, upper, mid)
        return (lower + upper) / 2
