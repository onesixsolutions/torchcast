"""
Mixture-of-regimes support for state-space models.

A measure can have one or more :class:`MixtureComponent` alternatives to its standard (state-dependent) measurement
model. Each component is *stateless*: an observation explained by it is scored against the component's own learned
mean and variance, and does not update the latent state.

The :class:`RegimeModel` owns the components and enumerates a fixed table of regime "combos" -- one entry per
combination of regimes across the mixture measures (the standard regime, or one of that measure's components). Regime
probabilities are tracked jointly over this table.
"""
import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


class MixtureComponent(torch.nn.Module):
    """
    An alternative ('weird') regime for a single measure, with a learnable mean, variance, and base-rate.

    :param measure: The measure this component applies to.
    :param mean_init: Initial value for the component's mean.
    :param prob_init: Initial value for the component's base-rate, i.e. the long-run probability that an observation
     for ``measure`` comes from this component (exact when it's the only component for this measure).
    :param id: A unique identifier (within the measure).
    """

    def __init__(self,
                 measure: str,
                 mean_init: float,
                 prob_init: float,
                 id: str):
        super().__init__()
        self.measure = measure
        self.id = id

        mean_init = torch.as_tensor(mean_init, dtype=torch.get_default_dtype())
        self.mean = torch.nn.Parameter(mean_init)
        self._log_std = torch.nn.Parameter(torch.zeros_like(mean_init))

        assert 0 < prob_init < 1
        prob_init = torch.as_tensor(prob_init, dtype=torch.get_default_dtype())
        # logit relative to the standard regime, whose logit is fixed at 0
        self.logit = torch.nn.Parameter(torch.log(prob_init) - torch.log1p(-prob_init))

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
    base-rates; implementations may use or ignore it.
    """

    def __init__(self, num_combos: int):
        super().__init__()
        self.num_combos = num_combos

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        :param base_probs: A ``(num_combos,)`` tensor.
        :param num_groups: The number of groups.
        :return: A ``(num_groups, num_combos)`` tensor of regime-probabilities.
        """
        raise NotImplementedError

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        """
        :param posterior: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the current timestep.
        :param base_probs: A ``(num_combos,)`` tensor.
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
    regime-prior is always ``base_probs``.

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
        return b / b.sum()

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        stay = self.stay
        # equivalent to `posterior @ self.matrix(base_probs)`, without materializing the matrix:
        leave = (posterior * (1 - stay)).sum(-1, keepdim=True)
        return posterior * stay + leave * self._jump_probs(base_probs)

    def matrix(self, base_probs: torch.Tensor) -> torch.Tensor:
        stay = self.stay
        return torch.diag(stay) + (1 - stay).unsqueeze(-1) * self._jump_probs(base_probs).unsqueeze(0)


class RegimeModel(torch.nn.Module):
    """
    Owns the :class:`MixtureComponent`s for a model and the fixed table of regime-combos.

    A combo is a tuple with one entry per mixture measure (in the order of ``self.mixture_measures``): either ``None``
    (the standard regime) or one of that measure's components. Measures without components are always in the standard
    regime, and do not appear in the table. The first combo is always all-standard.

    :param components: The mixture components.
    :param measures: All measures of the model (used to validate and order the mixture measures).
    :param transition: A :class:`RegimeTransition`. Defaults to :class:`StickyTransition`.
    """

    def __init__(self,
                 components: Sequence[MixtureComponent],
                 measures: Sequence[str],
                 transition: Optional[RegimeTransition] = None):
        super().__init__()
        if not components:
            raise ValueError("`components` cannot be empty.")

        by_measure = {}
        for component in components:
            if component.measure not in measures:
                raise ValueError(f"MixtureComponent '{component.id}' has measure '{component.measure}' not in `measures`")
            by_measure.setdefault(component.measure, []).append(component)
        for measure, comps in by_measure.items():
            ids = [c.id for c in comps]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Mixture components must have unique ids within a measure, but '{measure}' got {ids}")

        self.mixture_measures = [m for m in measures if m in by_measure]
        self.components = torch.nn.ModuleList([c for m in self.mixture_measures for c in by_measure[m]])
        self._by_measure = {m: by_measure[m] for m in self.mixture_measures}

        self.combos: list[tuple[Optional[MixtureComponent], ...]] = list(
            itertools.product(*[[None] + self._by_measure[m] for m in self.mixture_measures])
        )

        if transition is None:
            transition = StickyTransition(num_combos=self.num_combos)
        elif transition.num_combos != self.num_combos:
            raise ValueError(f"`transition` has {transition.num_combos} combos, but expected {self.num_combos}")
        self.transition = transition

    @property
    def num_combos(self) -> int:
        return len(self.combos)

    def log_base_probs(self) -> torch.Tensor:
        """
        :return: A ``(num_combos,)`` tensor with the log long-run probability of each combo. Regimes are independent
         across measures; within a measure, the standard regime has logit 0 and each component has its own logit.
        """
        per_measure = {}
        for measure, comps in self._by_measure.items():
            logits = torch.stack([torch.zeros_like(comps[0].logit)] + [c.logit for c in comps])
            per_measure[measure] = {
                None if i == 0 else comps[i - 1].id: lp for i, lp in enumerate(torch.log_softmax(logits, 0))
            }
        return torch.stack([
            sum(per_measure[m][None if c is None else c.id] for m, c in zip(self.mixture_measures, combo))
            for combo in self.combos
        ])

    def base_probs(self) -> torch.Tensor:
        return self.log_base_probs().exp()

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

    def validate_measures(self, measure_funs: dict, processes: Sequence) -> None:
        """
        Mixture measures must have a linear-Gaussian measurement model (no measure-function, no nonlinear processes).
        """
        for measure in self.mixture_measures:
            if measure in measure_funs:
                raise ValueError(f"Mixture components are not supported for '{measure}', which has a measure-function.")
            nonlinear = [p.id for p in processes if p.measure == measure and not p.linear_measurement]
            if nonlinear:
                raise ValueError(
                    f"Mixture components are not supported for '{measure}', which has nonlinear processes: {nonlinear}"
                )


@dataclass
class MixtureOfNormals:
    """
    A batch of univariate mixtures of normals: the predictive distribution of a single measure with mixture components.
    The last dimension of each tensor indexes the mixture's components; the first is the standard regime.

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
