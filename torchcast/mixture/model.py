import itertools
from typing import Sequence, Optional, Mapping

import torch

from torchcast.mixture import MixtureComponent, StickyTransition
from torchcast.mixture.regime_transition import RegimeTransition


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
        likelihood.

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

    def _get_component_X(self, kwargs: Mapping) -> tuple[dict[str, torch.Tensor], set[str]]:
        """
        Pick out the predictors of components' base-rates from the model's ``forward()`` kwargs.

        :param kwargs: The keyword-arguments.
        :return: A tuple of (1) a dictionary mapping the ``qualified_id`` of each component with predictors to its
         ``(num_groups, num_timesteps, num_predictors)`` tensor, and (2) the keys used.
        """
        out, used = {}, set()
        for component in self.components:
            if not component.predictors:
                continue
            name = component.model_mat_kwarg_name
            key = f'{component.id}__{name}'
            if key not in kwargs:
                key = name
            if key not in kwargs:
                raise TypeError(
                    f"MixtureComponent '{component.id}' has predictors, so expected a `{name}` (or "
                    f"`{component.id}__{name}`) keyword-argument."
                )
            X = kwargs[key]
            if X.ndim != 3 or X.shape[-1] != len(component.predictors):
                raise ValueError(
                    f"Expected `{key}` to have shape (num_groups, num_timesteps, {len(component.predictors)}) for "
                    f"MixtureComponent '{component.id}', got {tuple(X.shape)}."
                )
            out[component.qualified_id] = X
            used.add(key)
        return out, used

    def log_base_probs(self, component_X: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        :param component_X: Required if any components have predictors: the output of ``_get_component_X()`` (keyed
         by ``qualified_id``), or slices of it (e.g. for one timestep), all with the same leading dims.
        :return: The log long-run probability of each combo: a ``(num_combos,)`` tensor, or with predictors
         ``(*leading_dims, num_combos)``. Regimes are independent across measures; within a measure, the standard
         regime has logit 0 and each component has its own logit.
        """
        component_X = component_X or {}
        logits = {c.qualified_id: c.get_logit(component_X.get(c.qualified_id)) for c in self.components}
        batch_shape = torch.broadcast_shapes(*(lg.shape for lg in logits.values()))
        to = {'dtype': self.components[0].logit.dtype, 'device': self.components[0].logit.device}

        # per measure: the log-probability of each of its regimes (None = standard, then each component):
        regime_log_probs = {}
        for measure, comps in self._by_measure.items():
            measure_logits = [torch.zeros(batch_shape, **to)]
            measure_logits += [logits[c.qualified_id].expand(batch_shape) for c in comps]
            log_probs = torch.log_softmax(torch.stack(measure_logits, -1), -1).unbind(-1)
            regime_log_probs[measure] = dict(zip([None] + [c.qualified_id for c in comps], log_probs))

        # per combo: regimes are independent across measures, so their log-probabilities add:
        combo_log_probs = [
            sum(
                regime_log_probs[measure][None if c is None else c.qualified_id]
                for measure, c in zip(self.mixture_measures, combo)
            )
            for combo in self.combos
        ]
        return torch.stack(combo_log_probs, -1)

    def base_probs(self, component_X: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        return self.log_base_probs(component_X).exp()

    def prepare(self,
                kwargs: Mapping,
                init_regime_probs: Optional[torch.Tensor],
                num_groups: int,
                num_timesteps: int) -> tuple[torch.Tensor, Sequence[torch.Tensor], set[str]]:
        """
        For a model's ``forward()``: the regime-prior for the first timestep, and the base-rates for the regime-prior
        at each timestep (``base_probs[t]``; time-varying if any components have predictors).

        :param kwargs: The ``forward()`` kwargs, which may include the components' predictors.
        :param init_regime_probs: The ``regime_probs`` of ``forward()``'s ``initial_state``, if any (see
         ``_initial_prior``).
        :return: ``(regime_prior, base_probs, used_keys)``, where ``used_keys`` are the kwargs used.
        """
        component_X, used_keys = self._get_component_X(kwargs)
        if component_X:
            base_probs = self.base_probs(component_X).unbind(1)
            if len(base_probs) < num_timesteps:
                raise ValueError(
                    f"The mixture components' predictors have {len(base_probs)} timesteps, but expected (at least) "
                    f"{num_timesteps}."
                )
        else:
            base_probs = [self.base_probs()] * num_timesteps
        regime_prior = self._initial_prior(init_regime_probs, base_probs[0], num_groups)
        return regime_prior, base_probs, used_keys

    def _initial_prior(self,
                      init_regime_probs: Optional[torch.Tensor],
                      base_probs: torch.Tensor,
                      num_groups: int) -> torch.Tensor:
        """
        The regime-prior for the first timestep. ``init_regime_probs`` (from ``initial_state``) are treated like the
        mean/cov of ``initial_state``: as the state *before* the first timestep, so they're evolved one step by the
        transition. If not given, the transition's ``initial()`` is used.
        """
        if init_regime_probs is None:
            return self.transition.initial(base_probs, num_groups)
        if init_regime_probs.ndim != 2 or init_regime_probs.shape[-1] != self.num_combos:
            raise ValueError(
                f"Expected `initial_state.regime_probs` to have shape (num_groups, {self.num_combos}), "
                f"got {tuple(init_regime_probs.shape)}"
            )
        if init_regime_probs.shape[0] not in (1, num_groups):
            raise ValueError(f"Expected `initial_state.regime_probs.shape[0]` to be 1 or {num_groups}")
        return self.transition(init_regime_probs.expand(num_groups, -1), base_probs)

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
    def effective_offsets(effective_combo: tuple,
                          num_measures: int,
                          like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            shift[i], extra_var[i] = component.offset, component.extra_var
        return torch.stack(shift), torch.stack(extra_var)

    def combo_offsets(self, measures: Sequence[str], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param measures: The measures.
        :param like: A tensor whose dtype/device to use.
        :return: Two ``(num_combos, len(measures))`` tensors: for each combo, the offset to each measure's
         measured-mean, and the variance to add to its measurement-noise (zero for measures in the standard regime).
        """
        measures = list(measures)
        offsets = (
            self.effective_offsets(
                tuple(
                    (measures.index(m), c) for m, c in zip(self.mixture_measures, combo)
                    if c is not None and m in measures
                ),
                num_measures=len(measures),
                like=like,
            )
            for combo in self.combos
        )
        shift, extra_var = zip(*offsets)
        return torch.stack(shift), torch.stack(extra_var)

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
