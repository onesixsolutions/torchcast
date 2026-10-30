import itertools
from typing import Sequence, Optional, Mapping, Union

import torch

from torchcast.mixture import MixtureComponent, StandardRegime, StickyTransition
from torchcast.mixture.regime_transition import RegimeTransition


class MixtureModel(torch.nn.Module):
    """
    Experimental. Mixture components for a state-space model: pass as the ``mixture`` argument of
    :class:`.KalmanFilter` (or :class:`.BinomialFilter`). Owns the :class:`MixtureComponent` objects, the
    :class:`RegimeTransition`, and the fixed table of regime-combos.

    A combo is a tuple with one entry per mixture measure (in the order of ``self.mixture_measures``): one of that
    measure's regimes (see :func:`get_regimes`) -- its :class:`.StandardRegime`, or one of its components. Measures
    without components are always in the standard regime, and do not appear in the table. The first combo is always
    all-standard.

    :param components: The mixture components. Mixture measures are ordered by their first appearance here.
    :param transition: A :class:`RegimeTransition`, controlling how combo-probabilities evolve over time. Defaults to
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

        self.mixture_measures = list(by_measure)  # TODO: just rename to 'measures' after 55 is merged
        self.components = torch.nn.ModuleList([c for m in self.mixture_measures for c in by_measure[m]])
        self._components_by_id = {c.qualified_id: c for c in self.components}
        # (not in `components`: they have no parameters, but their buffers should follow `.to()`)
        self.standard_regimes = torch.nn.ModuleList([StandardRegime(m) for m in self.mixture_measures])
        self.univariate_prob = univariate_prob
        self.joseph_form = joseph_form

        self.combos: list[tuple[Union[StandardRegime, MixtureComponent], ...]] = list(
            itertools.product(*[self.get_regimes(m) for m in self.mixture_measures])
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
        """The number of regime-combos (``len(self.combos)``)."""
        return len(self.combos)

    def get_regimes(self, measure: str) -> list[Union[StandardRegime, MixtureComponent]]:
        """
        :param measure: A mixture measure.
        :return: The measure's regimes: its :class:`.StandardRegime`, then its components.
        """
        k = self.mixture_measures.index(measure)
        return [self.standard_regimes[k]] + [c for c in self.components if c.measure == measure]

    def get_base_probs(self, component_X: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        See ``get_log_base_probs()``.
        """
        return self.get_log_base_probs(component_X).exp()

    def get_log_base_probs(self, component_X: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Get the log-probability of each combo of regimes (like ``(measure1='standard', measure2='weird')``). This is
        just the product (the sum, on the log-scale) of each independent regime probability, which can be predicted by a
        model-matrix, ``component_X``.

        :param component_X: Required if any components have predictors: the output of ``_get_component_X()`` (keyed
         by ``qualified_id``), or slices of it (e.g. for one timestep), all with the same leading dims.
        :return: The log long-run probability of each combo: a ``(num_combos,)`` tensor, or with predictors
         ``(*leading_dims, num_combos)``. Regimes are independent across measures; within a measure, the standard
         regime has logit 0 and each component has its own logit.
        """
        component_X = component_X or {}

        # per measure: get the log-probability of each of its regimes by softmaxing across its logits:
        log_probs_per_regime = {}
        for measure in self.mixture_measures:
            regime_names, regime_logits = zip(*[
                (r.qualified_id, r.get_logit(component_X.get(r.qualified_id)))
                for r in self.get_regimes(measure)
            ])
            regime_log_probs = torch.log_softmax(torch.stack(torch.broadcast_tensors(*regime_logits), -1), -1)
            log_probs_per_regime.update(zip(regime_names, regime_log_probs.unbind(-1)))

        # now that we have probs for each regime individually, we compute the probability of each *combo* of regimes.
        # we treat them as independent here, so it's just the product (sum on the log-scale)
        logs_probs_per_combo = [sum(log_probs_per_regime[r.qualified_id] for r in combo) for combo in self.combos]
        return torch.stack(logs_probs_per_combo, -1)

    def get_effective_combos(self, measures: Sequence[str]) -> tuple[list[tuple[str, ...]], torch.Tensor]:
        """
        When only a subset of measures is observed, combos that differ only in unobserved measures are
        indistinguishable (they imply the same likelihood and the same state-update). So each 'effective' combo (aka a
        hypothesis) stands for one or more combos: those that are indistinguishable given ``measures``.

        :param measures: The measures observed.
        :return: A tuple of two items. The first is a list of the unique 'effective' combos -- each a tuple of the
         ``qualified_id`` of each component that's in effect, i.e. for the observed measures that are in a
         non-standard regime; the first is always the empty tuple (all-standard). The second is a ``(num_combos,)``
         long tensor mapping each item in ``self.combos`` to its effective combo (an index into the first item).
        """
        effective: list[tuple[str, ...]] = [()]
        mapping = []
        for combo in self.combos:
            eff = tuple(r.qualified_id for r in combo if not r.is_standard and r.measure in measures)
            if eff not in effective:
                effective.append(eff)
            # if it was added on an earlier iter, ``mapping`` tells us where to find the ith combo in ``effective``
            mapping.append(effective.index(eff))
        return effective, torch.as_tensor(mapping, dtype=torch.long)

    def get_combo_offsets(self,
                          components: tuple[str, ...],
                          measures: Sequence[str],
                          like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param components: The ``qualified_id`` of each component that's in effect.
        :param measures: The (observed) measures.
        :param like: A tensor whose dtype/device to use.
        :return: Two ``(num_measures,)`` tensors: the offset to the measured-mean, and the variance to add to the
         measurement-noise, for each measure.
        """
        zero = torch.zeros((), dtype=like.dtype, device=like.device)
        shift, extra_var = [zero] * len(measures), [zero] * len(measures)
        for qualified_id in components:
            component = self._components_by_id[qualified_id]
            midx = measures.index(component.measure)
            shift[midx], extra_var[midx] = component.offset, component.extra_var
        return torch.stack(shift), torch.stack(extra_var)

    def get_standard_probs(self, combo_probs: torch.Tensor, measures: Sequence[str]) -> torch.Tensor:
        """
        The probability that each measure is in its standard regime.

        :param combo_probs: A ``(num_groups, num_combos)`` tensor of combo-probabilities.
        :param measures: The measures to return probabilities for.
        :return: A ``(num_groups, len(measures))`` tensor with the probability that each measure is in its standard
         regime (always 1 for measures without mixture components).
        """
        mask = torch.ones((self.num_combos, len(measures)), dtype=combo_probs.dtype, device=combo_probs.device)
        for j, measure in enumerate(measures):
            if measure in self.mixture_measures:
                k = self.mixture_measures.index(measure)
                mask[:, j] = torch.as_tensor([float(combo[k].is_standard) for combo in self.combos])
        return combo_probs @ mask

    def prepare(self,
                kwargs: Mapping,
                init_combo_probs: Optional[torch.Tensor],
                num_groups: int,
                num_timesteps: int) -> tuple[torch.Tensor, Sequence[torch.Tensor], set[str]]:
        """
        For a model's ``forward()``: the combo-prior for the first timestep, and the base-rates for the combo-prior
        at each timestep (``base_probs[t]``; time-varying if any components have predictors).

        :param kwargs: The ``forward()`` kwargs, which may include the components' predictors.
        :param init_combo_probs: The ``combo_probs`` of ``forward()``'s ``initial_state``, if any (see
         ``_initial_prior``).
        :return: ``(combo_prior, base_probs, used_keys)``, where ``used_keys`` are the kwargs used.
        """
        component_X, used_keys = self._get_component_X(kwargs)

        # base-probs, the long run probability that the transitioning func converges to (for a given set of predictors)
        # in the absence of new observations/updates:
        if component_X:
            base_probs = self.get_base_probs(component_X).unbind(1)
            if len(base_probs) < num_timesteps:
                raise ValueError(
                    f"The mixture components' predictors have {len(base_probs)} timesteps, but expected (at least) "
                    f"{num_timesteps}."
                )
        else:
            base_probs = [self.get_base_probs()] * num_timesteps

        # combo-prior: for the most common case of a fresh run, it's just passing base-probs to the transition-func's
        # ``initial()`` method (and in the most common case that's just returning base_probs).
        # so in most cases, the prior == base_probs.
        if init_combo_probs is None:
            combo_prior = self.transition.initial(base_probs[0], num_groups)
        else:
            combo_prior = self._initial_prior(init_combo_probs, base_probs[0], num_groups)

        return combo_prior, base_probs, used_keys

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

    def _initial_prior(self,
                       init_combo_probs: torch.Tensor,
                       base_probs: torch.Tensor,
                       num_groups: int) -> torch.Tensor:
        """
        The combo-prior for the first timestep. ``init_combo_probs`` (from ``initial_state``) are treated like the
        mean/cov of ``initial_state``: as the state *before* the first timestep, so they're evolved one step by the
        transition.
        """

        if init_combo_probs.ndim != 2 or init_combo_probs.shape[-1] != self.num_combos:
            raise ValueError(
                f"Expected `initial_state.combo_probs` to have shape (num_groups, {self.num_combos}), "
                f"got {tuple(init_combo_probs.shape)}"
            )

        if init_combo_probs.shape[0] not in (1, num_groups):
            raise ValueError(f"Expected `initial_state.combo_probs.shape[0]` to be 1 or {num_groups}")

        return self.transition(init_combo_probs.expand(num_groups, -1), base_probs)
