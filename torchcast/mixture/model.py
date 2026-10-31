import itertools
import math
from typing import Sequence, Optional, Mapping, Union

import torch

from torchcast.mixture import LatentClass, MixtureComponent, StandardRegime, StickyTransition
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
    :param joseph_form: Whether the update-step for the *non-standard* regime-combos (those with any measure in a
     component's regime) uses the Joseph form of the covariance update. (The standard regime's update follows the
     model's ``joseph_form``.) With mixtures, the update runs once for each regime-combo, and the Joseph form's
     intermediate results (kept for the backward pass) can dominate memory-use during training; ``False`` uses the
     simpler ``P - K @ H @ P`` instead -- less memory (and compute), but less numerically robust. The non-standard
     combos are the safer place for this: a component's extra variance makes their update better-conditioned, and
     their covariances only matter in proportion to their posterior probability. Default True.
    :param classes: Optional :class:`LatentClass` objects (at least two): latent classes of groups, each with its own
     base-rate for each component (a discrete random effect on the base-rates). Default None (no classes; i.e. one).
     With ``K`` classes, the table of combos is repeated once per class (class-major: ``self.combos[i]`` is in class
     ``self.combo_class[i]``), and the base-rates are learned per class: each component's logit gets a per-class offset,
     and the classes have prior logits (both fixed at 0 for the first class, for identifiability). Otherwise the classes
     are the same: the ``transition`` is shared, applied within each class with that class's base-rates (see
     :class:`RegimeTransition`), so its ``num_combos`` is the number of combos per class (``self.num_class_combos``);
     and groups never change class. The initial values come from each class's ``membership_prob_init`` and
     ``relative_rate_init``; the latter is required for all but the first class, since otherwise the classes would be
     symmetric. Only the bookkeeping of combo-probabilities grows with ``K``: combos with the same observable regimes
     share a state-update and likelihood (see :func:`get_effective_combos`). Use :func:`get_class_probs` for the
     posterior probability of each class. Note that ``Predictions.sample()`` draws the regime-combo independently at
     each timestep (as it does the state; samples are not trajectories), so a sample's class can differ across
     timesteps: the per-timestep distributions are right, but a path from these samples doesn't stay in one class. For
     trajectories, use ``simulate()``, which simulates the regimes (and so the class) along with the state.
    """

    def __init__(self,
                 components: Sequence[MixtureComponent],
                 transition: Optional[RegimeTransition] = None,
                 joseph_form: bool = True,
                 classes: Optional[Sequence[LatentClass]] = None):
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
        self.joseph_form = joseph_form

        self.classes = list(classes or [])
        num_classes = max(len(self.classes), 1)
        self.num_classes = num_classes
        # the combos within a class; with latent classes, these are repeated for each class (class-major):
        class_combos = list(itertools.product(*[self.get_regimes(m) for m in self.mixture_measures]))
        self.combos: list[tuple[Union[StandardRegime, MixtureComponent], ...]] = class_combos * num_classes
        self.combo_class: list[int] = [c for c in range(num_classes) for _ in class_combos]

        self._class_prior_logits = self._class_offsets = None
        if self.classes:
            self._init_classes()

        if transition is None:
            transition = StickyTransition(num_combos=self.num_class_combos)
        elif transition.num_combos != self.num_class_combos:
            per_class = " per class" if num_classes > 1 else ""
            raise ValueError(
                f"`transition` has {transition.num_combos} combos, but expected {self.num_class_combos}{per_class}"
            )
        self.transition = transition

    def _init_classes(self) -> None:
        classes = self.classes
        if len(classes) < 2:
            raise ValueError("`classes` needs at least two classes (or None, for no classes).")
        ids = [c.id for c in classes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Latent classes must have unique ids, got {ids}")
        if classes[0].relative_rate_init:
            raise ValueError(
                f"LatentClass '{ids[0]}' is the first (reference) class, so can't have `relative_rate_init`: the other "
                f"classes' rates are relative to its."
            )
        component_ids = [c.qualified_id for c in self.components]
        for cls in classes[1:]:
            if not any(ratio != 1 for ratio in cls.relative_rate_init.values()):
                raise ValueError(
                    f"LatentClass '{cls.id}' needs a `relative_rate_init` with at least one ratio other than 1 (vs. "
                    f"the reference class '{ids[0]}'); otherwise the classes start out identical."
                )
            unknown = set(cls.relative_rate_init) - set(component_ids)
            if unknown:
                raise ValueError(
                    f"LatentClass '{cls.id}': `relative_rate_init` has unknown components {sorted(unknown)}; expected "
                    f"(a subset of) {component_ids}"
                )
        membership = [c.membership_prob_init for c in classes]
        if any(p is None for p in membership) and not all(p is None for p in membership):
            raise ValueError("`membership_prob_init` should be given for all classes, or for none.")

        to = {'dtype': torch.get_default_dtype()}
        # the prior logit of classes 1, ..., K-1 (class 0's is fixed at 0):
        log_membership = torch.zeros(len(classes), **to)
        if membership[0] is not None:
            log_membership = torch.as_tensor(membership, **to).log()
        self._class_prior_logits = torch.nn.Parameter(log_membership[1:] - log_membership[0])
        # for each component (in the order of self.components), the offset to its logit in classes 1, ..., K-1 (class
        # 0's is fixed at 0): the log of the ratio of its odds (vs. its measure's standard regime) to class 0's.
        offsets = torch.zeros(len(self.components), len(classes) - 1, **to)
        for k, cls in enumerate(classes[1:]):
            for qualified_id, ratio in cls.relative_rate_init.items():
                offsets[component_ids.index(qualified_id), k] = math.log(ratio)
        self._class_offsets = torch.nn.Parameter(offsets)

    def _class_log_prior(self) -> torch.Tensor:
        """
        :return: A ``(num_classes,)`` tensor: the log prior probability of each latent class (with classes).
        """
        logits = self._class_prior_logits
        return torch.log_softmax(torch.cat([torch.zeros_like(logits[:1]), logits]), 0)

    def get_class_probs(self, combo_probs: torch.Tensor) -> torch.Tensor:
        """
        :param combo_probs: A ``(..., num_combos)`` tensor of combo-probabilities (e.g. ``Predictions.combo_probs``).
        :return: A ``(..., num_classes)`` tensor: the probability of each latent class.
        """
        return combo_probs.unflatten(-1, (self.num_classes, self.num_class_combos)).sum(-1)

    def get_next_prior(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        """
        The combo-prior at the next timestep: ``self.transition`` applied to the combo-posterior. With latent
        classes, it's applied within each class (see :class:`RegimeTransition`), so the probability of each class is
        unchanged.

        :param posterior: A ``(num_groups, num_combos)`` tensor of combo-probabilities at the current timestep.
        :param base_probs: A ``(num_combos,)`` or ``(num_groups, num_combos)`` tensor, for the next timestep.
        :return: A ``(num_groups, num_combos)`` tensor of combo-probabilities at the next timestep.
        """
        if self.num_classes == 1:
            return self.transition(posterior, base_probs)
        class_mass, within = self._split_classes(posterior)
        _, within_base = self._split_classes(base_probs.expand_as(posterior))
        return self._merge_classes(class_mass, self.transition(within, within_base))

    def _initial_combo_probs(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        The transition's ``initial()`` (within each class, with latent classes).
        """
        if self.num_classes == 1:
            return self.transition.initial(base_probs, num_groups)
        class_mass, within_base = self._split_classes(base_probs.expand(num_groups, -1))
        return self._merge_classes(class_mass, self.transition.initial(within_base, num_groups * self.num_classes))

    def _split_classes(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param probs: A ``(num_groups, num_combos)`` tensor of combo-probabilities.
        :return: The ``(num_groups, num_classes, 1)`` probability of each class, and the ``(num_groups * num_classes,
         num_class_combos)`` probabilities of the combos given the class (the classes folded into the rows, so a
         transition sees them as more groups).
        """
        by_class = probs.unflatten(-1, (self.num_classes, self.num_class_combos))
        class_mass = by_class.sum(-1, keepdim=True)
        return class_mass, (by_class / class_mass.clamp_min(1e-30)).flatten(0, 1)

    def _merge_classes(self, class_mass: torch.Tensor, within: torch.Tensor) -> torch.Tensor:
        """
        The inverse of ``_split_classes()``: back to a ``(num_groups, num_combos)`` tensor.
        """
        return (class_mass * within.unflatten(0, class_mass.shape[:2])).flatten(-2)

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

    @property
    def num_class_combos(self) -> int:
        """The number of combos within each latent class (``num_combos`` without classes)."""
        return self.num_combos // self.num_classes

    def get_regimes(self, measure: str) -> list[Union[StandardRegime, MixtureComponent]]:
        """
        :param measure: A mixture measure.
        :return: The measure's regimes: its :class:`.StandardRegime`, then its components.
        """
        k = self.mixture_measures.index(measure)
        return [self.standard_regimes[k]] + [c for c in self.components if c.measure == measure]

    def get_base_probs(self, component_X: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        See ``_get_log_base_probs()``.
        """
        return self._get_log_base_probs(component_X).exp()

    def _get_log_base_probs(self, component_X: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Get the log-probability of each combo of regimes (like ``(measure1='standard', measure2='weird')``). This is
        just the product (the sum, on the log-scale) of each independent regime probability, which can be predicted by a
        model-matrix, ``component_X``.

        :param component_X: Required if any components have predictors: the output of ``_get_component_X()`` (keyed
         by ``qualified_id``), or slices of it (e.g. for one timestep), all with the same leading dims.
        :return: The log long-run probability of each combo: a ``(num_combos,)`` tensor, or with predictors
         ``(*leading_dims, num_combos)``. Regimes are independent across measures; within a measure, the standard
         regime has logit 0 and each component has its own logit. With latent classes, this is the log prior
         probability of the combo's class, plus the log-probability of its regimes given the class (whose logits add
         the class's offsets to the components' logits).
        """
        component_X = component_X or {}
        class_offsets = None
        if self.num_classes > 1:
            # for each component, its logit's offset in each class (the first class's is 0); added along a new last
            # dim, (..., num_classes). (the standard regime's logit stays 0 in every class.)
            class_offsets = torch.cat([torch.zeros_like(self._class_offsets[:, :1]), self._class_offsets], 1)
            component_idx = {c.qualified_id: i for i, c in enumerate(self.components)}

        # per measure: get the log-probability of each of its regimes by softmaxing across its logits:
        log_probs_per_regime = {}
        for measure in self.mixture_measures:
            regime_names, regime_logits = zip(*[
                (r.qualified_id, r.get_logit(component_X.get(r.qualified_id)))
                for r in self.get_regimes(measure)
            ])
            if class_offsets is not None:
                regime_logits = [
                    logit if r.is_standard else logit.unsqueeze(-1) + class_offsets[component_idx[r.qualified_id]]
                    for r, logit in zip(self.get_regimes(measure), regime_logits)
                ]
            regime_log_probs = torch.log_softmax(torch.stack(torch.broadcast_tensors(*regime_logits), -1), -1)
            log_probs_per_regime.update(zip(regime_names, regime_log_probs.unbind(-1)))

        # now that we have probs for each regime individually, we compute the probability of each *combo* of regimes.
        # we treat them as independent here, so it's just the product (sum on the log-scale)
        class_combos = self.combos[:self.num_class_combos]
        logs_probs_per_combo = [sum(log_probs_per_regime[r.qualified_id] for r in combo) for combo in class_combos]
        log_probs = torch.stack(logs_probs_per_combo, -1)
        if self.num_classes == 1:
            return log_probs
        # (..., num_classes, num_class_combos) -> class-major (..., num_combos):
        return (self._class_log_prior().unsqueeze(-1) + log_probs).flatten(-2)

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
        :param init_combo_probs: The ``combo_probs`` of ``forward()``'s ``initial_state``, if any (evolved one step by
         the transition, like the mean/cov of ``initial_state``).
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
        # ``initial()`` method (within each class, with latent classes; and in the most common case that's just
        # returning base_probs).
        # so in most cases, the prior == base_probs.
        if init_combo_probs is None:
            combo_prior = self._initial_combo_probs(base_probs[0], num_groups)
        else:
            # alternate case: init_combo_probs was passed as part of initial state. like the mean/cov of
            # ``initial_state``, we treat as the state *before* the first timestep, so they're evolved one step by the
            # transition.
            combo_prior = self.get_next_prior(init_combo_probs.expand(num_groups, -1), base_probs[0])

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
