"""
Mixture-of-regimes support for state-space models.

A measure can have one or more :class:`MixtureComponent` alternatives to its standard measurement model. Each
component is an *offset* from the standard regime: in the component's regime, an observation is the usual
(state-dependent) measured-mean plus the component's learned offset, with its learned variance added to the
measurement-noise. This is particularly useful for rare outliers with a clear data-generating process (e.g. forecasting
the activity of customers who usually have long visits but occasionally have very short ones), especially when
forecasting with data-transformations that make these outliers problematic in the back-transform (e.g. if the customer
duration data was log1p transformed, an outlier will increase the variance and so lead to implausibly large
back-transformed forecasts).

The :class:`MixtureModel` owns the components. Multivariate support is accomplished by enumerating the possible "combos"
 of regimes (e.g. [(measure1=standard, measure2=standard), (measure1=weird, measure2=standard), ...] and so on). Regime
probabilities are tracked jointly over this table, allowing p(measure_x=weird) to vary over time and across groups.

Optionally, groups can belong to latent classes (``MixtureModel(..., num_classes=K)``) with their own base-rates for
each component -- a discrete random effect on the base-rates, e.g. some customers have outliers much more often than
others. The table of combos is then repeated once per class. Classes share the transition (it applies within each
class), and a group never changes class, so its posterior over classes accumulates evidence over all of its timesteps,
and carries over into forecasts.
"""
import itertools
from typing import Optional, Sequence, Mapping

import torch


class MixtureComponent(torch.nn.Module):
    """
    An alternative ('weird') regime for a single measure: a learned offset from the usual state-mean, and a learned
    increase in measurement noise.

    :param measure: The measure this component applies to.
    :param offset_init: Initial value for the component's offset, relative to the standard regime's measured-mean. For
     example, if you know you're tracking anomalously low values, you can pick a value several standard-deviations
     below the global average).
    :param prob_init: Initial value for the component's base-rate, i.e. the long-run probability that an observation
     for ``measure`` comes from this component (exact when it's the only component for this measure). With
     ``predictors``, this is the base-rate when the predictors are zero.
    :param id: A unique identifier (within the measure).
    :param predictors: Optional names of predictors of the component's base-rate: each has a learned coefficient,
     added to the component's logit (vs. the standard regime). E.g. a treatment that makes this regime more common.
     The predictors are passed to the model's ``forward()`` as a ``(num_groups, num_timesteps, len(predictors))``
     tensor ``X`` (see ``model_mat_kwarg_name``) -- or, to use a different tensor than other consumers of ``X`` (e.g.
     a ``LinearModel``), as ``{id}__X``. Like the predictors of a ``LinearModel``, they need to cover the forecast
     horizon too: the regime-prior for each timestep uses that timestep's predictors.
    :param model_mat_kwarg_name: The name of the ``forward()`` keyword-argument with the predictors (as for
     ``LinearModel``). Default 'X'.
    """

    def __init__(self,
                 measure: str,
                 offset_init: float,
                 prob_init: float,
                 id: str,
                 predictors: Optional[Sequence[str]] = None,
                 model_mat_kwarg_name: str = 'X'):
        super().__init__()
        self.measure = measure
        self.id = id
        self.model_mat_kwarg_name = model_mat_kwarg_name
        if isinstance(predictors, str):
            raise ValueError("`predictors` should be a list of strings, not a string.")
        self.predictors = list(predictors or [])

        offset_init = torch.as_tensor(offset_init, dtype=torch.get_default_dtype())
        self.offset = torch.nn.Parameter(offset_init)
        self._log_extra_std = torch.nn.Parameter(torch.zeros_like(offset_init))

        assert 0 < prob_init < 1
        prob_init = torch.as_tensor(prob_init, dtype=torch.get_default_dtype())
        # logit relative to the standard regime, whose logit is fixed at 0
        self.logit = torch.nn.Parameter(torch.log(prob_init) - torch.log1p(-prob_init))
        self.coefs = None
        if self.predictors:
            self.coefs = torch.nn.Parameter(torch.zeros(len(self.predictors)))

    @property
    def qualified_id(self) -> str:
        """``'{measure}.{id}'``: unique across all of a model's components (``id`` is unique within a measure)."""
        return f'{self.measure}.{self.id}'

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
    def extra_var(self) -> torch.Tensor:
        """The variance this regime adds to the measurement-noise variance."""
        return self._log_extra_std.exp() ** 2

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id={repr(self.id)}, measure={repr(self.measure)})'


class RegimeTransition(torch.nn.Module):
    """
    Base-class for how regime-probabilities evolve over time. Subclasses implement :func:`initial` (the regime-prior
    at the first timestep) and :func:`forward` (the regime-prior at the next timestep, given the regime-posterior at
    the current timestep).

    Both receive ``base_probs``, the long-run probability of each combo implied by the :class:`MixtureComponent`
    base-rates: a ``(num_combos,)`` tensor (or, if any components have predictors, a ``(num_groups, num_combos)``
    tensor, for the timestep that the regime-prior is for).

    With latent classes (``MixtureModel(num_classes > 1)``), the transition only sees the combos *within* a class (so
    ``num_combos`` is the number per class): the :class:`MixtureModel` applies it to each class separately -- to the
    combo-probabilities given the class, with that class's base-rates (normalized within the class), as if each class
    of each group were a separate group (``(num_groups * num_classes, num_combos)`` tensors) -- and scales the result
    by the probability of the class. So groups never change class, and the classes share the transition's
    parameters, differing only in their base-rates.
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



class StaticTransition(RegimeTransition):
    """
    A static mixture: the regimes are independent across timesteps, so the regime-prior at every timestep is
    ``base_probs`` (whatever the previous posterior). It has no parameters. With latent classes
    (``MixtureModel(num_classes > 1)``), the regimes are independent across timesteps *given* the class, which stays
    fixed: each group's base-rates are a mixture over the classes, learned from all of its timesteps.

    :param num_combos: The number of regime-combos.
    """

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        return base_probs.expand(posterior.shape[0], -1)


class StickyTransition(RegimeTransition):
    """
    Each combo ``i`` persists with probability ``stay_i``; otherwise the next combo is drawn from a shared distribution
    ``b``. That is, ``T[i, j] = stay_i * (i == j) + (1 - stay_i) * b_j``.

    Rather than learning ``b`` directly, it's derived so that the stationary distribution of ``T`` is ``base_probs``
    (``b ∝ base_probs * (1 - stay)``). So ``base_probs`` keeps its interpretation as the long-run probability of each
    combo, and is also the initial distribution. With ``stay = 0`` this reduces to a static mixture where the
    regime-prior is always ``base_probs`` (as in :class:`StaticTransition`). (If the base-rates vary over time --
    components with predictors -- then ``base_probs`` is the stationary distribution for the current predictors.)

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

    def get_stay_prob(self) -> torch.Tensor:
        """
        :return: A ``(num_combos,)`` tensor: the probability of persisting in each combo, ``stay``.
        """
        return torch.sigmoid(self._stay_logit)

    def _jump_probs(self, base_probs: torch.Tensor) -> torch.Tensor:
        b = base_probs * (1 - self.get_stay_prob())
        return b / b.sum(-1, keepdim=True)

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        stay = self.get_stay_prob()
        # equivalent to `posterior @ self.matrix(base_probs)`, without materializing the matrix:
        leave = (posterior * (1 - stay)).sum(-1, keepdim=True)
        return posterior * stay + leave * self._jump_probs(base_probs)

    def matrix(self, base_probs: torch.Tensor) -> torch.Tensor:
        """
        The transition-matrix, for inspection and testing (:func:`forward` applies it without building it).

        :param base_probs: A ``(num_combos,)`` tensor (so not for components with predictors, whose base-rates vary
         by group).
        :return: The ``(num_combos, num_combos)`` transition-matrix.
        """
        stay = self.get_stay_prob()
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
    :param num_classes: The number of latent classes of groups, each with its own base-rate for each component (a
     discrete random effect on the base-rates). Default 1 (no classes). With ``K > 1`` classes, the table of combos is
     repeated once per class (class-major: ``self.combos[i]`` is in class ``self.combo_class[i]``), and the base-rates
     are learned per class: each component's logit gets a per-class offset, and the classes have prior logits (both
     fixed at 0 for class 0, for identifiability). Otherwise the classes are the same: the ``transition`` is shared,
     applied within each class with that class's base-rates (see :class:`RegimeTransition`), so its ``num_combos`` is
     the number of combos per class (``self.num_class_combos``); and groups never change class. The class-offsets are
     initialized at ``class_spread_init * c`` for class ``c``, so that the classes start in increasing order of their
     components' base-rates (otherwise, the classes would be symmetric). Only the bookkeeping of regime-probabilities
     grows with ``K``: combos with the same observable regimes share a state-update and likelihood (see
     :func:`effective_combos`). Use :func:`class_probs` for the posterior probability of each class. Note that
     ``Predictions.sample()`` draws the regime-combo independently at each timestep (as it does the state; samples
     are not trajectories), so a sample's class can differ across timesteps: the per-timestep distributions are
     right, but a path from these samples doesn't stay in one class. For trajectories, use ``simulate()``, which
     simulates the regimes (and so the class) along with the state.
    :param class_spread_init: See ``num_classes``. Default 1.
    """

    def __init__(self,
                 components: Sequence[MixtureComponent],
                 transition: Optional[RegimeTransition] = None,
                 univariate_prob: bool = False,
                 joseph_form: bool = True,
                 num_classes: int = 1,
                 class_spread_init: float = 1.):
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

        if num_classes < 1:
            raise ValueError(f"`num_classes` must be at least 1, got {num_classes}")
        self.num_classes = num_classes
        # the combos within a class; with latent classes, these are repeated for each class (class-major):
        class_combos = list(itertools.product(*[[None] + self._by_measure[m] for m in self.mixture_measures]))
        self.combos: list[tuple[Optional[MixtureComponent], ...]] = class_combos * num_classes
        self.combo_class: list[int] = [c for c in range(num_classes) for _ in class_combos]

        self._class_prior_logits = self._class_offsets = None
        if num_classes > 1:
            self._init_classes(class_spread_init)

        if transition is None:
            transition = StickyTransition(num_combos=self.num_class_combos)
        elif transition.num_combos != self.num_class_combos:
            per_class = " per class" if num_classes > 1 else ""
            raise ValueError(
                f"`transition` has {transition.num_combos} combos, but expected {self.num_class_combos}{per_class}"
            )
        self.transition = transition

    def _init_classes(self, class_spread_init: float) -> None:
        # the prior logit of classes 1, ..., K-1 (class 0's is fixed at 0):
        self._class_prior_logits = torch.nn.Parameter(torch.zeros(self.num_classes - 1))
        # for each component (in the order of self.components), the offset to its logit in classes 1, ..., K-1
        # (class 0's is fixed at 0). initialized in increasing order, to break the symmetry between classes:
        spread = class_spread_init * torch.arange(1, self.num_classes, dtype=torch.get_default_dtype())
        self._class_offsets = torch.nn.Parameter(spread.expand(len(self.components), -1).clone())

    @staticmethod
    def sample_combos(regime_probs: torch.Tensor) -> torch.Tensor:
        """
        For simulations: draw a combo for each row.

        :param regime_probs: A ``(num_groups, num_combos)`` tensor of regime-probabilities.
        :return: A ``(num_groups, num_combos)`` one-hot tensor of the drawn combos.
        """
        idx = torch.multinomial(regime_probs.detach(), 1)
        return torch.zeros_like(regime_probs).scatter_(-1, idx, 1.)

    def class_log_prior(self) -> torch.Tensor:
        """
        :return: A ``(num_classes,)`` tensor: the log prior probability of each latent class.
        """
        if self.num_classes == 1:
            return torch.zeros(1, dtype=self.components[0].logit.dtype, device=self.components[0].logit.device)
        logits = self._class_prior_logits
        return torch.log_softmax(torch.cat([torch.zeros_like(logits[:1]), logits]), 0)

    def class_probs(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        :param regime_probs: A ``(..., num_combos)`` tensor of regime-probabilities (e.g. ``Predictions.regime_probs``).
        :return: A ``(..., num_classes)`` tensor: the probability of each latent class.
        """
        return regime_probs.unflatten(-1, (self.num_classes, self.num_class_combos)).sum(-1)

    def next_prior(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        """
        The regime-prior at the next timestep: ``self.transition`` applied to the regime-posterior. With latent
        classes, it's applied within each class (see :class:`RegimeTransition`), so the probability of each class is
        unchanged.

        :param posterior: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the current timestep.
        :param base_probs: A ``(num_combos,)`` or ``(num_groups, num_combos)`` tensor, for the next timestep.
        :return: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the next timestep.
        """
        if self.num_classes == 1:
            return self.transition(posterior, base_probs)
        class_mass, within = self._split_classes(posterior)
        _, within_base = self._split_classes(base_probs.expand_as(posterior))
        return self._merge_classes(class_mass, self.transition(within, within_base))

    def _initial_regime_probs(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
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
        The inverse of :func:`_split_classes`: back to a ``(num_groups, num_combos)`` tensor.
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
        return len(self.combos)

    @property
    def num_class_combos(self) -> int:
        """The number of combos within each latent class (``num_combos`` without classes)."""
        return self.num_combos // self.num_classes

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
        :param component_X: Required if any components have predictors: the output of :func:`_get_component_X` (keyed
         by ``qualified_id``), or slices of it (e.g. for one timestep), all with the same leading dims.
        :return: The log long-run probability of each combo: a ``(num_combos,)`` tensor, or with predictors
         ``(*leading_dims, num_combos)``. Regimes are independent across measures; within a measure, the standard
         regime has logit 0 and each component has its own logit. With latent classes, this is the log prior
         probability of the combo's class, plus the log-probability of its regimes given the class (whose logits add
         the class's offsets to the components' logits).
        """
        component_X = component_X or {}
        logits = {c.qualified_id: c.get_logit(component_X.get(c.qualified_id)) for c in self.components}
        batch_shape = torch.broadcast_shapes(*(lg.shape for lg in logits.values()))
        to = {'dtype': self.components[0].logit.dtype, 'device': self.components[0].logit.device}
        if self.num_classes > 1:
            # each component's logit in each class, along a new last dim: (*batch_shape, num_classes)
            class_offsets = torch.cat([torch.zeros_like(self._class_offsets[:, :1]), self._class_offsets], 1)
            logits = {
                c.qualified_id: logits[c.qualified_id].unsqueeze(-1) + class_offsets[i]
                for i, c in enumerate(self.components)
            }
            batch_shape = (*batch_shape, self.num_classes)

        # per measure: the log-probability of each of its regimes (None = standard, then each component):
        regime_log_probs = {}
        for measure, comps in self._by_measure.items():
            measure_logits = [torch.zeros(batch_shape, **to)]
            measure_logits += [logits[c.qualified_id].expand(batch_shape) for c in comps]
            log_probs = torch.log_softmax(torch.stack(measure_logits, -1), -1).unbind(-1)
            regime_log_probs[measure] = dict(zip([None] + [c.qualified_id for c in comps], log_probs))

        # per combo: regimes are independent across measures, so their log-probabilities add:
        class_combos = self.combos[:self.num_class_combos]
        combo_log_probs = [
            sum(
                regime_log_probs[measure][None if c is None else c.qualified_id]
                for measure, c in zip(self.mixture_measures, combo)
            )
            for combo in class_combos
        ]
        combo_log_probs = torch.stack(combo_log_probs, -1)
        if self.num_classes == 1:
            return combo_log_probs
        # (*batch_shape, num_classes, num_class_combos) -> class-major (*batch_shape, num_combos):
        return (self.class_log_prior().unsqueeze(-1) + combo_log_probs).flatten(-2)

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
            return self._initial_regime_probs(base_probs, num_groups)
        if init_regime_probs.ndim != 2 or init_regime_probs.shape[-1] != self.num_combos:
            raise ValueError(
                f"Expected `initial_state.regime_probs` to have shape (num_groups, {self.num_combos}), "
                f"got {tuple(init_regime_probs.shape)}"
            )
        if init_regime_probs.shape[0] not in (1, num_groups):
            raise ValueError(f"Expected `initial_state.regime_probs.shape[0]` to be 1 or {num_groups}")
        return self.next_prior(init_regime_probs.expand(num_groups, -1), base_probs)

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
