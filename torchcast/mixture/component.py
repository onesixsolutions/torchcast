from typing import Mapping, Optional, Sequence

import torch

from torchcast.internals.utils import validate_scalar


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

        offset_init = validate_scalar(offset_init, 'offset_init')
        self.offset = torch.nn.Parameter(offset_init)
        self._log_extra_std = torch.nn.Parameter(torch.zeros_like(offset_init))

        prob_init = validate_scalar(prob_init, 'prob_init')
        if not 0 < prob_init < 1:
            raise ValueError(f"`prob_init` should be between 0 and 1, got {prob_init.item()}.")
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


class LatentClass:
    """
    A latent class of groups, for :class:`MixtureModel`'s ``classes``: groups in this class have their own base-rate
    for each component. Which class a group is in isn't known; the model learns each class's base-rates, and its
    posterior over each group's class accumulates evidence over all of the group's timesteps.

    The first of a model's classes is the reference: its base-rates are the components' own (see
    :class:`MixtureComponent`'s ``prob_init``). Each other class's base-rates are relative to the reference's, via
    ``relative_rate_init``.

    :param id: A unique identifier (among the model's classes).
    :param membership_prob_init: Initial value for the prior probability that a group is in this class. Either given
     for all of the model's classes (normalized to sum to 1), or for none (equal probabilities).
    :param relative_rate_init: For classes after the first (required, with at least one ratio other than 1): a
     dictionary mapping a component's ``qualified_id`` (``'{measure}.{id}'``) to the initial value of its rate in this
     class, relative to the reference class -- e.g. ``{'log_theo.high': .1}`` for a 'high' regime ~10x less common.
     Exactly, it multiplies the component's odds vs. its measure's standard regime (for a rare component, that's about
     the same as multiplying its rate). Components not listed are initialized as in the reference class.
    """

    def __init__(self,
                 id: str,
                 membership_prob_init: Optional[float] = None,
                 relative_rate_init: Optional[Mapping[str, float]] = None):
        self.id = id
        if membership_prob_init is not None and not 0 < membership_prob_init < 1:
            raise ValueError(f"LatentClass '{id}': `membership_prob_init` must be between 0 and 1.")
        self.membership_prob_init = membership_prob_init
        self.relative_rate_init = dict(relative_rate_init or {})
        for key, ratio in self.relative_rate_init.items():
            if not ratio > 0:
                raise ValueError(f"LatentClass '{id}': `relative_rate_init['{key}']` must be positive, got {ratio}.")

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id={repr(self.id)})'
