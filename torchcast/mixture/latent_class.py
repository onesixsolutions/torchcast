from typing import Mapping, Optional


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
