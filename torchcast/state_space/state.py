from typing import Iterator, Optional, Union, Mapping

import torch


class StateTuple:
    """
    The state of a :class:`.StateSpaceModel` at a single timestep: a ``(num_groups, state_dim)`` mean and a
    ``(num_groups, state_dim, state_dim)`` covariance. If the model has mixture components, also holds the
    ``(num_groups, num_combos)`` regime-probabilities (``regime_probs``). If the model has ``adaptive_scaling``, also
    holds the adaptive-scaler's state (``scaling``: a dict of ``(num_groups, ...)`` tensors, incl. the pending
    ``'multiplier'`` for the next step), so that a forecast continued from this state matches one made in a single
    pass.

    Behaves like the tuple ``(mean, cov)`` -- e.g. ``mean, cov = state`` -- so it can be passed anywhere a
    ``(mean, cov)`` tuple is accepted, such as the ``initial_state`` argument of :func:`StateSpaceModel.forward`.
    """

    def __init__(self,
                 mean: torch.Tensor,
                 cov: torch.Tensor,
                 regime_probs: Optional[torch.Tensor] = None,
                 scaling: Optional[Mapping[str, torch.Tensor]] = None):
        self.mean = mean
        self.cov = cov
        self.regime_probs = regime_probs
        self.scaling = None if scaling is None else dict(scaling)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.mean, self.cov))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: int) -> torch.Tensor:
        return (self.mean, self.cov)[item]

    def __repr__(self) -> str:
        regimes = '' if self.regime_probs is None else f', regime_probs={tuple(self.regime_probs.shape)}'
        scaling = '' if self.scaling is None else f', scaling={list(self.scaling)}'
        return f'{type(self).__name__}(mean={tuple(self.mean.shape)}, cov={tuple(self.cov.shape)}{regimes}{scaling})'


def _as_state_tuple(x: Union[StateTuple, tuple[torch.Tensor, torch.Tensor]]) -> StateTuple:
    # ``_update_step()`` may return a plain (mean, cov) tuple (e.g. ExpSmoother)
    return x if isinstance(x, StateTuple) else StateTuple(*x)
