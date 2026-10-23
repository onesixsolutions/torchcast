from typing import Optional, Sequence

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
