from warnings import warn

import torch

from typing import Sequence, Optional, Union, Collection

from torchcast.process import Process
from torchcast.process.utils import ProcessKwarg, StateElement, standardize_decay


class LinearModel(Process):
    """
    A process which takes a model-matrix of predictors, and each state corresponds to the coefficient on each.

    :param id: Unique identifier for the process
    :param predictors: A sequence of strings with predictor-names.
    :param measure: The name of the measure for this process.
    :param fixed: By default, the regression-coefficients are assumed to be fixed: we are initially
     uncertain about their value at the start of each series, but we gradually grow more confident. If
     ``fixed=False`` then we continue to inject uncertainty at each timestep so that uncertainty asymptotes
     at some nonzero value. This amounts to dynamic-regression where the coefficients evolve over-time. Note only
     ``KalmanFilter`` (but not ``ExpSmoother``) supports this.
    :param decay: By default, the coefficient-values will remain as the forecast horizon increases. An alternative is
     to allow these to decay (i.e. pass ``True``). If you'd like more fine-grained control over this decay,
     you can specify the min/max decay as a tuple (passing ``True`` uses a default value of ``(.98, 1.0)``).
    """

    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 measure: Optional[str] = None,
                 fixed: Union[bool, Collection[str]] = True,
                 decay: Optional[tuple[float, float]] = None,
                 model_mat_kwarg_name: str = 'X'):
        if isinstance(fixed, str):
            raise ValueError(f"`fixed` should be a collection of strings not a single string.")
        elif hasattr(fixed, '__contains__'):
            unexpected = set(fixed) - set(predictors)
            if unexpected:
                raise ValueError(f"fixed={fixed} contains elements not in predictors={predictors}: {unexpected}")
        else:
            fixed = list(predictors) if fixed else []

        super().__init__(
            id=id,
            state_elements=self._init_state_elements(predictors, fixed),
            measure=measure
        )
        self.model_mat_kwarg_name = model_mat_kwarg_name

        for se in self.state_elements.values():
            se_decay = standardize_decay(decay, lower=.98)
            _has_decay = not isinstance(se_decay, float) or se_decay < 1.0
            if _has_decay and se.name in fixed:
                warn(f"[{self.id}.{se.name}]: decay=True, fixed=True not recommended.")
            se.set_transition_to(se, multi=se_decay)

    @property
    def measurement_kwargs(self) -> Sequence[ProcessKwarg]:
        return [ProcessKwarg(name=self.model_mat_kwarg_name, is_group_time_tensor=True)]

    def _init_state_elements(self, predictors: Sequence[str], fixed: Sequence[str]) -> Sequence[StateElement]:
        if isinstance(predictors, str):
            raise ValueError(f"`predictors` should be a sequence of strings, not a single string: {predictors}")
        return [
            StateElement(name=p, measure_multi=None, has_process_variance=p not in fixed)
            for p in predictors
        ]

    def get_measurement_matrix(self, **kwargs) -> torch.Tensor:
        X = kwargs.pop(self.model_mat_kwarg_name, None)
        if X is None:
            raise TypeError(f"{self.id}.get_measurement_matrix() missing `{self.model_mat_kwarg_name}` argument")
        if kwargs:
            raise ValueError(f"{self.id}.get_measurement_matrix() received unexpected kwargs: {set(kwargs)}")
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()
        if X.shape[-1] != self.rank:
            raise ValueError(
                f"process '{self.id}' received X that has shape {X.shape}, but expected last dim to "
                f"match len(predictors) {self.rank}."
            )
        return X


class SaturatedLinearModel(LinearModel):
    linear_measurement = False

    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 measure: Optional[str] = None,
                 fixed: Union[bool, Collection[str]] = True,
                 fix_ceiling: bool = True,
                 decay: Optional[tuple[float, float]] = None,
                 model_mat_kwarg_name: str = 'X'):
        self.fix_ceiling = fix_ceiling
        super().__init__(
            id=id,
            predictors=predictors,
            measure=measure,
            fixed=fixed,
            decay=decay,
            model_mat_kwarg_name=model_mat_kwarg_name
        )

    def _init_state_elements(self,
                             predictors: Sequence[str],
                             fixed: Sequence[str]) -> Sequence[StateElement]:
        assert '_ceiling' not in predictors, f"`_ceiling` is a reserved name for {type(self).__name__}"
        coefs = [
            StateElement(name=p, measure_multi=None, has_process_variance=p not in fixed)
            for p in predictors
        ]
        return coefs + [StateElement(name='_ceiling', measure_multi=None, has_process_variance=not self.fix_ceiling)]

    def get_measurement_matrix(self, X: torch.Tensor) -> torch.Tensor:
        raise TypeError(f"This method should never be called because {self.id} has nonlinear measure-fun.")

    @property
    def num_predictors(self) -> int:
        return self.rank - 1

    def prepare_measurement_cache(self, X: torch.Tensor) -> dict:
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()
        if X.shape[-1] != self.num_predictors:
            raise ValueError(
                f"process '{self.id}' received X that has shape {X.shape}, but expected last dim to "
                f"match len(predictors) {self.num_predictors}."
            )
        return {'X': X.unbind(1)}

    def get_measured_mean(self, mean: torch.Tensor, time: int, cache: dict) -> torch.Tensor:
        # TODO: reparameterize
        X = cache['X'][time]
        coefs = mean[:, :self.num_predictors]
        ceiling = mean[:, self.num_predictors]
        cache['yhat'] = (X * coefs).sum(-1)
        return cache['yhat'] - torch.nn.functional.softplus(cache['yhat'] - ceiling)

    def get_measurement_jacobian(self, mean: torch.Tensor, time: int, cache: dict) -> torch.Tensor:
        # TODO: reparameterize
        X = cache['X'][time]
        ceiling = mean[:, self.num_predictors]
        ceil_derivs = torch.sigmoid((cache['yhat'] - ceiling).clamp(min=-10, max=10))
        coef_derivs = X * (1 - ceil_derivs.unsqueeze(-1))
        jacobian = torch.cat([coef_derivs, ceil_derivs.unsqueeze(-1)], dim=-1)
        return jacobian
