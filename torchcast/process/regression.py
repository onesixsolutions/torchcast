from warnings import warn

import torch

from typing import Sequence, Optional, Union, Collection

from torch.nn.functional import softplus

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

        self.predictors = predictors

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
    """
    Similar to :class:`.LinearModel`, except an additional ceiling state-element allows for saturating effects. That is,
     if ``yhat = X @ state`` and in a normal linear model ``measured_mean = y_hat``, the saturated linear model still
    has ``measured_mean = y_hat`` when far from the ceiling, but has ``measured_mean = ceiling`` when close.

    The measurement-function this process uses is:

    ```
    measured_mean = yhat - (1. / s) * softplus(s * (yhat - ceiling))
    ```

    With yhat defined above and ``s`` being a sharpness parameter. Part of this sharpness parameter is set by the user
    (see below), but this part is scaled to the inverse of the ceiling height, so that, as the ceiling lowers, sharpness
     increases. This allows the ``yhat -> measured_mean`` relationship to be consistent when yhat is far from the
    ceiling (i.e., the ceiling won't impact where yhat crosses the origin).

    :param id: Unique identifier for the process
    :param predictors: A sequence of strings with predictor-names.
    :param measure: The name of the measure for this process.
    :param fixed: By default, the regression-coefficients are assumed to be fixed: we are initially
     uncertain about their value at the start of each series, but we gradually grow more confident. See LinearModel.
    :param fix_ceiling: Like ``fixed``, but for the ceiling state-element.
    :param decay: See :class:`.LinearModel`
    :param model_mat_kwarg_name: See :class:`.LinearModel`
    :param ceiling_init_value: The initial value for the ceiling prior. Defaults to 1 +/- jitter. If your measure is
     very much not centered and/or scaled, optimization could be improved by putting an informative guess here.
    :param anchor: If we start with yhat near the ceiling and reduce it, ``anchor`` is the yhat value at which yhat
     converges to ``measured_mean``. Typically you want to leave this at zero, but that implicitly assumes your
     predictors are centered.
    """
    linear_measurement = False
    base_sharpness = 6.0

    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 measure: Optional[str] = None,
                 fixed: Union[bool, Collection[str]] = True,
                 fix_ceiling: bool = True,
                 decay: Optional[tuple[float, float]] = None,
                 model_mat_kwarg_name: str = 'X',
                 ceiling_init_value: Optional[float] = None,
                 anchor: float = 0.0):
        self.fix_ceiling = fix_ceiling
        super().__init__(
            id=id,
            predictors=predictors,
            measure=measure,
            fixed=fixed,
            decay=decay,
            model_mat_kwarg_name=model_mat_kwarg_name
        )

        # ceiling initial value:
        if ceiling_init_value is None:
            ceiling_init_value = 1.0 + torch.randn(1).item() / 10
        with torch.no_grad():
            assert list(self.state_elements)[-1] == '_ceiling'
            self.initial_mean[-1] = ceiling_init_value

        # yhat value below which y~=yhat (regardless of ceiling value)
        self.anchor = anchor

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
        return len(self.predictors)

    def prepare_measurement_cache(self, **kwargs) -> dict:
        X = kwargs.pop(self.model_mat_kwarg_name, None)
        if X is None:
            raise TypeError(f"{self.id}.prepare_measurement_cache() missing `{self.model_mat_kwarg_name}` argument")
        if kwargs:
            raise ValueError(f"{self.id}.prepare_measurement_cache() received unexpected kwargs: {set(kwargs)}")
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()
        if X.shape[-1] != self.num_predictors:
            raise ValueError(
                f"process '{self.id}' received X that has shape {X.shape}, but expected last dim to "
                f"match len(predictors) {self.num_predictors}."
            )
        return {
            'X': X.unbind(1),
            's': [None] * X.shape[1],
            'yhat': [None] * X.shape[1]
        }

    def get_measured_mean(self, mean: torch.Tensor, time: int, cache: dict) -> torch.Tensor:
        X = cache['X'][time]
        coefs = mean[:, :self.num_predictors]
        ceiling = mean[:, self.num_predictors]
        yhat = cache['yhat'][time] = (X * coefs).sum(-1)
        return _sat_measured_mean(yhat, ceiling=ceiling, sharpness=self.base_sharpness, anchor=self.anchor)

    def get_measurement_jacobian(self, mean: torch.Tensor, time: int, cache: dict) -> torch.Tensor:
        return _sat_jacobian(
            X=cache['X'][time],
            yhat=cache['yhat'][time],
            ceiling=mean[:, self.num_predictors],
            sharpness=self.base_sharpness,
            anchor=self.anchor
        )


def _sat_measured_mean(yhat: torch.Tensor,
                       ceiling: torch.Tensor,
                       sharpness: float,
                       anchor: float) -> torch.Tensor:
    d = (ceiling - anchor).clamp(min=1e-6)
    s = sharpness / d
    nl_mask = s * (ceiling - yhat) <= (4.6 - torch.log(s))
    adjustment = torch.zeros_like(yhat)
    adjustment[nl_mask] = softplus(s[nl_mask] * (yhat[nl_mask] - ceiling[nl_mask])) / s[nl_mask]
    return yhat - adjustment


def _sat_jacobian(X: torch.Tensor,
                  yhat: torch.Tensor,
                  ceiling: torch.Tensor,
                  sharpness: float,
                  anchor: float) -> torch.Tensor:
    d = ceiling - anchor
    ac_mask = d > 1e-6  # mask indicating ceiling is above anchor (below that `s` doesnt work)
    d = d.clamp(min=1e-6)
    s = sharpness / d

    u = yhat - ceiling
    su = (s * u).clamp(min=-20, max=20)
    sigma = torch.sigmoid(su)
    sp = softplus(su)

    nl_mask = s * (ceiling - yhat) <= (4.6 - torch.log(s))  # mask indicating we're in nonlinear regime

    # ceil derivs:
    _mask = nl_mask & ac_mask
    path2 = torch.zeros_like(sigma)
    path2[_mask] = -sp[_mask] / (s[_mask] * d[_mask]) + (u[_mask] / d[_mask]) * sigma[_mask]
    ceil_derivs = torch.zeros_like(sigma)
    ceil_derivs[nl_mask] = (sigma[nl_mask] + path2[nl_mask].clamp(min=-1., max=1.)).clamp(min=0.)

    # coef derivs:
    coef_deriv_multi = torch.ones_like(sigma)
    coef_deriv_multi[nl_mask] = (1. - sigma[nl_mask])
    coef_derivs = X * coef_deriv_multi.unsqueeze(-1)

    return torch.cat([coef_derivs, ceil_derivs.unsqueeze(-1)], dim=-1)
