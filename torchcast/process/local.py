import torch
from typing import Optional, Union

from .process import Process
from .utils import StateElement, standardize_decay


class LocalLevel(Process):
    """
    A process representing a random-walk.

    :param id: A unique identifier for this process.
    :param decay: If the process has decay, then the random walk will tend towards zero as we forecast out further
     (note that this means you should center your time-series, or you should include another process that does not
     have this decaying behavior). Decay can be between 0 and 1, but values < .50 (or even .90) can often be too
     rapid and you will run into trouble with vanishing gradients. When passing a pair of floats, the nn.Module will
     assign a parameter representing the decay as a learned parameter somewhere between these bounds. The default value
     means a half-life that can't go lower than ~13 timesteps; overriding this based on your domain-knowledge is
     encouraged.
    :param stationary_initial_variance: Only for a decaying level (whose decay is less than 1, e.g. bounds like
     ``(.5, .95)``). If ``True``, the level's initial variance is its stationary variance,
     ``sigma^2 / (1 - decay^2)`` (``sigma^2`` its process-variance), rather than a free parameter of the
     ``initial_covariance``; its initial correlations with other state elements are zero. With a learned decay this
     is recommended: otherwise the decay trades off against the initial variance (a faster decay with a larger initial
     variance fits the start of a series about as well as a slower decay with a smaller one), so the decay is
     identified by the first few timesteps rather than the whole series; the :class:`.KalmanFilter` warns about this.
     ``False`` keeps the free initial variance and silences that warning. Note: the initial covariance must be
     created *after* this process (as it is by default), since this removes a parameter from it.
    """

    def __init__(self,
                 id: str,
                 measure: Optional[str] = None,
                 decay: Optional[Union[torch.nn.Module, tuple[float, float], float]] = None,
                 stationary_initial_variance: Optional[bool] = None):
        decay = standardize_decay(decay)
        self._has_decay = not isinstance(decay, float) or decay < 1.0
        if stationary_initial_variance:
            upper = getattr(decay, 'upper', decay)  # (``Bounded`` or float; other modules are checked at forward)
            if not self._has_decay or (isinstance(upper, (float, int)) and upper >= 1.0):
                raise ValueError(
                    f"`{id}`: ``stationary_initial_variance=True`` needs a decay that's less than 1 (e.g. bounds like "
                    f"``decay=(.5, .95)``); without decay there's no stationary variance."
                )
        level = StateElement(
            name='level',
            measure_multi=1.0,
            has_process_variance=True,
            stationary_initial_variance=stationary_initial_variance
        )
        super().__init__(id=id, state_elements=[level], measure=measure)
        self.state_elements['level'].set_transition_to(self.state_elements['level'], multi=decay)

    @property
    def intercept_state_element(self) -> Optional[str]:
        return None if self._has_decay else 'level'


class LocalTrend(Process):
    """
    A process representing an evolving trend.

    :param id: A unique identifier for this process.
    :param decay_velocity: If set, then the trend will decay to zero as we forecast out further. The default is
     to allow the trend to decay somewhere between .95 (moderate decay) and 1.00 (no decay), with the exact value
     being a learned parameter.
    :param decay_position: See `decay` in :class:`LocalLevel`. Default is no decay.
    """
    _velocity_multi = 0.1  # trend has large effect on the prediction -- large values can lead to exploding predictions

    def __init__(self,
                 id: str,
                 measure: Optional[str] = None,
                 decay_velocity: Optional[Union[torch.nn.Module, tuple[float, float]]] = (0.95, 1.00),
                 decay_position: Optional[Union[torch.nn.Module, tuple[float, float]]] = None):
        # observed, transitions to itself (but with possible decay):
        position = StateElement(name='position', measure_multi=1.0, has_process_variance=True)
        decay_position = standardize_decay(decay_position)
        self._has_position_decay = not isinstance(decay_position, float) or decay_position < 1.0
        position.set_transition_to(position, multi=decay_position)

        # unobserved, transitions not only to itself but into position:
        velocity = StateElement(name='velocity', measure_multi=0.0, has_process_variance=True)
        decay_velocity = standardize_decay(decay_velocity)
        velocity.set_transition_to(velocity, multi=decay_velocity)
        velocity.set_transition_to(position, multi=self._velocity_multi)

        super().__init__(id=id, state_elements=[position, velocity], measure=measure)

    @property
    def intercept_state_element(self) -> Optional[str]:
        return None if self._has_position_decay else 'position'
