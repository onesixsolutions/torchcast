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
     assign a parameter representing the decay as a learned parameter somewhere between these bounds.
    """

    def __init__(self,
                 id: str,
                 measure: Optional[str] = None,
                 decay: Optional[Union[torch.nn.Module, tuple[float, float], float]] = None):
        super().__init__(
            id=id,
            state_elements=[StateElement(name='level', measure_multi=1.0, has_process_variance=True)],
            measure=measure
        )

        decay = standardize_decay(decay)
        self._has_decay = not isinstance(decay, float) or decay < 1.0
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
