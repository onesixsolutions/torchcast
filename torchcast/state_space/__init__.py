"""
The base-class for time-series modeling with state-space models. Generates forecasts in the form of
:class:`torchcast.state_space.Predictions`, which can be used for training
(:func:`~torchcast.state_space.Predictions.log_prob()`), evaluation
(:func:`~torchcast.state_space.Predictions.to_dataframe()`) or visualization
(:func:`~torchcast.state_space.Predictions.plot()`).

This class is abstract; see :class:`torchcast.kalman_filter.KalmanFilter` for the go-to forecasting model.
"""

from .state_space import StateSpaceModel
from .predictions import Predictions
