"""
This module includes:

* Data-utils, such as those for converting time-series data from a Pandas DataFrame into a PyTorch
  :class:`torch.utils.data.Dataset` and/or :class:`torch.utils.data.DataLoader`, as well as a function for handling
  implicit missing data.
* A function for adding calendar-features: i.e. weekly/daily/yearly season dummy-features for usage as predictors.
* A function for creating a simple baseline model, against which to compare more sophisticated forecasting models.
* Simple trainer classes for PyTorch models, with specialized subclasses for torchcast's model-classes, as well as a
  special class for training neural networks to embed complex seasonal patterns into lower dimensional embeddings.
* A 'Stopping' class for controlling convergence/stopping for the `fit()` method in state-space models.
---
"""

from .features import add_season_features
from .data import TimeSeriesDataset, TimeSeriesDataLoader, complete_times
from .baseline import make_baseline
from .training import SimpleTrainer, StateSpaceTrainer, SeasonalEmbeddingsTrainer
from .stopping import Stopping
