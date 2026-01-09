"""
`Process` modules are used to specify the latent-states (and their temporal dynamics) underlying your time-series(es):

* :class:`.LocalLevel` - a random-walk.
* :class:`.LocalTrend` - a random-walk with (optionally damped) velocity.
* :class:`.Season` - a process with seasonal structure, implementing the fourier-series based model from
  `De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011)`.
* :class:`.LinearModel` - a linear-model allowing for external predictors.

----------
"""

from .process import Process
from .regression import LinearModel
from .local import LocalLevel, LocalTrend
from .season import Season
