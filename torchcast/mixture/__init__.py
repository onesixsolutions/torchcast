"""
Mixture-of-regimes support for state-space models.

A measure can have one or more :class:`MixtureComponent` alternatives to its standard measurement model. Each
component is an *offset* from the standard regime: in the component's regime, an observation is the usual
(state-dependent) measured-mean plus the component's learned offset, with its learned variance added to the
measurement-noise. This is particularly useful for rare outliers with a clear data-generating process (e.g. forecasting
the activity of customers who usually have long visits but occasionally have very short ones), especially when
forecasting with data-transformations that make these outliers problematic in the back-transform (e.g. if the customer
duration data was log1p transformed, an outlier will increase the variance and so lead to implausibly large
back-transformed forecasts).

The :class:`MixtureModel` owns the components. Multivariate support is accomplished by enumerating the possible "combos"
of regimes (e.g. [(measure1=standard, measure2=standard), (measure1=weird, measure2=standard), ...] and so on). Regime
probabilities are tracked jointly over this table, allowing p(measure_x=weird) to vary over time and across groups.

Optionally, groups can belong to latent classes (``MixtureModel(..., classes=[LatentClass(...), ...])``) with their own
base-rates for each component -- a discrete random effect on the base-rates, e.g. some customers have outliers much more
often than others. The table of combos is then repeated once per class. Classes share the transition (it applies within
each class), and a group never changes class, so its posterior over classes accumulates evidence over all of its
timesteps, and carries over into forecasts.
"""
from .regime_transition import RegimeTransition, StaticTransition, StickyTransition
from .component import LatentClass, MixtureComponent
from .model import MixtureModel