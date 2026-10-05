# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
# ---

# %% {"nbsphinx": "hidden"}
import warnings

import numpy as np
import pandas as pd
import torch

from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalLevel
from torchcast.state_space import MixtureComponent
from torchcast.utils.data import TimeSeriesDataset
from plotnine.exceptions import PlotnineWarning

warnings.filterwarnings('ignore', category=PlotnineWarning)

# %% [markdown]
# # Mixture Components: Outliers and Regimes
#
# *Experimental.* A standard Kalman filter assumes every observation comes from the same process: a measurement of the
# latent state, plus gaussian noise. Real data often have observations that come from a different process entirely.
# In this example, we have weekly (log-transformed) spend for a set of customers. Most visits reflect a customer's
# typical level of spend, but occasionally a customer makes a "quick visit" with very low spend -- this isn't
# informative about their typical level, and we don't want it to be treated as such.
#
# A standard model has two bad options for these observations: move the state towards them (dragging down the
# forecast), or inflate the variance to accommodate them. The latter is especially damaging when forecasting on the
# original (non-log) scale, since back-transforming requires `exp(mean + var / 2)`, which is very sensitive to the
# variance.
#
# With the `mixture` argument, a measure can have one or more alternative *regimes*. Each is described by a learned
# mean and variance, and a learned probability. An observation that's explained by a component's regime is scored
# against that component, and doesn't update the latent state.

# %% [markdown]
# ### Simulated Data
#
# Each customer has a slowly-drifting typical (log) spend. Some weeks they don't visit (missing values), and
# about 8% of visits are "quick visits" with low spend, unrelated to the customer's typical level.

# %%
rs = np.random.RandomState(1234)
NUM_GROUPS, NUM_TIMES, SPLIT = 60, 80, 60
QUICK_PROB, QUICK_MEAN, QUICK_STD, NOISE_STD = .08, 1.0, .5, .3

level = 5 + rs.randn(NUM_GROUPS, 1) + np.cumsum(.05 * rs.randn(NUM_GROUPS, NUM_TIMES), axis=1)
is_quick = rs.rand(NUM_GROUPS, NUM_TIMES) < QUICK_PROB
log_spend = np.where(
    is_quick,
    QUICK_MEAN + QUICK_STD * rs.randn(NUM_GROUPS, NUM_TIMES),
    level + NOISE_STD * rs.randn(NUM_GROUPS, NUM_TIMES)
)
log_spend[rs.rand(NUM_GROUPS, NUM_TIMES) < .3] = np.nan  # weeks without a visit

# the true expected spend (on the original scale) for each customer-week:
true_expected = (
        (1 - QUICK_PROB) * np.exp(level + NOISE_STD ** 2 / 2) +
        QUICK_PROB * np.exp(QUICK_MEAN + QUICK_STD ** 2 / 2)
)

y = torch.as_tensor(log_spend, dtype=torch.float32).unsqueeze(-1)
y_train = y[:, :SPLIT]
START = np.datetime64('2024-01-01')
dataset = TimeSeriesDataset(
    y,
    group_names=[f'customer_{i}' for i in range(NUM_GROUPS)],
    start_times=np.full(NUM_GROUPS, START),
    measures=[['log_spend']],
    dt_unit='W'
)
SPLIT_DT = START + np.timedelta64(SPLIT, 'W')

# %% [markdown]
# ### Models
#
# We fit a standard model and one with a mixture-component for 'quick visits'. The component's `mean_init` and
# `prob_init` are just starting values; they're learned during training (as is the component's variance, and how
# "sticky" the regime is from one timestep to the next).

# %%
torch.manual_seed(1)
kf_standard = KalmanFilter(
    processes=[LocalLevel(id='level')],
    measures=['log_spend']
)
kf_standard.fit(y_train, verbose=0);

torch.manual_seed(1)
kf_mixture = KalmanFilter(
    processes=[LocalLevel(id='level')],
    measures=['log_spend'],
    mixture=[MixtureComponent(measure='log_spend', mean_init=0., prob_init=.05, id='quick')]
)
kf_mixture.fit(y_train, verbose=0);

# %% [markdown]
# The learned component closely matches the data-generating process (mean 1.0, std 0.5, probability 0.08). There is
# no persistence in these simulated quick visits, so the learned stickiness stays low. Meanwhile the measurement noise
# (std 0.3 in the simulation) is learned accurately by the mixture model, but is badly inflated in the standard model,
# which has to accommodate the quick visits:

# %%
component = kf_mixture.mixture.components[0]
pd.Series({
    'component mean': component.mean.item(),
    'component std': component.var.sqrt().item(),
    'component probability': kf_mixture.mixture.base_probs()[1].item(),
    'stickiness (standard, quick)': kf_mixture.mixture.transition.stay.detach().numpy().round(3),
    'measurement std (mixture model)': kf_mixture.measure_covariance({}, 1, 1)[0, 0, 0, 0].sqrt().item(),
    'measurement std (standard model)': kf_standard.measure_covariance({}, 1, 1)[0, 0, 0, 0].sqrt().item(),
})

# %% [markdown]
# ### Forecasting on the Original Scale
#
# We forecast the holdout period (the last 20 weeks), and back-transform to get expected spend.
#
# - For the standard model, the forecast is gaussian on the log scale, so `exp(mean + var / 2)`.
# - For the mixture model, the forecast on the log-scale is a mixture: `Predictions.get_mixture()` returns the
#   probability, mean and variance of each regime. The correct way to back-transform is to back-transform each regime,
#   *then* mix. (Back-transforming the collapsed `mean`/`var` of the mixture would be badly wrong, since the mixture's
#   variance includes the gap between the regimes.)

# %%
with torch.no_grad():
    pred_standard = kf_standard(y_train, out_timesteps=NUM_TIMES)
    pred_mixture = kf_mixture(y_train, out_timesteps=NUM_TIMES).set_metadata(dataset)

    mean, cov = pred_standard
    fcast_standard = torch.exp(mean[..., 0] + cov[..., 0, 0] / 2)

    mix = pred_mixture.get_mixture('log_spend')
    fcast_mixture = (mix.probs * torch.exp(mix.means + mix.vars / 2)).sum(-1)
    # (``pred_mixture.to_dataframe(transform=LogTransform())`` does this for you -- see below.)
    # for comparison, back-transforming the collapsed moments:
    fcast_collapsed = torch.exp(mix.mean() + mix.var() / 2)


def pct_error(fcast: torch.Tensor) -> float:
    fcast = fcast.numpy()[:, SPLIT:]
    return 100 * np.mean(np.abs(fcast - true_expected[:, SPLIT:]) / true_expected[:, SPLIT:])


pd.Series({
    'standard model': pct_error(fcast_standard),
    'mixture model': pct_error(fcast_mixture),
    'mixture model, collapsed (wrong)': pct_error(fcast_collapsed),
}, name='mean abs. % error of expected spend')

# %% [markdown]
# The standard model's inflated variance inflates its back-transformed forecasts. The mixture model does much better;
# most of its remaining error is irreducible here, from the random drift in each customer's level over the 20-week
# holdout (a forecast that knew each customer's true level at the start of the holdout, and the true regime
# parameters, would still have about 13% error).
#
# `to_dataframe(transform=LogTransform())` does this back-transformation for you, for both the mean and the
# intervals. On the log scale, `Predictions.to_dataframe()` (and so `plot()`) uses the exact quantiles of the mixture
# for the prediction intervals -- note the long lower tail from the 'quick' regime. Similarly, the plotted mean is the mean of
# the mixture, which sits below this customer's typical spend, since it averages in the chance of a quick visit. (The
# small dips right after each quick visit reflect the learned stickiness: a quick visit makes another one slightly
# more likely.)

# %%
df_pred = pred_mixture.to_dataframe(dataset)
pred_mixture.plot(df_pred.query("group == 'customer_3'"), split_dt=SPLIT_DT, figure_size=(8, 4))

# %% [markdown]
# ### Other Notes
#
# - Mixture components are also supported in the `BinomialFilter`, on its non-binary measures. For example, with a
#   binary 'visited' measure and a 'log-spend' measure that's missing whenever there was no visit. Then
#   `to_dataframe(transform={'log_spend': LogTransform()}, derived={'spend': lambda s: s['visited'] *
#   s['log_spend'].nan_to_num()})` gives forecasts of weekly spend (from joint samples of both measures). The binary
#   measure doesn't influence the regime-probabilities (its gaussian approximation in the update is crude).
# - A mixture measure can have a nonlinear measurement -- e.g. a `SaturatedLinearModel` process, or a sigmoid
#   measure-function with a gaussian likelihood. Then `log_prob()`, `means`, and `to_dataframe()` use monte-carlo for
#   it (as for nonlinear measures without mixtures), and `get_mixture()` warns that its standard regime is the
#   linearized approximation (its regime-probabilities are still exact).
# - With multiple measures, each measure can have its own components. Regime-probabilities are tracked jointly across
#   measures; `Predictions.sample()` gives joint draws of all measures (incl. the regimes).
# - To carry regime-probabilities over into a later forecast, pass the output of `Predictions.get_state_at_times()`
#   as `initial_state` (it includes the regime-probabilities as well as the state mean/covariance).
# - A component's base-rate can depend on predictors, e.g. a treatment that makes 'quick visits' more common:
#   `MixtureComponent(..., predictors=['treated'])`, then pass `X` (or `quick__X`, to keep it separate from another
#   `LinearModel`'s `X`) to the model, covering the forecast horizon.
# - Passing a list of components is shorthand for `mixture=MixtureModel(components)`. To configure it, pass the
#   `MixtureModel` yourself: e.g. `MixtureModel(components, transition=...)`. The default `StickyTransition` learns
#   a "stickiness" that controls how regime-probabilities persist over time; you can also write your own
#   `RegimeTransition`.
