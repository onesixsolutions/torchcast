# CHANGELOG

## Unreleased

### New feature: mixture components (experimental)

A measure can now have one or more alternative *regimes* -- e.g. for outliers that shouldn't update the state or
inflate the variance. See the new [mixture components example](https://docs.strong.io/torchcast/examples/mixture_components.html).

- `KalmanFilter(mixture=MixtureModel([MixtureComponent(...), ...]))` (or just `mixture=[MixtureComponent(...), ...]`):
  each `MixtureComponent` is an *offset* from the standard regime: a learned offset to the (state-dependent)
  measured-mean, extra variance (added to the measurement-noise), and base-rate. E.g. a 'quick visit' component means
  'spend well below *this* customer's usual level'. An observation in a component's regime updates the state as
  usual, but accounting for the offset and the extra noise -- so it doesn't drag the state towards it. Components are supported on any measure with a gaussian likelihood, including the non-binary measures of a
  `BinomialFilter`, and measures with a nonlinear measurement (a nonlinear process such as `SaturatedLinearModel`, or
  a measure-function): for these, the update-step uses the linearized (EKF) measurement, while `log_prob()`, `means`,
  and `to_dataframe()` use monte-carlo.
- Regime-probabilities are tracked jointly across measures and carried through time. How they persist is controlled by
  a `RegimeTransition`; the default `StickyTransition` learns a "stickiness" for each regime (and reduces to a static
  mixture when that's zero). A custom transition can be passed via `MixtureModel(..., transition=)`.
- `MixtureModel(..., univariate_prob=True)` computes the per-timestep regime-probabilities from the mixture measures' likelihood
  only (an approximation, but cheaper). Binary measures never influence the regime-probabilities (their gaussian
  approximation in the update-step is crude).
- `Predictions.log_prob()` is the exact marginal likelihood of the mixture.
- Outputs: `Predictions.get_mixture(measure)` returns a `MixtureOfNormals` (probability, mean, and variance of each
  regime, with `mean()`, `var()`, `cdf()`, `quantile()`), e.g. for back-transforming each regime before mixing. (For
  a measure with a nonlinear measurement, the standard regime's mean and variance are the linearized approximation,
  with a warning.)
  `means`/`covs` are the mixture's exact moments (accessing `covs` warns, since it's easy to misuse), and
  `to_dataframe()`/`plot()` intervals use the mixture's exact quantiles.
- With `adaptive_scaling`, residuals explained by a mixture component don't inflate the scaling.
- `MixtureModel(..., joseph_form=False)` uses the simpler covariance update (`P - K @ H @ P`) instead of the Joseph
  form in the mixture update-step, which runs once per regime-combo: less memory during training (~30% in a test) and
  faster, but less numerically robust. The default is the Joseph form, as without mixtures.
- A component's base-rate can depend on predictors: `MixtureComponent(..., predictors=['treated'])` learns a
  coefficient for each (added to its logit), e.g. for a treatment that makes the regime more common. Pass them to
  `forward()` as `X` (or `{component_id}__X`), as a `(num_groups, num_timesteps, num_predictors)` tensor covering the
  forecast horizon. Custom `RegimeTransition`s then receive `base_probs` with shape `(num_groups, num_combos)`.

### New feature: back-transforming predictions

- `Predictions.to_dataframe(transform=...)` maps predictions of transformed measures back to the original scale,
  e.g. `transform=LogTransform()` (for all measures) or `transform={'sales': LogTransform(base=10)}`. Intervals are
  back-transformed exactly (quantiles pass through monotone transforms), and so is the mean: `E[inverse(Y)]` rather
  than `inverse(E[Y])`, via closed form where available and gauss-hermite quadrature otherwise. For models with
  mixture components, each regime is back-transformed and then mixed. Actuals are back-transformed too.
- `Transform` is the base class: subclasses only need to implement `inverse()`. `LogTransform` and
  `BoxCoxTransform` (with `lmbda >= 0`) are provided.
  `bias_adjust` (0-1) controls how much bias-adjustment is applied to the back-transformed mean (0 = the
  back-transformed median; default = the full mean); it doesn't affect intervals.
- `Predictions.samples_to_dataframe()` summarizes samples of arbitrary quantities into the same format as
  `to_dataframe()`.
- `SmearingTransform(base, residuals)` wraps a transform so that back-transformed means use the empirical
  distribution of standardized residuals instead of assuming gaussian noise (Duan's smearing estimator); intervals
  are unaffected. `SmearingTransform.from_predictions(base, predictions, y, measure)` builds one from a model's
  residuals. For a measure with mixture components, it smears each regime with its own residuals, weighted by the
  probability that each observation came from that regime -- so a component's back-transformed mean uses the
  residuals attributed to it, rather than e.g. a lognormal mean that's very sensitive to a large component
  variance. Custom transforms can override `Transform.expected_inverse()` (a closed form) or
  `Transform.noise_nodes()` (a different noise distribution).
- `RegimeTransform(standard, components={component_id: Transform})` chooses how each regime of a mixture measure is
  back-transformed, e.g. `components={'quick': LogTransform(bias_adjust=0)}`.

### New feature: sampling predictions, and derived quantities

- `Predictions.sample(num_samples, observation_noise=True)` draws from the predictive distribution, jointly across
  measures (independently per group/timestep; for trajectories, see `simulate()`). Each draw samples the state (and
  regime, for mixture models), giving the conditional `means`/`covs` of the measures; with `observation_noise=True`,
  observations are sampled too (binomial draws for binary measures). Returns a `PredictionSamples`.
- `Predictions.to_dataframe(derived={'total': lambda s: s['a'] + s['b']})` adds quantities computed from several
  measures, from joint samples (so correlations between measures are accounted for), on the scale given by
  `transform`. E.g. expected weekly spend from a binary 'visited' measure and a log-spend measure:
  `derived={'weekly_spend': lambda s: s['visited'] * s['log_spend'].nan_to_num()}` with
  `transform={'log_spend': LogTransform()}`.

### Other changes

- `Predictions.get_state_at_times()` returns a `StateTuple` rather than a tuple. It behaves like the `(mean, cov)`
  tuple it replaces (unpacking, `len()`, indexing), and also carries the regime-probabilities of models with mixture
  components, so that passing it as `initial_state` continues a forecast where it left off. (Code that checks
  `isinstance(..., tuple)` will need updating.)
- `AdaptiveScaler.forward()` takes an optional `weights` argument. It's only passed for models with mixture
  components, so custom subclasses only need to accept it to be used with mixtures.
- For subclasses of `KalmanFilter`: the update-step is now split into `_prepare_update()` (a hook to adjust
  inputs, called once per step), `_kalman_update()`, and `_mixture_update()`. Subclasses that previously overrode
  `_update_step()` to adjust its inputs (as `BinomialFilter` did) should override `_prepare_update()` instead.
- Adds `benchmarks/profile_simple_model.py`, for checking performance against other git refs.

### Bug fix that changes predictions: `to_dataframe(conf=None)`

- The `std` column of `Predictions.to_dataframe(conf=None)` was 0.76x the actual standard-deviation. **Code that
  uses this column -- e.g. for a manual bias-corrected back-transform like `exp(mean + std**2 / 2)` -- will now get
  larger (correct) values.** Consider using `to_dataframe(transform=...)` instead, which back-transforms correctly.

### Behavior change: adaptive scaling starts at "no adjustment"

`EWMAdaptiveScaler` (used by `adaptive_scaling=True`) now starts its running mean of squared residuals at 1 instead of 0. Before, a group with no observations kept a running variance of `eps`, so its standard deviations were multiplied by `sqrt(eps) ** weight` (often ~0.2–0.3): its forecast intervals were arbitrarily narrow. And groups with only a few observations were shrunk towards zero variance unless the learned initial step-size (`rho`) was close to 1. Now a group with no observations gets a multiplier of exactly 1 (no adjustment), and a group with few observations is shrunk towards no adjustment.

Models saved by older versions keep the old behavior, whether loaded by unpickling (`torch.load` of the whole model) or via `load_state_dict()` (state-dicts without the new `adaptive_scaling._extra_state` entry). Note that state-dicts saved by this version have that extra entry, so they can't be loaded with `strict=True` into older versions of torchcast.

## v1.2.0 (2026-06-08)

### New Features

**EKF**

The KalmanFilter's support for extended-kalman-filtering, introduced silently in [v1.1.0](https://github.com/onesixsolutions/torchcast/releases/tag/v1.1.0), is now public, focusing on two newly documented features:

- `SaturatedLinearModel` process is now documented and exported from the `torchcast.process` top-level namespace.
- `BinomialFilter` is now documented and exported from ``torchcast.kalman_filter``.

**`TimeSeriesDataset and DataLoader`**

- Adds `standardize()` method to `TimeSeriesDataset` that centers and scales one or more tensors in a dataset. 
- `TimeSeriesDataLoader` has been refactored to improve support for subclasses, which can override ``_collate_fn`` to determine how each group's DataFrame gets collated and then transformed into a `TimeSeriesDataset`.

### Deprecations / Breaking Changes

- **Covariance module restructured**: The `torchcast/covariance` module directory has been consolidated into a single `torchcast/covariance.py` module. Existing imports from `torchcast.covariance.base` (typically in pickled models) will continue to work via a backwards-compatibility shim but will emit a `DeprecationWarning`.

## v1.1.1 (2026-04-17)

### Bug fix: LBFGS default optimizer regression with PyTorch >= 2.10

The default LBFGS optimizer now explicitly sets `max_eval=25`. This restores correct training behavior after [pytorch/pytorch#161488](https://github.com/pytorch/pytorch/pull/161488) (shipped in PyTorch 2.10) fixed a bug where `max_eval` was silently ignored by the strong Wolfe line search. Prior to that fix, `max_eval` defaulted to `2` (from `max_iter * 1.25 + 1` with `max_iter=1`), which was effectively ignored — the line search ran freely. After the fix, the cap was correctly enforced, causing the optimizer to converge after only a handful of epochs with a poor loss.

## v1.1.0 (2026-04-06)

### Refactor of `Process` API and internals

Rewrite of the `Process` class and its subclasses to improve maintainability and support extended-kalman-filter 
processes. Note the external API behavior is fully backwards-compatible, but models created in an earlier version of 
torchcast cannot be loaded into this newer version (and vice versa) due to renaming/reorganization of the 
`state_dict`.

### Updates to Utils: Data-Loading and Trainer

- The ``TimeSeriesDataLoader`` class has been updated to support batchwise transformations. Its ``from_dataframe()`` method now optionally accepts a function for `X_colnames`, which should take a dataframe for a batch and return the model-matrix for that batch (i.e. a dataframe of predictors). This is useful for memory-intensive transformations, since they can be applied just-in-time to single batch of the data instead of being applied to the entire dataframe before sending it to the dataloader. See the electricity example in the documentation for an example of usage.
- The ``SeasonalEmbeddingsTrainer`` (used in the electricity example) has been deprecated in favor of the more general ``ModelMatEmbeddingsTrainer``, which embeds any high-dimensional model-matrix into a lower dimensional space. See the electricity example in the documentation for an example of usage.

### Experimental

- State-space models (like ``KalmanFilter``) now support an ``adaptive_scaling`` argument. If set to ``True``, then the model will use a learned exponential moving average model to dynamically adjust the model's variance.

### Other

- Python 3.9 or greater is now required.
- Pandas is currently pinned to <3, as support for 3.* has not yet been tested.
- The ``to_dataframe()`` method of ``Predictions`` supports 'predictions',  'states', or 'observed_states'. The last of these replaces ``type='components'``, which is now deprecated.

## v0.6.0 (2025-04-25)

### Updated default `fit()` behavior

The `fit()` method of `torchcast.state_space.StateSpaceModel` has been updated:

* The default `LBFGS` settings have been updated to avoid the unnecessary inner loop (see [here](https://discuss.pytorch.org/t/unclear-purpose-of-max-iter-kwarg-in-the-lbfgs-optimizer/65695/4)).
* The default convergence settings have been updated to increase `patience` to 2 (instead of 1) and increase `max_iter` to 300 (instead of 200).
* To restore the old behavior, pass `optimizer=lambda p: torch.optim.LBFGS(p, max_eval=8, line_search_fn='strong_wolfe'), stopping={'patience' : 1, 'max_iter' : 200}`.
* Convergence is now controlled by a `torch.utils.Stopping` instance (or kwargs for one). This means passing `tol`, `patience`, and `max_iter` directly to `fit` is deprecated; instead call `fit(stopping={'patience' : ... etc})`.

### Updated default `Covariance` behavior

* The 'low_rank' method is never chosen by default; if desired it must be selected manually using the 'method' kwarg (previously would automatically be chosen if rank was >= 10). This was based on poor performance empirically.
* The starting values for the covariance diagonal have been increased.
* Added `initial_covariance` kwarg to `KalmanFilter` and subclasses.

### Updates to `BinomialFilter`

* Added the `observed_counts` argument, allowing the user to specify whether observations are counts or proportions. If `num_obs==1` then this argument is not required (since they are the same). 
* Fix bug in BinomialStep's kalman-gain calculation when num_obs > 1
* Fix issues with BinomialFilter on the GPU.
* Fix `__getitem__()` for BinomialPredictions.
* Fix monte-carlo `BinomialPredictions.log_prob()` to properly marginalize over samples.

### Other Fixes

* Fix `get_durations()` on GPU.
* Remove redundant matmul in `KalmanStep._update()`
* `ss_step` is no longer a property but is instead an attribute, avoids unnecessary re-instantiation on each timestep

## v0.5.1 (2025-01-09)

### Documentation

* New [Using NN’s for Long-Range Forecasts: Electricity Data](https://docs.strong.io/torchcast/examples/electricity.html#Using-NN's-for-Long-Range-Forecasts:-Electricity-Data) example
* Documentation/README cleanup

### Trainers

Add `torchcast.utils.training` module with...

* `SimpleTrainer` for training simple `nn.Module`s
* `SeasonalEmbeddingsTrainer` for training `nn.Module`s to embed seasonal patterns.
* `StateSpaceTrainer` for training torchcast's `StateSpaceModel`s (when data are too big for the `fit()` method)

### Baseline

* Add `make_baseline` helper to generate baseline forecasts using a simple n-back method 3641e7c137fb7574d13eb744312584dafc622650

### Fixes

* Ensure consistent column-ordering and default RangeIndex in output of `Predictions.to_dataframe()` 0a0fc810a78d4508b029d580483484790005cc6b, f33c6380b94548be5158eb2e2344bd7277a05e48
* Fix default behavior in how `TimeSeriesDataLoader` forward-fills nans for the `X` tensor 0a0fc810a78d4508b029d580483484790005cc6b
* Fix seasonal initial values when passing `initial_value` to forward cae28795095110da56f081b3e2ec4fd942c546d1
* Fix behavior of `StateSpaceModel.simulate()` when num_sims > 1 cae28795095110da56f081b3e2ec4fd942c546d1
* Fix extra arg in `ExpSmoother._generate_predictions()` b55324879384e30517045b51d475cf5c9d2cf5e2
* Make `TimeSeriesDataset.split_measures()` usable by removing `which` argument 8f1001b039ce5e6c901774bafe0ffac78b40f02f


## v0.4.1 (2024-10-09)

### Continuous Integration

* ci: Update actions/checkout version ([`ed64632`](https://github.com/onesixsolutions/torchcast/commit/ed646329cfc2665b2c1732b2c05e7ef30b1f80f6))

* ci: Clone repo using PAT ([`d0adaca`](https://github.com/onesixsolutions/torchcast/commit/d0adacac743986e97317ea499b72aee7e6724fc0))

* ci: Enable repo push ([`f565d2a`](https://github.com/onesixsolutions/torchcast/commit/f565d2ac262f7096b304b8aac482303492c37895))

* ci: Use SSH Key ([`469d531`](https://github.com/onesixsolutions/torchcast/commit/469d53114417314ee28d5fa655b67a6b3310d7e5))

* ci: Fix docs job permissions ([`e6e2e34`](https://github.com/onesixsolutions/torchcast/commit/e6e2e346d68725b2ff7eaec57f859079221901cb))

* ci: Pick python version form pyproject.toml ([`2a9eef7`](https://github.com/onesixsolutions/torchcast/commit/2a9eef7ca5f609a7fb14197286f72bc6ff095bff))

* ci: Setup auto-release ([`9df4f26`](https://github.com/onesixsolutions/torchcast/commit/9df4f2642fe74e6016c2b2a3980ca0eba3c77403))

### Documentation

* docs: Fix examples ([`6f5a2dc`](https://github.com/onesixsolutions/torchcast/commit/6f5a2dc5cb8fea4895f44bdba67c60f27e09a84b))

* docs: AirQuality datasets [skip ci] ([`c675f04`](https://github.com/onesixsolutions/torchcast/commit/c675f04bb244a73155fbd98c110590bd735808bc))

* docs: Self-hosted docs and fixtures ([`baca184`](https://github.com/onesixsolutions/torchcast/commit/baca184beeb53ff44065b6753b220029c5467f9b))

### Fixes

* fix: AQ Dataset ([`9b6e23e`](https://github.com/onesixsolutions/torchcast/commit/9b6e23e0ac1a0a7511f490f5b92d3b5e5c69fb59))

### Refactoring

* refactor: Switch to pyproject.toml ([`6de2f27`](https://github.com/onesixsolutions/torchcast/commit/6de2f279d82d6fb78e1464aecabdd642969e86e0))
