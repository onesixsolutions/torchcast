# CHANGELOG

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

* ci: Update actions/checkout version ([`ed64632`](https://github.com/strongio/torchcast/commit/ed646329cfc2665b2c1732b2c05e7ef30b1f80f6))

* ci: Clone repo using PAT ([`d0adaca`](https://github.com/strongio/torchcast/commit/d0adacac743986e97317ea499b72aee7e6724fc0))

* ci: Enable repo push ([`f565d2a`](https://github.com/strongio/torchcast/commit/f565d2ac262f7096b304b8aac482303492c37895))

* ci: Use SSH Key ([`469d531`](https://github.com/strongio/torchcast/commit/469d53114417314ee28d5fa655b67a6b3310d7e5))

* ci: Fix docs job permissions ([`e6e2e34`](https://github.com/strongio/torchcast/commit/e6e2e346d68725b2ff7eaec57f859079221901cb))

* ci: Pick python version form pyproject.toml ([`2a9eef7`](https://github.com/strongio/torchcast/commit/2a9eef7ca5f609a7fb14197286f72bc6ff095bff))

* ci: Setup auto-release ([`9df4f26`](https://github.com/strongio/torchcast/commit/9df4f2642fe74e6016c2b2a3980ca0eba3c77403))

### Documentation

* docs: Fix examples ([`6f5a2dc`](https://github.com/strongio/torchcast/commit/6f5a2dc5cb8fea4895f44bdba67c60f27e09a84b))

* docs: AirQuality datasets [skip ci] ([`c675f04`](https://github.com/strongio/torchcast/commit/c675f04bb244a73155fbd98c110590bd735808bc))

* docs: Self-hosted docs and fixtures ([`baca184`](https://github.com/strongio/torchcast/commit/baca184beeb53ff44065b6753b220029c5467f9b))

### Fixes

* fix: AQ Dataset ([`9b6e23e`](https://github.com/strongio/torchcast/commit/9b6e23e0ac1a0a7511f490f5b92d3b5e5c69fb59))

### Refactoring

* refactor: Switch to pyproject.toml ([`6de2f27`](https://github.com/strongio/torchcast/commit/6de2f279d82d6fb78e1464aecabdd642969e86e0))
