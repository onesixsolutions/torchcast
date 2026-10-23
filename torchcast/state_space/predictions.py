import math
import numbers
from math import log

from dataclasses import dataclass, fields
from typing import Any, Tuple, Union, Optional, Sequence, TYPE_CHECKING, Callable, Iterator, Mapping
from warnings import warn

import torch

import numpy as np
import pandas as pd

from scipy import stats

from .transforms import Transform, _IdentityTransform
from torchcast.internals.utils import get_nan_groups, class_or_instancemethod, ragged_cat, mvnorm_log_prob

if TYPE_CHECKING:
    from .state import StateTuple
    from torchcast.mixture import MixtureModel
    from torchcast.utils import TimeSeriesDataset
    from torchcast.internals.batch_design import MeasurementModel
    from torchcast.internals.monte_carlo import FixedWhiteNoise

# seed for sample-based outputs (``to_dataframe()``'s monte-carlo means and intervals, ``derived``, states), drawn at
# import: repeated calls within a process give the same results. (Separate from the model's ``mc_sampling``, which is
# only for the monte-carlo log-prob, fixed so that it's a deterministic function of the parameters.) Draws of the
# state, and draws given the state, use separate seeds (see ``Predictions._output_generator``).
_OUTPUT_SEED = int(np.random.RandomState().randint(0, 2 ** 31 - 1))
# the default number of samples for sample-based outputs:
_DEFAULT_NUM_SAMPLES = 1000
# sample-based outputs are computed in chunks of groups, so that their samples have at most about this many elements
# (~64MB per tensor in float32). since their noise is shared across rows, the results don't depend on the chunking.
_MAX_OUTPUT_SAMPLE_ELEMENTS = 2 ** 24


class Predictions:
    """
    The output of the :class:`.StateSpaceModel` forward pass, containing the underlying state means and covariances, as
    well as methods such as ``log_prob()``, ``to_dataframe()``, and ``plot()``.

    The states (``state_means``, ``state_covs``) are those that each timestep's observations are predicted from: for
    filtering/forecasting, the predicted (prior) states, before that timestep's update; for :func:`.simulate`, the
    simulated states. (``update_means`` and ``update_covs``, if included, are the states after each update.)
    """
    _means = None
    _covs = None

    # Construction & metadata ------------------------------------------------------------------------------------------

    def __init__(self,
                 measurement_model: 'MeasurementModel',
                 states: Union[Sequence['StateTuple'], 'StateTuple', tuple],
                 measure_covs: Union[Sequence[torch.Tensor], torch.Tensor],
                 updates: Union[Sequence['StateTuple'], 'StateTuple', tuple, None] = None,
                 mc_white_noise: Optional['FixedWhiteNoise'] = None,
                 mixture: Optional['MixtureModel'] = None):
        """
        :param measurement_model: The ``MeasurementModel``.
        :param states: The predicted states: a sequence (one per timestep) of :class:`.StateTuple`; or a single
         ``StateTuple`` whose tensors have leading dims ``(num_groups, num_timesteps)``; or a tuple of means and covs.
         For models with mixture components, the ``StateTuple`` objects carry the ``regime_probs``.
        :param measure_covs: The measure-covariance for each timestep.
        :param updates: Optionally, the updated (filtered) states, in the same format as ``states``.
        :param mc_white_noise: Required if the measurement-model is nonlinear.
        :param mixture: The model's :class:`.MixtureModel`, if any.
        """
        self.state_means, self.state_covs, self.regime_probs, self.scaling_states = _unpack_states(states)
        self.measure_covs = _maybe_stack(measure_covs, 1)

        self.measurement_model = measurement_model
        self.measurement_model_flat = self.measurement_model.flattened()

        self.update_means = self.update_covs = self.update_regime_probs = self.update_scaling_states = None
        if updates is not None:
            (self.update_means, self.update_covs, self.update_regime_probs,
             self.update_scaling_states) = _unpack_states(updates)

        if mc_white_noise is None and self.measurement_model.is_nonlinear:
            raise ValueError(
                "Since the measurement model is nonlinear, the `mc_white_noise` argument must be specified."
            )

        self.mc_white_noise = mc_white_noise

        self._dataset_metadata = None
        self._state_means_flat = None
        self._state_covs_flat = None
        self._mcovs_flat = None

        self.mixture = mixture
        if self.mixture is not None and self.regime_probs is None:
            raise ValueError("If `mixture` is passed, `states` must carry `regime_probs`.")

    @property
    def num_groups(self) -> int:
        return len(self.state_means)

    @property
    def num_timesteps(self) -> int:
        return self.state_means.shape[1]

    def set_metadata(self,
                     dataset: Optional['TimeSeriesDataset'] = None,
                     group_names: Optional[Sequence[str]] = None,
                     start_offsets: Optional[np.ndarray] = None,
                     group_colname: str = 'group',
                     time_colname: str = 'time',
                     dt_unit: Optional[str] = None) -> 'Predictions':
        if dataset is not None:
            group_names = dataset.group_names
            start_offsets = dataset.start_offsets
            dt_unit = dataset.dt_unit

        if isinstance(dt_unit, str):
            dt_unit = np.timedelta64(1, dt_unit)

        if group_names is not None and len(group_names) != self.num_groups:
            raise ValueError("`group_names` must have the same length as the number of groups.")
        if start_offsets is not None and len(start_offsets) != self.num_groups:
            raise ValueError("`start_offsets` must have the same length as the number of groups.")

        kwargs = {
            'group_names': group_names,
            'start_offsets': start_offsets,
            'dt_unit': dt_unit,
            'group_colname': group_colname,
            'time_colname': time_colname
        }
        if self._dataset_metadata is not None:
            self._dataset_metadata.update(**kwargs)
        else:
            self._dataset_metadata = DatasetMetadata(**kwargs)
        return self

    @property
    def dataset_metadata(self) -> 'DatasetMetadata':
        if self._dataset_metadata is None:
            raise RuntimeError("Metadata not set. Pass the dataset or call `set_metadata()`.")
        return self._dataset_metadata

    # Predicted means & covariances ------------------------------------------------------------------------------------

    @property
    def means(self) -> torch.Tensor:
        """
        Returns the observed means of the predictions, i.e. the measured means of the state.

        For models with mixture components, ``means`` and ``covs`` are the exact mean and covariance of the mixture
        over regimes -- but the predictive distribution is not gaussian, so e.g. don't use them to construct intervals,
        or to back-transform a transformed measure. Use :func:`get_mixture` (or ``_get_regime_combos()``) instead.
        """
        if self._means is None:
            self._means, self._covs = self._observe()
        return self._means

    @property
    def covs(self) -> Optional[torch.Tensor]:
        if self._means is None:
            self._means, self._covs = self._observe()
        if self._covs is None:
            if not _warn_once.get('cov', False):
                warn("The measurement model is nonlinear, so no closed-form covariance is available, returning None.")
                _warn_once['cov'] = True
        elif self.mixture is not None and not _warn_once.get('mixture_cov', False):
            warn(
                "With mixture components, `covs` is the covariance of a (non-gaussian) mixture: e.g. don't use it to "
                "construct intervals, sample, or back-transform. See `get_mixture()`, `to_dataframe(transform=...)`, "
                "and `sample()`."
            )
            _warn_once['mixture_cov'] = True
        return self._covs

    def __iter__(self) -> Iterator[Optional[torch.Tensor]]:
        # so that we can do ``mean, cov = predictions``
        yield self.means
        yield self.covs

    def __array__(self) -> np.ndarray:
        # for numpy.asarray
        return self.means.detach().cpu().numpy()

    def _observe(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_shape = self.state_means.shape[0:2]

        if self.measurement_model.is_nonlinear:
            # monte-carlo for the means of nonlinear measures; there's no closed form cov
            return self._map_group_chunks(
                lambda chunk: chunk._get_means(use_map=False), num_samples=_DEFAULT_NUM_SAMPLES
            ), None
        elif self.mixture is not None:
            # the exact mean and covariance of the mixture over regime-combos:
            _, probs, means, covs = self._get_regime_combos()
            mean = (probs.unsqueeze(-1) * means).sum(-2)
            second_moment = (probs[..., None, None] * (covs + means.unsqueeze(-1) * means.unsqueeze(-2))).sum(-3)
            cov = second_moment - mean.unsqueeze(-1) * mean.unsqueeze(-2)
            return mean, cov
        else:
            measured_mean, system_cov = self._measured_moments_flat()
            return measured_mean.view(*batch_shape, -1), system_cov.view(*batch_shape, *self.measure_covs.shape[-2:])

    def get_means(self,
                  transform: Union['Transform', dict[str, 'Transform'], None] = None,
                  use_map: Optional[bool] = None,
                  num_samples: int = _DEFAULT_NUM_SAMPLES) -> torch.Tensor:
        """
        The predicted means, as in the ``mean`` column of :func:`to_dataframe` with the same ``transform``,
        ``use_map`` and ``num_samples`` -- without computing any intervals.

        :param transform: See :func:`to_dataframe`.
        :param use_map: See :func:`to_dataframe`.
        :param num_samples: See :func:`to_dataframe`.
        :return: A ``(num_groups, num_timesteps, num_measures)`` tensor.
        """
        use_map = self._resolve_use_map(use_map)
        transforms = self._standardize_transforms(transform)
        return self._map_group_chunks(
            lambda chunk: chunk._get_means(use_map=use_map, transforms=transforms, mmean_samples=num_samples),
            num_samples=num_samples if not use_map else 0,
        )

    def _get_means(self,
                   use_map: bool,
                   transforms: Optional[dict[str, 'Transform']] = None,
                   mmean_samples: Union[torch.Tensor, int, None] = None,
                   adjust_bias: bool = True) -> torch.Tensor:
        """
        The means, given already-resolved arguments -- separate from :func:`get_means` so that callers can share one
        set of samples (e.g. ``to_dataframe``: between the means, the intervals and ``derived``).

        :param mmean_samples: Pass already-computed samples to save compute (and to share them), otherwise will
         compute if needed (or for the latter this can be an int for num_draws).
        :param adjust_bias: Whether the transforms' ``bias_adjust`` applies, for measures with a linear measured-mean
         (False: the mean of the back-transformed distribution).
        :return: A ``(num_groups, num_timesteps, num_measures)`` tensor: the back-transformed measured mean
         (handling mixtures if needed).
        """
        transforms = transforms or {}
        identity = _IdentityTransform()
        all_measures = list(self.measurement_model.measures)
        batch_shape = self.state_means.shape[0:2]

        # divide measures into (as in `_get_intervals`):
        # (1) nonlinear-measures (mixture or not),
        # (2) mixture-but-linear-measures,
        # (3) everything else
        # with `use_map`, nonlinear measures are treated like linear ones, with the moments of the linearized
        # measurement-model (the MAP measured-mean, and H P H^T + R) -- so there's no monte-carlo:
        nonlinear_measures = [] if use_map else list(self._nonlinear_measures)
        _mix_measures = self.mixture.mixture_measures if self.mixture is not None else []
        mixture_measures = [m for m in _mix_measures if m not in nonlinear_measures]
        simple_measures = [m for m in all_measures if m not in nonlinear_measures and m not in mixture_measures]

        # exact for linear measures; the MAP and the linearized variance for nonlinear ones:
        measured_mean, system_cov = self._measured_moments_flat()

        out = {}

        # simple measures: closed form (or quadrature), given the mean and variance:
        for measure in simple_measures:
            midx = all_measures.index(measure)
            transform = transforms.get(measure, identity)
            mean = transform.inverse_mean(measured_mean[:, midx], system_cov[:, midx, midx], adjust_bias=adjust_bias)
            out[measure] = mean.view(*batch_shape)

        # mixture measures: the same, for each regime, mixed by the regime-probabilities:
        for measure in mixture_measures:
            midx = all_measures.index(measure)
            # (not `get_mixture()`, which warns for nonlinear measures: here, with `use_map`, that's intended)
            mixture = self._get_mixture(
                measure,
                standard_mean=measured_mean[:, midx].view(*batch_shape),
                standard_var=system_cov[:, midx, midx].view(*batch_shape),
            )
            transform = transforms.get(measure, identity)
            regime_means = [
                transform.inverse_mean(mixture.means[..., k], mixture.vars[..., k], adjust_bias, regime=label)
                for k, (label, _, _) in enumerate(self._regimes(measure))
            ]
            out[measure] = (mixture.probs * torch.stack(regime_means, -1)).sum(-1)

        # nonlinear measures (without `use_map`): for each regime, monte-carlo over the state, mixed by the
        # regime-probabilities (all ones without mixture components). `bias_adjust` scales the variance, including the
        # state's: so the state is sampled with its deviations scaled -- once per measure, so its regimes' nonzero
        # values must be the same. A regime with `bias_adjust=0` needs no sampling: it's the plug-in `inverse(MAP)`.
        var_multis = {m: self._nonlinear_var_multis(m, transforms.get(m, identity)) for m in nonlinear_measures}
        sampled = {m: [v for v in vm.values() if v is not None] for m, vm in var_multis.items()}
        if any(sampled.values()):
            deviation_scales = [math.sqrt(sampled[m][0]) if sampled.get(m) else 1. for m in all_measures]
            mmean_samples = self._resolve_mmean_samples(mmean_samples, deviation_scales=deviation_scales)

        for measure in nonlinear_measures:
            midx = all_measures.index(measure)
            transform = transforms.get(measure, identity)
            noise_var = self.measure_covs_flat[:, midx, midx]
            regime_means = []
            for label, offset, extra in self._regimes(measure):
                var_multi = var_multis[measure][label]
                if var_multi is None:  # (bias_adjust=0)
                    regime_means.append(transform.inverse(measured_mean[:, midx] + offset))
                else:
                    regime_means.append(transform.inverse_mean(
                        mmean_samples[..., midx] + offset,
                        (noise_var + extra) * var_multi,
                        adjust_bias=False,
                        regime=label,
                    ).mean(0))
            if len(regime_means) == 1:
                out[measure] = regime_means[0].view(*batch_shape)
                continue
            probs = self._measure_regime_probs(measure).view(-1, len(regime_means))
            out[measure] = (probs * torch.stack(regime_means, -1)).sum(-1).view(*batch_shape)

        return torch.stack([out[m] for m in all_measures], -1)

    def _needs_mc_means(self, use_map: bool, transforms: dict[str, 'Transform']) -> bool:
        """
        Whether the means need monte-carlo samples: for measures with a nonlinear measured-mean, without ``use_map``,
        unless every regime has ``bias_adjust=0`` (the plug-in, see ``_get_means``).
        """
        if use_map:
            return False
        identity = _IdentityTransform()
        return any(
            not np.isclose(transforms.get(m, identity).var_multi(label), 0)
            for m in self._nonlinear_measures
            for label, _, _ in self._regimes(m)
        )

    def _nonlinear_var_multis(self, measure: str, transform: 'Transform') -> dict[Optional[str], Optional[float]]:
        """
        For the monte-carlo means of a measure with a nonlinear measured-mean: each regime's ``bias_adjust`` (see
        ``Transform.var_multi``), or None where it's 0 (no sampling needed). The state is sampled once per measure, so
        the other regimes' values must be the same -- if not, they're ignored (with a warning), i.e. set to 1.
        """
        out = {}
        for label, _, _ in self._regimes(measure):
            var_multi = transform.var_multi(label)
            out[label] = None if np.isclose(var_multi, 0) else var_multi
        nonzero = {v for v in out.values() if v is not None}
        if len(nonzero) > 1:
            warn(
                f"'{measure}' has a nonlinear measured-mean, so its regimes' `bias_adjust` values must be 0 or the "
                f"same, got {transform.bias_adjust}; using 1 for those that aren't 0."
            )
            out = {k: (None if v is None else 1.) for k, v in out.items()}
        return out

    # Log-probability --------------------------------------------------------------------------------------------------

    def log_prob(self,
                 obs: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 nan_groups_flat: Optional[Sequence[tuple[torch.Tensor, Optional[torch.Tensor]]]] = None
                 ) -> torch.Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the ``StateSpaceModel``).

        :param obs: A Tensor that could be used in the ``StateSpaceModel`` forward pass.
        :param weights: If specified, will be used to weight the log-probability of each group X timestep.
        :param nan_groups_flat: used by StateSpaceModel.fit() for speeding up computations, pre-computing nan-masks at
         the start of fitting rather than doing so on each call to log_prob().
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        assert len(obs.shape) == 3
        measure_rank = obs.shape[-1]
        state_rank = self.state_means.shape[-1]

        obs_flat = obs.reshape(-1, measure_rank)
        if weights is None:
            weights = torch.ones(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)
        else:
            assert weights.shape == obs.shape[0:-1]
            weights = weights.view(-1)
        state_means_flat = self.state_means.view(-1, state_rank)
        state_covs_flat = self.state_covs.view(-1, state_rank, state_rank)
        measure_covs_flat = self.measure_covs.view(-1, measure_rank, measure_rank)

        lp_flat = torch.zeros(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)

        if nan_groups_flat is None:
            nan_groups_flat = get_nan_groups(torch.isnan(obs_flat))

        for gt_idx, masks in nan_groups_flat:
            if masks is None:
                val_idx = None
                gt_obs = obs_flat[gt_idx]
                gt_mcov = measure_covs_flat[gt_idx]
                gt_mmodel = self.measurement_model_flat.subset(gt_idx)
            else:
                val_idx, m1d, m2d = masks
                gt_mmodel = self.measurement_model_flat.subset(gt_idx, measures=val_idx)
                gt_mcov = measure_covs_flat[m2d]
                gt_obs = obs_flat[m1d]
            _kwargs = self._get_log_prob_kwargs(gt_idx, val_idx)
            lp_flat[gt_idx] = self._log_prob(
                obs=gt_obs,
                state_means=state_means_flat[gt_idx],
                state_covs=state_covs_flat[gt_idx],
                measure_cov=gt_mcov,
                measurement_model=gt_mmodel,
                **_kwargs
            )

        lp_flat = lp_flat * weights

        return lp_flat.view(obs.shape[0:2])

    def _get_log_prob_kwargs(self, group_idx: torch.Tensor, measure_idx: Optional[torch.Tensor]) -> dict:
        """
        :param group_idx: Indices into the flattened (group*time) predictions.
        :param measure_idx: The observed measures, or None if all are observed.
        """
        out = {}
        if self.regime_probs is not None:
            regime_probs_flat = self.regime_probs.reshape(-1, self.regime_probs.shape[-1])
            out['regime_log_prior'] = regime_probs_flat[group_idx].clamp_min(1e-30).log()
        return out

    def _log_prob(self,
                  obs: torch.Tensor,
                  state_means: torch.Tensor,
                  state_covs: torch.Tensor,
                  measure_cov: torch.Tensor,
                  measurement_model: 'MeasurementModel',
                  regime_log_prior: Optional[torch.Tensor] = None,
                  **kwargs) -> torch.Tensor:
        if kwargs:
            raise TypeError(f"`_log_prob()` does not accept additional keyword arguments, got {set(kwargs)}")
        assert measurement_model.num_timesteps == 1

        monte_carlo = measurement_model.is_nonlinear
        if monte_carlo:
            # samples of the measured-mean, each with the measurement-noise:
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=measurement_model,
                state_means=state_means,
                state_covs=state_covs,
                white_noise=self._log_prob_white_noise(measurement_model),
            )
            resid = obs - mmean_samples
            ll_cov = measure_cov
        else:
            measured_mean, measure_mat = measurement_model(mean=state_means, time=0)
            resid = obs - measured_mean
            ll_cov = measure_mat @ state_covs @ measure_mat.transpose(-1, -2) + measure_cov

        if not self._is_mixture_model(measurement_model):
            return _log_lik(resid, ll_cov, monte_carlo=monte_carlo)

        # the marginal log-lik over the (effective) regime-combos. these all share the predicted state, so each just
        # offsets the mean and adds to the diagonal:
        terms = [
            log_prior + _log_lik(resid - shift, ll_cov + torch.diag(extra_var), monte_carlo=monte_carlo)
            for shift, extra_var, log_prior
            in self._regime_terms(measurement_model.measures, log_prior=regime_log_prior, like=obs)
        ]
        return torch.logsumexp(torch.stack(terms, -1), -1)

    def _is_mixture_model(self, measurement_model: 'MeasurementModel') -> bool:
        if self.mixture is None:
            return False
        return any(m in measurement_model.measures for m in self.mixture.mixture_measures)

    def _regime_terms(self,
                      measures: Sequence[str],
                      like: torch.Tensor,
                      log_prior: Optional[torch.Tensor] = None
                      ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        For the mixture's marginal log-likelihood, ``logsumexp_c(log_prior_c + log_lik_c)``: the regime-combos that
        are distinguishable given ``measures`` (see ``MixtureModel.effective_combos``).

        :param like: A ``(batch, ...)`` tensor, for the number of rows, dtype and device.
        :param log_prior: A ``(batch, num_combos)`` tensor of log prior regime-probabilities. Defaults to the
         regime-model's base-probs.
        :return: For each effective combo, two ``(len(measures),)`` tensors -- the offset to each measure's
         measured-mean and the variance to add to its measurement-noise -- and the ``(batch,)`` log prior probability
         (pooled over the combos that are indistinguishable given ``measures``).
        """
        if log_prior is None:
            log_prior = self.mixture.log_base_probs().expand(like.shape[0], -1)
        effective, mapping = self.mixture.effective_combos(measures)
        for e, eff in enumerate(effective):
            shift, extra_var = self.mixture.effective_offsets(eff, len(measures), like=like)
            yield shift, extra_var, torch.logsumexp(log_prior[:, (mapping == e).to(log_prior.device)], dim=-1)

    # Dataframes & plotting --------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def to_dataframe(self,
                     dataset: Optional['TimeSeriesDataset'] = None,
                     type: str = 'predictions',
                     group_colname: Optional[str] = None,
                     time_colname: Optional[str] = None,
                     conf: Union[float, None, bool] = .95,
                     use_map: Optional[bool] = None,
                     transform: Union['Transform', dict[str, 'Transform'], None] = None,
                     derived: Optional[dict[str, Callable[[dict[str, torch.Tensor]], torch.Tensor]]] = None,
                     num_samples: int = _DEFAULT_NUM_SAMPLES) -> pd.DataFrame:
        """
        :param dataset: If not provided, will use the metadata set by ``set_metadata()``.
        :param type: What type of dataframe to return, either 'predictions',  'states', or 'observed_states'.
        :param group_colname: The name of the column to use for groups, defaults to the metadata's `group_colname`.
        :param time_colname: The name of the column to use for time, defaults to the metadata's `time_colname`.
        :param conf: The confidence level for the confidence intervals, defaults to 0.95. ``None`` returns a ``std``
         column instead of the intervals; ``False`` returns only the means, skipping all of the work of computing
         intervals (for predictions; see also :func:`get_means`).
        :param use_map: For measures with a nonlinear measured-mean (e.g. a measure-function, or nonlinear processes):
         whether the mean averages over the state's uncertainty by monte-carlo (``use_map=False``), or plugs in the
         state's mean (``use_map=True``; the MAP), treating the measure like a linear one, with the moments of the
         linearized measurement-model. (The intervals are monte-carlo either way.) Reasons to use the MAP: for a
         monotone measure-function (e.g. a sigmoid), it's the predictive *median* of the measured-mean rather than its
         mean (better for e.g. absolute error); it doesn't depend on the state's variance, so it's robust to that
         being miscalibrated (an overestimated variance pulls the monte-carlo mean of a sigmoid towards .5); and
         it's cheap. With a ``transform``, the back-transformed mean then works as for linear measures: its
         ``bias_adjust`` sets how much of the (linearized) variance is used -- e.g. ``bias_adjust=0`` gives the
         plug-in ``inverse(MAP)``, a median-like forecast throughout. (``bias_adjust`` works the same way for the
         monte-carlo mean, where it scales the state's variance too.)
         Ignored for models without nonlinear measures, whose means are exact.
        :param transform: For measures that were transformed before modeling (e.g. log), a :class:`.Transform` to
         map the predictions (and actuals) back to the original scale: either a single ``Transform`` for all measures,
         or a dictionary of ``{measure: Transform}``. The mean is back-transformed exactly (not just the mean on the
         modeled scale, back-transformed), as are the intervals. Only for ``type='predictions'``.
        :param derived: Quantities computed from multiple measures, e.g. ``{'total': lambda s: s['a'] + s['b']}``.
         Each function receives a dictionary of ``{measure: (num_samples, num_groups, num_timesteps)}`` samples from
         the (joint) predictive distribution, including observation noise and on the scale given by ``transform``, and
         should return a ``(num_samples, num_groups, num_timesteps)`` tensor, computing each group and timestep from
         its own values (e.g. not a total across groups: for memory, it's applied to chunks of groups at a time).
         Each is added to the output with its key as the 'measure'; the mean and intervals are computed across samples
         (see :func:`sample`). The mean uses the measures whose means are exact (those with a linear measured-mean) as
         control variates, which reduces its monte-carlo error -- e.g. a sum of these measures is exact. If there are
         actuals, the function is applied to them too (so it should handle missing values, which are nan; measures
         that aren't in ``dataset`` are all-nan). Only for ``type='predictions'``.
        :param num_samples: The number of monte-carlo samples, for the outputs that need them: the means (with
         ``use_map=False`` or a ``transform``) and intervals of measures with a nonlinear measured-mean, ``derived``,
         and ``type='states'``/``'observed_states'`` for nonlinear models. Drawn with a fixed seed, so repeated calls
         agree. (Separate from the model's ``mc_sampling``, which is only for the log-prob, in training.)
        """
        dataset = self._resolve_dataset(dataset)
        group_colname = group_colname or self.dataset_metadata.group_colname
        time_colname = time_colname or self.dataset_metadata.time_colname

        if conf is not None and conf is not False:
            assert conf >= .50

        type = type.casefold()
        if conf is False and not type.startswith('pred'):
            raise ValueError("`conf=False` is only supported for ``type='predictions'``.")
        if (transform is not None or derived) and not type.startswith('pred'):
            raise ValueError("`transform` and `derived` are only supported for ``type='predictions'``.")
        if type.startswith('pred'):
            return_std = False
            if conf is None:
                if transform is not None or derived:
                    raise ValueError("`conf=None` (i.e. returning `std`) is not supported with `transform`/`derived`.")
                # the conf-level for which upper - lower == 1 std-dev (i.e. +/- .5 std-devs):
                conf = 1 - 2 * stats.norm.cdf(-.5)
                return_std = True

            df = self._to_dataframe(
                dataset=dataset,
                group_colname=group_colname,
                time_colname=time_colname,
                conf=conf,
                use_map=use_map,
                transform=transform,
                derived=derived,
                num_samples=num_samples,
            )
            if return_std:
                df['std'] = df.pop('upper') - df.pop('lower')
            return df
        elif type in ('components', 'states', 'observed_states'):
            if use_map:
                raise NotImplementedError("``use_map=True`` not yet implemented for type!='predictions'")
            if type == 'components':
                warn("`type='components'` is deprecated, use `type='observed_states'` instead.", DeprecationWarning)
            return self._to_components_dataframe(
                dataset=dataset,
                group_colname=group_colname,
                time_colname=time_colname,
                conf=conf,
                measured=type in ('observed_states', 'components'),
                num_samples=num_samples,
            )
        else:
            raise ValueError(f"Expected type to be 'predictions', 'states', or 'observed_states', got '{type}'.")

    @torch.inference_mode()
    def samples_to_dataframe(self,
                             samples: dict[str, torch.Tensor],
                             dataset: Optional['TimeSeriesDataset'] = None,
                             conf: float = .95,
                             actuals: Optional[dict[str, torch.Tensor]] = None,
                             group_colname: Optional[str] = None,
                             time_colname: Optional[str] = None) -> pd.DataFrame:
        """
        Summarize samples (e.g. of some quantity computed from ``sample()``) into a dataframe with the same format as
        ``to_dataframe()``: the mean and the ``conf`` interval across samples, for each group and timestep.

        :param samples: A dictionary with ``(num_samples, num_groups, num_timesteps)`` tensors. The keys become the
         'measure' column of the output.
        :param dataset: If not provided, will use the metadata set by ``set_metadata()``.
        :param conf: The confidence level for the intervals.
        :param actuals: Optionally, a dictionary with ``(num_groups, num_timesteps)`` tensors of actual values for some
         keys of ``samples``.
        :param group_colname: The name of the column to use for groups, defaults to the metadata's `group_colname`.
        :param time_colname: The name of the column to use for time, defaults to the metadata's `time_colname`.
        """
        assert .5 <= conf < 1
        alpha = (1 - conf) / 2
        by_name = {}
        for name, x in samples.items():
            if x.shape[1:] != (self.num_groups, self.num_timesteps):
                raise ValueError(
                    f"Expected `samples['{name}']` to have shape (num_samples, {self.num_groups}, {self.num_timesteps})"
                    f", got {tuple(x.shape)}"
                )
            by_name[name] = (x.mean(0), *_quantile_interval(x, alpha))
        return self._summaries_to_dataframe(
            by_name,
            actuals=actuals or {},
            dataset=self._resolve_dataset(dataset),
            group_colname=group_colname or self.dataset_metadata.group_colname,
            time_colname=time_colname or self.dataset_metadata.time_colname,
        )

    @class_or_instancemethod
    def plot(cls,
             df: Optional[Union[pd.DataFrame, 'TimeSeriesDataset']] = None,
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs):
        """
        :param df: A dataset, or the output of :func:`Predictions.to_dataframe()`.
        :param group_colname: The name of the group-column.
        :param time_colname: The name of the time-column.
        :param max_num_groups: Max. number of groups to plot; if the number of groups in the dataframe is greater than
         this, a random subset will be taken.
        :param split_dt: If supplied, will draw a vertical line at this date (useful for showing pre/post validation).
        :param kwargs: Further keyword arguments to pass to ``plotnine.theme`` (e.g. ``figure_size=(x,y)``)
        :return: A plot of the predicted and actual values.
        """

        from plotnine import (
            ggplot, aes, geom_line, geom_ribbon, facet_grid, facet_wrap, theme_bw, theme, ylab, geom_vline
        )
        from torchcast.utils import TimeSeriesDataset

        if isinstance(cls, Predictions):  # using it as an instance-method
            group_colname = group_colname or cls.dataset_metadata.group_colname
            time_colname = time_colname or cls.dataset_metadata.time_colname
            if df is None:
                df = cls.to_dataframe()
        elif not group_colname or not time_colname:
            raise TypeError("Please specify group_colname and time_colname")
        elif df is None:
            raise TypeError("Please specify a dataframe `df`")

        if group_colname is None:
            group_colname = 'group'
            if group_colname not in getattr(df, 'columns', []):
                raise TypeError("Please specify group_colname")
        if time_colname is None:
            time_colname = 'time'
            if 'time' not in getattr(df, 'columns', []):
                raise TypeError("Please specify time_colname")

        if isinstance(df, TimeSeriesDataset):
            df = cls.to_dataframe(dataset=df, group_colname=group_colname, time_colname=time_colname)

        is_components = 'process' in df.columns
        if is_components and 'state_element' not in df.columns:
            df = df.assign(state_element='all')

        df = df.copy()
        if 'upper' not in df.columns and 'std' in df.columns:
            raise RuntimeError("Please convert your 'std' column into lower/upper columns.")
        if df[group_colname].nunique() > max_num_groups:
            subset_groups = df[group_colname].drop_duplicates().sample(max_num_groups).tolist()
            if len(subset_groups) < df[group_colname].nunique():
                print("Subsetting to groups: {}".format(subset_groups))
            df = df.loc[df[group_colname].isin(subset_groups), :]
        num_groups = df[group_colname].nunique()

        aes_kwargs = {'x': time_colname}
        if is_components:
            aes_kwargs['group'] = 'state_element'

        plot = (
                ggplot(df, aes(**aes_kwargs)) +
                geom_line(aes(y='mean'), color='#4C6FE7', size=1.5, alpha=.75) +
                geom_ribbon(aes(ymin='lower', ymax='upper'), color=None, alpha=.25) +
                ylab("")
        )

        assert 'measure' in df.columns
        if is_components:
            num_processes = df['process'].nunique()
            if num_groups > 1 and num_processes > 1:
                raise ValueError("Cannot plot components for > 1 group and > 1 processes.")
            elif num_groups == 1:
                plot = plot + facet_wrap(f"~ measure + process", scales='free_y', labeller='label_both')
                if 'figure_size' not in kwargs:
                    from plotnine.facets.facet_wrap import wrap_dims
                    nrow, _ = wrap_dims(len(df[['process', 'measure']].drop_duplicates().index))
                    kwargs['figure_size'] = (12, nrow * 2.5)
            else:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
                if 'figure_size' not in kwargs:
                    kwargs['figure_size'] = (12, num_groups * 2.5)

            if (df.groupby('measure')['process'].nunique() <= 1).all():
                plot = plot + geom_line(aes(y='mean', color='state_element'), size=1.5)

        else:
            if 'actual' in df.columns:
                plot = plot + geom_line(aes(y='actual'))
            if num_groups > 1:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
            else:
                plot = plot + facet_wrap("~measure", scales='free_y', labeller='label_both')

            if 'figure_size' not in kwargs:
                kwargs['figure_size'] = (12, 5)

        if split_dt:
            plot = plot + geom_vline(xintercept=np.datetime64(split_dt), linetype='dashed')

        return plot + theme_bw() + theme(**kwargs)

    @torch.inference_mode()
    def _to_dataframe(self,
                      dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                      group_colname: str,
                      time_colname: str,
                      conf: float,
                      use_map: bool,
                      transform: Union['Transform', dict[str, 'Transform'], None] = None,
                      derived: Optional[dict[str, Callable]] = None,
                      num_samples: int = _DEFAULT_NUM_SAMPLES) -> pd.DataFrame:

        alpha = None if conf is False else (1 - conf) / 2
        transforms = self._standardize_transforms(transform)
        use_map = self._resolve_use_map(use_map)
        if derived:
            overlap = set(derived) & set(self.measurement_model.measures)
            if overlap:
                raise ValueError(f"`derived` names can't be the same as measures: {overlap}")

        # (intervals for nonlinear measures are always monte-carlo)
        needs_samples = (not use_map) or (alpha is not None and self.measurement_model.is_nonlinear) or derived
        by_measure = self._map_group_chunks(
            lambda chunk: chunk._output_summaries(
                alpha=alpha, transforms=transforms, use_map=use_map, derived=derived, num_samples=num_samples
            ),
            num_samples=num_samples if needs_samples else 0,
        )

        from torchcast.utils import TimeSeriesDataset

        actuals = {}
        if isinstance(dataset, TimeSeriesDataset):
            for mgroup, tens in zip(dataset.measures, dataset.tensors):
                for m in mgroup:
                    if m not in self.measurement_model.measures:
                        continue
                    actuals[m] = tens[..., mgroup.index(m)]
                    if m in transforms:
                        actuals[m] = transforms[m].inverse(actuals[m])
            missing = set(self.measurement_model.measures) - set(dataset.all_measures)
            if missing:
                warn(
                    f"The following measures in your model are not present in your dataset, please double-check that "
                    f"the names you passed to the dataset match the `measures` you passed to the model:\n{missing}"
                )

        if derived:
            actuals.update(self._derived_actuals(derived, actuals))

        return self._summaries_to_dataframe(
            by_measure,
            actuals=actuals,
            dataset=dataset,
            group_colname=group_colname,
            time_colname=time_colname,
        )

    def _output_summaries(self,
                          alpha: Optional[float],
                          transforms: dict[str, 'Transform'],
                          use_map: bool,
                          derived: Optional[dict[str, Callable]],
                          num_samples: int) -> dict[str, tuple[torch.Tensor, ...]]:
        """
        For ``to_dataframe()``: for each measure and ``derived`` quantity, ``(num_groups, num_timesteps)`` tensors with
        the mean and (unless ``alpha`` is None) the lower and upper bounds.
        """
        measures = list(self.measurement_model.measures)

        # samples of the state's measured-mean, shared by the means, the intervals and `derived` -- only if needed
        # (intervals for nonlinear measures are always monte-carlo):
        mmean_samples = None
        needs_mc_intervals = (alpha is not None) and self.measurement_model.is_nonlinear
        if needs_mc_intervals or self._needs_mc_means(use_map, transforms):
            mmean_samples = self._get_output_samples(num_samples, generator=self._output_generator(state=True))

        means = self._get_means(use_map=use_map, transforms=transforms, mmean_samples=mmean_samples)
        by_measure = {m: (means[..., j],) for j, m in enumerate(measures)}
        if alpha is not None:
            intervals = self._get_intervals(alpha, transforms=transforms, mmean_samples=mmean_samples)
            by_measure = {m: (*v, *intervals[m]) for m, v in by_measure.items()}
        if derived:
            by_measure.update(self._derived_summaries(
                derived,
                alpha=alpha,
                transforms=transforms,
                mmean_samples=num_samples if mmean_samples is None else mmean_samples,
            ))
        return by_measure

    def _map_group_chunks(self, fun: Callable[['Predictions'], Any], num_samples: int) -> Any:
        """
        Applies ``fun`` to chunks of groups (``self[start:end]``), each small enough that its sample-based outputs'
        samples have at most about ``_MAX_OUTPUT_SAMPLE_ELEMENTS`` elements, and concatenates the results along the
        groups. Sample-based outputs share their noise across rows (with a fixed seed), so the results are the same as
        for all groups at once -- just with bounded memory. (The exception: ``BinomialPredictions`` draws binary
        observations independently per row, so ``derived`` quantities of binary measures depend on the chunking --
        each row's draws are still valid, and reproducible.)

        :param fun: Takes a ``Predictions`` object, returns a ``(num_groups, ...)`` tensor, or a dict of tuples of them.
        :param num_samples: The number of samples ``fun`` draws (0 if none, in which case there's no chunking).
        """
        dim = self.measurement_model_flat.extended_measure_mat.shape[1]
        group_size = max(1, _MAX_OUTPUT_SAMPLE_ELEMENTS // max(1, num_samples * self.num_timesteps * dim))
        if not num_samples or group_size >= self.num_groups:
            return fun(self)
        results = [fun(self[start:start + group_size]) for start in range(0, self.num_groups, group_size)]
        if isinstance(results[0], torch.Tensor):
            return torch.cat(results)
        return {k: tuple(torch.cat(parts) for parts in zip(*(r[k] for r in results))) for k in results[0]}

    def _get_intervals(self,
                       alpha: float,
                       transforms: dict[str, 'Transform'],
                       mmean_samples: Union[torch.Tensor, int, None]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        The ``alpha`` and ``1 - alpha`` quantiles of each measure's predictive distribution, back-transformed by
        ``transforms``:

        - Measures with a linear measured-mean (also in a model with other, nonlinear measures): closed form -- the
          gaussian's quantiles, or the mixture's, for measures with mixture components.
        - Nonlinear measures: quantiles of the (fixed) samples of the state's measured-mean, ``mmean_samples`` (the
          same as for the means), plus observation-noise -- and, for measures with mixture components, the regime's
          offset and extra variance, with the regime drawn per sample from its probability. Each measure is sampled on
          its own (its marginal is all an interval needs), with noise shared across rows, drawn with a fixed seed so
          repeated calls agree. For non-gaussian measures (e.g. binary), the interval is for the mean (e.g. the
          probability), so there's no observation-noise.

        :param mmean_samples: Only used if there are nonlinear measures: see ``_resolve_mmean_samples``. Pass
         already-computed samples to save compute (and to share them). Ignored with a warning if not needed.
        :return: For each measure, ``(num_groups, num_timesteps)`` lower and upper bounds.
        """
        all_measures = list(self.measurement_model.measures)
        batch_shape = self.state_means.shape[0:2]
        no_op_transform = _IdentityTransform()

        # divide measures into:
        # (1) nonlinear-measures (mixture or not),
        # (2) mixture-but-linear-measures,
        # (3) everything else
        nonlinear_measures = list(self._nonlinear_measures)
        _mix_measures = self.mixture.mixture_measures if self.mixture is not None else []
        mixture_measures = [m for m in _mix_measures if m not in nonlinear_measures]
        simple_measures = [m for m in all_measures if m not in nonlinear_measures and m not in mixture_measures]

        out = {}

        # simple measures:
        if simple_measures:
            measured_mean, system_cov = self._measured_moments_flat()
            multi = -stats.norm.ppf(alpha)
            for measure in simple_measures:
                midx = all_measures.index(measure)
                mean = measured_mean[:, midx].view(*batch_shape)
                std = system_cov[:, midx, midx].sqrt().view(*batch_shape)

                transform = transforms.get(measure, no_op_transform)
                lower = transform.inverse(mean - multi * std)
                upper = transform.inverse(mean + multi * std)
                out[measure] = (lower, upper)

        # mixture measures:
        for measure in mixture_measures:
            mixture = self.get_mixture(measure)
            transform = transforms.get(measure, no_op_transform)
            lower = transform.inverse(mixture.quantile(alpha))
            upper = transform.inverse(mixture.quantile(1 - alpha))
            out[measure] = (lower, upper)

        # for nonlinear measures: observation-noise and uniforms (for the regimes), shared across rows:
        if nonlinear_measures:
            mmean_samples = self._resolve_mmean_samples(mmean_samples)
            generator = self._output_generator(state=False)
            to = {'dtype': self.state_means.dtype, 'device': self.state_means.device}
            noise = torch.randn((mmean_samples.shape[0], len(all_measures)), generator=generator, **to)
            uniforms = torch.rand((mmean_samples.shape[0], len(all_measures)), generator=generator, **to)

            for measure in nonlinear_measures:
                midx = all_measures.index(measure)
                this_measure_samples = mmean_samples[..., midx]  # (num_samples, num_rows)
                # (a non-gaussian measure, e.g. binary, has no gaussian observation-noise -- its `measure_covs` entry is
                # zero -- and no mixture components, so its interval is for the mean, e.g. the probability)
                noise_var = self.measure_covs_flat[:, midx, midx]
                regimes = self._regimes(measure)
                if len(regimes) > 1:
                    # draw a regime for each sample and row, by "inverse-cdf sampling": e.g. with regime-probabilities
                    # (.7, .2, .1), the cdf's boundaries are (.7, .9), and a uniform u is in regime 0 if u < .7, regime
                    # 1 if .7 <= u < .9, else regime 2 -- i.e. the regime is the number of boundaries below u. (the
                    # last boundary, 1, is dropped: u is always below it.)
                    cdf = self._measure_regime_probs(measure).reshape(-1, len(regimes)).cumsum(-1)[:, :-1]
                    regime = (uniforms[:, midx, None, None] > cdf).sum(-1)  # (num_samples, num_rows)
                    # each draw then gets its regime's offset (added to the mean) and extra variance (added to the
                    # noise's):
                    _, offsets, extras = zip(*regimes)
                    offsets = torch.stack([torch.as_tensor(o, **to) for o in offsets])  # (num_regimes,)
                    extras = torch.stack([torch.as_tensor(e, **to) for e in extras])
                    this_measure_samples = this_measure_samples + offsets[regime]
                    noise_var = noise_var + extras[regime]
                this_measure_samples = this_measure_samples + noise_var.sqrt() * noise[:, midx, None]

                transform = transforms.get(measure, no_op_transform)
                this_measure_samples = transform.inverse(this_measure_samples)
                out[measure] = tuple(b.view(*batch_shape) for b in _quantile_interval(this_measure_samples, alpha))
        elif mmean_samples is not None:
            warn("`mmean_samples` argument to `_get_intervals()` ignored, no sampling needed.")
        return out

    def _derived_summaries(self,
                           derived: dict[str, Callable],
                           alpha: Optional[float],
                           transforms: dict[str, 'Transform'],
                           mmean_samples: Union[torch.Tensor, int, None] = None) -> dict[str, tuple[torch.Tensor, ...]]:
        """
        :param mmean_samples: Samples of the state's measured-mean (see ``_resolve_mmean_samples``): pass the ones the
         means and intervals use, to save compute (and to share them). The rest of each sample (the regime, the
         observations) is drawn given these, as in :func:`sample`.
        :return: For each derived quantity, ``(num_groups, num_timesteps)`` tensors with the mean and (unless ``alpha``
         is None) the lower and upper bounds.
        """
        measures = list(self.measurement_model.measures)
        mmean_samples = self._resolve_mmean_samples(mmean_samples)
        # summaries are per-row, so shared noise is fine (and uses less memory):
        num_samples = mmean_samples.shape[0]
        _, _, observations, _ = self._sample_given_state(
            mmean_samples,
            generator=self._output_generator(state=False),
            shared_noise=True,
            observation_measures=measures,
        )
        observations = observations.view(num_samples, self.num_groups, self.num_timesteps, len(measures))
        values = {m: observations[..., j] for j, m in enumerate(measures)}
        for m, t in transforms.items():
            if t.bias_adjust is not None:
                warn(f"`{type(t).__name__}(bias_adjust=...)` is ignored for `derived`.")
            values[m] = t.inverse(values[m])

        # measures whose samples have a known (exact) mean: control variates, for the derived quantities' means
        identity = _IdentityTransform()
        control_measures = [
            m for m in measures
            if m not in self._nonlinear_measures and _has_gaussian_noise(transforms.get(m, identity))
        ]
        controls = control_means = None
        if control_measures:
            cidx = [measures.index(m) for m in control_measures]
            controls = torch.stack([values[m] for m in control_measures], -1)
            control_means = self._get_means(use_map=True, transforms=transforms, adjust_bias=False)[..., cidx]

        out = {}
        expected_shape = (num_samples, self.num_groups, self.num_timesteps)
        for name, fun in derived.items():
            samples = fun(values)
            if tuple(samples.shape) != expected_shape:
                raise ValueError(
                    f"`derived['{name}']` returned shape {tuple(samples.shape)}, expected {expected_shape}."
                )
            if controls is None:
                out[name] = (samples.mean(0),)
            else:
                out[name] = (_control_variate_mean(samples, controls, control_means),)
            if alpha is not None:
                out[name] += _quantile_interval(samples, alpha)
        return out

    def _derived_actuals(self,
                         derived: dict[str, Callable],
                         actuals: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        The ``derived`` functions applied to the actuals (as a batch of one sample), if there are any. Measures that
        aren't in the dataset are nan, like missing values.
        """
        measures = list(self.measurement_model.measures)
        measure_actuals = {m: a for m, a in actuals.items() if m in measures}
        if not measure_actuals:
            return {}
        nans = torch.full_like(next(iter(measure_actuals.values())), float('nan'))
        measure_actuals = {m: measure_actuals.get(m, nans).unsqueeze(0) for m in measures}
        return {name: fun(measure_actuals).squeeze(0) for name, fun in derived.items()}

    def _summaries_to_dataframe(self,
                                by_measure: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                actuals: dict[str, torch.Tensor],
                                dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                                group_colname: str,
                                time_colname: str) -> pd.DataFrame:
        """
        :param by_measure: For each measure (or other named quantity), a tuple of ``(num_groups, num_timesteps)``
         tensors: mean, lower, upper -- or just mean.
        :param actuals: Optionally, ``(num_groups, num_timesteps)`` actuals for some measures/quantities (can have
         fewer timesteps).
        :return: A long-format dataframe with columns for group, time, mean, lower, upper, (actual,) and measure.
        """
        from torchcast.utils import TimeSeriesDataset

        out = []
        times = TimeSeriesDataset.get_dataset_times(
            dataset.start_offsets, num_timesteps=self.state_means.shape[1], dt_unit=dataset.dt_unit
        )
        for measure, (mean, *interval) in by_measure.items():
            _to_stack = {'mean': mean.unsqueeze(-1)}
            if interval:
                lower, upper = interval
                _to_stack.update(lower=lower.unsqueeze(-1), upper=upper.unsqueeze(-1))
            mactuals = actuals.get(measure, None)
            if mactuals is not None:
                _to_stack['actual'] = mactuals.unsqueeze(-1)
            out.append(
                TimeSeriesDataset.tensor_to_dataframe(
                    tensor=ragged_cat(list(_to_stack.values()), cat_dim=-1, ragged_dim=1),
                    times=times,
                    group_names=dataset.group_names,
                    group_colname=group_colname,
                    time_colname=time_colname,
                    measures=list(_to_stack)
                )
            )
            out[-1]['measure'] = measure
        out = pd.concat(out)

        return out

    @torch.inference_mode()
    def _to_components_dataframe(self,
                                 dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                                 group_colname: str,
                                 time_colname: str,
                                 conf: float,
                                 measured: bool,
                                 num_samples: int = _DEFAULT_NUM_SAMPLES) -> pd.DataFrame:
        alpha = (1 - conf) / 2
        batch_shape = self.state_means.shape[0:2]

        if self.measurement_model.is_nonlinear:
            # sample from the state distribution:
            chol = torch.linalg.cholesky(self.state_covs_flat)
            white_noise = torch.randn(
                (num_samples, *self.state_means_flat.shape),
                generator=self._output_generator(state=True),
                dtype=chol.dtype,
                device=chol.device,
            )
            state_mean_samples = self.state_means_flat + _matmul_per_row(chol, white_noise)

            # pass each sample to the `get_components` function, organize by process:
            samples_by_proc = {}
            for smean_samp in state_mean_samples:
                for pid, se, comp_mean in self.measurement_model_flat.get_components(smean_samp, measured=measured):
                    key = (pid, se)
                    if key not in samples_by_proc:
                        samples_by_proc[key] = []
                    samples_by_proc[key].append(comp_mean)
            # compute CIs:
            cis_by_proc = {}
            for key, samples in samples_by_proc.items():
                stacked = torch.stack(samples, dim=0).view(num_samples, *batch_shape)
                lower = torch.quantile(stacked, q=alpha, dim=0)
                upper = torch.quantile(stacked, q=1 - alpha, dim=0)
                cis_by_proc[key] = (lower, upper)
        else:
            cis_by_proc = {}
            for q in (alpha, 1 - alpha):
                multi = -stats.norm.ppf(q)
                offset = self.state_means_flat + multi * torch.sqrt(self.state_covs_flat.diagonal(dim1=-2, dim2=-1))
                for pid, se, comp_mean in self.measurement_model_flat.get_components(offset, measured=measured):
                    key = (pid, se)
                    if key not in cis_by_proc:
                        cis_by_proc[key] = []
                    cis_by_proc[key].append(comp_mean.view(*batch_shape))

        from torchcast.utils import TimeSeriesDataset

        # for each process, get mean/quantiles:
        times = TimeSeriesDataset.get_dataset_times(
            dataset.start_offsets, num_timesteps=batch_shape[-1], dt_unit=dataset.dt_unit
        )
        out = []
        for pid, se, comp_mean in self.measurement_model_flat.get_components(self.state_means_flat, measured=measured):
            mean = comp_mean.view(*batch_shape)
            lower, upper = cis_by_proc[(pid, se)]

            # to dataframe:
            _df = TimeSeriesDataset.tensor_to_dataframe(
                tensor=torch.stack([mean, lower, upper], -1),
                times=times,
                group_names=dataset.group_names,
                group_colname=group_colname,
                time_colname=time_colname,
                measures=['mean', 'lower', 'upper']
            )

            _df['process'] = pid
            _df['state_element'] = se
            _df['measure'] = self.measurement_model.processes[pid].measure
            out.append(_df)

        if isinstance(dataset, TimeSeriesDataset):
            for mgroup, tens in zip(dataset.measures, dataset.tensors):
                for m in mgroup:
                    if m not in self.measurement_model.measures:
                        continue
                    actuals = tens[:, :, [mgroup.index(m)]]
                    preds = self.means[:, 0:actuals.shape[1], [self.measurement_model.measures.index(m)]]
                    _df = TimeSeriesDataset.tensor_to_dataframe(
                        tensor=preds - actuals,
                        times=times,
                        group_names=dataset.group_names,
                        group_colname=group_colname,
                        time_colname=time_colname,
                        measures=['mean'],
                    )
                    _df['measure'] = m
                    _df['process'] = 'residuals'
                    _df['state_element'] = 'residuals'
                    out.append(_df)

        out = pd.concat(out)
        return out

    def _resolve_dataset(self,
                         dataset: Optional['TimeSeriesDataset']) -> Union['TimeSeriesDataset', 'DatasetMetadata']:
        if dataset is not None:
            return dataset
        dataset = self.dataset_metadata.copy()
        if dataset.group_names is None:
            warn(
                "This ``Predictions`` object doesn't have access to the group-names, consider calling "
                "``predictions.set_metadata()``."
            )
            dataset.group_names = [f"group_{i}" for i in range(self.num_groups)]
        if dataset.start_offsets.dtype.name.startswith('date') and not dataset.dt_unit:
            raise ValueError(
                "Unable to infer `dt_unit`, please call ``predictions.set_metadata(dt_unit=X)``, or pass `dataset` "
                "to ``predictions.to_dataframe()``"
            )
        if dataset.dt_unit and not dataset.start_offsets.dtype.name.startswith('date'):
            raise ValueError(
                "Expected `start_offsets` to be a datetime64 array, but got a different dtype. If you don't have "
                "dates, then set `dt_unit=None`."
            )
        return dataset

    # Sampling ---------------------------------------------------------------------------------------------------------

    def sample(self,
               num_samples: int,
               shared_noise: bool,
               observation_noise: bool = True,
               generator: Optional[torch.Generator] = None,
               regime_noise_scale: Optional[Mapping[str, float]] = None) -> 'PredictionSamples':
        """
        Draw samples from the predictive distribution: jointly across measures, for each group and timestep. These
        are *not* trajectories -- the model's timesteps are correlated (via the state), while samples for different
        timesteps are not -- so quantities that combine timesteps shouldn't be computed from them; see
        :func:`StateSpaceModel.simulate` for that.

        Each sample is a draw of the state (and, for models with mixture components, of the regime), giving a
        conditional distribution for the measures: ``means`` and ``covs``. These capture uncertainty about the state,
        but not the observation noise around it. With ``observation_noise=True``, each sample also includes a draw of
        the observations from that conditional distribution (for binary measures: binomial draws).

        :param num_samples: The number of samples.
        :param shared_noise: If True, the underlying random draws (the standard-normal noise for the state and for
         the gaussian observation-noise, and the uniforms for the regime) are the same for every group and timestep
         (binomial draws are still independent). Each row's distribution is still correct, so this is fine
         for per-row summaries (means, quantiles, functions of several measures), and it uses less memory. But
         samples are then correlated across rows. If False, samples are independent across rows, so they can be
         aggregated across groups (e.g. summed), since groups are independent in the model -- but still not across
         timesteps (see above).
        :param observation_noise: If True (the default), also sample the observations.
        :param generator: An optional :class:`torch.Generator`, for reproducibility.
        :param regime_noise_scale: For models with mixture components: scales the randomness of draws in a
         component's regime, ``{component_id: scale}`` (or ``'{measure}.{component_id}'``; others get 1). A draw's
         value for that measure is its regime's center -- the measured-mean of the state's mean (the MAP), plus the
         component's offset -- plus ``scale`` times its deviation from that center (from the state, the observation
         noise, and the component's extra variance). E.g. ``{'low': 0.}`` pins draws in a 'low' regime at its center
         (on the modeled scale), while the standard regime is sampled as usual: e.g. to check how much a component's
         variance contributes to back-transformed outputs (a sample-based analogue of a transform's per-regime
         ``bias_adjust=0``). The regimes are drawn as usual. The state is shared across measures, so this also weakens
         the draws' correlation with other measures. The scales are in ``PredictionSamples.noise_scales`` (and
         reflected in ``covs``).
        :return: A :class:`PredictionSamples`.
        """
        batch_shape = self.state_means.shape[0:2]
        measures = list(self.measurement_model.measures)
        noise_scale_table = self._regime_noise_scale_table(regime_noise_scale) if regime_noise_scale else None
        # sample the measured-mean (via the state), then the rest given it (the same generator draws the state's noise
        # first, then the regimes and observations):
        means = self._get_output_samples(num_samples, generator=generator, shared_noise=shared_noise)
        means, extra_vars, observations, combo_idx = self._sample_given_state(
            means,
            generator=generator,
            shared_noise=shared_noise,
            observation_measures=measures if observation_noise else None,
        )
        noise_scales = None
        if noise_scale_table is not None:
            # each draw's deviation from its regime's center, scaled:
            noise_scales = noise_scale_table[combo_idx]  # (num_samples, num_rows, num_measures)
            shift, _ = self.mixture.combo_offsets(measures, like=means)
            centers = self._measured_moments_flat()[0] + shift[combo_idx]
            means = centers + noise_scales * (means - centers)
            if observations is not None:
                observations = centers + noise_scales * (observations - centers)
            noise_scales = noise_scales.view(num_samples, *batch_shape, len(measures))
        if combo_idx is not None:
            combo_idx = combo_idx.view(num_samples, *batch_shape)
        if observations is not None:
            observations = observations.view(num_samples, *batch_shape, len(measures))

        means = means.view(num_samples, *batch_shape, len(measures))
        if extra_vars is not None:
            extra_vars = extra_vars.view(num_samples, *batch_shape, len(measures))

        return PredictionSamples(
            measures=measures,
            means=means,
            measure_covs=self.measure_covs,
            extra_vars=extra_vars,
            observations=observations,
            regimes=combo_idx,
            noise_scales=noise_scales,
        )

    def _regime_noise_scale_table(self, regime_noise_scale: Mapping[str, float]) -> torch.Tensor:
        """
        For :func:`sample`'s ``regime_noise_scale``: a ``(num_combos, num_measures)`` table with each measure's scale
        in each regime-combo.
        """
        if self.mixture is None:
            raise ValueError("`regime_noise_scale` is only for models with mixture components.")
        components = list(self.mixture.components)
        known = {c.id for c in components} | {c.qualified_id for c in components}
        unknown = set(regime_noise_scale) - known
        if unknown:
            raise ValueError(f"`regime_noise_scale` has unknown components: {unknown}")
        if any(v < 0 for v in regime_noise_scale.values()):
            raise ValueError("`regime_noise_scale` values must be non-negative.")
        measures = list(self.measurement_model.measures)
        table = torch.ones(
            (self.mixture.num_combos, len(measures)), dtype=self.state_means.dtype, device=self.state_means.device
        )
        for c, combo in enumerate(self.mixture.combos):
            for measure, component in zip(self.mixture.mixture_measures, combo):
                if component is None or measure not in measures:
                    continue
                scale = regime_noise_scale.get(component.qualified_id, regime_noise_scale.get(component.id))
                if scale is not None:
                    table[c, measures.index(measure)] = scale
        return table

    def _sample_given_state(self,
                            means: torch.Tensor,
                            generator: Optional[torch.Generator],
                            shared_noise: bool,
                            observation_measures: Optional[Sequence[str]]
                            ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
                                       Optional[torch.Tensor]]:
        """
        The rest of a sample, given samples of the measured-mean (one per sample of the state): the regime (for
        models with mixture components), then the observations.

        :param means: A ``(num_samples, num_rows, num_measures)`` tensor of measured-means.
        :param observation_measures: The measures whose observations to sample (the others' observations are their
         means); or None to skip the observations.
        :return: The means (with the regimes' offsets), the per-sample variances added to the diagonal of
         ``measure_covs`` (or None), the observations (or None), and the drawn regime-combos (or None).
        """
        # the observations' covariance given the sampled state is `measure_covs` plus a per-sample diagonal (e.g. a
        # binary measure's binomial variance; a regime's extra variance) -- tracked as just that diagonal, rather
        # than a (num_samples, num_rows, num_measures, num_measures) tensor:
        extra_vars = self._conditional_measure_vars(means)

        # sample the regime:
        combo_idx = None
        if self.mixture is not None:
            means, extra_vars, combo_idx = self._sample_regimes(
                means, extra_vars, generator=generator, shared_noise=shared_noise
            )

        observations = None
        if observation_measures is not None:
            observations = self._sample_observations(
                means,
                extra_vars,
                generator=generator,
                measures=observation_measures,
                shared_noise=shared_noise,
            )
        return means, extra_vars, observations, combo_idx

    def _conditional_measure_vars(self, means: torch.Tensor) -> Optional[torch.Tensor]:
        """
        The covariance of the observations given a sampled state is ``measure_covs`` plus a per-sample diagonal; this
        returns that diagonal. For gaussian measures there is none (their observation-noise doesn't depend on the
        state); subclasses whose observation-noise *does* depend on the state override this (e.g.
        ``BinomialPredictions``: ``p * (1 - p) / num_obs`` for binary measures, whose ``measure_covs`` entries are
        empty).

        :param means: A ``(num_samples, num_rows, num_measures)`` tensor of measured means, each from a sampled state.
        :return: A ``(num_samples, num_rows, num_measures)`` tensor of variances to add to the diagonal of
         ``measure_covs``, or None if there are none.
        """
        return None

    def _sample_regimes(self,
                        means: torch.Tensor,
                        extra_vars: Optional[torch.Tensor],
                        generator: Optional[torch.Generator],
                        shared_noise: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each sample, draw a regime-combo; for measures in a non-standard regime, add the component's offset to
        their mean and its variance to their (conditional) variance -- i.e. to ``extra_vars``, the per-sample diagonal
        added to ``measure_covs`` (see ``_conditional_measure_vars``). Also returns the ``(num_samples, num_rows)``
        indices of the drawn combos.

        :param shared_noise: If True, the draws use the same uniforms for every row (inverting each row's cdf of the
         regime-probabilities); otherwise they're independent.
        """
        rm = self.mixture
        num_samples = means.shape[0]
        shift, extra_var = rm.combo_offsets(self.measurement_model.measures, like=means)  # (num_combos, num_measures)
        regime_probs = self.regime_probs.reshape(-1, rm.num_combos)
        if shared_noise:
            uniforms = torch.rand(num_samples, generator=generator, dtype=means.dtype, device=means.device)
            cdf = regime_probs.cumsum(-1)[:, :-1].contiguous()  # (num_rows, num_combos - 1)
            # (the number of cdf-values below each uniform)
            combo_idx = torch.searchsorted(cdf, uniforms.expand(cdf.shape[0], -1).contiguous()).T  # (samples, rows)
        else:
            combo_idx = torch.multinomial(regime_probs, num_samples, replacement=True, generator=generator).T
        extra = extra_var[combo_idx]  # (num_samples, num_rows, num_measures)
        return means + shift[combo_idx], extra if extra_vars is None else extra_vars + extra, combo_idx

    def _sample_observations(self,
                             means: torch.Tensor,
                             extra_vars: Optional[torch.Tensor],
                             generator: Optional[torch.Generator],
                             measures: Optional[Sequence[str]] = None,
                             shared_noise: bool = False) -> torch.Tensor:
        """
        Samples gaussian observation-noise around ``means``, for the requested measures. All other measures returned
        as-is. The noise's covariance is ``measure_covs`` plus ``diag(extra_vars)`` (per-sample), drawn as the sum of
        two independent terms -- ``chol(measure_covs) @ z1 + sqrt(extra_vars) * z2`` -- so it needs neither a
        per-sample covariance nor a per-sample cholesky.

        :param shared_noise: If True, the same standard-normal noise for every row; otherwise independent.
        """
        all_measures = list(self.measurement_model.measures)
        if measures is None:
            measures = all_measures

        non_gaussian = set(measures).intersection(self._non_gaussian_measures)
        if non_gaussian:
            raise ValueError(
                f"{type(self).__name__}._sample_observations() only works for gaussian measures, but got {non_gaussian}"
            )

        idx = [i for i, m in enumerate(all_measures) if m in measures]
        if not idx:
            return means.clone()

        z_shape = (means.shape[0], len(idx)) if shared_noise else (*means.shape[:-1], len(idx))
        to = {'dtype': means.dtype, 'device': means.device}
        chol = torch.linalg.cholesky(self.measure_covs_flat[:, idx][:, :, idx])  # (num_rows, len(idx), len(idx))
        z = torch.randn(z_shape, generator=generator, **to)
        noise = _matmul_per_row(chol, z)
        if extra_vars is not None:
            z = torch.randn(z_shape, generator=generator, **to)
            noise = noise + extra_vars[..., idx].sqrt() * (z.unsqueeze(1) if shared_noise else z)
        if len(idx) == len(all_measures):
            return means + noise
        # (out-of-place; much faster than a clone plus in-place updates via advanced indexing)
        return means.index_add(-1, torch.as_tensor(idx, device=means.device), noise)

    # Mixture components -----------------------------------------------------------------------------------------------

    def get_mixture(self, measure: str) -> 'MixtureOfNormals':
        """
        For a measure with mixture components, the predictive distribution as a (univariate) mixture of normals: the
        standard regime, then each of the measure's components.

        For a measure with a nonlinear measurement (a measure-function or nonlinear processes), the standard regime
        isn't actually normal: its mean and variance are from the linearized (EKF) measurement-model, so are
        approximate (with a warning). The regime-probabilities are still exact. (``to_dataframe()`` and ``means`` use
        monte-carlo for these measures instead.)

        :param measure: The name of the measure.
        :return: A :class:`.MixtureOfNormals` whose tensors have shape ``(num_groups, num_timesteps, num_components)``.
        """
        if measure in self._nonlinear_measures:
            warn(
                f"'{measure}' has a nonlinear measurement, so the standard regime's mean and variance are from the "
                f"linearized measurement-model (approximate). The regime-probabilities are exact."
            )
        j = list(self.measurement_model.measures).index(measure)
        measured_mean, system_cov = self._measured_moments_flat()
        batch_shape = self.state_means.shape[0:2]
        return self._get_mixture(
            measure,
            standard_mean=measured_mean[:, j].view(*batch_shape),
            standard_var=system_cov[:, j, j].view(*batch_shape),
        )

    def _get_mixture(self,
                     measure: str,
                     standard_mean: torch.Tensor,
                     standard_var: torch.Tensor) -> 'MixtureOfNormals':
        """
        :param standard_mean: A ``(num_groups, num_timesteps)`` tensor with the mean of the standard regime.
        :param standard_var: A ``(num_groups, num_timesteps)`` tensor with the variance of the standard regime.
        """

        if self.mixture is None or measure not in self.mixture.mixture_measures:
            raise ValueError(f"'{measure}' has no mixture components.")
        labels, offsets, extravars = zip(*self._regimes(measure))
        return MixtureOfNormals(
            labels=[(label or 'standard') for label in labels],
            probs=self._measure_regime_probs(measure),
            means=torch.stack([standard_mean + offset for offset in offsets], -1),
            vars=torch.stack([standard_var + extra_var for extra_var in extravars], -1),
        )

    def _get_regime_combos(self) -> tuple[list[tuple[str, ...]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For models with mixture components: the predictive distribution as a mixture over regime-combos (see
        :class:`.MixtureModel`). Within each combo the prediction is multivariate normal, with the usual
        (state-dependent) mean and covariance -- plus, for measures in a mixture-component's regime, that component's
        offset (to the mean) and variance (on the diagonal).

        Note that for measures with a nonlinear measurement (a measure-function, e.g. binary measures, or nonlinear
        processes), the standard-regime moments are from the linearized measurement-model, so are approximate (with a
        warning, if any of these are mixture measures).

        :return: A tuple of (1) a label for each combo: a tuple with the regime of each mixture measure ('standard' or
         the component id), (2) a ``(num_groups, num_timesteps, num_combos)`` tensor of combo probabilities, (3) a
         ``(num_groups, num_timesteps, num_combos, num_measures)`` tensor of means, and (4) a ``(num_groups,
         num_timesteps, num_combos, num_measures, num_measures)`` tensor of covariances.
        """
        if self.mixture is None:
            raise RuntimeError("This model has no mixture components.")
        rm = self.mixture
        nonlinear = [m for m in rm.mixture_measures if m in self._nonlinear_measures]
        if nonlinear:
            warn(
                f"Mixture measures {nonlinear} have a nonlinear measurement, so the standard regime's moments are from "
                f"the linearized measurement-model (approximate). The regime-probabilities are exact."
            )
        measures = list(self.measurement_model.measures)
        measured_mean, system_cov = self._measured_moments_flat()
        shift, extra_var = rm.combo_offsets(measures, like=measured_mean)  # (num_combos, num_measures)
        means = [measured_mean + shift[c] for c in range(rm.num_combos)]
        covs = [system_cov + torch.diag(extra_var[c]) for c in range(rm.num_combos)]
        batch_shape = self.state_means.shape[0:2]
        labels = [tuple('standard' if c is None else c.id for c in combo) for combo in rm.combos]
        return (
            labels,
            self.regime_probs,
            torch.stack(means, 1).view(*batch_shape, rm.num_combos, len(measures)),
            torch.stack(covs, 1).view(*batch_shape, rm.num_combos, len(measures), len(measures)),
        )

    def _regimes(self, measure: str) -> list[tuple[Optional[str], Union[torch.Tensor, float], ...]]:
        """
        The regimes of a measure: ``(component_id, offset, extra_var)`` for the standard regime (``(None, 0, 0)``),
        then each of its mixture components (if any).
        """
        return [(None, 0., 0.)] + [(c.id, c.offset, c.extra_var) for c in self._mixture_components(measure)]

    def _measure_regime_probs(self, measure: str) -> torch.Tensor:
        """
        :return: A ``(num_groups, num_timesteps, num_regimes)`` tensor with the probability of each of ``measure``'s
         regimes (see ``_regimes``): all ones for a measure without mixture components.
        """
        if self.mixture is None or measure not in self.mixture.mixture_measures:
            return torch.ones(
                (*self.state_means.shape[0:2], 1),
                dtype=self.state_means.dtype,
                device=self.state_means.device
            )
        rm = self.mixture
        k = rm.mixture_measures.index(measure)
        probs = []
        for component in [None] + self._mixture_components(measure):
            in_regime = torch.as_tensor([combo[k] is component for combo in rm.combos], device=self.regime_probs.device)
            probs.append(self.regime_probs[..., in_regime].sum(-1))
        return torch.stack(probs, -1)

    def _mixture_components(self, measure: str) -> list:
        if self.mixture is None:
            return []
        return [c for c in self.mixture.components if c.measure == measure]

    # States at specific times, and indexing ---------------------------------------------------------------------------

    def get_state_at_times(self,
                           times: Union[np.ndarray, np.datetime64],
                           type_: str = 'update',
                           **kwargs) -> 'StateTuple':
        """
        For each group, get the state (tuple of (mean, cov)) for a timepoint. This is often useful since predictions
        are right-aligned and padded, so that the final prediction for each group is arbitrarily padded and does not
        correspond to a timepoint of interest -- e.g. for simulation (i.e., calling
        ``StateSpaceModel.simulate(initial_state=get_state_at_times(...))``).

        :param times: An array/sequence containing the time for each group; or a single datetime to apply to all groups.
          If the model/predictions are dateless (no dt_unit) then simply an array of indices
        :param type_: What type of state? Since this method is typically used for getting an `initial_state` for
         another call to :func:`StateSpaceModel.forward()`, this should generally be 'update' (the default); other
         option is 'prediction'.
        :return: A :class:`.StateTuple`, appropriate for forecasting by passing as `initial_state` for
         :func:`StateSpaceModel.forward()`. This behaves like a tuple of ``(state_means, state_covs)``; if the model has
         mixture components, it also carries the ``regime_probs`` at those times.
        """
        from .state import StateTuple

        preds = self.with_new_start_times(start_times=times, n_timesteps=1, **kwargs)
        single_time = (slice(None), 0)  # (like `.squeeze(1)`)

        if type_.startswith('pred'):
            return StateTuple(
                preds.state_means.squeeze(1),
                preds.state_covs.squeeze(1),
                regime_probs=_index_states(preds.regime_probs, single_time),
                scaling=_index_states(preds.scaling_states, single_time),
            )
        elif type_.startswith('update'):
            if preds.update_means is None:
                raise RuntimeError("No updates available; call the model with ``include_updates_in_output=True``.")
            return StateTuple(
                preds.update_means.squeeze(1),
                preds.update_covs.squeeze(1),
                regime_probs=_index_states(preds.update_regime_probs, single_time),
                scaling=_index_states(preds.update_scaling_states, single_time),
            )
        else:
            raise ValueError("Unrecognized `type_`, expected 'prediction' or 'update'.")

    def with_new_start_times(self,
                             start_times: Union[np.ndarray, np.datetime64],
                             n_timesteps: int,
                             **kwargs) -> 'Predictions':
        """
        :param start_times: An array/sequence containing the start time for each group; or a single datetime to apply
          to all groups. If the model/predictions are dateless (no dt_unit) then simply an array of indices.
        :param n_timesteps: Each group will be sliced to this many timesteps, so times is start and times + n_timesteps
          is end.
        :return: A new ``Predictions`` object, with the state and measurement tensors sliced to the given times.
        """
        start_indices = self._standardize_times(times=start_times, *kwargs)
        time_indices = np.arange(n_timesteps)[None, ...] + start_indices[:, None, ...]
        return self[np.arange(self.num_groups)[:, None, ...], time_indices]

    def _standardize_times(self,
                           times: Union[np.ndarray, np.datetime64],
                           start_offsets: Optional[np.ndarray] = None,
                           dt_unit: Optional[str] = None) -> np.ndarray:
        if start_offsets is not None:
            warn(
                "Passing `start_offsets` as an argument is deprecated, first call ``set_metadata()``",
                DeprecationWarning
            )
        if dt_unit is not None:
            warn(
                "Passing `dt_unit` as an argument is deprecated, first call ``set_metadata()``",
                DeprecationWarning
            )
        if self.dataset_metadata.start_offsets is not None:
            start_offsets = self.dataset_metadata.start_offsets
        if self.dataset_metadata.dt_unit is not None:
            dt_unit = self.dataset_metadata.dt_unit

        if not isinstance(times, (list, tuple, np.ndarray)):
            times = [times] * self.num_groups
        times = np.asanyarray(times, dtype='datetime64' if dt_unit else 'int')

        if start_offsets is None:
            if dt_unit is not None:
                raise ValueError("If `dt_unit` is specified, then `start_offsets` must also be specified.")
        else:
            if isinstance(dt_unit, str):
                dt_unit = np.timedelta64(1, dt_unit)
            times = times - start_offsets
            if dt_unit is not None:
                times = times // dt_unit  # todo: validate int?
            else:
                assert times.dtype.name.startswith('int')

        assert len(times.shape) == 1
        assert times.shape[0] == self.num_groups

        return times

    def __getitem__(self, item) -> 'Predictions':
        kwargs = self._getitem_helper(item)
        cls = type(self)
        return cls(**kwargs)

    def _getitem_helper(self, item: tuple) -> dict:
        if not isinstance(item, tuple):
            item = (item,)
        from .state import StateTuple

        kwargs = {
            'measurement_model': self.measurement_model.subset(*item),
            'states': StateTuple(
                self.state_means[item],
                self.state_covs[item],
                regime_probs=_index_states(self.regime_probs, item),
                scaling=_index_states(self.scaling_states, item),
            ),
            'measure_covs': self.measure_covs[item],
            # indexing only can impact group/time (ensured by measurementModel.subset), so no impact:
            'mc_white_noise': self.mc_white_noise,
            'mixture': self.mixture,
        }
        if self.update_means is not None:
            kwargs['updates'] = StateTuple(
                self.update_means[item],
                self.update_covs[item],
                regime_probs=_index_states(self.update_regime_probs, item),
                scaling=_index_states(self.update_scaling_states, item),
            )

        return kwargs

    # Shared internals (used by several of the above) ------------------------------------------------------------------

    def _resolve_use_map(self, use_map: Optional[bool]) -> bool:
        """
        :return: Whether the means of nonlinear measures use the MAP (the measured-mean of the state-mean) rather than
         monte-carlo. Always True for a linear model, whose (closed-form) means are the MAP -- so ``not use_map``
         implies there are nonlinear measures.
        """
        if not self.measurement_model.is_nonlinear:
            if use_map:
                warn("``use_map`` disregarded, no monte-carlo")
            return True
        if use_map is None:
            warn(
                "Will use MCMC for intervals but MAP for mean; to keep this behavior and suppress this warning, "
                "pass ``use_map=True``; to use MCMC for the mean as well pass ``use_map=False``."
            )
            return True
        return bool(use_map)

    def _standardize_transforms(self,
                                transform: Union['Transform', dict[str, 'Transform'], None]) -> dict[str, 'Transform']:
        from .transforms import Transform

        measures = list(self.measurement_model.measures)
        if transform is None:
            return {}
        if isinstance(transform, Transform):
            transforms = {m: transform for m in measures}
        elif isinstance(transform, dict):
            transforms = dict(transform)
            unknown = set(transforms) - set(measures)
            if unknown:
                raise ValueError(f"`transform` has measures not in the model: {unknown}")
        else:
            raise TypeError(f"Expected `transform` to be a `Transform` or a dict of them, got {type(transform)}")
        # note: measures with a nonlinear measurement (measure-fun or nonlinear process) are fine -- the model is
        # `T(y) = g(state) + noise`, and the monte-carlo path back-transforms samples of `g(state) + noise`. but
        # measures with a non-gaussian likelihood aren't:
        non_gaussian = [m for m in transforms if m in self._non_gaussian_measures]
        if non_gaussian:
            raise ValueError(
                f"`transform` is not supported for measures with a non-gaussian likelihood (e.g. binary measures): "
                f"{non_gaussian}. To transform only some measures, pass a dict of `{{measure: Transform}}`."
            )
        return transforms

    @property
    def _non_gaussian_measures(self) -> Sequence[str]:
        """
        Measures whose likelihood isn't gaussian (e.g. the binary measures of ``BinomialPredictions``).
        """
        return ()

    @property
    def _nonlinear_measures(self) -> set[str]:
        """
        Measures whose measured-mean is a nonlinear function of the state (via a measure-function or a nonlinear
        process), so their predictive distribution isn't gaussian even with a gaussian likelihood.
        """
        mm = self.measurement_model
        return set(mm.measure_funs) | {p.measure for p in mm.nonlinear_processes}

    def _measured_moments_flat(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: The (flattened) measured-mean and system-covariance of the (linearized) measurement model -- i.e.,
         for models with mixture components, of the standard regime.
        """
        measured_mean, measure_mat = self.measurement_model_flat(self.state_means_flat, time=0)
        system_cov = measure_mat @ self.state_covs_flat @ measure_mat.permute(0, 2, 1) + self.measure_covs_flat
        return measured_mean, system_cov

    def _get_measured_mean_samples(self,
                                   measurement_model: 'MeasurementModel',
                                   state_means: torch.Tensor,
                                   state_covs: torch.Tensor,
                                   white_noise: torch.Tensor,
                                   deviation_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Samples of the measured-mean, from samples of the state. Rather than sampling the full state, this samples a
        lower-dimensional gaussian: the measured-mean of the linear processes, concatenated with the state-elements of
        any nonlinear processes (see ``MeasurementModel.extended_measure_mat``). The nonlinear processes'
        contributions, then any measure-functions, are applied to those samples.

        :param measurement_model: A flattened measurement-model (possibly a subset of the measures, for nans), with
         ``num_rows`` groups.
        :param state_means: A ``(num_rows, state_rank)`` tensor.
        :param state_covs: A ``(num_rows, state_rank, state_rank)`` tensor.
        :param white_noise: Standard-normal noise for ``measurement_model``'s extended dimensions, either
         ``(num_samples, dim)`` -- the same draws for every row -- or ``(num_samples, num_rows, dim)``. E.g.
         ``_log_prob_white_noise()`` (fixed, for the log-prob), or ``_get_output_samples()``'s.
        :param deviation_scale: Optional scales for the sampled deviations from the mean, one per extended dimension
         (before any nonlinearity; see ``_extended_dim_scale``).
        :return: A ``(num_samples, num_rows, num_measures)`` tensor.
        """
        nmeasures = len(measurement_model.measures)

        extended_measure_mat = measurement_model.extended_measure_mat  # (raises if not flattened)
        num_rows, state_rank = measurement_model.num_groups, extended_measure_mat.shape[-1]
        if state_means.shape != (num_rows, state_rank) or state_covs.shape != (num_rows, state_rank, state_rank):
            raise ValueError(
                f"Expected `state_means` with shape {(num_rows, state_rank)} and `state_covs` with shape "
                f"{(num_rows, state_rank, state_rank)} (`measurement_model` has {num_rows} rows), got "
                f"{tuple(state_means.shape)} and {tuple(state_covs.shape)}."
            )
        partial_measured_mean = (extended_measure_mat @ state_means.unsqueeze(-1)).squeeze(-1)
        partial_measured_cov = extended_measure_mat @ state_covs @ extended_measure_mat.permute(0, 2, 1)

        # some dims might be all-zero (e.g. a measure with no linear processes), which means we can't take the
        # cholesky for those, so only factorize the nonzero block:
        nonzero = (extended_measure_mat != 0).any(0).any(1).cpu().nonzero(as_tuple=True)[0]
        m2d = torch.meshgrid(torch.arange(measurement_model.num_groups), nonzero, nonzero, indexing='ij')
        chol = torch.zeros_like(partial_measured_cov)
        chol[m2d] = torch.linalg.cholesky(partial_measured_cov[m2d])

        em_dim = extended_measure_mat.shape[1]
        if white_noise.ndim not in (2, 3) or white_noise.shape[-1] != em_dim or (
                white_noise.ndim == 3 and white_noise.shape[1] != state_means.shape[0]
        ):
            raise ValueError(
                f"Expected `white_noise` to have shape (num_samples, {em_dim}) or (num_samples, "
                f"{state_means.shape[0]}, {em_dim}), got {tuple(white_noise.shape)}."
            )
        deviations = _matmul_per_row(chol, white_noise)
        if deviation_scale is not None:
            deviations = deviations * deviation_scale
        sampled_pmmeans = partial_measured_mean.unsqueeze(0) + deviations

        # each of these samples is a draw of (1) the measured-mean of the linear processes, concatenated with (2) the
        # nonlinear processes' state-means. use (2) to add the nonlinear processes' contributions to (1):
        mmean_samples = sampled_pmmeans[..., 0:nmeasures]
        if measurement_model.nonlinear_processes:
            # (processes' measured-means aren't vectorized over samples, so loop)
            mmean_samples = torch.stack([
                measurement_model.apply_process_adjustments(
                    sampled_pmean[..., 0:nmeasures],
                    [
                        (proc, sampled_pmean[..., measurement_model.extended_mmat_slices[proc.id]])
                        for proc in measurement_model.nonlinear_processes
                    ],
                    time=0
                )
                for sampled_pmean in sampled_pmmeans.unbind(0)
            ])
        if measurement_model.measure_funs:
            # (elementwise, so no need to loop)
            mmean_samples = measurement_model.get_measure_wide_adjustments(mmean_samples)
        return mmean_samples

    def _output_generator(self, state: bool) -> torch.Generator:
        """
        A generator with a fixed seed, so that sample-based outputs are the same on repeated calls.

        :param state: True for draws of the state; False for draws given the state (regimes, observation-noise). These
         use different seeds, so they're independent (with one seed, e.g. the observation-noise would be the first of
         the state's noise draws).
        """
        return torch.Generator(device=self.state_means.device).manual_seed(_OUTPUT_SEED + (0 if state else 1))

    def _log_prob_white_noise(self, measurement_model: 'MeasurementModel') -> torch.Tensor:
        """
        The model's ``mc_white_noise`` for the monte-carlo log-prob (see ``_get_measured_mean_samples``):
        ``(num_samples, dim)``, fixed (so that the log-prob is a deterministic function of the parameters) and shared
        across rows.

        The noise has one column per extended dimension of the *full* model (each measure, then the nonlinear
        processes' state-elements). For a ``measurement_model`` that's a subset of the measures (for nans), the
        columns of the missing measures are dropped, so that each remaining dimension keeps its own column. This isn't
        needed for correctness (any standard-normal noise of the right size gives a valid monte-carlo estimate), but
        it keeps the draws consistent: each measure always uses the same column of the fixed draws, whichever other
        measures are missing in a row.
        """
        em_dim = self.measurement_model_flat.extended_measure_mat.shape[1]
        white_noise = self.mc_white_noise(num_dim=em_dim, dtype=self.state_means.dtype, device=self.state_means.device)
        missing_midx = {i for i, m in enumerate(self.measurement_model.measures) if m not in measurement_model.measures}
        return white_noise[..., [i for i in range(em_dim) if i not in missing_midx]]

    def _resolve_mmean_samples(self,
                               mmean_samples: Union[torch.Tensor, int, None],
                               deviation_scales: Optional[Sequence[float]] = None) -> torch.Tensor:
        """
        For methods that take ``mmean_samples`` to save compute: already-computed samples from
        ``_get_output_samples()`` are returned as-is; otherwise they're drawn here -- an int is the number of samples,
        None the default number (with the fixed output seed, so they're the same draws as any computed elsewhere).

        :param deviation_scales: See ``_get_output_samples``. If any aren't 1, the samples are drawn here (with the same
         draws, and the same number of samples, as any passed), since passed samples are unscaled.
        """
        if deviation_scales is not None and all(scale == 1 for scale in deviation_scales):
            deviation_scales = None
        if mmean_samples is None:
            mmean_samples = _DEFAULT_NUM_SAMPLES
        elif not isinstance(mmean_samples, numbers.Integral):
            if deviation_scales is None:
                return mmean_samples
            mmean_samples = mmean_samples.shape[0]
        return self._get_output_samples(
            int(mmean_samples),
            generator=self._output_generator(state=True),
            deviation_scales=deviation_scales,
        )

    def _get_output_samples(self,
                            num_samples: int,
                            generator: Optional[torch.Generator],
                            shared_noise: bool = True,
                            deviation_scales: Optional[Sequence[float]] = None) -> torch.Tensor:
        """
        Samples of the measured-mean, from ``num_samples`` draws of the state: for everything that produces samples
        for users -- :func:`sample`, and sample-based outputs (monte-carlo means and intervals, ``derived``). (The
        log-prob calls ``_get_measured_mean_samples`` directly, with the model's fixed ``mc_white_noise``.)

        :param generator: For the noise: e.g. ``_output_generator(state=True)`` (a fixed seed, so that repeated calls
         agree), or :func:`sample`'s (None: torch's global random state).
        :param shared_noise: If True (the default), the same noise for every row; otherwise independent.
        :param deviation_scales: One per measure: scales its (sampled) deviations from its mean, before any
         nonlinearity -- so each measure's samples are from the state with that part of its variance scaled by the
         square, from the same draws.
        :return: A ``(num_samples, num_rows, num_measures)`` tensor.
        """
        em_dim = self.measurement_model_flat.extended_measure_mat.shape[1]
        num_rows = self.state_means_flat.shape[0]
        white_noise = torch.randn(
            (num_samples, em_dim) if shared_noise else (num_samples, num_rows, em_dim),
            generator=generator,
            dtype=self.state_means.dtype,
            device=self.state_means.device,
        )
        return self._get_measured_mean_samples(
            measurement_model=self.measurement_model_flat,
            state_means=self.state_means_flat,
            state_covs=self.state_covs_flat,
            white_noise=white_noise,
            deviation_scale=None if deviation_scales is None else self._extended_dim_scale(deviation_scales),
        )

    def _extended_dim_scale(self, measure_scales: Sequence[float]) -> torch.Tensor:
        """
        Per-measure scales, for each of the extended dimensions of ``_get_measured_mean_samples``: each measure's
        own, then the state-elements of each nonlinear process, which get its measure's.
        """
        mm = self.measurement_model_flat
        measures = list(mm.measures)
        to = {'dtype': self.state_means.dtype, 'device': self.state_means.device}
        scale = torch.ones(mm.extended_measure_mat.shape[1], **to)
        scale[:len(measures)] = torch.as_tensor(measure_scales, **to)
        for proc in mm.nonlinear_processes:
            scale[mm.extended_mmat_slices[proc.id]] = measure_scales[measures.index(proc.measure)]
        return scale

    @property
    def state_means_flat(self):
        if self._state_means_flat is None:
            self._state_means_flat, self._state_covs_flat, self._mcovs_flat = self._flatten()
        return self._state_means_flat

    @property
    def state_covs_flat(self):
        if self._state_covs_flat is None:
            self._state_means_flat, self._state_covs_flat, self._mcovs_flat = self._flatten()
        return self._state_covs_flat

    @property
    def measure_covs_flat(self) -> torch.Tensor:
        if self._mcovs_flat is None:
            self._state_means_flat, self._state_covs_flat, self._mcovs_flat = self._flatten()
        return self._mcovs_flat

    def _flatten(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nmeasures = self.measure_covs.shape[-1]
        state_rank = self.state_means.shape[-1]
        state_means_flat = self.state_means.view(-1, state_rank)
        state_covs_flat = self.state_covs.view(-1, state_rank, state_rank)
        measure_covs_flat = self.measure_covs.view(-1, nmeasures, nmeasures)
        return state_means_flat, state_covs_flat, measure_covs_flat


@dataclass
class StateSpaceModelMetadata:
    measures: Sequence[str]
    all_state_elements: Sequence[Tuple[str, str]]


@dataclass
class PredictionSamples:
    """
    Samples from the predictive distribution, from :func:`Predictions.sample`.

    :param measures: The names of the measures (the last dimension of the tensors).
    :param means: A ``(num_samples, num_groups, num_timesteps, num_measures)`` tensor: for each sample (of the state,
     and regime if applicable), the mean of the measures.
    :param measure_covs: A ``(num_groups, num_timesteps, num_measures, num_measures)`` tensor: the observation-noise
     covariance that's the same for every sample.
    :param extra_vars: Optionally, a ``(num_samples, num_groups, num_timesteps, num_measures)`` tensor of per-sample
     variances added to the diagonal of ``measure_covs`` (e.g. a binary measure's binomial variance, or a regime's
     extra variance). See :attr:`covs`.
    :param observations: If ``observation_noise=True``, a ``(num_samples, num_groups, num_timesteps, num_measures)``
     tensor of sampled observations.
    :param regimes: For models with mixture components, a ``(num_samples, num_groups, num_timesteps)`` tensor with
     the index of each sample's regime-combo (into ``MixtureModel.combos``).
    :param noise_scales: With ``sample(regime_noise_scale=...)``, a ``(num_samples, num_groups, num_timesteps,
     num_measures)`` tensor of the scales applied to each draw's deviation from its regime's center.
    """
    measures: list
    means: torch.Tensor
    measure_covs: torch.Tensor
    extra_vars: Optional[torch.Tensor] = None
    observations: Optional[torch.Tensor] = None
    regimes: Optional[torch.Tensor] = None
    noise_scales: Optional[torch.Tensor] = None

    @property
    def covs(self) -> torch.Tensor:
        """
        A ``(num_samples, num_groups, num_timesteps, num_measures, num_measures)`` tensor: for each sample, the
        covariance of the observations around ``means`` (i.e. the observation-noise). Computed on access (a view, if
        there are no ``extra_vars``; otherwise a new, potentially large, tensor).
        """
        covs = self.measure_covs.expand(self.means.shape[0], *self.measure_covs.shape)
        if self.extra_vars is not None:
            covs = covs + torch.diag_embed(self.extra_vars)
        if self.noise_scales is not None:
            covs = covs * self.noise_scales.unsqueeze(-1) * self.noise_scales.unsqueeze(-2)
        return covs

    def __getitem__(self, measure: str) -> torch.Tensor:
        """
        The ``(num_samples, num_groups, num_timesteps)`` samples for a measure: of the observations if available,
        otherwise of the means.
        """
        values = self.means if self.observations is None else self.observations
        return values[..., self.measures.index(measure)]


@dataclass
class MixtureOfNormals:
    """
    A batch of univariate mixtures of normals: the predictive distribution of a single measure with mixture components.
    The last dimension of each tensor indexes the mixture's components; the first is the standard regime. (Each
    component's mean and variance are the standard regime's plus that component's offset and extra variance.)

    For example, to get the mean on the original scale of a log-transformed measure, back-transform each component and
    then mix: ``(mix.probs * torch.exp(mix.means + mix.vars / 2)).sum(-1)``.

    :param labels: A name for each component (``'standard'``, then the :class:`MixtureComponent` ids).
    :param probs: The probability of each component.
    :param means: The mean of each component.
    :param vars: The variance of each component.
    """
    labels: list
    probs: torch.Tensor
    means: torch.Tensor
    vars: torch.Tensor

    def mean(self) -> torch.Tensor:
        return (self.probs * self.means).sum(-1)

    def var(self) -> torch.Tensor:
        return (self.probs * (self.vars + self.means ** 2)).sum(-1) - self.mean() ** 2

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        z = (value.unsqueeze(-1) - self.means) / self.vars.sqrt()
        return (self.probs * torch.special.ndtr(z)).sum(-1)

    def quantile(self, q: float, num_iter: int = 60) -> torch.Tensor:
        """
        The ``q``-th quantile, found by bisection on the cdf.
        """
        assert 0 < q < 1
        sd = self.vars.sqrt()
        lower = (self.means - 10 * sd).min(-1).values
        upper = (self.means + 10 * sd).max(-1).values
        for _ in range(num_iter):
            mid = (lower + upper) / 2
            below = self.cdf(mid) < q
            lower = torch.where(below, mid, lower)
            upper = torch.where(below, upper, mid)
        return (lower + upper) / 2


@dataclass
class DatasetMetadata:
    group_names: Optional[Sequence[str]]
    start_offsets: Optional[np.ndarray]
    dt_unit: Optional[np.timedelta64]
    group_colname: str = 'group'
    time_colname: str = 'time'

    def update(self, **kwargs) -> 'DatasetMetadata':
        for f in fields(self):
            v = kwargs.pop(f.name, None)
            if v is not None:
                setattr(self, f.name, v)
        if kwargs:
            raise TypeError(f"Unrecognized kwargs: {list(kwargs)}")
        return self

    def copy(self) -> 'DatasetMetadata':
        return DatasetMetadata(
            group_names=self.group_names,
            start_offsets=self.start_offsets,
            dt_unit=self.dt_unit,
            group_colname=self.group_colname,
            time_colname=self.time_colname
        )


def _log_lik(resid: torch.Tensor, cov: torch.Tensor, monte_carlo: bool) -> torch.Tensor:
    """
    The gaussian log-likelihood of the residuals: exact; or, with ``monte_carlo``, averaged over the leading
    dimension of ``resid`` (one residual per monte-carlo sample of the measured-mean).
    """
    if not monte_carlo:
        return mvnorm_log_prob(resid, cov)
    # we don't want log_prob(x).mean(0), we want prob(x).mean(0).log(); numerically stable version:
    mc_log_probs = _mc_mvnorm_log_prob(resid, cov)
    return torch.logsumexp(mc_log_probs, dim=0) - log(mc_log_probs.shape[0])


def _mc_mvnorm_log_prob(resid: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    :param resid: A ``(num_samples, batch, d)`` tensor of residuals, one per monte-carlo sample.
    :param cov: A ``(batch, d, d)`` covariance, shared across samples.
    :return: A ``(num_samples, batch)`` tensor of log-densities of a zero-mean multivariate normal.
    """
    d = resid.shape[-1]
    if d == 1:
        return mvnorm_log_prob(resid, cov)
    # (solve for all samples at once, rather than broadcasting -- which would copy the cholesky for each sample)
    L = torch.linalg.cholesky(0.5 * (cov + cov.transpose(-2, -1)))
    z = torch.linalg.solve_triangular(L, resid.permute(1, 2, 0), upper=False)  # (batch, d, num_samples)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (d * math.log(2 * math.pi) + logdet.unsqueeze(0) + (z ** 2).sum(1).T)


def _matmul_per_row(mats: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    """
    ``mats[r] @ vecs[n, r]`` for each sample ``n`` and row ``r`` (or ``mats[r] @ vecs[n]``, if ``vecs`` is shared
    across rows) -- e.g. turning standard-normal noise into correlated noise with each row's cholesky factor.

    Like ``mats @ vecs.unsqueeze(-1)``, but without that broadcast's copy of ``mats`` for every sample (which, for
    many samples, can be much larger than the inputs or the output).

    :param mats: A ``(num_rows, d, d)`` tensor.
    :param vecs: A ``(num_samples, num_rows, d)`` tensor, or ``(num_samples, d)`` (the same for every row).
    :return: A ``(num_samples, num_rows, d)`` tensor.
    """
    # (in einsum's notation, r = row, n = sample, i/j = the matrix's row/column; j is summed over)
    if vecs.ndim == 2:
        return torch.einsum('rij,nj->nri', mats, vecs)
    return torch.einsum('rij,nrj->nri', mats, vecs)


def _has_gaussian_noise(transform: 'Transform') -> bool:
    """
    Whether ``transform.inverse_mean`` is the mean of back-transformed *gaussian* samples (e.g. not a
    :class:`.SmearingTransform`, which uses the empirical distribution of the residuals instead).
    """
    from .transforms import Transform

    return type(transform)._noise_nodes is Transform._noise_nodes


def _control_variate_mean(samples: torch.Tensor, controls: torch.Tensor, control_means: torch.Tensor) -> torch.Tensor:
    """
    The mean of ``samples``, with variance reduced by control variates: quantities sampled alongside them, whose exact
    means are known. Per row, regress the samples on the controls (across samples); the estimate is the samples' mean,
    minus the coefficients times how far the controls' sample-means are from their exact means. E.g. for the sum of two
    measures, with the measures as controls, the coefficients are 1 and the estimate is exact.

    :param samples: A ``(num_samples, *batch_shape)`` tensor.
    :param controls: A ``(num_samples, *batch_shape, num_controls)`` tensor.
    :param control_means: A ``(*batch_shape, num_controls)`` tensor of the controls' exact means.
    :return: A ``batch_shape`` tensor.
    """
    control_sample_means = controls.mean(0)
    xc = controls - control_sample_means
    # (standardized, so the regression is well-conditioned whatever the controls' scales; constant controls drop out)
    scale = xc.square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    xc = xc / scale
    fc = samples - samples.mean(0)
    cov_xx = torch.einsum('n...i,n...j->...ij', xc, xc)
    cov_xf = torch.einsum('n...i,n...->...i', xc, fc)
    coefs = (torch.linalg.pinv(cov_xx) @ cov_xf.unsqueeze(-1)).squeeze(-1) / scale
    return samples.mean(0) - (coefs * (control_sample_means - control_means)).sum(-1)


def _quantile_interval(x: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The ``alpha`` and ``1 - alpha`` quantiles along the first dimension (the samples), from a single sort.
    """
    lower, upper = torch.quantile(x, torch.tensor([alpha, 1 - alpha], dtype=x.dtype, device=x.device), dim=0)
    return lower, upper


def _unpack_states(states) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
    """
    :return: Stacked (num_groups, num_timesteps, ...) means, covs, regime-probs (or None), and adaptive-scaling states
     (a dict of tensors, or None).
    """
    from .state import StateTuple

    if isinstance(states, StateTuple):  # already stacked
        return states.mean, states.cov, states.regime_probs, states.scaling
    if len(states) and all(isinstance(s, StateTuple) for s in states):
        regime_probs = scaling = None
        if states[0].regime_probs is not None:
            regime_probs = torch.stack([s.regime_probs for s in states], 1)
        if states[0].scaling is not None:
            scaling = {k: torch.stack([s.scaling[k] for s in states], 1) for k in states[0].scaling}
        means, covs = (torch.stack(x, 1) for x in zip(*states))  # (a StateTuple iterates as (mean, cov))
        return means, covs, regime_probs, scaling
    means, covs = states
    return _maybe_stack(means, 1), _maybe_stack(covs, 1), None, None


def _index_states(x: Union[torch.Tensor, dict, None], item: tuple) -> Union[torch.Tensor, dict, None]:
    """Index a tensor, or each tensor in a dict (e.g. adaptive-scaling states); None stays None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: v[item] for k, v in x.items()}
    return x[item]


def _maybe_stack(x: Union[torch.Tensor, Sequence[torch.Tensor]], dim: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(x, dim=dim)


_warn_once = {}
