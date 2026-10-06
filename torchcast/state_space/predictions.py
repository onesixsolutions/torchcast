import math
from math import log

from dataclasses import dataclass, fields
from typing import Tuple, Union, Optional, Sequence, TYPE_CHECKING, Callable
from warnings import warn

import torch

import numpy as np
import pandas as pd

from scipy import stats
from torch.distributions import MultivariateNormal

from torchcast.internals.utils import get_nan_groups, class_or_instancemethod, ragged_cat, mvnorm_log_prob

if TYPE_CHECKING:
    from .mixture import MixtureModel, MixtureOfNormals
    from .transforms import Transform
    from .state import StateTuple
    from torchcast.utils import TimeSeriesDataset
    from torchcast.internals.batch_design import MeasurementModel
    from torchcast.internals.monte_carlo import FixedWhiteNoise

# seed for sample-based outputs (``to_dataframe()`` intervals for nonlinear measures, ``derived``, states), drawn at
# import: repeated calls within a process give the same results. (Separate from the model's ``mc_sampling``,
# which is fixed so that the monte-carlo log-prob -- and ``means`` -- are deterministic functions of the parameters.)
_OUTPUT_SEED = int(np.random.RandomState().randint(0, 2 ** 31 - 1))


class Predictions:
    """
    The output of the :class:`.StateSpaceModel` forward pass, containing the underlying state means and covariances, as
    well as methods such as ``log_prob()``, ``to_dataframe()``, and ``plot()``.
    """
    _means = None
    _covs = None


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
        self.state_means, self.state_covs, self.regime_probs = _unpack_states(states)
        self.measure_covs = _maybe_stack(measure_covs, 1)

        self.measurement_model = measurement_model
        self.measurement_model_flat = self.measurement_model.flattened()

        self.update_means = self.update_covs = self.update_regime_probs = None
        if updates is not None:
            self.update_means, self.update_covs, self.update_regime_probs = _unpack_states(updates)

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

    @torch.inference_mode()
    def to_dataframe(self,
                     dataset: Optional['TimeSeriesDataset'] = None,
                     type: str = 'predictions',
                     group_colname: Optional[str] = None,
                     time_colname: Optional[str] = None,
                     conf: Optional[float] = .95,
                     use_map: Optional[bool] = None,
                     transform: Union['Transform', dict[str, 'Transform'], None] = None,
                     derived: Optional[dict[str, Callable[[dict[str, torch.Tensor]], torch.Tensor]]] = None,
                     derived_num_samples: int = 1000) -> pd.DataFrame:
        """
        :param dataset: If not provided, will use the metadata set by ``set_metadata()``.
        :param type: What type of dataframe to return, either 'predictions',  'states', or 'observed_states'.
        :param group_colname: The name of the column to use for groups, defaults to the metadata's `group_colname`.
        :param time_colname: The name of the column to use for time, defaults to the metadata's `time_colname`.
        :param conf: The confidence level for the confidence intervals, defaults to 0.95.
        :param use_map: If the model requires MCMC, this controls whether the mean uses mcmc to marginalize over the
         state distribution (``use_map=False``) or whether the MAP is used to apply any non-linearities to the
         state-mean directly (``use_map=True``). The latter can sometimes exhibit better predictive performance on
         traditional supervised learning metrics.
        :param transform: For measures that were transformed before modeling (e.g. log), a :class:`.Transform` to
         map the predictions (and actuals) back to the original scale: either a single ``Transform`` for all measures,
         or a dictionary of ``{measure: Transform}``. The mean is back-transformed exactly (not just the mean on the
         modeled scale, back-transformed), as are the intervals. Only for ``type='predictions'``.
        :param derived: Quantities computed from multiple measures, e.g. ``{'total': lambda s: s['a'] + s['b']}``.
         Each function receives a dictionary of ``{measure: (num_samples, num_groups, num_timesteps)}`` samples from
         the (joint) predictive distribution, including observation noise and on the scale given by ``transform``, and
         should return a ``(num_samples, num_groups, num_timesteps)`` tensor. Each is added to the output with its
         key as the 'measure'; the mean and intervals are computed across samples (see :func:`sample`). If there are
         actuals, the function is applied to them too (so it should handle missing values, which are nan; measures
         that aren't in ``dataset`` are all-nan). Only for
         ``type='predictions'``.
        :param derived_num_samples: The number of samples used for ``derived``.
        """
        dataset = self._resolve_dataset(dataset)
        group_colname = group_colname or self.dataset_metadata.group_colname
        time_colname = time_colname or self.dataset_metadata.time_colname

        if conf is not None:
            assert conf >= .50

        type = type.casefold()
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
                derived_num_samples=derived_num_samples,
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
                measured=type in ('observed_states', 'components')
            )
        else:
            raise ValueError(f"Expected type to be 'predictions', 'states', or 'observed_states', got '{type}'.")

    def _add_derived(self,
                     derived: dict[str, Callable],
                     num_samples: int,
                     alpha: float,
                     transforms: dict[str, 'Transform'],
                     by_measure: dict,
                     actuals: dict[str, torch.Tensor]) -> None:
        """
        Adds summaries of derived quantities to ``by_measure``, and (if there are any actuals, i.e. a dataset) the
        derived actuals to ``actuals``. Modifies both in place.
        """
        measures = list(self.measurement_model.measures)
        overlap = set(derived) & set(measures)
        if overlap:
            raise ValueError(f"`derived` names can't be the same as measures: {overlap}")
        samples = self.sample(num_samples, observation_noise=True, generator=self._output_generator())
        values = {m: samples[m] for m in measures}
        for m, t in transforms.items():
            _warn_bias_adjust_ignored(t)
            values[m] = t.inverse(values[m])

        # the functions are also applied to the actuals (as a batch of one sample), if there's a dataset. measures that
        # aren't in the dataset are nan, like missing values:
        measure_actuals = {m: a for m, a in actuals.items() if m in measures}
        if measure_actuals:
            nans = torch.full_like(next(iter(measure_actuals.values())), float('nan'))
            measure_actuals = {m: measure_actuals.get(m, nans).unsqueeze(0) for m in measures}

        expected_shape = (num_samples, self.num_groups, self.num_timesteps)
        for name, fun in derived.items():
            out = fun(values)
            if tuple(out.shape) != expected_shape:
                raise ValueError(f"`derived['{name}']` returned shape {tuple(out.shape)}, expected {expected_shape}.")
            by_measure[name] = (out.mean(0), _quantile(out, alpha), _quantile(out, 1 - alpha))
            if measure_actuals:
                actuals[name] = fun(measure_actuals).squeeze(0)

    def sample(self,
               num_samples: int,
               observation_noise: bool = True,
               generator: Optional[torch.Generator] = None) -> 'PredictionSamples':
        """
        Draw samples from the predictive distribution: jointly across measures, independently for each group and
        timestep. (Note that these are *not* trajectories -- samples for different timesteps are independent -- so
        quantities that combine timesteps shouldn't be computed from them; see :func:`StateSpaceModel.simulate` for
        that.)

        Each sample is a draw of the state (and, for models with mixture components, of the regime), giving a
        conditional distribution for the measures: ``means`` and ``covs``. These capture uncertainty about the state,
        but not the observation noise around it. With ``observation_noise=True``, each sample also includes a draw of
        the observations from that conditional distribution (for binary measures: binomial draws).

        :param num_samples: The number of samples.
        :param observation_noise: If True (the default), also sample the observations.
        :param generator: An optional :class:`torch.Generator`, for reproducibility.
        :return: A :class:`PredictionSamples`.
        """
        batch_shape = self.state_means.shape[0:2]
        measures = list(self.measurement_model.measures)
        num_rows = self.state_means_flat.shape[0]
        to = {'dtype': self.state_means.dtype, 'device': self.state_means.device}

        # sample the measured-mean (via the state). independent noise for each row (unlike the fixed noise used for
        # the monte-carlo log-prob, which is shared across rows):
        em_dim = self.measurement_model_flat.extended_measure_mat.shape[1]
        white_noise = torch.randn((num_samples, num_rows, em_dim), generator=generator, **to)
        means = self._get_measured_mean_samples(
            measurement_model=self.measurement_model_flat,
            state_means=self.state_means_flat,
            state_covs=self.state_covs_flat,
            white_noise=white_noise,
        )
        covs = self._conditional_measure_covs(means)

        # sample the regime:
        if self.mixture is not None:
            means, covs = self._sample_regimes(means, covs, generator=generator)

        observations = None
        if observation_noise:
            observations = self._sample_observations(means, covs, generator=generator)
            observations = observations.view(num_samples, *batch_shape, len(measures))
        return PredictionSamples(
            measures=measures,
            means=means.view(num_samples, *batch_shape, len(measures)),
            covs=covs.view(num_samples, *batch_shape, len(measures), len(measures)),
            observations=observations,
        )

    def _conditional_measure_covs(self, means: torch.Tensor) -> torch.Tensor:
        """
        The covariance of the observations given a sampled state. For gaussian measures this is just ``measure_covs``
        (it doesn't depend on the state); subclasses whose observation-noise *does* depend on the state override this
        (e.g. ``BinomialPredictions``: ``p * (1 - p) / num_obs`` for binary measures, whose ``measure_covs`` entries
        are empty).

        :param means: A ``(num_samples, num_rows, num_measures)`` tensor of measured means, each from a sampled state.
        :return: A ``(num_samples, num_rows, num_measures, num_measures)`` tensor with the covariance of the measures
         given the state.
        """
        return self.measure_covs_flat.expand(means.shape[0], -1, -1, -1)

    def _sample_regimes(self,
                        means: torch.Tensor,
                        covs: torch.Tensor,
                        generator: Optional[torch.Generator]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each sample, draw a regime-combo; for measures in a non-standard regime, replace their mean/variance with
        the component's (and zero their covariance with other measures).
        """
        rm = self.mixture
        num_samples = means.shape[0]
        measures = list(self.measurement_model.measures)
        zero = torch.zeros((), dtype=means.dtype, device=means.device)
        comp_means, comp_vars, weird = [], [], []
        for combo in rm.combos:
            row_mean, row_var, row_weird = [zero] * len(measures), [zero] * len(measures), [False] * len(measures)
            for measure, component in zip(rm.mixture_measures, combo):
                if component is not None:
                    j = measures.index(measure)
                    row_mean[j], row_var[j], row_weird[j] = component.mean, component.var, True
            comp_means.append(torch.stack(row_mean))
            comp_vars.append(torch.stack(row_var))
            weird.append(row_weird)
        comp_means, comp_vars = torch.stack(comp_means), torch.stack(comp_vars)  # (num_combos, num_measures)
        weird = torch.as_tensor(weird, device=means.device)

        regime_probs = self.regime_probs.reshape(-1, rm.num_combos)
        combo_idx = torch.multinomial(regime_probs, num_samples, replacement=True, generator=generator).T
        is_weird = weird[combo_idx]  # (num_samples, num_rows, num_measures)
        means = torch.where(is_weird, comp_means[combo_idx], means)
        keep = (~is_weird).to(covs.dtype)
        covs = covs * keep.unsqueeze(-1) * keep.unsqueeze(-2) + torch.diag_embed(comp_vars[combo_idx] * is_weird)
        return means, covs

    def _sample_observations(self,
                             means: torch.Tensor,
                             covs: torch.Tensor,
                             generator: Optional[torch.Generator]) -> torch.Tensor:
        """
        Samples gaussian observation-noise around ``means``, for the gaussian measures. Non-gaussian measures (see
        ``_non_gaussian_measures``) are left at their means, for subclasses to sample.
        """
        measures = list(self.measurement_model.measures)
        idx = [i for i, m in enumerate(measures) if m not in self._non_gaussian_measures]
        observations = means.clone()
        if not idx:
            return observations
        z = torch.randn((*means.shape[:-1], len(idx)), generator=generator, dtype=means.dtype, device=means.device)
        if covs.stride(0) == 0:
            # the same covariance for every sample (an expanded view): only decompose it once
            chol = torch.linalg.cholesky(covs[0][:, idx][:, :, idx])
            observations[..., idx] += torch.einsum('rij,nrj->nri', chol, z)
        else:
            chol = torch.linalg.cholesky(covs[..., idx, :][..., idx])
            observations[..., idx] += torch.einsum('nrij,nrj->nri', chol, z)
        return observations

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
            by_name[name] = (x.mean(0), _quantile(x, alpha), _quantile(x, 1 - alpha))
        return self._summaries_to_dataframe(
            by_name,
            actuals=actuals or {},
            dataset=self._resolve_dataset(dataset),
            group_colname=group_colname or self.dataset_metadata.group_colname,
            time_colname=time_colname or self.dataset_metadata.time_colname,
        )

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

    @torch.inference_mode()
    def _to_components_dataframe(self,
                                 dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                                 group_colname: str,
                                 time_colname: str,
                                 conf: float,
                                 measured: bool) -> pd.DataFrame:
        alpha = (1 - conf) / 2
        batch_shape = self.state_means.shape[0:2]

        if self.mc_white_noise is not None:
            # sample from the state distribution (einsum rather than a broadcasting matmul, which would materialize a
            # (samples, rows, dim, dim) copy of the cholesky):
            chol = torch.linalg.cholesky(self.state_covs_flat)
            white_noise = torch.randn(
                (self.mc_white_noise.num_samples, *self.state_means_flat.shape),
                generator=self._output_generator(),
                dtype=chol.dtype,
                device=chol.device,
            )
            state_mean_samples = self.state_means_flat + torch.einsum('rij,nrj->nri', chol, white_noise)

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
                stacked = torch.stack(samples, dim=0).view(self.mc_white_noise.num_samples, *batch_shape)
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

    def _output_generator(self) -> torch.Generator:
        """
        A generator with a fixed seed, so that sample-based outputs are the same on repeated calls.
        """
        return torch.Generator(device=self.state_means.device).manual_seed(_OUTPUT_SEED)

    def _get_mc_pred_intervals(self,
                               alpha: float,
                               use_map: bool,
                               transforms: Optional[dict[str, 'Transform']] = None
                               ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        For measures with a nonlinear measured-mean (see ``_get_pred_intervals`` for the rest): the intervals are
        quantiles of samples from :func:`sample`; the mean is from :func:`_get_means_nonlinear`.
        """
        transforms = transforms or {}
        measures = list(self.measurement_model.measures)
        nonlinear = [m for m in measures if m in self._nonlinear_measures]
        if not nonlinear:
            return {}
        if not use_map and self.mc_white_noise.num_samples < 1000:
            warn("Consider at least ``my_model.mc_sampling = 1000`` if use_map=False")

        means = self._get_means_nonlinear(use_map=use_map, transforms=transforms)
        samples = self.sample(
            self.mc_white_noise.num_samples, observation_noise=True, generator=self._output_generator()
        )
        by_measure = {}
        for measure in nonlinear:
            j = measures.index(measure)
            # (for non-gaussian measures, e.g. binary, the interval is for the mean -- e.g. the probability -- rather
            # than for the observations)
            x = samples.means[..., j] if measure in self._non_gaussian_measures else samples.observations[..., j]
            if measure in transforms:
                _warn_bias_adjust_ignored(transforms[measure])
                x = transforms[measure].inverse(x)
            by_measure[measure] = (means[..., j], _quantile(x, alpha), _quantile(x, 1 - alpha))
        return by_measure

    def _get_means_nonlinear(self,
                             use_map: bool = False,
                             transforms: Optional[dict[str, 'Transform']] = None) -> torch.Tensor:
        """
        The predicted means for a model with nonlinear measures. Measures with a linear measured-mean have a closed
        form. For the rest, the (standard regime's) mean is a monte-carlo mean over samples of the state, from the
        fixed ``mc_white_noise`` (so it's deterministic) -- or, with ``use_map``, the measured-mean of the state-mean.
        For mixture measures, this is mixed with the components' means using the (known) regime-probabilities.

        :param transforms: Back-transforms for measures with a nonlinear measured-mean. (``use_map`` doesn't apply to
         these: the back-transformed mean of the MAP isn't meaningful.)
        :return: A ``(num_groups, num_timesteps, num_measures)`` tensor.
        """
        batch_shape = self.state_means.shape[0:2]
        measures = list(self.measurement_model.measures)
        nonlinear = self._nonlinear_measures
        transforms = {m: t for m, t in (transforms or {}).items() if m in nonlinear}

        # exact for linear measures; the MAP for nonlinear ones:
        standard, _ = self._measured_moments_flat()
        standard = standard.clone()
        mc_idx = [j for j, m in enumerate(measures) if m in nonlinear and (not use_map or m in transforms)]
        if mc_idx:
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=self.measurement_model_flat,
                state_means=self.state_means_flat,
                state_covs=self.state_covs_flat,
            )
            for j in mc_idx:
                if measures[j] in transforms:
                    # E[inverse(g(state) + noise)]: monte-carlo over the state, quadrature over the noise
                    t = transforms[measures[j]]
                    var = self.measure_covs_flat[:, j, j]
                    standard[:, j] = torch.stack(
                        [t.expected_inverse(x, var) for x in mmean_samples[..., j].unbind(0)]
                    ).mean(0)
                else:
                    standard[:, j] = mmean_samples[..., j].mean(0)
        standard = standard.view(*batch_shape, -1)

        if self.mixture is None:
            return standard
        out = standard.clone()
        for measure in self.mixture.mixture_measures:
            j = measures.index(measure)
            mixture = self._get_mixture(measure, standard[..., j], torch.zeros_like(standard[..., j]))
            component_means = mixture.means[..., 1:]
            if measure in transforms:
                # (each regime with its own transform -- see `Transform.for_regime` -- and the full back-transformed
                # mean, like the standard regime's: bias_adjust is ignored for MC)
                component_means = []
                for k, label in enumerate(mixture.labels[1:], start=1):
                    t = transforms[measure].for_regime(label)
                    _warn_bias_adjust_ignored(t)
                    component_means.append(t.expected_inverse(mixture.means[..., k], mixture.vars[..., k]))
                component_means = torch.stack(component_means, -1)
            out[..., j] = mixture.probs[..., 0] * standard[..., j] + (mixture.probs[..., 1:] * component_means).sum(-1)
        return out

    def _get_pred_intervals(self,
                            alpha: float,
                            transforms: Optional[dict[str, 'Transform']] = None
                            ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Closed-form means and intervals, for measures with a linear measured-mean (whose predictive distribution is
        gaussian, or a mixture of gaussians -- see ``_get_mixture_intervals``) -- including in a model with other,
        nonlinear measures (see ``_get_mc_pred_intervals``).
        """
        transforms = transforms or {}
        measured_mean, system_cov = self._measured_moments_flat()

        batch_shape = self.state_means.shape[0:2]
        multi = -stats.norm.ppf(alpha)

        by_measure = {}
        for i, measure in enumerate(self.measurement_model.measures):
            if measure in self._nonlinear_measures:
                continue
            mean = measured_mean[..., i]
            var = system_cov[..., i, i]
            lower = mean - multi * torch.sqrt(var)
            upper = mean + multi * torch.sqrt(var)
            if measure in transforms:
                t = transforms[measure]
                mean, lower, upper = t.inverse_mean(mean, var), t.inverse(lower), t.inverse(upper)
            by_measure[measure] = (
                mean.view(*batch_shape),
                lower.view(*batch_shape),
                upper.view(*batch_shape)
            )
        by_measure.update(self._get_mixture_intervals(alpha, transforms))
        return by_measure

    def _get_mixture_intervals(self,
                               alpha: float,
                               transforms: Optional[dict[str, 'Transform']] = None
                               ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.mixture is None:
            return {}
        transforms = transforms or {}
        out = {}
        for measure in self.mixture.mixture_measures:
            if measure in self._nonlinear_measures:
                continue  # see _get_mc_pred_intervals
            mixture = self.get_mixture(measure)
            lower, upper = mixture.quantile(alpha), mixture.quantile(1 - alpha)
            if measure in transforms:
                t = transforms[measure]
                # back-transform each regime (with its own transform, see `Transform.for_regime`), then mix:
                regime_means = [t.inverse_mean(mixture.means[..., 0], mixture.vars[..., 0])] + [
                    t.for_regime(label).inverse_mean(mixture.means[..., k], mixture.vars[..., k])
                    for k, label in enumerate(mixture.labels[1:], start=1)
                ]
                mean = (mixture.probs * torch.stack(regime_means, -1)).sum(-1)
                out[measure] = (mean, t.inverse(lower), t.inverse(upper))
            else:
                out[measure] = (mixture.mean(), lower, upper)
        return out

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
        from .transforms import RegimeTransform
        for measure, t in transforms.items():
            if isinstance(t, RegimeTransform):
                ids = {c.id for c in self.mixture.components if c.measure == measure} if self.mixture else set()
                unknown = set(t.components) - ids
                if unknown:
                    raise ValueError(f"`RegimeTransform` for '{measure}' has components that aren't its: {unknown}")
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

    def _get_regime_combos(self) -> tuple[list[tuple[str, ...]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For models with mixture components: the predictive distribution as a mixture over regime-combos (see
        :class:`.MixtureModel`). Within each combo the prediction is multivariate normal: measures in the standard
        regime have the usual (state-dependent) mean and covariance; measures in a mixture-component's regime have
        that component's mean and variance, and are uncorrelated with the other measures.

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
        means, covs = [], []
        for combo in rm.combos:
            mean = measured_mean.clone()
            cov = system_cov.clone()
            for measure, component in zip(rm.mixture_measures, combo):
                if component is None:
                    continue
                j = measures.index(measure)
                mean[:, j] = component.mean
                cov[:, j, :] = 0
                cov[:, :, j] = 0
                cov[:, j, j] = component.var
            means.append(mean)
            covs.append(cov)
        batch_shape = self.state_means.shape[0:2]
        labels = [tuple('standard' if c is None else c.id for c in combo) for combo in rm.combos]
        return (
            labels,
            self.regime_probs,
            torch.stack(means, 1).view(*batch_shape, rm.num_combos, len(measures)),
            torch.stack(covs, 1).view(*batch_shape, rm.num_combos, len(measures), len(measures)),
        )

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
        from .mixture import MixtureOfNormals

        if self.mixture is None or measure not in self.mixture.mixture_measures:
            raise ValueError(f"'{measure}' has no mixture components.")
        rm = self.mixture
        k = rm.mixture_measures.index(measure)
        components = [None] + [c for c in rm.components if c.measure == measure]
        batch_shape = self.state_means.shape[0:2]

        probs, means, vars_ = [], [], []
        for component in components:
            in_regime = torch.as_tensor(
                [combo[k] is component for combo in rm.combos], device=self.regime_probs.device
            )
            probs.append(self.regime_probs[..., in_regime].sum(-1))
            means.append(standard_mean if component is None else component.mean.expand(*batch_shape))
            vars_.append(standard_var if component is None else component.var.expand(*batch_shape))
        return MixtureOfNormals(
            labels=['standard' if c is None else c.id for c in components],
            probs=torch.stack(probs, -1),
            means=torch.stack(means, -1),
            vars=torch.stack(vars_, -1),
        )

    @torch.inference_mode()
    def _to_dataframe(self,
                      dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                      group_colname: str,
                      time_colname: str,
                      conf: float,
                      use_map: bool,
                      transform: Union['Transform', dict[str, 'Transform'], None] = None,
                      derived: Optional[dict[str, Callable]] = None,
                      derived_num_samples: int = 1000) -> pd.DataFrame:

        alpha = (1 - conf) / 2
        transforms = self._standardize_transforms(transform)

        # closed-form for measures with a linear measured-mean; monte-carlo for the rest:
        by_measure = self._get_pred_intervals(alpha, transforms=transforms)
        if self.mc_white_noise is not None:
            if use_map is None:
                warn(
                    "Will use MCMC for intervals but MAP for mean; to keep this behavior and suppress this warning, "
                    "pass ``use_map=True``; to use MCMC for the mean as well pass ``use_map=False``."
                )
                use_map = True
            by_measure.update(self._get_mc_pred_intervals(alpha, use_map=use_map, transforms=transforms))
            by_measure = {m: by_measure[m] for m in self.measurement_model.measures}  # (keep the measures' order)
        elif use_map:
            warn("``use_map`` disregarded, no monte-carlo")

        from torchcast.utils import TimeSeriesDataset

        actuals = {}
        if isinstance(dataset, TimeSeriesDataset):
            for mgroup, tens in zip(dataset.measures, dataset.tensors):
                for m in mgroup:
                    if m not in by_measure:
                        continue
                    actuals[m] = tens[..., mgroup.index(m)]
                    if m in transforms:
                        actuals[m] = transforms[m].inverse(actuals[m])
            missing = set(by_measure) - set(dataset.all_measures)
            if missing:
                warn(
                    f"The following measures in your model are not present in your dataset, please double-check that "
                    f"the names you passed to the dataset match the `measures` you passed to the model:\n{missing}"
                )
        if derived:
            self._add_derived(derived, num_samples=derived_num_samples, alpha=alpha, transforms=transforms,
                              by_measure=by_measure, actuals=actuals)
        return self._summaries_to_dataframe(
            by_measure,
            actuals=actuals,
            dataset=dataset,
            group_colname=group_colname,
            time_colname=time_colname,
        )

    def _summaries_to_dataframe(self,
                                by_measure: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                actuals: dict[str, torch.Tensor],
                                dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                                group_colname: str,
                                time_colname: str) -> pd.DataFrame:
        """
        :param by_measure: For each measure (or other named quantity), a tuple of ``(num_groups, num_timesteps)``
         tensors: mean, lower, upper.
        :param actuals: Optionally, ``(num_groups, num_timesteps)`` actuals for some measures/quantities (can have
         fewer timesteps).
        :return: A long-format dataframe with columns for group, time, mean, lower, upper, (actual,) and measure.
        """
        from torchcast.utils import TimeSeriesDataset

        out = []
        times = TimeSeriesDataset.get_dataset_times(
            dataset.start_offsets, num_timesteps=self.state_means.shape[1], dt_unit=dataset.dt_unit
        )
        for measure, (mean, lower, upper) in by_measure.items():
            _to_stack = {'mean': mean.unsqueeze(-1), 'lower': lower.unsqueeze(-1), 'upper': upper.unsqueeze(-1)}
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

    def _observe(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_shape = self.state_means.shape[0:2]

        if self.measurement_model.is_nonlinear:
            # monte-carlo for the means of nonlinear measures; there's no closed form cov
            return self._get_means_nonlinear(), None
        elif self.mixture is not None:
            # the exact mean and covariance of the mixture over regime-combos:
            _, probs, means, covs = self._get_regime_combos()
            mean = (probs.unsqueeze(-1) * means).sum(-2)
            second_moment = (probs[..., None, None] * (covs + means.unsqueeze(-1) * means.unsqueeze(-2))).sum(-3)
            return mean, second_moment - mean.unsqueeze(-1) * mean.unsqueeze(-2)
        else:
            measured_mean, system_cov = self._measured_moments_flat()
            return measured_mean.view(*batch_shape, -1), system_cov.view(*batch_shape, *self.measure_covs.shape[-2:])

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

    def _flatten(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nmeasures = self.measure_covs.shape[-1]
        state_rank = self.state_means.shape[-1]
        state_means_flat = self.state_means.view(-1, state_rank)
        state_covs_flat = self.state_covs.view(-1, state_rank, state_rank)
        measure_covs_flat = self.measure_covs.view(-1, nmeasures, nmeasures)
        return state_means_flat, state_covs_flat, measure_covs_flat

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

        has_mixture = self.mixture is not None and any(
            m in measurement_model.measures for m in self.mixture.mixture_measures
        )

        if measurement_model.is_nonlinear:
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=measurement_model,
                state_means=state_means,
                state_covs=state_covs,
            )
            if has_mixture:
                def standard_log_lik(idx: torch.Tensor) -> torch.Tensor:
                    # the monte-carlo marginal likelihood of the standard-regime measures:
                    mc_log_probs = _mc_mvnorm_log_prob(
                        resid=obs[:, idx] - mmean_samples[..., idx],
                        cov=measure_cov[:, idx.unsqueeze(-1), idx.unsqueeze(0)]
                    )
                    return torch.logsumexp(mc_log_probs, dim=0) - log(mc_log_probs.shape[0])

                return self._mixture_log_prob(obs=obs, measures=measurement_model.measures,
                                              standard_log_lik=standard_log_lik, log_prior=regime_log_prior)

            # evaluate the log-prob of the observations under each sampled measured-mean:
            mc_log_probs = MultivariateNormal(
                loc=mmean_samples,
                covariance_matrix=measure_cov.unsqueeze(0),
                validate_args=False
            ).log_prob(obs)
            # we don't want log_prob(x).mean(0), we want prob(x).mean(0).log()
            # this is a numerically stable way to do that:
            return torch.logsumexp(mc_log_probs, dim=0) - log(mc_log_probs.shape[0])

        measured_mean, measure_mat = measurement_model(mean=state_means, time=0)
        system_cov = measure_mat @ state_covs @ measure_mat.permute(0, 2, 1) + measure_cov
        if not has_mixture:
            return MultivariateNormal(measured_mean, system_cov, validate_args=False).log_prob(obs)

        def standard_log_lik(idx: torch.Tensor) -> torch.Tensor:
            # all regime-combos share the same predicted state, so this is just a sub-block of `system_cov`:
            return mvnorm_log_prob(
                resid=obs[:, idx] - measured_mean[:, idx],
                cov=system_cov[:, idx.unsqueeze(-1), idx.unsqueeze(0)]
            )

        return self._mixture_log_prob(obs=obs, measures=measurement_model.measures,
                                      standard_log_lik=standard_log_lik, log_prior=regime_log_prior)

    def _mixture_log_prob(self,
                          obs: torch.Tensor,
                          measures: Sequence[str],
                          standard_log_lik: Callable[[torch.Tensor], torch.Tensor],
                          log_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The marginal log-likelihood under the mixture: ``logsumexp_c(log_prior_c + log_lik_c)``. Each combo's
        likelihood is the (joint) likelihood of the measures in the standard regime, plus each non-standard measure's
        component likelihood.

        :param standard_log_lik: A function that takes the indices (into ``measures``) of the measures in the standard
         regime, and returns their ``(batch,)`` marginal log-likelihood.
        :param log_prior: A ``(batch, num_combos)`` tensor of log prior regime-probabilities. Defaults to the
         regime-model's base-probs.
        """
        num_rows = obs.shape[0]
        if log_prior is None:
            log_prior = self.mixture.log_base_probs().expand(num_rows, -1)

        effective, mapping = self.mixture.effective_combos(measures)
        out = []
        for e, eff in enumerate(effective):
            weird_idx = [i for i, _ in eff]
            normal_idx = [i for i in range(len(measures)) if i not in weird_idx]
            log_lik = torch.zeros(num_rows, dtype=obs.dtype, device=obs.device)
            if normal_idx:
                log_lik = log_lik + standard_log_lik(torch.as_tensor(normal_idx, dtype=torch.long, device=obs.device))
            for i, component in eff:
                log_lik = log_lik + mvnorm_log_prob(
                    (obs[:, i] - component.mean).unsqueeze(-1),
                    component.var.expand(num_rows, 1, 1)
                )
            # combos that are indistinguishable given `measures` pool their prior probability:
            log_prior_e = torch.logsumexp(log_prior[:, (mapping == e).to(log_prior.device)], dim=-1)
            out.append(log_prior_e + log_lik)
        return torch.logsumexp(torch.stack(out, -1), -1)

    def _get_measured_mean_samples(self,
                                   measurement_model: 'MeasurementModel',
                                   state_means: torch.Tensor,
                                   state_covs: torch.Tensor,
                                   white_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Samples of the measured-mean, from samples of the state. Rather than sampling the full state, this samples a
        lower-dimensional gaussian: the measured-mean of the linear processes, concatenated with the state-elements of
        any nonlinear processes (see ``MeasurementModel.extended_measure_mat``). The nonlinear processes'
        contributions, then any measure-functions, are applied to those samples.

        :param measurement_model: A flattened measurement-model (possibly a subset of the measures, for nans).
        :param state_means: A ``(num_rows, state_rank)`` tensor.
        :param state_covs: A ``(num_rows, state_rank, state_rank)`` tensor.
        :param white_noise: Standard-normal noise for the *full* model's extended dimensions (dims for measures that
         ``measurement_model`` drops are dropped here), either ``(num_samples, dim)`` -- the same draws for every row
         -- or ``(num_samples, num_rows, dim)``. Defaults to ``mc_white_noise``, which is fixed (so that the
         monte-carlo log-prob is a deterministic function of the parameters) and shared across rows.
        :return: A ``(num_samples, num_rows, num_measures)`` tensor.
        """
        nmeasures = len(measurement_model.measures)

        extended_measure_mat = measurement_model.extended_measure_mat
        partial_measured_mean = (extended_measure_mat @ state_means.unsqueeze(-1)).squeeze(-1)
        partial_measured_cov = extended_measure_mat @ state_covs @ extended_measure_mat.permute(0, 2, 1)

        # some dims might be all-zero (e.g. a measure with no linear processes), which means we can't take the
        # cholesky for those, so only factorize the nonzero block:
        nonzero = (extended_measure_mat != 0).any(0).any(1).cpu().nonzero(as_tuple=True)[0]
        m2d = torch.meshgrid(torch.arange(measurement_model.num_groups), nonzero, nonzero, indexing='ij')
        chol = torch.zeros_like(partial_measured_cov)
        chol[m2d] = torch.linalg.cholesky(partial_measured_cov[m2d])

        # take care to drop missing measures:
        missing_midx = [i for i, m in enumerate(self.measurement_model.measures) if m not in measurement_model.measures]
        em_dim = self.measurement_model_flat.extended_measure_mat.shape[1]
        em_idx = [i for i in range(em_dim) if i not in missing_midx]
        if white_noise is None:
            white_noise = self.mc_white_noise(num_dim=em_dim, dtype=chol.dtype, device=chol.device)
        white_noise = white_noise[..., em_idx]
        # (einsum rather than a broadcasting matmul, which would materialize a (samples, rows, dim, dim) copy of chol)
        if white_noise.ndim == 2:
            offsets = torch.einsum('rij,nj->nri', chol, white_noise)
        else:
            offsets = torch.einsum('rij,nrj->nri', chol, white_noise)
        sampled_pmmeans = partial_measured_mean.unsqueeze(0) + offsets

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
        if type_.startswith('pred'):
            regime_probs = None if preds.regime_probs is None else preds.regime_probs.squeeze(1)
            return StateTuple(preds.state_means.squeeze(1), preds.state_covs.squeeze(1), regime_probs=regime_probs)
        elif type_.startswith('update'):
            if preds.update_means is None:
                raise RuntimeError("No updates available; call the model with ``include_updates_in_output=True``.")
            regime_probs = None if preds.update_regime_probs is None else preds.update_regime_probs.squeeze(1)
            return StateTuple(preds.update_means.squeeze(1), preds.update_covs.squeeze(1), regime_probs=regime_probs)
        else:
            raise ValueError("Unrecognized `type_`, expected 'prediction' or 'update'.")

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

    def __iter__(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # so that we can do ``mean, cov = predictions``
        yield self.means
        yield self.covs

    def __array__(self) -> np.ndarray:
        # for numpy.asarray
        return self.means.detach().numpy()

    def __getitem__(self, item) -> 'Predictions':
        kwargs = self._getitem_helper(item)
        cls = type(self)
        return cls(**kwargs)

    def _getitem_helper(self, item: tuple) -> dict:
        if not isinstance(item, tuple):
            item = (item,)
        from .state import StateTuple

        def _slice(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x[item]

        kwargs = {
            'measurement_model': self.measurement_model.subset(*item),
            'states': StateTuple(self.state_means[item], self.state_covs[item], regime_probs=_slice(self.regime_probs)),
            'measure_covs': self.measure_covs[item],
            # indexing only can impact group/time (ensured by measurementModel.subset), so no impact:
            'mc_white_noise': self.mc_white_noise,
            'mixture': self.mixture,
        }
        if self.update_means is not None:
            kwargs['updates'] = StateTuple(
                self.update_means[item], self.update_covs[item], regime_probs=_slice(self.update_regime_probs)
            )

        return kwargs


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
    :param covs: A ``(num_samples, num_groups, num_timesteps, num_measures, num_measures)`` tensor: the corresponding
     covariance (i.e. the observation noise).
    :param observations: If ``observation_noise=True``, a ``(num_samples, num_groups, num_timesteps, num_measures)``
     tensor of sampled observations.
    """
    measures: list
    means: torch.Tensor
    covs: torch.Tensor
    observations: Optional[torch.Tensor] = None

    def __getitem__(self, measure: str) -> torch.Tensor:
        """
        The ``(num_samples, num_groups, num_timesteps)`` samples for a measure: of the observations if available,
        otherwise of the means.
        """
        values = self.means if self.observations is None else self.observations
        return values[..., self.measures.index(measure)]


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


def _warn_bias_adjust_ignored(transform: 'Transform') -> None:
    if transform.bias_adjust is not None:
        warn(f"`{type(transform).__name__}(bias_adjust=...)` is ignored for monte-carlo predictions (the mean is the "
             f"mean of back-transformed samples).")


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


def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """
    Quantile along the first dimension, with linear interpolation (like ``torch.quantile``, which has an input-size
    limit).
    """
    x = x.sort(0).values
    pos = q * (x.shape[0] - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, x.shape[0] - 1)
    return x[lo] + (pos - lo) * (x[hi] - x[lo])


def _unpack_states(states) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    :return: Stacked (num_groups, num_timesteps, ...) means, covs, and regime-probs (or None).
    """
    from .state import StateTuple

    if isinstance(states, StateTuple):  # already stacked
        return states.mean, states.cov, states.regime_probs
    if len(states) and all(isinstance(s, StateTuple) for s in states):
        regime_probs = None
        if states[0].regime_probs is not None:
            regime_probs = torch.stack([s.regime_probs for s in states], 1)
        return torch.stack([s.mean for s in states], 1), torch.stack([s.cov for s in states], 1), regime_probs
    means, covs = states
    return _maybe_stack(means, 1), _maybe_stack(covs, 1), None


def _maybe_stack(x: Union[torch.Tensor, Sequence[torch.Tensor]], dim: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(x, dim=dim)


_warn_once = {}
