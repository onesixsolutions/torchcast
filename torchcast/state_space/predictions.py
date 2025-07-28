from functools import cached_property

from math import log

from dataclasses import dataclass, fields
from typing import Tuple, Union, Optional, Sequence, TYPE_CHECKING, Iterator
from warnings import warn

import torch

import numpy as np
import pandas as pd

from scipy import stats

from torchcast.internals.utils import get_nan_groups, class_or_instancemethod, get_meshgrids

if TYPE_CHECKING:
    from torchcast.utils import TimeSeriesDataset
    from torchcast.internals.batch_design import MeasurementModel


class Predictions:
    """
    The output of the :class:`.StateSpaceModel` forward pass, containing the underlying state means and covariances, as
    well as methods such as ``log_prob()``, ``to_dataframe()``, and ``plot()``.
    """
    _means = None
    _covs = None

    def __init__(self,
                 measurement_model: 'MeasurementModel',
                 state_means: Sequence[torch.Tensor],
                 state_covs: Sequence[torch.Tensor],
                 measure_covs: Union[Sequence[torch.Tensor], torch.Tensor],
                 white_noise: Optional[torch.Tensor] = None):
        # TODO: white_noise, updates
        self.measurement_model = measurement_model
        self.state_means = _maybe_stack(state_means, 1)
        self.state_covs = _maybe_stack(state_covs, 1)
        self.measure_covs = _maybe_stack(measure_covs, 1)

        self._dataset_metadata = None

    @property
    def num_groups(self) -> int:
        return len(self.state_means)

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
                     mc_draws: Optional[int] = None) -> pd.DataFrame:
        if dataset is None:
            dataset = self.dataset_metadata.copy()
            if dataset.group_names is None:
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

        group_colname = group_colname or self.dataset_metadata.group_colname
        time_colname = time_colname or self.dataset_metadata.time_colname

        if type.startswith('pred'):
            df = self._to_dataframe(
                dataset=dataset,
                group_colname=group_colname,
                time_colname=time_colname,
                conf=conf,
                mc_draws=mc_draws
            )
            return df
        else:
            assert type.startswith('comp'), "Expected `type` to be 'predictions' or 'components'."
            return self._to_components_dataframe()

    @classmethod
    def _flatten(cls,
                 state: tuple[torch.Tensor, torch.Tensor],
                 measure_cov: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        nmeasures = measure_cov.shape[-1]
        state_rank = state[0].shape[-1]
        state_means_flat = state[0].view(-1, state_rank)
        state_covs_flat = state[1].view(-1, state_rank, state_rank)
        measure_covs_flat = measure_cov.view(-1, nmeasures, nmeasures)
        return (state_means_flat, state_covs_flat), measure_covs_flat

    def _observe(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_shape = self.state_means.shape[0:2]

        (smeans_flat, scovs_flat), mcovs_flat = self._flatten(
            state=(self.state_means, self.state_covs),
            measure_cov=self.measure_covs
        )
        measurement_model_flat = self.measurement_model.flattened()
        measured_mean, measure_mat = measurement_model_flat(mean=smeans_flat, time=0)
        if self.measurement_model.is_nonlinear:
            # in this case, we need to use monte-carlo to get samples/distribution, there's no closed form cov
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=measurement_model_flat,
                state_means=smeans_flat,
                state_covs=scovs_flat,
                white_noise=self.white_noise
            )
            mvnorm = torch.distributions.MultivariateNormal(
                loc=mmean_samples,
                covariance_matrix=mcovs_flat.unsqueeze(0),
                validate_args=False
            )
            samples = mvnorm.sample().squeeze(0)
            measured_mean = torch.mean(samples, dim=0)
            return measured_mean, None
        else:
            system_cov = measure_mat @ scovs_flat @ measure_mat.permute(0, 2, 1) + mcovs_flat
            return measured_mean.view(*batch_shape, -1), system_cov.view(*batch_shape, *self.measure_covs.shape[-2:])

    @cached_property
    def means(self) -> torch.Tensor:
        """
        Returns the observed means of the predictions, i.e. the measured means of the state.
        """
        if self._means is None:
            self._means, self._covs = self._observe()
        return self._means

    @cached_property
    def covs(self) -> Optional[torch.Tensor]:
        if self._means is None:
            self._means, self._covs = self._observe()
        if self._covs is None:
            if _warn_once.get('cov', False):
                warn("The measurement model is nonlinear, so no closed-form covariance is available, returning None.")
                _warn_once['cov'] = True
        return self._covs

    def _to_dataframe(self,
                      dataset: Union['TimeSeriesDataset', 'DatasetMetadata'],
                      group_colname: str,
                      time_colname: str,
                      conf: float,
                      mc_draws: Optional[int]) -> pd.DataFrame:
        batch_shape = self.state_means.shape[0:2]
        if mc_draws is None:
            mc_draws = 500 if self.measurement_model.is_nonlinear else 0
        if not mc_draws and self.measurement_model.is_nonlinear:
            raise RuntimeError("Since there are nonlinearities in the measurement-model, must use `monte_carlo`")

        (smeans_flat, scovs_flat), mcovs_flat = self._flatten(
            state=(self.state_means, self.state_covs),
            measure_cov=self.measure_covs
        )
        measurement_model_flat = self.measurement_model.flattened()
        if conf is not None:
            assert conf >= .50

        if mc_draws:
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=measurement_model_flat,
                state_means=smeans_flat,
                state_covs=scovs_flat,
                white_noise=mc_draws
            )
            mvnorm = torch.distributions.MultivariateNormal(
                loc=mmean_samples,
                covariance_matrix=mcovs_flat.unsqueeze(0),
                validate_args=False
            )
            samples = mvnorm.sample().squeeze(0)

            alpha = (1 - conf) / 2
            by_measure = {}
            for i, measure in enumerate(measurement_model_flat.measures):
                mean = torch.mean(samples[..., i], dim=0)
                lower = torch.quantile(samples[..., i], q=alpha, dim=0)
                upper = torch.quantile(samples[..., i], q=1 - alpha, dim=0)
                by_measure[measure] = (
                    mean.view(*batch_shape),
                    lower.view(*batch_shape),
                    upper.view(*batch_shape)
                )
        else:
            multi = -stats.norm.ppf((1 - conf) / 2) if conf else .5
            measured_mean, measure_mat = measurement_model_flat(mean=smeans_flat, time=0)
            system_cov = measure_mat @ scovs_flat @ measure_mat.permute(0, 2, 1) + mcovs_flat

            by_measure = {}
            for i, measure in enumerate(measurement_model_flat.measures):
                mean = measured_mean[..., i]
                var = system_cov[..., i, i]
                lower = mean - multi * torch.sqrt(var)
                upper = mean + multi * torch.sqrt(var)
                by_measure[measure] = (
                    mean.view(*batch_shape),
                    lower.view(*batch_shape),
                    upper.view(*batch_shape)
                )

        from torchcast.utils import TimeSeriesDataset

        actuals = {}
        if isinstance(dataset, TimeSeriesDataset):
            for mgroup, tens in zip(dataset.measures, dataset.tensors):
                for m in mgroup:
                    if m not in by_measure:
                        continue
                    actuals[m] = tens[..., mgroup.index(m)]
        out = []
        times = TimeSeriesDataset.get_dataset_times(
            dataset.start_offsets, num_timesteps=batch_shape[-1], dt_unit=dataset.dt_unit
        )
        for measure, (mean, lower, upper) in by_measure.items():
            _to_stack = {'mean': mean, 'lower': lower, 'upper': upper}
            mactuals = actuals.get(measure, None)
            if mactuals is not None:
                _to_stack['actual'] = mactuals
            out.append(
                TimeSeriesDataset.tensor_to_dataframe(
                    tensor=torch.stack(list(_to_stack.values()), -1),
                    times=times,
                    group_names=dataset.group_names,
                    group_colname=group_colname,
                    time_colname=time_colname,
                    measures=list(_to_stack)
                )
            )
            out[-1]['measure'] = measure
        out = pd.concat(out)

        if conf is None:
            out['std'] = out.pop('upper') - out.pop('lower')
        return out

    def log_prob(self, obs: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the ``StateSpaceModel``).

        :param obs: A Tensor that could be used in the ``StateSpaceModel`` forward pass.
        :param weights: If specified, will be used to weight the log-probability of each group X timestep.
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        assert len(obs.shape) == 3
        measure_rank = obs.shape[-1]
        state_rank = self.state_means.shape[-1]

        obs_flat = obs.view(-1, measure_rank)
        if weights is None:
            weights = torch.ones(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)
        else:
            weights = weights.view(-1, measure_rank)
        state_means_flat = self.state_means.view(-1, state_rank)
        state_covs_flat = self.state_covs.view(-1, state_rank, state_rank)
        measure_covs_flat = self.measure_covs.view(-1, measure_rank, measure_rank)
        measurement_model_flat = self.measurement_model.flattened()

        lp_flat = torch.zeros(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)
        for gt_idx, valid_idx in get_nan_groups(torch.isnan(obs_flat)):
            if valid_idx is None:
                gt_obs = obs_flat[gt_idx]
                gt_mcov = measure_covs_flat[gt_idx]
                gt_mmodel = measurement_model_flat.subset(gt_idx)
            else:
                gt_mmodel = measurement_model_flat.subset(gt_idx, measures=valid_idx)
                mask1d, mask2d = get_meshgrids(gt_idx, valid_idx)
                gt_mcov = measure_covs_flat[mask2d]
                gt_obs = obs_flat[mask1d]
            _kwargs = self._get_log_prob_kwargs(gt_idx, valid_idx)
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
        return {}

    def _log_prob(self,
                  obs: torch.Tensor,
                  state_means: torch.Tensor,
                  state_covs: torch.Tensor,
                  measure_cov: torch.Tensor,
                  measurement_model: 'MeasurementModel',
                  mc_draws: Union[int, torch.Tensor] = 0) -> torch.Tensor:
        assert measurement_model.num_timesteps == 1

        if isinstance(mc_draws, torch.Tensor) or mc_draws:
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=measurement_model,
                state_means=state_means,
                state_covs=state_covs,
                white_noise=mc_draws
            )

            # evaluate the log-prob of the observations under each sampled measured-mean:
            mc_log_probs = torch.distributions.MultivariateNormal(
                loc=mmean_samples,
                covariance_matrix=measure_cov.unsqueeze(0),
                validate_args=False
            ).log_prob(obs)
            # we don't want log_prob(x).mean(0), we want prob(x).mean(0).log()
            # this is a numerically stable way to do that:
            return torch.logsumexp(mc_log_probs, dim=0) - log(mc_draws)
        else:
            if measurement_model.is_nonlinear:
                raise RuntimeError("Since there are nonlinearities in the measurement-model, must use `mc_draws` > 0.")
            measured_mean, measure_mat = measurement_model(mean=state_means, time=0)
            system_cov = measure_mat @ state_covs @ measure_mat.permute(0, 2, 1) + measure_cov
            return torch.distributions.MultivariateNormal(measured_mean, system_cov, validate_args=False).log_prob(obs)

    def _get_measured_mean_samples(self,
                                   measurement_model: 'MeasurementModel',
                                   state_means: torch.Tensor,
                                   state_covs: torch.Tensor,
                                   white_noise: Union[int, torch.Tensor]):
        nmeasures = len(measurement_model.measures)

        measured_mean, measure_mat = measurement_model(mean=state_means, time=0)

        # first, create an extended measure-mat
        extended_measure_mat = measure_mat
        eprocess2_slice = {}
        start_ = nmeasures
        for process in self.measurement_model.nonlinear_processes:
            if process.measure not in measurement_model.measures:
                continue  # see note in MeasurementModel._initialize_measure_mats
            identity_mat = torch.eye(process.rank, dtype=measure_mat.dtype)
            extended_measure_mat = torch.stack([extended_measure_mat, identity_mat], dim=-2)
            eprocess2_slice[process.id] = slice(start_, start_ + process.rank)
            start_ += process.rank

        # now, use the extended measure-mat to reduce dimensionality
        partial_measured_mean = extended_measure_mat @ state_means  # todo: squeeze?
        partial_measured_cov = extended_measure_mat @ state_covs @ extended_measure_mat.permute(0, 2, 1)

        # then we sample from that multivariate distribution
        chol = torch.linalg.cholesky(partial_measured_cov)
        if not isinstance(white_noise, torch.Tensor):
            white_noise = torch.randn((white_noise, *state_means.shape[:-1], chol.shape[-1]), device=chol.device)
        partial_mmean_samples = chol.unsqueeze(0) @ white_noise + partial_measured_mean

        # each of these samples represents a draw from a concatenated set of means: (1) the measured-mean of the
        # linear processes with (2) the nonlinear processes' state-means.
        # for each sample, we take those draws from the (nonlinear) state distribution and use them to apply
        # adjustment to the linear measured-mean.
        mmean_samples = []
        for sampled_pmean in partial_mmean_samples.unbind(0):
            procs_and_means = [
                (proc, sampled_pmean[..., eprocess2_slice[proc.id]])
                for proc in self.measurement_model.nonlinear_processes
            ]
            mmean_samples.append(
                sampled_pmean[..., :nmeasures]
                +
                measurement_model.get_measured_mean_adjustment(procs_and_means, time=0)
            )
        return torch.stack(mmean_samples, dim=0)

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
                           **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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
        :return: A tuple of state-means and state-covs, appropriate for forecasting by passing as `initial_state`
         for :func:`StateSpaceModel.forward()`.
        """
        preds = self.with_new_start_times(start_times=times, n_timesteps=1, **kwargs)
        if type_.startswith('pred'):
            return preds.state_means.squeeze(1), preds.state_covs.squeeze(1)
        elif type_.startswith('update'):
            return preds.update_means.squeeze(1), preds.update_covs.squeeze(1)
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
            df['lower'], df['upper'] = conf2bounds(df['mean'], df.pop('std'), conf=.95)
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
        kwargs = {
            'measurement_model': self.measurement_model.subset(*item),
            'state_means': self.state_means[item],
            'state_covs': self.state_covs[item],
            'measure_covs': self.measure_covs[item]
        }
        # if self._update_means is not None:
        #     kwargs.update({'update_means': self.update_means[item], 'update_covs': self.update_covs[item]})

        return kwargs


@dataclass
class StateSpaceModelMetadata:
    measures: Sequence[str]
    all_state_elements: Sequence[Tuple[str, str]]


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


def _maybe_stack(x: Union[torch.Tensor, Sequence[torch.Tensor]], dim: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(x, dim=dim)


_warn_once = {}
