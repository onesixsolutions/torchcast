from dataclasses import dataclass, fields
from typing import Tuple, Union, Optional, Dict, Iterator, Sequence, TYPE_CHECKING
from warnings import warn

import torch
from torch import nn, Tensor

import numpy as np
import pandas as pd

from functools import cached_property
from scipy import stats

from torchcast.internals.utils import get_nan_groups, is_near_zero, transpose_last_dims, class_or_instancemethod

if TYPE_CHECKING:
    from torchcast.state_space import StateSpaceModel
    from torchcast.utils import TimeSeriesDataset


class Predictions:
    """
    The output of the :class:`.StateSpaceModel` forward pass, containing the underlying state means and covariances, as
    well as the predicted observations and covariances.
    """

    def __init__(self,
                 state_means: Union[Sequence[Tensor], Tensor],
                 state_covs: Union[Sequence[Tensor], Tensor],
                 R: Union[Sequence[Tensor], Tensor],
                 H: Union[Sequence[Tensor], Tensor],
                 model: Union['StateSpaceModel', 'StateSpaceModelMetadata'],
                 update_means: Optional[Sequence[Tensor]] = None,
                 update_covs: Optional[Sequence[Tensor]] = None):

        # predictions state:
        self._state_means = state_means
        self._state_covs = state_covs

        # updates state:
        self._update_means = update_means
        self._update_covs = update_covs

        # design mats:
        self._H = H
        self._R = R

        # some model attributes are needed for `log_prob` method and for names for plotting
        if not isinstance(model, StateSpaceModelMetadata):
            all_state_elements = []
            for pid in model.processes:
                process = model.processes[pid]
                for state_element in process.state_elements:
                    all_state_elements.append((pid, state_element))
            self._model_attributes = StateSpaceModelMetadata(
                measures=model.measures,
                all_state_elements=all_state_elements,
            )

        # for lazily populated properties:
        self._means = self._covs = None

        # metadata
        self.num_groups, self.num_timesteps, self.state_size = self.state_means.shape
        self._dataset_metadata = None

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

    @cached_property
    def R(self) -> torch.Tensor:
        if not isinstance(self._R, torch.Tensor):
            self._R = torch.stack(self._R, 1)
        return self._R

    @cached_property
    def H(self) -> torch.Tensor:
        if not isinstance(self._H, torch.Tensor):
            self._H = torch.stack(self._H, 1)
        return self._H

    @property
    def measures(self) -> Sequence[str]:
        return self._model_attributes.measures

    @cached_property
    def state_means(self) -> torch.Tensor:
        if not isinstance(self._state_means, torch.Tensor):
            self._state_means = torch.stack(self._state_means, 1)
        if torch.isnan(self._state_means).any():
            if torch.isnan(self._state_means).all():
                raise ValueError("`nans` in all groups' `state_means`")
            else:
                groups, *_ = zip(*torch.isnan(self._state_means).nonzero().tolist())
                raise ValueError(f"`nans` in `state_means` for group-indices: {set(groups)}")
        return self._state_means

    @cached_property
    def state_covs(self) -> torch.Tensor:
        if not isinstance(self._state_covs, torch.Tensor):
            self._state_covs = torch.stack(self._state_covs, 1)
        if torch.isnan(self._state_covs).any():
            raise ValueError("`nans` in `state_covs`")
        return self._state_covs

    @cached_property
    def update_means(self) -> Optional[torch.Tensor]:
        if self._update_means is None:
            raise RuntimeError(
                "Cannot get ``update_means`` because update mean/cov was not passed when creating this "
                "``Predictions`` object. This usually means you have to include ``include_updates_in_output=True`` "
                "when calling ``StateSpaceModel()``."
            )
        if not isinstance(self._update_means, torch.Tensor):
            self._update_means = torch.stack(self._update_means, 1)
        if torch.isnan(self._update_means).any():
            raise ValueError("`nans` in `state_means`")
        return self._update_means

    @cached_property
    def update_covs(self) -> Optional[torch.Tensor]:
        if self._update_covs is None:
            raise RuntimeError(
                "Cannot get ``update_covs`` because update mean/cov was not passed when creating this "
                "``Predictions`` object. This usually means you have to include ``include_updates_in_output=True`` "
                "when calling ``StateSpaceModel()``."
            )
        if not isinstance(self._update_covs, torch.Tensor):
            self._update_covs = torch.stack(self._update_covs, 1)
        if torch.isnan(self._update_covs).any():
            raise ValueError("`nans` in `update_covs`")
        return self._update_covs

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
                           **kwargs) -> Tuple[Tensor, Tensor]:
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

    @classmethod
    def observe(cls, state_means: Tensor, state_covs: Tensor, R: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert latent states into observed predictions (and their uncertainty).

        :param state_means: The latent state means
        :param state_covs: The latent state covs.
        :param R: The measure-covariance matrices.
        :param H: The measurement matrix.
        :return: A tuple of `means`, `covs`.
        """
        means = H.matmul(state_means.unsqueeze(-1)).squeeze(-1)
        Ht = transpose_last_dims(H)
        covs = H.matmul(state_covs).matmul(Ht) + R
        return means, covs

    @property
    def means(self) -> Tensor:
        if self._means is None:
            # TODO: in ExpSmooth, _state_covs, _R, and _H will often not be time-varying. if we could generate them s.t.
            #  we could perform a fast check of this (self._R[0] is self._R[1]) then could speed up slowest step:
            #  `H.matmul(state_covs).matmul(Ht) + R`
            self._means, self._covs = self.observe(self.state_means, self.state_covs, self.R, self.H)
        return self._means

    @property
    def covs(self) -> Tensor:
        if self._covs is None:
            self._means, self._covs = self.observe(self.state_means, self.state_covs, self.R, self.H)
        return self._covs

    def log_prob(self, obs: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the ``StateSpaceModel``).

        :param obs: A Tensor that could be used in the ``StateSpaceModel`` forward pass.
        :param weights: If specified, will be used to weight the log-probability of each group X timestep.
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        assert len(obs.shape) == 3
        assert obs.shape[-1] == self.means.shape[-1]
        n_measure_dim = obs.shape[-1]
        n_state_dim = self.state_means.shape[-1]

        obs_flat = obs.reshape(-1, n_measure_dim)
        if weights is None:
            weights = torch.ones(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)
        else:
            weights = weights.reshape(-1, n_measure_dim)
        state_means_flat = self.state_means.view(-1, n_state_dim)
        state_covs_flat = self.state_covs.view(-1, n_state_dim, n_state_dim)
        H_flat = self.H.view(-1, n_measure_dim, n_state_dim)
        R_flat = self.R.view(-1, n_measure_dim, n_measure_dim)

        lp_flat = torch.zeros(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)
        for gt_idx, valid_idx in get_nan_groups(torch.isnan(obs_flat)):
            if valid_idx is None:
                gt_obs = obs_flat[gt_idx]
                gt_R = R_flat[gt_idx]
                gt_H = H_flat[gt_idx]
            else:
                mask1d = torch.meshgrid(gt_idx, valid_idx, indexing='ij')
                mask2d = torch.meshgrid(gt_idx, valid_idx, valid_idx, indexing='ij')
                gt_R = R_flat[mask2d]
                gt_H = H_flat[mask1d]
                gt_obs = obs_flat[mask1d]
            _kwargs = self._get_log_prob_kwargs(gt_idx, valid_idx)
            lp_flat[gt_idx] = self._log_prob(
                obs=gt_obs,
                state_means=state_means_flat[gt_idx],
                state_covs=state_covs_flat[gt_idx],
                R=gt_R,
                H=gt_H,
                **_kwargs
            )

        lp_flat = lp_flat * weights

        return lp_flat.view(obs.shape[0:2])

    def _get_log_prob_kwargs(self, groups: Tensor, valid_idx: Tensor) -> dict:
        return {}

    def _log_prob(self,
                  obs: Tensor,
                  state_means: Tensor,
                  state_covs: Tensor,
                  R: Tensor,
                  H: Tensor,
                  **kwargs) -> Tensor:
        means, covs = self.observe(state_means, state_covs, R, H)
        return torch.distributions.MultivariateNormal(means, covs, validate_args=False).log_prob(obs)

    def to_dataframe(self,
                     dataset: Optional['TimeSeriesDataset'] = None,
                     type: str = 'predictions',
                     group_colname: Optional[str] = None,
                     time_colname: Optional[str] = None,
                     conf: Optional[float] = .95,
                     **kwargs) -> pd.DataFrame:
        """
        :param dataset: The dataset which generated the predictions. If not supplied, will use the metadata set at
         prediction time, but the group-names will be replaced by dummy group names, and the output will not include
         actuals.
        :param type: Either 'predictions' or 'components'.
        :param group_colname: Column-name for 'group'
        :param time_colname: Column-name for 'time'
        :param conf: If set, specifies the confidence level for the 'lower' and 'upper' columns in the output. Default
         of 0.95 means these are 0.025 and 0.975. If ``None``, then will just include 'std' column instead.
        :return: A pandas DataFrame with 'group', 'time', 'measure', 'mean', 'lower', 'upper'. For ``type='components'``
         additionally includes: 'process' and 'state_element'.
        """
        from torchcast.utils import TimeSeriesDataset

        multi = kwargs.pop('multi', False)
        if multi is not False:
            msg = "`multi` is deprecated, please use `conf` instead."
            if multi is None:  # old way of specifying "just return std", for backwards-compatibility
                warn(msg, DeprecationWarning)
                conf = None
            else:
                raise TypeError(msg)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {set(kwargs)}")

        named_tensors = {}
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
        else:
            for measure_group, tensor in zip(dataset.measures, dataset.tensors):
                for i, measure in enumerate(measure_group):
                    if measure in self.measures:
                        named_tensors[measure] = tensor[..., [i]]
            missing = set(self.measures) - set(dataset.all_measures)
            if missing:
                raise ValueError(
                    f"Some measures in the design aren't in the dataset.\n"
                    f"Design: {missing}\nDataset: {dataset.all_measures}"
                )

        group_colname = group_colname or self.dataset_metadata.group_colname
        time_colname = time_colname or self.dataset_metadata.time_colname

        def _tensor_to_df(tens, measures):
            offsets = np.arange(0, tens.shape[1]) * (dataset.dt_unit if dataset.dt_unit else 1)
            times = dataset.start_offsets[:, None] + offsets

            return TimeSeriesDataset.tensor_to_dataframe(
                tensor=tens,
                times=times,
                group_names=dataset.group_names,
                group_colname=group_colname,
                time_colname=time_colname,
                measures=measures
            )

        assert group_colname not in {'mean', 'lower', 'upper', 'std'}
        assert time_colname not in {'mean', 'lower', 'upper', 'std'}

        out = []
        if type.startswith('pred'):

            stds = torch.diagonal(self.covs, dim1=-1, dim2=-2).sqrt()
            for i, measure in enumerate(self.measures):
                # predicted:
                df = _tensor_to_df(torch.stack([self.means[..., i], stds[..., i]], 2), measures=['mean', 'std'])
                if conf is not None:
                    df['lower'], df['upper'] = conf2bounds(df['mean'], df.pop('std'), conf=conf)

                # actual:
                orig_tensor = named_tensors.get(measure, None)
                if orig_tensor is not None and not torch.isnan(orig_tensor).all():
                    df_actual = _tensor_to_df(orig_tensor, measures=['actual'])
                    df = df.merge(df_actual, on=[group_colname, time_colname], how='left')

                out.append(df.assign(measure=measure))

        elif type.startswith('comp'):
            for (measure, process, state_element), (m, std) in self._components().items():
                df = _tensor_to_df(torch.stack([m, std], 2), measures=['mean', 'std'])
                if conf is not None:
                    df['lower'], df['upper'] = conf2bounds(df['mean'], df.pop('std'), conf=conf)
                df['process'], df['state_element'], df['measure'] = process, state_element, measure
                out.append(df)

            # residuals:
            for i, measure in enumerate(self.measures):
                orig_tensor = named_tensors.get(measure, None)
                if orig_tensor is None:
                    continue
                predictions = self.means[..., [i]]
                if orig_tensor.shape[1] < predictions.shape[1]:
                    orig_aligned = predictions.data.clone()
                    orig_aligned[:] = float('nan')
                    orig_aligned[:, 0:orig_tensor.shape[1], :] = orig_tensor
                else:
                    orig_aligned = orig_tensor[:, 0:predictions.shape[1], :]

                df = _tensor_to_df(predictions - orig_aligned, ['mean'])
                df['process'], df['state_element'], df['measure'] = 'residuals', 'residuals', measure
                out.append(df)

        else:
            raise ValueError("Expected `type` to be 'predictions' or 'components'.")

        out = pd.concat(out).reset_index(drop=True)
        _out_cols = [group_colname, time_colname, 'measure', 'mean']
        if conf is None:
            _out_cols.append('std')
        else:
            _out_cols.extend(['lower', 'upper'])
        if type.startswith('comp'):
            _out_cols = _out_cols[0:3] + ['process', 'state_element'] + _out_cols[3:]
        if 'actual' in out.columns:
            _out_cols.append('actual')
        return out[_out_cols]

    @torch.no_grad()
    def _components(self) -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
        out = {}
        for midx, measure in enumerate(self.measures):
            H = self.H[..., midx, :]
            means = H * self.state_means
            stds = H * torch.diagonal(self.state_covs, dim1=-2, dim2=-1).sqrt()

            for se_idx, (process, state_element) in enumerate(self._model_attributes.all_state_elements):
                if not is_near_zero(means[:, :, se_idx]).all():
                    out[(measure, process, state_element)] = (means[:, :, se_idx], stds[:, :, se_idx])

        return out

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

    def __iter__(self) -> Iterator[Tensor]:
        # so that we can do ``mean, cov = predictions``
        yield self.means
        yield self.covs

    def __array__(self) -> np.ndarray:
        # for numpy.asarray
        return self.means.detach().numpy()

    def __getitem__(self, item: Tuple) -> 'Predictions':
        kwargs = self._getitem_helper(item)
        cls = type(self)
        return cls(**kwargs, model=self._model_attributes)

    def _getitem_helper(self, item: tuple) -> dict:
        kwargs = {
            'state_means': self.state_means[item],
            'state_covs': self.state_covs[item],
            'H': self.H[item],
            'R': self.R[item]
        }
        if self._update_means is not None:
            kwargs.update({'update_means': self.update_means[item], 'update_covs': self.update_covs[item]})

        for k in list(kwargs):
            expected_shape = getattr(self, k).shape
            v = kwargs[k]
            if len(v.shape) != len(expected_shape):
                raise TypeError(f"Expected {k} to have ndims {len(expected_shape)}, but got {len(v.shape)}")
            if v.shape[2:] != expected_shape[2:]:
                raise TypeError(f"Cannot index into non-batch dims of {type(self).__name__}")
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


def conf2bounds(mean, std, conf) -> tuple:
    assert conf >= .50
    multi = -stats.norm.ppf((1 - conf) / 2)
    lower = mean - multi * std
    upper = mean + multi * std
    return lower, upper
