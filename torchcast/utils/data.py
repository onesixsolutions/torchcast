import datetime
import itertools
import pandas as pd

from typing import Sequence, Any, Union, Optional, Tuple, Callable, TYPE_CHECKING
from warnings import warn

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset

from torchcast.internals.utils import ragged_cat, true1d_idx

if TYPE_CHECKING:
    from pandas import DataFrame


class TimeSeriesDataset(TensorDataset):
    """
    :class:`.TimeSeriesDataset` includes additional information about each of the Tensors' dimensions: the name for
    each group in the first dimension, the start (date)time (and optionally datetime-unit) for the second dimension,
    and the name of the measures for the third dimension.

    Note that unlike :class:`torch.utils.data.TensorDataset`, indexing a :class:`.TimeSeriesDataset` returns another
    :class:`.TimeSeriesDataset`, not a tuple of tensors. So when using :class:`.TimeSeriesDataset`, use
    :class:`.TimeSeriesDataLoader`.
    """
    _repr_attrs = ('sizes', 'measures')

    def __init__(self,
                 *tensors: Tensor,
                 group_names: Sequence[Any],
                 start_times: Union[np.ndarray, Sequence],
                 measures: Sequence[Sequence[str]],
                 dt_unit: Optional[str]):

        if not isinstance(group_names, np.ndarray):
            group_names = np.array(group_names)

        assert len(group_names) == len(set(group_names))
        assert len(group_names) == len(start_times)
        assert len(tensors) == len(measures)

        for i, (tensor, tensor_measures) in enumerate(zip(tensors, measures)):
            if isinstance(tensor_measures, str):
                raise ValueError(f"Expected measures to be a list of lists/tuples, but element-{i} is a string.")
            if len(tensor.shape) < 3:
                raise ValueError(f"Tensor {i} has < 3 dimensions")
            if tensor.shape[0] != len(group_names):
                raise ValueError(f"Tensor {i}'s first dimension has length != {len(group_names)}.")
            if tensor.shape[2] != len(tensor_measures):
                raise ValueError(f"Tensor {i}'s 3rd dimension has length != len({tensor_measures}).")

        self.measures = tuple(tuple(m) for m in measures)

        self.group_names = group_names
        self.dt_unit = None
        if dt_unit:
            if not isinstance(dt_unit, np.timedelta64):
                dt_unit = np.timedelta64(1, dt_unit)
            self.dt_unit = dt_unit
        start_times = np.asanyarray(start_times)
        if self.dt_unit:
            assert len(start_times.shape) == 1
            if isinstance(start_times[0], (np.datetime64, datetime.datetime)):
                start_times = np.array(start_times, dtype='datetime64')
            else:
                raise ValueError("`dt_unit` is not None but `start_times` is not an array of datetimes")
        else:
            if not isinstance(start_times[0], int) and not float(start_times[0]).is_integer():
                raise ValueError(
                    f"`dt_unit` is None but `start_times` does not appear to be integers "
                    f"(e.g. start_times[0] is {start_times[0]})."
                )
        self.start_times = start_times
        super().__init__(*tensors)

    @property
    def all_measures(self) -> tuple:
        return tuple(itertools.chain.from_iterable(self.measures))

    def to(self, *args, **kwargs) -> 'TimeSeriesDataset':
        new_tensors = [x.to(*args, **kwargs) for x in self.tensors]
        return self.with_new_tensors(*new_tensors)

    def __repr__(self) -> str:
        kwargs = []
        for k in self._repr_attrs:
            v = getattr(self, k)
            if isinstance(v, Tensor):
                v = v.size()
            kwargs.append("{}={!r}".format(k, v))
        return "{}({})".format(type(self).__name__, ", ".join(kwargs))

    @property
    def sizes(self) -> Sequence:
        return [t.size() for t in self.tensors]

    @property
    def num_timesteps(self) -> int:
        return max(tensor.shape[1] for tensor in self.tensors)  # todo: why are we supporting this?

    # Subsetting ------------------------:
    @torch.no_grad()
    def train_val_split(self,
                        *args,
                        train_frac: float = None,
                        dt: Union[np.datetime64, dict] = None,
                        quiet: bool = False) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        """
        :param train_frac: The proportion of the data to keep for training. This is calculated on a per-group basis, by
         taking the last observation for each group (i.e., the last observation that a non-nan value on any measure). If
         neither `train_frac` nor `dt` are passed, ``train_frac=.75`` is used.
        :param dt: A datetime to use in dividing train/validation (first datetime for validation), or a dictionary of
         group-names : date-times.
        :param quiet: If True, will not emit a warning for groups having only `nan` after `dt`
        :return: Two ``TimeSeriesDatasets``, one with data before the split, the other with >= the split.
        """
        if args:
            raise TypeError(
                f"`train_val_split` takes no positional args but {len(args)} {'were' if len(args) > 1 else 'was'} provided"
            )

        # get split times:
        if dt is None:
            if train_frac is None:
                train_frac = .75
            assert 0 < train_frac < 1
            # for each group, find the last non-nan, take `frac` of that to find the train/val split point:
            split_idx = np.array([int((dur - 1) * train_frac) for dur in self.get_durations()], dtype='int')
            _times = self.times(0)
            split_times = np.array([_times[i, t] for i, t in enumerate(split_idx)])
        else:
            if train_frac is not None:
                raise TypeError("Can pass only one of `train_frac`, `dt`.")
            if isinstance(dt, dict):
                split_times = np.array([dt[group_name] for group_name in self.group_names],
                                       dtype='datetime64[ns]' if self.dt_unit else 'int')
            else:
                if self.dt_unit:
                    if isinstance(dt, datetime.datetime):  # base
                        dt = np.datetime64(dt)
                    if hasattr(dt, 'to_datetime64'):  # pandas
                        dt = dt.to_datetime64()
                    if not isinstance(dt, np.datetime64):
                        raise ValueError(f"`dt` is not datetimelike, but dataset has non-null dt_unit `{self.dt_unit}`")
                split_times = np.full(shape=len(self.group_names), fill_value=dt)

        # val:
        val_dataset = self.with_new_start_times(split_times, quiet=quiet)

        # train:
        train_tensors = []
        for i, tens in enumerate(self.tensors):
            train = tens.clone()
            train[np.where(self.times(i) >= split_times[:, None])] = float('nan')
            if i == 0:
                not_all_nan = (~torch.isnan(train)).sum((0, 2))
                last_good_idx = true1d_idx(not_all_nan).max()
            train = train[:, :(last_good_idx + 1), :]
            train_tensors.append(train)
        # TODO: replace padding nans for all but first tensor?
        # TODO: reduce width of 0> tensors based on width of 0 tensor?
        train_dataset = self.with_new_tensors(*train_tensors)

        return train_dataset, val_dataset

    def with_new_start_times(self,
                             start_times: Union[datetime.datetime, np.datetime64, np.ndarray, Sequence],
                             n_timesteps: Optional[int] = None,
                             quiet: bool = False) -> 'TimeSeriesDataset':
        """
        Subset a :class:`.TimeSeriesDataset` so that some/all of the groups have later start times.

        :param start_times: An array/list of new datetimes, or a single datetime that will be used for all groups.
        :param n_timesteps: The number of timesteps in the output (nan-padded).
        :param quiet: If True, will not emit a warning for groups having only `nan` after the start-time.
        :return: A new :class:`.TimeSeriesDataset`.
        """
        if isinstance(start_times, (datetime.datetime, np.datetime64)):
            start_times = np.full(len(self.group_names), start_times, dtype='datetime64[ns]' if self.dt_unit else 'int')
        new_tensors = []
        for i, tens in enumerate(self.tensors):
            times = self.times(i)
            new_tens = []
            for g, (new_time, old_times) in enumerate(zip(start_times, times)):
                if (old_times <= new_time).all():
                    if not quiet:
                        warn(f"{new_time} is later than all the times for group {self.group_names[g]}")
                    new_tens.append(tens[[g], 0:0])
                    continue
                elif (old_times > new_time).all():
                    if not quiet:
                        warn(f"{new_time} is earlier than all the times for group {self.group_names[g]}")
                    new_tens.append(tens[[g], 0:0])
                    continue
                # drop if before new_time:
                g_tens = tens[g, true1d_idx(old_times >= new_time)]
                # drop if after last nan:
                all_nan, _ = torch.min(torch.isnan(g_tens), 1)
                if all_nan.all():
                    if not quiet:
                        warn(f"Group '{self.group_names[g]}' (tensor {i}) has only `nans` after {new_time}")
                    end_idx = 0
                else:
                    end_idx = true1d_idx(~all_nan).max() + 1
                new_tens.append(g_tens[:end_idx].unsqueeze(0))
            new_tens = ragged_cat(new_tens, ragged_dim=1, cat_dim=0)
            if n_timesteps:
                if new_tens.shape[1] > n_timesteps:
                    new_tens = new_tens[:, :n_timesteps, :]
                else:
                    tmp = torch.empty((new_tens.shape[0], n_timesteps, new_tens.shape[2]), dtype=new_tens.dtype)
                    tmp[:] = float('nan')
                    tmp[:, :new_tens.shape[1], :] = new_tens
                    new_tens = tmp
            new_tensors.append(new_tens)
        return type(self)(
            *new_tensors,
            group_names=self.group_names,
            start_times=start_times,
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    def get_groups(self, groups: Sequence[Any]) -> 'TimeSeriesDataset':
        """
        Get the subset of the batch corresponding to groups. Note that the ordering in the output will match the
        original ordering (not that of `group`), and that duplicates will be dropped.
        """
        return self[np.isin(self.group_names, groups)]

    def split_measures(self, *measure_groups) -> 'TimeSeriesDataset':
        """
        Take a dataset and split it into a dataset with multiple tensors.

        :param measure_groups: Each argument should be a list of measure-names.
        :return: A :class:`.TimeSeriesDataset`, now with multiple tensors for the measure-groups.
        """
        concat_tensors = torch.cat(self.tensors, dim=2)

        idx_groups = []
        for measure_group in measure_groups:
            idx_groups.append([])
            for measure in measure_group:
                if measure not in self.all_measures:
                    raise ValueError(f"Measure '{measure}' not in dataset measures:\n{self.all_measures}")
                idx_groups[-1].append(self.all_measures.index(measure))

        return type(self)(
            *(concat_tensors[:, :, idxs] for idxs in idx_groups),
            start_times=self.start_times,
            group_names=self.group_names,
            measures=measure_groups,
            dt_unit=self.dt_unit
        )

    def __getitem__(self, item: Union[int, Sequence, slice]) -> 'TimeSeriesDataset':
        if isinstance(item, int):
            item = [item]
        return type(self)(
            *super().__getitem__(item),
            group_names=self.group_names[item],
            start_times=self.start_times[item],
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    # Creation/Transformation ------------------------:
    def to_dataframe(self,
                     group_colname: str = 'group',
                     time_colname: str = 'time'
                     ) -> 'DataFrame':

        return self.tensor_to_dataframe(
            tensor=ragged_cat(self.tensors, ragged_dim=1, cat_dim=2),
            times=self.times(),
            group_names=self.group_names,
            group_colname=group_colname,
            time_colname=time_colname,
            measures=self.all_measures
        )

    @staticmethod
    @torch.no_grad()
    def tensor_to_dataframe(tensor: Tensor,
                            times: np.ndarray,
                            group_names: Sequence,
                            group_colname: str,
                            time_colname: str,
                            measures: Sequence[str]) -> 'DataFrame':
        # TODO: this could probably be better
        from pandas import DataFrame, concat

        tensor = tensor.cpu().numpy()
        assert tensor.shape[0] == len(group_names)
        assert tensor.shape[0] == len(times)
        assert tensor.shape[1] <= times.shape[1]
        assert tensor.shape[2] == len(measures)

        _all_nan_groups = []
        dfs = []
        for g, group_name in enumerate(group_names):
            # get values, don't store trailing nans:
            values = tensor[g]
            all_nan_per_row = np.min(np.isnan(values), axis=1)
            if all_nan_per_row.all():
                _all_nan_groups.append(group_name)
                continue
            end_idx = true1d_idx(~all_nan_per_row).max() + 1
            # convert to dataframe:
            df = DataFrame(data=values[:end_idx, :], columns=measures)
            df[group_colname] = group_name
            df[time_colname] = np.nan
            df[time_colname] = times[g, 0:len(df.index)]
            dfs.append(df)
        if _all_nan_groups:
            warn(f"Groups have only missing values:{_all_nan_groups}")

        if dfs:
            return concat(dfs)
        else:
            return DataFrame(columns=list(measures) + [group_colname, time_colname])

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: Optional[str],
                       time_colname: str,
                       dt_unit: Optional[str],
                       measure_colnames: Optional[Sequence[str]] = None,
                       X_colnames: Optional[Sequence[str]] = None,
                       y_colnames: Optional[Sequence[str]] = None,
                       pad_X: Union[float, str, None] = 'ffill',
                       dtype: torch.dtype = torch.float32,
                       device: Optional[torch.device] = None) -> 'TimeSeriesDataset':
        """
        :param dataframe: A pandas ``DataFrame``
        :param group_colname: Name for the group-column name.
        :param time_colname: Name for the time-column name.
        :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
         used -- i.e., how far we advance with every timestep. Can be `None` if the data are in arbitrary (non-datetime)
         units.
        :param measure_colnames: A list of names of columns that include the actual time-series data in the dataframe.
         Optional if `X_colnames` and `y_colnames` are passed.
        :param X_colnames: In many settings we have a set of columns corresponding to predictors and a set of columns
         corresponding to the actual time-series data. The former should be passed as `X_colnames` and the latter as
         `y_colnames`.
        :param y_colnames: See above.
        :param pad_X: When stacking time-serieses of unequal length, we left-align them and so get trailing missings.
         Setting ``pad_X`` allows you to select the padding value for these. Default 0-padding.
        :return: A :class:`TimeSeriesDataset`.
        """

        if X_colnames is not None:
            X_colnames = list(X_colnames)
        if y_colnames is not None:
            y_colnames = list(y_colnames)
        if measure_colnames is not None:
            measure_colnames = list(measure_colnames)

        if measure_colnames is None:
            if y_colnames is None or X_colnames is None:
                raise ValueError("Must pass either `measure_colnames` or `X_colnames` & `y_colnames`")
            if isinstance(y_colnames, str):
                y_colnames = [y_colnames]
                warn(f"`y_colnames` should be a list of strings not a string; interpreted as `{y_colnames}`.")

            measure_colnames = list(y_colnames) + list(X_colnames)
        else:
            if X_colnames is not None or y_colnames is not None:
                raise ValueError("If passing `measure_colnames` do not pass `X_colnames` or `y_colnames`.")
            if isinstance(measure_colnames, str):
                measure_colnames = [measure_colnames]
                warn(f"`measure_colnames` should be list of strings not a string; interpreted as `{measure_colnames}`.")

        if group_colname is None:
            dataframe = dataframe.assign(_group=1)
            group_colname = '_group'
        assert isinstance(group_colname, str)
        assert isinstance(time_colname, str)
        assert len(measure_colnames) == len(set(measure_colnames))

        # sort by time:
        dataframe = dataframe.sort_values(time_colname)

        for measure_colname in measure_colnames:
            if measure_colname not in dataframe.columns:
                raise ValueError(f"'{measure_colname}' not in dataframe.columns:\n{dataframe.columns}'")

        # first pass for info:
        arrays, time_idxs, group_names, start_times = [], [], [], []
        for g, df in dataframe.groupby(group_colname, sort=True):
            # group-names:
            group_names.append(g)

            # times:
            times = df[time_colname].values
            assert len(times) == len(set(times)), f"Group {g} has duplicate times"
            min_time = times[0]
            start_times.append(min_time)
            if dt_unit is None:
                time_idx = (times - min_time).astype('int64')
            else:
                if not isinstance(dt_unit, np.timedelta64):
                    dt_unit = np.timedelta64(1, dt_unit)
                time_idx = (times - min_time) // dt_unit
            time_idxs.append(time_idx)

            # values:
            arrays.append(df.loc[:, measure_colnames].values)

        # second pass organizes into tensor
        time_len = max(time_idx[-1] + 1 for time_idx in time_idxs)
        tens = torch.empty((len(arrays), time_len, len(measure_colnames)), dtype=dtype)
        tens[:] = float('nan')
        for i, (array, time_idx) in enumerate(zip(arrays, time_idxs)):
            tens[i, time_idx, :] = torch.tensor(array, dtype=dtype)
        tens = tens.to(device=device)

        dataset = cls(
            tens,
            group_names=group_names,
            start_times=start_times,
            measures=[measure_colnames],
            dt_unit=dt_unit
        )

        if X_colnames is not None:
            dataset = dataset.split_measures(y_colnames, X_colnames)
            y, X = dataset.tensors
            if isinstance(pad_X, str) and pad_X == 'ffill':
                for i, time_idx in enumerate(time_idxs):
                    max_idx = time_idx.max()
                    X[i, (max_idx + 1):, :] = X[i, max_idx, :]
            elif pad_X is not None:
                for i, time_idx in enumerate(time_idxs):
                    pad_idx = time_idx.max() + 1
                    X[i, pad_idx:, :] = pad_X

        return dataset

    def with_new_tensors(self, *tensors: Tensor) -> 'TimeSeriesDataset':
        """
        Create a new Batch with a different Tensor, but all other attributes the same.
        """
        return type(self)(
            *tensors,
            group_names=self.group_names,
            start_times=self.start_times,
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    # Util/Private ------------------------
    def times(self, which: Optional[int] = None) -> np.ndarray:
        """
        A 2D array of datetimes (or integers if dt_unit is None) for this dataset.

        :param which: If this dataset has multiple tensors of different number of timesteps, which should be used for
         constructing the `times` array? Defaults to the one with the most timesteps.
        :return: A 2D numpy array of datetimes (or integers if dt_unit is None).
        """
        return self.get_dataset_times(
            start_times=self.start_times,
            num_timesteps=self.num_timesteps if which is None else self.tensors[which].shape[1],
            dt_unit=self.dt_unit
        )

    @classmethod
    def get_dataset_times(cls, start_times: np.ndarray, num_timesteps: int, dt_unit: Optional[np.timedelta64]):
        offsets = np.arange(0, num_timesteps) * (dt_unit if dt_unit else 1)
        return start_times[:, None] + offsets

    def datetimes(self) -> np.ndarray:
        return self.times()

    @property
    def start_datetimes(self) -> np.ndarray:
        return self.start_times

    @property
    def start_offsets(self) -> np.ndarray:
        return self.start_times

    @torch.no_grad()
    def get_durations(self, which: int = 0) -> np.ndarray:
        """
        Get an array (algined with self.group_names) with the number of 'duration' for each group, defined as the
        number of timesteps until the last measurement (i.e. the last timestep after which all measures are `nan`).

        Since TimeSeriesDatasets are padded, this can be a helpful way to get the length of each time-series.
        """
        any_measured_bool = ~torch.isnan(self.tensors[which]).all(2).cpu()
        last_measured_idx = np.array(
            [np.max(true1d_idx(any_measured_bool[g]).numpy(), initial=0) for g in range(len(self.group_names))],
            dtype='int'
        )
        return last_measured_idx + 1

    @torch.no_grad()
    def trimmed(self, using: int = 0) -> 'TimeSeriesDataset':
        """
        Return a new TimeSeriesDataset with unneeded padding removed. This is useful if we've subset a dataset and the
        remaining time-serieses are all shorter than the previous longest time-series' length.

        For example, this method combined with ``get_durations()`` can be helpful for splitting a single dataset with
        time-series of heterogeneous lengths into multiple datasets:

        >>> ds_all = TimeSeriesDataset(x, group_names=group_names, start_times=start_dts)
        >>> durations = ds_all.get_durations()
        >>> ds_long = ds_all[durations >= 8784]  # >= a year
        >>> ds_short = ds_all[durations < 8784].trimmed()  # shorter than a year
        >>> assert ds_short.tensors.shape[0] < ds_long.tensors.shape[0]
        """
        last_dur = self.get_durations(using).max()
        new_tensors = [tens[:, :last_dur] for i, tens in enumerate(self.tensors)]
        return self.with_new_tensors(*new_tensors)


class TimeSeriesDataLoader(DataLoader):
    """
    A ``DataLoader`` that  produces ``TimeSeriesDataset``s. Generally should be created with the ``from_dataframe()``
    classmethod.
    """

    def __init__(self,
                 dataset: '_DataFrameGroupByDataset',
                 batch_size: Optional[int],
                 **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self._collate,
            **kwargs
        )

    @staticmethod
    def _collate(batch: Sequence['TimeSeriesDataset']) -> 'TimeSeriesDataset':
        if len(batch) == 1:
            return batch[0]
        raise NotImplementedError

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: str,
                       time_colname: str,
                       dt_unit: Optional[str],
                       measure_colnames: Optional[Sequence[str]] = None,
                       X_colnames: Optional[Union[Sequence[str], Callable]] = None,
                       y_colnames: Optional[Sequence[str]] = None,
                       pad_X: Union[float, str, None] = 'ffill',
                       dtype: torch.dtype = torch.float32,
                       device: Optional[torch.device] = None,
                       **kwargs) -> 'TimeSeriesDataLoader':
        """
        :param dataframe: A pandas ``DataFrame``
        :param group_colname: Name for the group-column name.
        :param time_colname: Name for the time-column name.
        :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
         used -- i.e., how far we advance with every timestep. Can be `None` if the data are in arbitrary (non-datetime)
         units.
        :param measure_colnames: A list of names of columns that include the actual time-series data in the dataframe.
         Optional if `X_colnames` and `y_colnames` are passed.
        :param X_colnames: Column-names for predictors; or a function that takes the original dataframe and returns a
         dataframe with the same number of rows which contains predictors (and only predictors). The latter option is
         useful so that memory-heavy transforms can be applied just as the batch is yielded, instead of to the whole
         dataset.
        :param y_colnames: Alias for ``measure_colnames``; preferred if ``X_colnames`` is used.
        :param pad_X: When stacking time-serieses of unequal length, we left-align them and so get trailing missings.
         Setting ``pad_X`` allows you to select the padding value for these. Default 0-padding.
        :param dtype: The dtype for returned tensors.
        :param device: The device for returned tensors. Note that if you are working with a GPU, it may be more
         efficient to leave this parameter as-is, then send each batch to the GPU within your training loop.
        :return: An iterable that yields :class:`TimeSeriesDataset`.
        """

        return cls(
            dataset=_DataFrameGroupByDataset(
                df=dataframe,
                group_colname=group_colname,
                time_colname=time_colname,
                dt_unit=dt_unit,
                y_colnames=measure_colnames or y_colnames,
                X_colnames=X_colnames,
                pad_X=pad_X,
                device=device,
                dtype=dtype
            ),
            **kwargs
        )


class _DataFrameGroupByDataset(Dataset):
    """
    Util class for ``TimeSeriesDataLoader``, to allow lazy (and so memory-efficient) transformations to the data before
    converting to a ``TimeSeriesDataset``.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 group_colname: str,
                 time_colname: str,
                 dt_unit: Optional[str],
                 y_colnames: Sequence[str],
                 X_colnames: Optional[Union[Sequence[str], Callable]],
                 pad_X: Union[float, str, None] = 'ffill',
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None):
        self.group_colname = group_colname
        self.time_colname = time_colname
        self.dt_unit = dt_unit
        self.y_colnames = y_colnames
        self.X_colnames = X_colnames
        self.pad_X = pad_X
        self.dtype = dtype
        self.device = device
        self.group_dfs = {g: dfg for g, dfg in df.groupby(self.group_colname, sort=False)}
        self.groups = df[group_colname].unique()

    def __len__(self) -> int:
        return len(self.group_dfs)

    def __getitem__(self, idx) -> TimeSeriesDataset:
        raise NotImplementedError

    def __getitems__(self, indices: list) -> Sequence[TimeSeriesDataset]:
        group_names = [self.groups[idx] for idx in indices]
        df_group = pd.concat([self.group_dfs[group_name] for group_name in group_names]).reset_index(drop=True)

        # extract/create model-matrix:
        X_colnames = dfX = None
        if callable(self.X_colnames):
            dfX = self.X_colnames(df_group)
            if not dfX.index.equals(df_group.index):
                raise ValueError("`X_colnames` function must return a dataframe with the same index as the input.")
        elif self.X_colnames:
            dfX = df_group[self.X_colnames]

        df = df_group[[self.group_colname, self.time_colname] + list(self.y_colnames)]
        if dfX is not None:
            X_colnames = dfX.columns.tolist()
            df = pd.concat([df, dfX], axis=1)

        # return a batch of size 1 since this is what collate_fn expects, but basically it's already pre-collated
        return [TimeSeriesDataset.from_dataframe(
            df,
            group_colname=self.group_colname,
            time_colname=self.time_colname,
            dt_unit=self.dt_unit,
            y_colnames=self.y_colnames,
            X_colnames=X_colnames,
            pad_X=self.pad_X,
            dtype=self.dtype,
            device=self.device
        )]


def complete_times(data: 'DataFrame',
                   group_colnames: Sequence[str] = None,
                   time_colname: Optional[str] = None,
                   dt_unit: Optional[str] = None,
                   max_dt_colname: Optional[str] = None,
                   global_max: Union[bool, datetime.datetime] = False,
                   group_colname: Optional[str] = None):
    """
    Given a dataframe time-serieses, convert implicit missings within each time-series to explicit missings.

    :param data: A pandas dataframe.
    :param group_colnames: The column name(s) for the groups.
    :param time_colname: The column name for the times. Will attempt to guess based on common labels.
    :param dt_unit: Passed to ``pandas.date_range``. If not passed, will attempt to guess based on the minimum
     difference between times.
    :param max_dt_colname: Optional, a column-name that indicates the maximum time for each group. If not supplied, the
     actual maximum time for each group will be used.
    :return: A dataframe where implicit missings are converted to explicit missings, but the min/max time for each
     group is preserved.
    """
    import pandas as pd

    if isinstance(group_colnames, str):
        group_colnames = [group_colnames]
    elif group_colnames is None:
        if group_colname is None:
            raise TypeError("Missing required argument `group_colnames`")
        warn("Please pass `group_colnames` instead of `group_colname`", DeprecationWarning)
        group_colnames = [group_colname]
    if max_dt_colname and max_dt_colname not in group_colnames:
        assert (data.groupby(group_colnames)[max_dt_colname].nunique() == 1).all()
        group_colnames.append(max_dt_colname)

    if time_colname is None:
        for col in ('datetime', 'date', 'timestamp', 'time', 'dt'):
            if col in data.columns:
                time_colname = col
                break
        if time_colname is None:
            raise ValueError("Unable to guess `time_colname`, please pass")
    if dt_unit is None:
        diffs = data[time_colname].drop_duplicates().sort_values().diff()
        dt_unit = diffs.min()
        if dt_unit != diffs.value_counts().index[0]:
            raise ValueError("Unable to guess dt_unit, please pass")
    elif dt_unit == 'W':
        # pd.date_range behaves oddly with freq='W'
        # (e.g. does not match behavior of `my_dates.to_period('W').dt.to_timestamp()`)
        dt_unit = pd.Timedelta('7 days 00:00:00')

    if global_max:
        warn("`global_max=True` is deprecated, use `max_dt_colname` instead.", DeprecationWarning)

    df_group_summary = (data
                        .groupby(group_colnames)
                        .agg(_min=(time_colname, 'min'), _max=(time_colname, 'max'))
                        .reset_index())
    if max_dt_colname:
        df_group_summary['_max'] = df_group_summary[max_dt_colname]

    max_of_maxes = df_group_summary['_max'].max()

    df_grid = pd.DataFrame({time_colname: pd.date_range(data[time_colname].min(), max_of_maxes, freq=dt_unit)})

    # cross-join for all times to all groups (todo: not very memory efficient)
    df_cj = df_grid.merge(df_group_summary, how='cross')
    # filter to min/max for each group
    df_cj = (df_cj
             .loc[df_cj[time_colname].between(df_cj['_min'], df_cj['_max']), group_colnames + [time_colname]]
             .reset_index(drop=True))
    return df_cj.merge(data, how='left', on=group_colnames + [time_colname])
