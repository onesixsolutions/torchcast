from .design_model import DesignModel
from .measure_funs import MeasureFun

from functools import cached_property

from typing import TYPE_CHECKING, Sequence, Optional, Union, Collection, Iterable

import torch

from torchcast.internals.utils import normalize_index, compute_index_result_shape

if TYPE_CHECKING:
    from torchcast.process import Process


class MeasurementModel(DesignModel):

    def __init__(self,
                 processes: torch.nn.ModuleDict,
                 measures: Sequence[str],
                 num_groups: int,
                 num_timesteps: int,
                 measure_funs: dict[str, MeasureFun] = None,
                 **kwargs):
        super().__init__(
            processes=processes,
            num_groups=num_groups,
            num_timesteps=num_timesteps
        )
        self.measures = measures
        if measure_funs is None:
            measure_funs = {}
        self.measure_funs = measure_funs

        self._kwargs_per_process, self.used_keys = self._get_kwargs_per_process(**kwargs)

        self._extended_mmat_slices = None

    @property
    def is_nonlinear(self) -> bool:
        return bool(self.nonlinear_processes) or self.measure_funs

    @cached_property
    def nonlinear_processes(self) -> list['Process']:
        return [p for p in self.processes.values() if not p.linear_measurement]

    def __call__(self,
                 mean: torch.Tensor,
                 time: int,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        measure_mat = self._get_linear_measure_mat(time)
        measured_mean = (measure_mat @ mean.unsqueeze(-1)).squeeze(-1)

        nl_procs_and_means = list(self._get_nonlinear_processes_and_means(mean))

        if self.is_nonlinear:
            measured_mean = self.adjust_measured_mean(measured_mean, nl_procs_and_means, time)
            measure_mat = self._adjust_measure_mat(measure_mat, nl_procs_and_means, measured_mean, time)

        return measured_mean, measure_mat

    @cached_property
    def extended_measure_mat(self) -> torch.Tensor:
        if self.num_timesteps != 1:
            raise ValueError("Can only get extended measure-mat for flattened measurement-model.")

        linear_mmat = self._get_linear_measure_mat(0)

        nonlinear_rank = sum(p.rank for p in self.nonlinear_processes)
        extension = torch.zeros(
            (self.num_groups, nonlinear_rank, self.state_rank),
            dtype=self.dtype,
            device=self.device
        )
        if nonlinear_rank:
            extension[:, :, -nonlinear_rank:] = torch.eye(
                nonlinear_rank, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
        return torch.cat([linear_mmat, extension], dim=1)

    @property
    def extended_mmat_slices(self) -> dict[str, slice]:
        if self._extended_mmat_slices is None:
            self._extended_mmat_slices = {}
            start_ = len(self.measures)
            for process in self.nonlinear_processes:
                self._extended_mmat_slices[process.id] = slice(start_, start_ + process.rank)
                start_ += process.rank
        return self._extended_mmat_slices

    def _get_linear_measure_mat(self, time: int) -> torch.Tensor:
        assert time >= 0
        return self._measure_mats[time]

    def _get_nonlinear_processes_and_means(self, mean: torch.Tensor) -> Iterable[tuple['Process', torch.Tensor]]:
        """
        Returns an iterable of tuples (process, mean) for each process in this model.
        """
        for process in self.nonlinear_processes:
            pidx = self.process2slice[process.id]
            yield process, mean[..., pidx]

    def adjust_measured_mean(self,
                             linear_measured_mean: torch.Tensor,
                             nl_processes_and_means: Iterable[tuple['Process', torch.Tensor]],
                             time: int) -> Union[torch.Tensor, float]:
        # process-level adjustments:
        out = linear_measured_mean.clone()
        for pid, this_mm in self._get_measured_mean_adjustments(nl_processes_and_means, time):
            out = out + this_mm

        # measure-wide adjustments:
        if self.measure_funs:
            out = self._get_measure_wide_adjustments(out)
        return out

    def _get_measure_wide_adjustments(self, measured_mean: torch.Tensor) -> torch.Tensor:
        assert self.measure_funs
        measured_mean = list(measured_mean.unbind(-1))
        for i, measure in enumerate(self.measures):
            if measure in self.measure_funs:
                measured_mean[i] = self.measure_funs[measure](measured_mean[i])
        return torch.stack(measured_mean, dim=-1)

    def _get_measured_mean_adjustments(self,
                                       nl_processes_and_means: Iterable[tuple['Process', torch.Tensor]],
                                       time: int) -> Iterable[tuple[str, torch.Tensor]]:
        for process, mean in nl_processes_and_means:
            midx = self.measure2idx.get(process.measure, None)
            if midx is None:
                continue  # see note in _measure_mats
            this_mm = torch.zeros((self.num_groups, len(self.measures)), device=mean.device, dtype=mean.dtype)
            this_mm[..., midx] = process.get_measured_mean(
                mean,
                time=time,
                cache=self._cache_per_process[process.id]
            )
            yield process.id, this_mm

    def _adjust_measure_mat(self,
                            measure_mat: torch.Tensor,
                            processes_and_means: Iterable[tuple['Process', torch.Tensor]],
                            measured_mean: torch.Tensor,
                            time: int) -> torch.Tensor:
        to_kwargs = {'device': measured_mean.device, 'dtype': measured_mean.dtype}
        for process, mean in processes_and_means:
            midx = self.measure2idx.get(process.measure)
            if midx is None:
                continue  # see note in _measure_mats
            pidx = self.process2slice[process.id]
            jacobian = process.get_measurement_jacobian(mean, time, self._cache_per_process[process.id])
            pH = torch.zeros((self.num_groups, len(self.measures), self.state_rank), **to_kwargs)
            pH[..., midx, pidx] = jacobian
            measure_mat = measure_mat + pH

        if not self.measure_funs:
            return measure_mat

        measured_mean = measured_mean.unbind(-1)
        measure_mat = list(measure_mat.unbind(-2))
        for i, measure in enumerate(self.measures):
            if measure not in self.measure_funs:
                continue
            # apply measure-wide adjustment
            measure_mat[i] = self.measure_funs[measure].adjust_measure_mat(measure_mat[i], measured_mean[i])

        return torch.stack(measure_mat, dim=-2)

    @cached_property
    def measure2idx(self) -> dict[str, int]:
        return {m: i for i, m in enumerate(self.measures)}

    def flattened(self) -> 'MeasurementModel':
        return self._copy(flattened=True)

    def subset(self, *item, measures: Optional[Union[Sequence[str], torch.Tensor]] = None) -> 'MeasurementModel':
        return self._copy(item=item, measures=measures)

    def _get_kwargs_per_process(self, **kwargs) -> tuple[dict[str, dict], set]:
        used = set()
        kwargs_per_process = {}
        for pid, proc in self.processes.items():
            keys = {
                k.name: (f'{pid}__{k.name}' if f'{pid}__{k.name}' in kwargs else k.name)
                for k in proc.measurement_kwargs
            }
            kwargs_per_process[pid] = {k1: kwargs[k2] for k1, k2 in keys.items()}
            used.update(set(keys.values()))
        return kwargs_per_process, used

    @cached_property
    def _cache_per_process(self) -> dict[str, dict]:
        cache_per_process = {}
        for pid, process in self.processes.items():
            if process.linear_measurement:
                continue
            cache_per_process[pid] = process.prepare_measurement_cache(**self._kwargs_per_process[pid])
        return cache_per_process

    @cached_property
    def _measure_mats(self) -> Sequence[torch.Tensor]:
        is_time_varying = any(p.measurement_kwargs for p in self.processes.values())
        n_times = self.num_timesteps if is_time_varying else 1
        H = torch.zeros(
            (self.num_groups, n_times, len(self.measures), self.state_rank),
            device=self.device,
            dtype=self.dtype
        )
        for pid, process in self.processes.items():
            midx = self.measure2idx.get(process.measure, None)
            if not process.linear_measurement or midx is None:
                # we can have a process whose measure isn't in this model if this model is a subset of another.
                # this would happen if we had to drop that measure b/c it was nan
                continue
            pidx = self.process2slice[pid]
            value = process.get_measurement_matrix(**self._kwargs_per_process[pid])
            if len(value.shape) == 1:
                value = value.unsqueeze(0).unsqueeze(0)
            elif len(value.shape) != 3:
                assert not is_time_varying
                raise ValueError(f"for process {pid}, measurement matrix expected to be a vector or have shape"
                                 f"(num_groups, num_times, rank). Instead got {value.shape}. ")
            H[:, :, midx, pidx] = value

        if not is_time_varying:
            # much faster for backward-step
            H0 = H.squeeze(1)
            return [H0] * self.num_timesteps

        return H.unbind(1)

    def _copy(self,
              item: Optional[tuple[torch.Tensor, ...]] = None,
              measures: Optional[Union[Collection[str], torch.Tensor]] = None,
              flattened: bool = False) -> 'MeasurementModel':

        num_groups = self.num_groups
        num_timesteps = self.num_timesteps
        if item is not None and flattened:
            raise ValueError("Cannot pass both `item` and `flattened` arguments at the same time.")
        elif flattened:
            num_groups = self.num_groups * self.num_timesteps
            num_timesteps = 1
        elif item is not None:
            item = normalize_index(item)
            if any(isinstance(i, int) for i in item):
                raise ValueError("Cannot drop a dimension, but received an integer index. ")
            num_groups, num_timesteps, *other = compute_index_result_shape(item, (self.num_groups, self.num_timesteps))
            if other:
                raise ValueError(f"{type(self).__name__} only supports indexing the first two dimensions")

        if measures is None:
            measures = self.measures
        else:
            if isinstance(measures, torch.Tensor):  # support tensor of integers like what get_nan_groups returns
                measures = [self.measures[m] for m in measures]
            assert set(self.measures).issuperset(measures)
            measures = [m for m in self.measures if m in measures]  # preserve order

        # create a new instance, but without processes, so that _get_kwargs_per_process isn't called
        new = type(self)(
            processes={},  # noqa
            measures=measures,
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            measure_funs={m: self.measure_funs[m] for m in measures if m in self.measure_funs}
        )

        # reshape/subset any group-time tensors as needed, to manually set the _kwargs_per_process attr
        new._kwargs_per_process = {}
        for pid, pkwargs in self._kwargs_per_process.items():
            new._kwargs_per_process[pid] = {}
            for pkwarg in self.processes[pid].measurement_kwargs:
                value = pkwargs[pkwarg.name]
                if pkwarg.is_group_time_tensor:
                    value = value.reshape(num_groups, num_timesteps, *value.shape[2:]) if flattened else value[item]
                new._kwargs_per_process[pid][pkwarg.name] = value
        new.used_keys = self.used_keys
        new.processes = self.processes
        return new

    @torch.no_grad()
    def get_components(self,
                       mean: torch.Tensor,
                       time: Optional[int] = None,
                       measured: bool = True) -> Iterable[tuple[str, str, torch.Tensor]]:
        """
        :param mean: The mean state vector at the given time step.
        :param time: The time step for which to get the components.
        :param measured: If False, returns the underlying state-means; if True, first applies the measurement-matrix
         elementwise to the state-means, so as to show the contribution of each to the measured mean. For processes
         that do not have a linear measurement model, this isn't possible so the process is combined into a single
         state-element in the output (called '__combined__').
        """
        if time is None:
            if self.num_timesteps != 1:
                raise ValueError("Must specify `time`.")
            time = 0

        mmean_adjustments = {}
        if measured:
            mmean_adjustments.update(
                self._get_measured_mean_adjustments(self._get_nonlinear_processes_and_means(mean), time)
            )

        measure_mat = self._get_linear_measure_mat(time)
        for pid, process in self.processes.items():
            midx = self.measure2idx.get(process.measure, None)
            if midx is None:
                continue  # see note in _measure_mats
            pidx = self.process2slice[pid]

            state_elements = [se.name for se in process.state_elements.values()]
            proc_mean = mean[..., pidx]
            if measured:
                if process.linear_measurement:
                    h = measure_mat[..., midx, pidx]
                    proc_mean = proc_mean * h
                    # drop state-elements that are not observable:
                    mask = (h != 0).any(dim=0)
                    proc_mean = proc_mean[..., mask]
                    state_elements = [s for i, s in enumerate(state_elements) if mask[i]]
                else:
                    state_elements = ['__combined__']
                    proc_mean = mmean_adjustments[pid].sum(-1, keepdim=True)
            assert len(state_elements) == proc_mean.shape[1]
            for se, se_mean in zip(state_elements, proc_mean.T):
                yield pid, se, se_mean
