from functools import cached_property

from typing import TYPE_CHECKING, Sequence, Optional, Union, Collection, Iterable

import torch

from torchcast.internals.utils import update_tensor, normalize_index, compute_index_result_shape

if TYPE_CHECKING:
    from torchcast.process import Process


class DesignModel:
    def __init__(self,
                 processes: torch.nn.ModuleDict,
                 num_groups: int,
                 num_timesteps: int):
        self.num_groups = num_groups
        self.num_timesteps = num_timesteps
        self.processes: dict[str, 'Process'] = processes
        self._cache_per_process = {}

    @property
    def device(self) -> torch.device:
        device = None
        for param in self.processes.parameters():
            if device is None:
                device = param.device
            elif device != param.device:
                raise RuntimeError("Multiple devices!")
        return device

    def __call__(self,
                 mean: torch.Tensor,
                 time: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @cached_property
    def rank(self) -> int:
        return sum(p.rank for p in self.processes.values())

    @cached_property
    def process2slice(self) -> dict[str, slice]:
        """
        Returns a mapping from process id to the slice of the state vector that contains its state elements.
        """
        start_ = 0
        process2slice = {}
        for pid, process in self.processes.items():
            end_ = start_ + process.rank
            process2slice[pid] = slice(start_, end_)
            start_ = end_
        return process2slice


class MeasurementModel(DesignModel):

    def __init__(self,
                 processes: torch.nn.ModuleDict,
                 measures: Sequence[str],
                 num_groups: int,
                 num_timesteps: int,
                 measure_funs: dict[str, callable] = None,
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
        if self.measure_funs:
            raise NotImplementedError("TODO")

        self._kwargs_per_process, self.used_keys = self._get_kwargs_per_process(**kwargs)

    @property
    def is_nonlinear(self) -> bool:
        return bool(self.nonlinear_processes) or self.measure_funs

    @property
    def nonlinear_processes(self) -> list['Process']:
        return [p for p in self.processes.values() if not p.linear_measurement]

    def __call__(self,
                 mean: torch.Tensor,
                 time: int,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        measure_mat = self._get_linear_measure_mat(time)
        measured_mean = (measure_mat @ mean.unsqueeze(-1)).squeeze(-1)

        nl_procs_and_means = list(self._get_nonlinear_processes_and_means(mean))

        measured_mean = measured_mean + self.get_measured_mean_adjustment(nl_procs_and_means, time=time)
        measure_mat = measure_mat + self._get_measure_mat_adjustment(nl_procs_and_means, time)

        return measured_mean, measure_mat

    def _get_linear_measure_mat(self, time: int) -> torch.Tensor:
        assert time >= 0
        return self._measure_mats[time]  # [..., measure_idx, :]

    def _get_nonlinear_processes_and_means(self, mean: torch.Tensor) -> Iterable[tuple['Process', torch.Tensor]]:
        """
        Returns an iterable of tuples (process, mean) for each process in this model.
        """
        for process in self.nonlinear_processes:
            midx = self.measure2idx.get(process.measure, None)
            if midx is None:
                continue  # see note in _initialize_measure_mats()
            pidx = self.process2slice[process.id]
            yield process, mean[..., pidx, midx]

    def get_measured_mean_adjustment(self,
                                     nl_processes_and_means: Iterable[tuple['Process', torch.Tensor]],
                                     time: int) -> torch.Tensor:
        adjustment = 0
        for process, mean in nl_processes_and_means:
            midx = self.measure2idx.get(process.measure, None)
            if midx is None:
                continue  # see note in _initialize_measure_mats()
            this_mm = torch.zeros((self.num_groups, self.num_timesteps, len(self.measures)), device=mean.device)
            this_mm[..., midx] = process.get_measured_mean(
                mean,
                time=time,
                cache=self._cache_per_process[process.id]
            )
            adjustment = adjustment + this_mm

        # TODO: measure-wide adjustments (e.g. sigmoid transform that is post H-dot-state) go here

        return adjustment

    def _get_measure_mat_adjustment(self,
                                    processes_and_means: Iterable[tuple['Process', torch.Tensor]],
                                    time: int) -> torch.Tensor:
        adjustment = 0
        for process, mean in processes_and_means:
            midx = self.measure2idx[process.measure]
            pidx = self.process2slice[process.id]
            jacobian = process.get_measurement_jacobian(mean, time, self._cache_per_process[process.id])
            pH = torch.zeros((self.num_groups, self.num_timesteps, len(self.measures), self.rank), device=mean.device)
            pH[..., midx, pidx] = jacobian
            adjustment = adjustment + pH
        return adjustment

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
        for pid, process in self.processes.items():
            # TODO: support {pid}__{kwargs} syntax
            kwargs_per_process[pid] = {k.name: kwargs[k.name] for k in process.measurement_kwargs}
            used.update(set(kwargs_per_process[pid]))
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
        H = torch.zeros((self.num_groups, self.num_timesteps, len(self.measures), self.rank), device=self.device)
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
                raise ValueError(f"for process {pid}, measurement matrix expected to be a vector or have shape"
                                 f"(num_groups, num_times, rank). Instead got {value.shape}. ")
            # todo is all this masking inefficient?
            H[:, :, midx, pidx] = value

        return H.unbind(1)

    def _copy(self,
              item: Optional[tuple[torch.Tensor, ...]] = None,
              measures: Optional[Union[Collection[str], torch.Tensor]] = None,
              flattened: bool = False) -> 'MeasurementModel':

        num_groups = self.num_groups
        num_timesteps = self.num_timesteps
        if item is not None and flattened:
            raise ValueError("Cannot pass both `group_idx` and `flattened` arguments at the same time.")
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


class TransitionModel(DesignModel):
    def __init__(self,
                 processes: torch.nn.ModuleDict,
                 measures: Sequence[str],
                 num_groups: int,
                 num_timesteps: int):
        super().__init__(
            processes=processes,
            num_groups=num_groups,
            num_timesteps=num_timesteps
        )
        self.measures = measures

        zeros = torch.zeros((self.num_groups, self.num_timesteps, self.rank, self.rank), device=self.device)
        F = []
        for pid, process in self.processes.items():
            if process.linear_transition:
                pidx = self.process2slice[pid]
                # note: as in other parts, assuming autograd makes it more efficient to create clones then sum vs.
                # repeated masks on the same tensor. should verify that
                thisF = zeros.clone()
                thisF[:, :, pidx, pidx] = process.get_transition_matrix()
                F.append(thisF)
            else:
                raise NotImplementedError
        self.transition_mats = torch.stack(F, dim=0).sum(0).unbind(1)

    def __call__(self,
                 mean: torch.Tensor,
                 time: int,
                 mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert time >= 0
        if mask is None or mask.all():
            mask = slice(None)
        F = self.transition_mats[time]
        new_mean = update_tensor(mean, new=(F[mask] @ mean[mask].unsqueeze(-1)).squeeze(-1), mask=mask)
        return new_mean.squeeze(-1), F
