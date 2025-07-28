import math
from typing import Optional, Tuple, Sequence, Union
from warnings import warn

import numpy as np

import torch

from torchcast.internals.utils import update_tensor
from torchcast.process.process import Process
from torchcast.process.utils import Multi, standardize_decay, StateElement, NoInputSequential


class Season(Process):
    """
    Method from `De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011)`, specifically the novel approach to modeling
    seasonality that they proposed.

    :param id: Unique identifier for this process.
    :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
     used in the kalman-filter -- i.e., how far we advance with every timestep. Can be `None` if the data are in
     arbitrary (non-datetime) units.
    :param period: The number of timesteps it takes to get through a full seasonal cycle. Does not have to be an
     integer (e.g. 365.25 for yearly to account for leap-years). Can also be a ``numpy.timedelta64`` (or string that
     will be converted to one).
    :param K: The number of the fourier components.
    :param measure: The name of the measure for this process.
    :param fixed: Whether the seasonal-structure is allowed to evolve over time, or is fixed (default:
     ``fixed=False``). Setting this to ``True`` can be helpful for limiting the uncertainty of long-range forecasts.
    :param decay: By default, the seasonal structure will remain as the forecast horizon increases. An alternative is
     to allow this structure to decay (i.e. pass ``True``). If you'd like more fine-grained control over this decay,
     you can specify the min/max decay as a tuple (passing ``True`` uses a default value of ``(.98, 1.0)``).
    """

    def __init__(self,
                 id: str,
                 period: Union[float, str],
                 dt_unit: Optional[str],
                 K: int,
                 measure: Optional[str] = None,
                 fixed: bool = False,
                 decay: Optional[Tuple[float, float]] = None):

        self.dt_unit_ns = None if dt_unit is None else self._get_dt_unit_ns(dt_unit)
        self.period = self._standardize_period(period, self.dt_unit_ns)
        if self.period.is_integer() and self.period < K * 2:
            warn(f"K is larger than necessary given a period of {self.period}.")

        if isinstance(decay, tuple) and (decay[0] ** self.period) < .01:
            warn(
                f"Given the seasonal period, the lower bound on `{id}`'s `decay` ({decay}) may be too low to "
                f"generate useful gradient information for optimization."
            )
        decay = standardize_decay(decay)

        super().__init__(
            id=id,
            state_elements=self._init_state_elements(K=K, period=self.period, decay=decay, fixed=fixed),
            measure=measure
        )

    @property
    def dt_unit(self) -> Optional[np.timedelta64]:
        if self.dt_unit_ns is None:
            return None
        return np.timedelta64(self.dt_unit_ns, 'ns')  # todo: promote

    @staticmethod
    def _standardize_period(period: Union[str, np.timedelta64], dt_unit_ns: Optional[float]) -> float:
        if dt_unit_ns is None:
            if not isinstance(period, (float, int)):
                raise ValueError(f"period is {type(period)}, but expected float/int since dt_unit is None.")
        else:
            if not isinstance(period, (float, int)):
                if isinstance(period, str):
                    period = np.timedelta64(1, period)
                period = period / (dt_unit_ns * np.timedelta64(1, 'ns'))
        return float(period)

    @staticmethod
    def _get_dt_unit_ns(dt_unit_str: str) -> int:
        if isinstance(dt_unit_str, np.timedelta64):
            dt_unit = dt_unit_str
        else:
            dt_unit = np.timedelta64(1, dt_unit_str)
        dt_unit_ns = dt_unit / np.timedelta64(1, 'ns')
        assert dt_unit_ns.is_integer()
        return int(dt_unit_ns)

    def _standardize_offsets(self, offsets: np.ndarray) -> np.ndarray:
        if self.dt_unit_ns is None:
            return np.asanyarray(offsets) % self.period
        offsets = np.asanyarray(offsets, dtype='datetime64[ns]')
        ns_since_epoch = (offsets - np.datetime64(0, 'ns')).view('int64')
        offsets = ns_since_epoch % (self.period * self.dt_unit_ns) / self.dt_unit_ns  # todo: cancels out?
        return offsets

    @staticmethod
    def _init_state_elements(K: int,
                             period: float,
                             decay: Union[float, torch.nn.Module],
                             fixed: bool) -> Sequence[StateElement]:

        state_elements = []
        for j in range(1, K + 1):
            sj = StateElement(
                name=f"s{j}",
                measure_multi=1.0,
                has_process_variance=not fixed
            )
            s_star_j = StateElement(
                name=f"s*{j}",
                measure_multi=0.,
                has_process_variance=not fixed
            )

            lam = torch.tensor(2. * math.pi * j / period)
            _transitions = {
                sj: {
                    sj: torch.cos(lam),
                    s_star_j: -torch.sin(lam)
                },
                s_star_j: {
                    sj: torch.sin(lam),
                    s_star_j: torch.cos(lam)
                }
            }
            # todo: previously supported different decays per element, any need for that?
            for se_from, se_from_transitions in _transitions.items():
                for se_to, multi in se_from_transitions.items():
                    if isinstance(decay, torch.nn.Module):
                        multi = NoInputSequential(decay, Multi(multi))
                    else:
                        multi = decay * multi
                    se_from.set_transition_to(se_to, multi=multi)
                state_elements.append(se_from)
        return state_elements

    def get_initial_mean(self, start_offsets: np.ndarray) -> torch.Tensor:
        if start_offsets is None:
            raise RuntimeError(f"Process '{self.id}' requires `start_offsets`")

        start_offsets = self._standardize_offsets(start_offsets)
        # TODO: this is imprecise for non-integer periods
        start_offsets = start_offsets.round()
        num_groups = len(start_offsets)

        if self.linear_transition:
            out = []
            zeros = torch.zeros((num_groups, self.rank), device=self.initial_mean.device)
            F = self.get_transition_matrix().expand(len(start_offsets), -1, -1)
            mean = self.initial_mean.expand(len(start_offsets), -1)
            for i in range(int(self.period) + 1):
                maski = (start_offsets == i)
                out.append(update_tensor(zeros, new=mean[maski], mask=maski))
                mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
            return torch.stack(out, 0).sum(0)
        else:
            raise NotImplementedError
