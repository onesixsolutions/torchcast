"""
Arguments that were deprecated (with a ``FutureWarning``) before 2.0, and now raise.
"""
import pandas as pd
import pytest
import torch

from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalLevel
from torchcast.utils import complete_times


def _make_kf() -> KalmanFilter:
    return KalmanFilter(processes=[LocalLevel(id='level', measure='y')], measures=['y'])


@pytest.mark.parametrize('name', ['tol', 'patience', 'max_iter'])
def test_fit_stopping_kwargs_raise(name: str):
    kf = _make_kf()
    with pytest.raises(TypeError, match=f"no longer takes \\['{name}'\\]; pass them via `stopping`"):
        kf.fit(torch.randn(3, 10, 1), **{name: 1}, verbose=0)


def test_stopping_dict():
    kf = _make_kf()
    with pytest.raises(TypeError, match="'tol'"):
        kf.fit(torch.randn(3, 10, 1), stopping={'tol': .01}, verbose=0)
    kf.fit(torch.randn(3, 10, 1), stopping={'abstol': .01, 'max_iter': 2}, verbose=0)


def _dates(days: list) -> pd.Series:
    return pd.Timestamp('2020-01-01') + pd.to_timedelta(days, unit='D')


def test_complete_times_removed_kwargs():
    df = pd.DataFrame({'group': ['a', 'a', 'b'], 'time': _dates([0, 2, 0]), 'value': [1., 2., 3.]})
    with pytest.raises(TypeError, match='group_colname'):
        complete_times(df, group_colname='group', time_colname='time', dt_unit='D')
    with pytest.raises(TypeError, match='global_max'):
        complete_times(df, 'group', time_colname='time', dt_unit='D', global_max=True)
    out = complete_times(df, 'group', time_colname='time', dt_unit='D')
    assert len(out) == 4  # (a gets its missing time=1)


def test_complete_times_does_not_modify_group_colnames():
    df = pd.DataFrame({'group': ['a', 'a', 'b'], 'time': _dates([0, 2, 0]), 'max_time': _dates([2, 2, 1])})
    group_colnames = ['group']
    complete_times(df, group_colnames, time_colname='time', dt_unit='D', max_dt_colname='max_time')
    assert group_colnames == ['group']


def test_to_dataframe_components_raises():
    kf = _make_kf()
    with torch.no_grad():
        pred = kf(torch.randn(2, 5, 1))
        with pytest.raises(ValueError, match="use `type='observed_states'`"):
            pred.to_dataframe({'group': ['a', 'b'], 'start_offsets': None}, type='components')


def test_start_times_without_extra_kwargs():
    kf = _make_kf()
    with torch.no_grad():
        pred = kf(torch.randn(2, 5, 1), include_updates_in_output=True)
    state = pred.get_state_at_times([1, 3])
    means, _ = state
    assert torch.equal(means, torch.stack([pred.update_means[0, 1], pred.update_means[1, 3]]))
    assert pred.with_new_start_times([0, 2], n_timesteps=1).num_timesteps == 1
    with pytest.raises(TypeError, match='start_offsets'):
        pred.with_new_start_times([0, 2], n_timesteps=1, start_offsets=None)
