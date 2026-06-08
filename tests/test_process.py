import torch

import numpy as np
from torchcast.kalman_filter import KalmanFilter
from torchcast.process.season import Season
import pytest


@torch.no_grad()
def test_fourier_season():
    series = torch.cos(2. * 3.1415 * torch.arange(1., 8.) / 7.)
    data = torch.stack([series.roll(-i).repeat(3) for i in range(6)]).unsqueeze(-1)
    start_datetimes = np.array([np.datetime64('2019-04-18') + np.timedelta64(i, 'D') for i in range(6)])
    kf = KalmanFilter(
        processes=[Season(id='day_of_week', period='7D', dt_unit='D', K=3, fixed=True)],
        measures=['y']
    )
    kf.state_dict()['processes.day_of_week.initial_mean'][:] = torch.tensor([1., 0., 0., 0., 0., 0.])
    kf.state_dict()['measure_covariance.cholesky_log_diag'] -= 2
    pred = kf(data, start_offsets=start_datetimes)
    for g in range(6):
        assert torch.abs(pred.means[g] - data[g]).mean() < .01, f"Group {g} failed"


@pytest.mark.parametrize("period,K,fixed,decay", [
    (12.0, 2, True, None),    # no decay (fixed season)
    (12.0, 2, False, None),   # no decay (evolving season)
    (12.0, 2, False, True),   # learnable decay
    (24.0, 4, True, None),    # larger K, no decay
])
@torch.no_grad()
def test_season_get_initial_mean_integer_offsets(period, K, fixed, decay):
    """get_initial_mean([i]) == F^i @ initial_mean for integer-unit offsets."""
    torch.manual_seed(0)
    season = Season(id='test', period=period, dt_unit=None, K=K, fixed=fixed, decay=decay)
    season.initial_mean.data = torch.randn(season.rank)

    F = season.get_transition_matrix()
    offsets = np.array([0., 1., 4., 7., int(period) - 1])

    result = season.get_initial_mean(offsets)

    for g, offset in enumerate(offsets):
        expected = torch.matrix_power(F, int(offset)) @ season.initial_mean
        assert torch.allclose(result[g], expected, atol=1e-5), (
            f"Mismatch at offset={offset}, period={period}, K={K}, fixed={fixed}, decay={decay}"
        )


@torch.no_grad()
def test_season_get_initial_mean_datetime_offsets():
    """get_initial_mean works correctly when start_offsets are np.datetime64 values."""
    torch.manual_seed(0)
    season = Season(id='test', period='7D', dt_unit='D', K=3, fixed=True)
    season.initial_mean.data = torch.randn(season.rank)

    F = season.get_transition_matrix()

    # Days 0–6 relative to the Unix epoch; day 7 wraps back to 0.
    epoch = np.datetime64('1970-01-01')
    offsets = np.array([epoch + np.timedelta64(i, 'D') for i in range(7)])
    result = season.get_initial_mean(offsets)

    for day in range(7):
        expected = torch.matrix_power(F, day) @ season.initial_mean
        assert torch.allclose(result[day], expected, atol=1e-5), (
            f"Mismatch at day={day}"
        )


@torch.no_grad()
def test_season_get_initial_mean_multiple_groups():
    """get_initial_mean handles a batch of groups, each potentially at a different offset."""
    torch.manual_seed(0)
    season = Season(id='test', period=24.0, dt_unit=None, K=3, fixed=True)
    season.initial_mean.data = torch.randn(season.rank)

    F = season.get_transition_matrix()
    offsets = np.array([0., 3., 7., 12., 18., 23.])
    result = season.get_initial_mean(offsets)

    assert result.shape == (len(offsets), season.rank)
    for g, offset in enumerate(offsets):
        expected = torch.matrix_power(F, int(offset)) @ season.initial_mean
        assert torch.allclose(result[g], expected, atol=1e-5), (
            f"Mismatch for group {g} at offset={offset}"
        )
