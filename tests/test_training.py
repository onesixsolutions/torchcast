from typing import Type

import time
import warnings

import numpy as np
import pytest
import torch

from torchcast.exp_smooth import ExpSmoother
from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalLevel, LinearModel, LocalTrend, Season
from torchcast.state_space import StateSpaceModel
from torchcast.utils.data import TimeSeriesDataset

MAX_TRIES = 3  # we set the seed but not tested across different platforms


@pytest.mark.parametrize("ndim", [1, 2, 3])
@torch.no_grad()
def test_log_prob_with_missings(ndim: int, num_groups: int = 1, num_times: int = 5):
    data = torch.randn((num_groups, num_times, ndim))
    mask = torch.randn_like(data) > 1.
    while mask.all() or not mask.any():
        mask = torch.randn_like(data) > 1.
    data[mask.nonzero(as_tuple=True)] = float('nan')
    kf = KalmanFilter(
        processes=[LocalTrend(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
        measures=[str(i) for i in range(ndim)]
    )
    pred = kf(data)
    lp_method1 = pred.log_prob(data)
    lp_method1_sum = lp_method1.sum().item()

    lp_method2_sum = 0
    for g in range(num_groups):
        data_g = data[[g]]
        pred_g = kf(data_g)
        for t in range(num_times):
            pred_gt = pred_g[:, [t]]
            data_gt = data_g[:, [t]]
            isvalid_gt = ~torch.isnan(data_gt).squeeze(0).squeeze(0)
            if not isvalid_gt.any():
                continue
            if isvalid_gt.all():
                lp_gt = torch.distributions.MultivariateNormal(*pred_gt).log_prob(data_gt)
            else:
                mmeans = pred_gt.means[..., isvalid_gt]
                mcovs = pred_gt.covs[..., isvalid_gt, :][..., isvalid_gt]
                lp_gt = torch.distributions.MultivariateNormal(mmeans, mcovs).log_prob(data_gt[..., isvalid_gt])
            lp_gt = lp_gt.item()
            assert abs(lp_method1[g, t].item() - lp_gt) < 1e-4
            lp_method2_sum += lp_gt
    assert abs(lp_method1_sum - lp_method2_sum) < 1e-3


def test_training1(ndim: int = 2, num_groups: int = 150, num_times: int = 24, compile: bool = False):
    """
    simulated data with known parameters, fitted loss should approach the loss given known params
    """
    torch.manual_seed(123)

    def _make_kf():
        kf = KalmanFilter(
            processes=[
                              LocalLevel(id=f'll{i + 1}', measure=str(i + 1))
                              for i in range(ndim)
                          ] + [
                              LinearModel(id=f'lm{i + 1}',
                                          predictors=['x1', 'x2', 'x3', 'x4', 'x5'],
                                          measure=str(i + 1))
                              for i in range(ndim)
                          ],
            measures=[str(i + 1) for i in range(ndim)]
        )
        if compile:
            kf = torch.jit.script(kf)
        return kf

    # simulate:
    X = torch.randn((num_groups, num_times, 5))
    kf_generator = _make_kf()
    with torch.no_grad():
        sim = kf_generator.simulate(out_timesteps=num_times, num_groups=num_groups, X=X)
        y = torch.distributions.MultivariateNormal(*sim).sample()
    assert not y.requires_grad
    # y[0, 1, :] = float('nan')  # introduce a missing value
    # y[1, 2, 0] = float('nan')  # introduce another missing value

    # train:
    kf_learner = _make_kf()
    optimizer = torch.optim.LBFGS(kf_learner.parameters(), lr=.5, max_iter=10)
    forward_times = []
    backward_times = []

    def closure():
        optimizer.zero_grad()
        _start = time.time()
        # print(f'[{datetime.datetime.now().time()}] forward...')
        pred = kf_learner(y, X=X)
        forward_times.append(time.time() - _start)
        loss = -pred.log_prob(y).mean()
        _start = time.time()
        # print(f'[{datetime.datetime.now().time()}] backward...')
        loss.backward()
        backward_times.append(time.time() - _start)
        # print(f'[{datetime.datetime.now().time()}] {loss.item()}')
        return loss

    print("\nTraining for 6 epochs...")
    for i in range(6):
        loss = optimizer.step(closure)
        print("loss:", loss.item())

    # print("forward time:", np.nanmean(forward_times))
    # print("backward time:", np.nanmean(backward_times))

    oracle_loss = -kf_generator(y, X=X).log_prob(y).mean()
    assert abs(oracle_loss.item() - loss.item()) < 0.1


@pytest.mark.parametrize("klass", [ExpSmoother, KalmanFilter])
def test_training2(klass: Type[StateSpaceModel], num_groups: int = 50, compile: bool = False):
    """
    # manually generated data (sin-wave, trend, etc.) with virtually no noise: MSE should be near zero
    """
    torch.manual_seed(123)

    weekly = torch.sin(2. * 3.1415 * torch.arange(0., 7.) / 7.)
    data = torch.stack([
        weekly.roll(-i).repeat(3) + torch.linspace(0, 10, 7 * 3) for i in range(num_groups)
    ]).unsqueeze(-1)
    # data[0, 1, :] = float('nan')  # introduce a missing value
    # data[1, 2, 0] = float('nan')  # introduce another missing value
    start_datetimes = np.array([np.datetime64('2019-04-14') + np.timedelta64(i, 'D') for i in range(num_groups)])

    def _train(num_epochs: int = 12):
        kf = klass(
            processes=[
                LocalTrend(id='trend'),
                Season(id='day_of_week', period='7D', dt_unit='D', K=3, fixed=True)
            ],
            measures=['y']
        )
        if compile:
            kf = torch.jit.script(kf)

        # train:
        optimizer = torch.optim.LBFGS([p for n, p in kf.named_parameters() if 'measure_covariance' not in n],
                                      lr=.20,
                                      line_search_fn='strong_wolfe',
                                      max_iter=10)

        def closure():
            optimizer.zero_grad()
            _start = time.time()
            # print(f'[{datetime.datetime.now().time()}] forward...')
            pred = kf(data, start_offsets=start_datetimes)
            loss = -pred.log_prob(data).mean()
            _start = time.time()
            loss.backward()
            return loss

        print(f"\nTraining for {num_epochs} epochs...")
        for i in range(num_epochs):
            loss = optimizer.step(closure)
            print("loss:", loss.item())

        return kf

    kf = None
    for i in range(MAX_TRIES):
        try:
            kf = _train()
        except (RuntimeError, ValueError) as e:
            if 'cholesky' not in str(e) and 'has invalid values' not in str(e):
                raise e
        if kf is not None:
            break
    if kf is None:
        raise RuntimeError("MAX_TRIES exceeded")

    pred = kf(data, start_offsets=start_datetimes)
    # MSE should be virtually zero
    assert torch.mean((pred.means - data) ** 2) < .01
    # trend should be identified:
    assert abs(pred.state_means[..., 1].mean().item() - 5.) < 0.1


def test_training3(compile: bool = False):
    """
    Test TBATS and TimeSeriesDataset integration
    """
    import pandas as pd

    torch.manual_seed(123)
    df = pd.DataFrame({'sin': np.sin(2. * 3.1415 * np.arange(0., 5 * 7.) / 7.),
                       'cos': np.cos(2. * 3.1415 * np.arange(0., 5 * 7.) / 7.)})
    df['y'] = df['cos'].where(df.index < 12, other=df['sin'])

    # create multiple groups. make sure we're testing the `offset_initial_mean`
    df = pd.concat([
        df.assign(
            observed=lambda df: df['y'] + np.random.normal(scale=.2, size=len(df.index)),
            group=str(i + 1),
            time=lambda df: np.array(df.index.tolist(), dtype='datetime64[D]')
        ).iloc[i:, :]
        for i in range(10)
    ]).reset_index(drop=True)

    dataset = TimeSeriesDataset.from_dataframe(
        df,
        group_colname='group',
        time_colname='time',
        dt_unit='D',
        measure_colnames=['y']
    )

    def _train(num_epochs: int = 15):

        kf = KalmanFilter(
            processes=[
                Season(id='day_of_week', period='7D', dt_unit='D', K=1, decay=(.85, 1.))
            ],
            measures=['y']
        )
        if compile:
            kf = torch.jit.script(kf)

        # train:
        optimizer = torch.optim.LBFGS(kf.parameters(), lr=.15, max_iter=10)

        def closure():
            optimizer.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = kf(dataset.tensors[0], start_offsets=dataset.start_datetimes)
            loss = -pred.log_prob(dataset.tensors[0]).mean()
            loss.backward()
            return loss

        print(f"\nTraining for {num_epochs} epochs...")
        for i in range(num_epochs):
            loss = optimizer.step(closure)
            print("loss:", loss.item())

        return kf

    kf = None
    for i in range(MAX_TRIES):
        try:
            kf = _train()
        except (RuntimeError, ValueError) as e:
            if 'cholesky' not in str(e) and 'has invalid values' not in str(e) and 'nans' not in str(e):
                raise e
        if kf is not None:
            break
    if kf is None:
        raise RuntimeError("MAX_TRIES exceeded")

    with torch.no_grad():
        pred = kf(dataset.tensors[0], start_offsets=dataset.start_datetimes)
    df_pred = pred.to_dataframe(dataset)
    assert np.mean((df_pred['actual'] - df_pred['mean']) ** 2) < .05
