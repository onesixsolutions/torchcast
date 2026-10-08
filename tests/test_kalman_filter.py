import copy
import itertools
from collections import defaultdict
from typing import Callable

import pytest
import torch

from torchcast.internals.batch_design import TransitionModel, MeasurementModel
from torchcast.internals.utils import get_nan_groups

from torchcast.kalman_filter import KalmanFilter
from torchcast.exp_smooth import ExpSmoother

import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

from torchcast.process import LocalTrend, LinearModel, LocalLevel


@pytest.mark.parametrize("ndim,n_step", list(itertools.product([1, 2, 3], [1, 2, 3])))
@torch.no_grad()
def test_nans(ndim: int, n_step: int):
    ntimes = 4 + n_step
    data = torch.ones((5, ntimes, ndim)) * 10
    data[0, 2, 0:(ndim - 1)] = float('nan')
    data[2, 2, 0] = float('nan')

    # test critical helper fun:
    nan_groups = {2}
    if ndim > 1:
        nan_groups.add(0)
    for t in range(ntimes):
        for group_idx, masks in get_nan_groups(torch.isnan(data[:, t])):
            if t == 2:
                if masks is None:
                    assert len(group_idx) == data.shape[0] - len(nan_groups)
                    assert not bool(set(group_idx.tolist()).intersection(nan_groups))
                else:
                    valid_idx, m1d, m2d = masks
                    assert len(valid_idx) < ndim
                    assert len(valid_idx) > 0
                    if len(valid_idx) == 1:
                        if ndim == 2:
                            assert set(valid_idx.tolist()) == {1}
                            assert set(group_idx.tolist()) == nan_groups
                        else:
                            assert set(valid_idx.tolist()) == {ndim - 1}
                            assert set(group_idx.tolist()) == {0}
                    else:
                        assert set(valid_idx.tolist()) == {1, 2}
                        assert set(group_idx.tolist()) == {2}
            else:
                assert masks is None

    # test `update`
    # TODO: measure dim vs. state-dim

    # test integration:
    # TODO: make missing dim highly correlated with observed dims. upward trend in observed should get reflected in
    #       unobserved state
    kf = KalmanFilter(
        processes=[LocalLevel(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
        measures=[str(i) for i in range(ndim)]
    )
    obs_means, obs_covs = kf(data, n_step=n_step)
    assert not torch.isnan(obs_means).any()
    assert not torch.isnan(obs_covs).any()
    assert tuple(obs_means.shape) == (5, ntimes, ndim)


@torch.no_grad()
def test_equations_decay():
    data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)
    num_times = data.shape[1]

    # make torch kf:
    torch_kf = KalmanFilter(
        processes=[LinearModel(id='lm', predictors=['x1', 'x2', 'x3'], fixed=False, decay=(.95, 1.))],
        measures=['y']
    )
    tmodel = TransitionModel(
        processes=torch_kf.processes,
        measures=torch_kf.measures,
        num_groups=1,
        num_timesteps=num_times
    )
    F = tmodel.transition_mats[0].squeeze(0)

    #
    assert (torch.diag(F) > .95).all()
    assert (torch.diag(F) < 1.00).all()
    assert len(set(torch.diag(F).tolist())) > 1
    for r in range(F.shape[-1]):
        for c in range(F.shape[-1]):
            if r == c:
                continue
            assert F[r, c] == 0

    # confirm decay works in forward pass
    # also tests that kf.forward works with `out_timesteps > input.shape[1]`
    pred = torch_kf(
        initial_state=torch_kf._prepare_initial_state(None, start_offsets=np.zeros(1)),
        X=torch.randn(1, num_times, 3),
        out_timesteps=num_times
    )
    for t in range(1, num_times):
        for i in range(3):
            assert pred.state_means[:, t, i].abs() < pred.state_means[:, t - 1, i].abs()


@torch.no_grad()
def test_equations():
    data = torch.tensor([[-5.]]).unsqueeze(-1)
    num_times = data.shape[1]

    # make torch kf:
    _oldval = LocalTrend._velocity_multi
    try:
        LocalTrend._velocity_multi = 1.0
        torch.manual_seed(123)
        torch_kf = KalmanFilter(
            processes=[LocalTrend(id='lt', decay_velocity=None, measure='y')],
            measures=['y']
        )
        expectedF = torch.tensor([[1., 1.], [0., 1.]])
        expectedH = torch.tensor([[1., 0.]])

        tmodel = TransitionModel(
            processes=torch_kf.processes,
            measures=torch_kf.measures,
            num_groups=1,
            num_timesteps=num_times
        )
        F = tmodel.transition_mats[0]
        mmodel = MeasurementModel(
            processes=torch_kf.processes,
            measures=torch_kf.measures,
            num_groups=1,
            num_timesteps=num_times
        )
        H = mmodel._get_linear_measure_mat(0)

        R = torch_kf.measure_covariance(inputs={}, num_groups=1, num_times=1)[:, 0]
        predict_kwargs = torch_kf._parse_kwargs(1, 1, R)[0]
        Q = predict_kwargs['Q'][0]

        assert torch.isclose(expectedF, F).all()
        assert torch.isclose(expectedH, H).all()

        # make filterpy kf:
        filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
        filter_kf.x, filter_kf.P = torch_kf._prepare_initial_state(None)
        filter_kf.x = filter_kf.x.detach().numpy().T
        filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
        filter_kf.Q = Q.numpy().squeeze(0)
        filter_kf.R = R.numpy().squeeze(0)
        filter_kf.F = F.numpy().squeeze(0)
        filter_kf.H = H.numpy().squeeze(0)

        # compare:
        sb = torch_kf(data)
    finally:
        LocalTrend._velocity_multi = _oldval

    #
    filter_kf.state_means = []
    filter_kf.state_covs = []
    for t in range(num_times):
        # 1step:
        filter_kf.predict()
        # append:
        filter_kf_copy = copy.deepcopy(filter_kf)
        filter_kf.state_means.append(filter_kf_copy.x)
        filter_kf.state_covs.append(filter_kf_copy.P)
        # update:
        filter_kf.update(data[:, t, :])

    assert np.isclose(sb.state_means.numpy().squeeze(), np.stack(filter_kf.state_means).squeeze(), rtol=1e-4).all()
    assert np.isclose(sb.state_covs.numpy().squeeze(), np.stack(filter_kf.state_covs).squeeze(), rtol=1e-4).all()


@torch.no_grad()
def test_equations_preds(n_step: int = 1):
    from torchcast.utils.data import TimeSeriesDataset
    from pandas import DataFrame

    class LinearModelFixed(LinearModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for se in self.state_elements.values():
                se.has_initial_variance = False

    kf = KalmanFilter(
        processes=[
            LinearModelFixed(id='lm', predictors=['x1', 'x2'])
        ],
        measures=['y']
    )

    kf.state_dict()['processes.lm.initial_mean'][:] = torch.tensor([1.5, -0.5])
    kf.state_dict()['measure_covariance.cholesky_log_diag'][0] = np.log(.1 ** .5)

    num_times = 100
    df = DataFrame({'x1': np.random.randn(num_times), 'x2': np.random.randn(num_times)})
    df['y'] = 1.5 * df['x1'] + -.5 * df['x2'] + .1 * np.random.randn(num_times)
    df['time'] = df.index.values
    df['group'] = '1'
    dataset = TimeSeriesDataset.from_dataframe(
        dataframe=df,
        group_colname='group',
        time_colname='time',
        dt_unit=None,
        X_colnames=['x1', 'x2'],
        y_colnames=['y']
    )
    y, X = dataset.tensors
    #
    from pandas import Series

    if n_step == 0:
        with pytest.raises(AssertionError):
            kf(y, X=X, n_step=n_step)
        return

    pred = kf(y, X=X, out_timesteps=X.shape[1], n_step=n_step)
    y_series = Series(y.squeeze().numpy())
    for shift in range(-2, 3):
        resid = y_series.shift(shift) - Series(pred.means.squeeze().numpy())
        if shift:
            # check there's no misalignment in internal n_step logic (i.e., realigning the input makes things worse)
            assert (resid ** 2).mean() > 1.
        else:
            assert (resid ** 2).mean() < .02


@pytest.mark.parametrize(
    "klass,n_step,every_step",
    list(itertools.product([KalmanFilter, ExpSmoother], [2, 3, 5], [True, False]))
)
@torch.no_grad()
def test_n_step_matches_nan_forecast(klass: type, n_step: int, every_step: bool):
    """
    An h-step-ahead prediction for time t is a forecast from the update at t - h. So it should exactly match the
    1-step-ahead prediction for time t when the observations between t - h and t are missing.

    With every_step=True, h=n_step (except for the first n_step timesteps, which forecast from the initial state). With
    every_step=False, the horizon cycles through 1...n_step.
    """
    torch.manual_seed(123)
    measures = ['y1', 'y2']
    model = klass(
        processes=[LocalTrend(id=f'trend_{m}', measure=m) for m in measures],
        measures=measures
    )
    if isinstance(model, ExpSmoother):
        # default init gives K~0 (so covs~0), which would make this test trivially pass
        model.smoothing_matrix.init_bias = 0
    num_times = 12
    y = torch.randn((3, num_times, len(measures))).cumsum(1)
    pred_n = model(y, n_step=n_step, every_step=every_step)
    assert (pred_n.state_covs.diagonal(dim1=-2, dim2=-1) > .01).any()

    for t in range(num_times):
        h = min(t + 1, n_step) if every_step else (t % n_step) + 1
        y_nan = y.clone()
        y_nan[:, (t - h + 1):t] = float('nan')
        pred_1 = model(y_nan, n_step=1)
        assert torch.allclose(pred_n.state_means[:, t], pred_1.state_means[:, t], atol=1e-5)
        assert torch.allclose(pred_n.state_covs[:, t], pred_1.state_covs[:, t], atol=1e-5)

    # a prediction should never depend on observations after it:
    y_later = y.clone()
    y_later[:, -1] = float('nan')
    pred_later = model(y_later, n_step=n_step, every_step=every_step)
    assert torch.allclose(pred_n.state_covs[:, :-1], pred_later.state_covs[:, :-1], atol=1e-5)


def test_keyword_dispatch():
    _counter = defaultdict(int)

    def check_input(func: Callable, expected: torch.Tensor) -> Callable:
        def outfunc(**inputs):
            x = inputs.get('X')
            _counter[func.__name__] += 1
            assert x is not None
            _bool = (x == expected)
            if hasattr(_bool, 'all'):
                _bool = _bool.all().item()
            assert _bool
            return func(**inputs)

        return outfunc

    data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)

    def _make_kf():
        return KalmanFilter(
            processes=[
                LinearModel(id='lm1', predictors=['x1', 'x2']),
                LinearModel(id='lm2', predictors=['x1', 'x2'])
            ],
            measures=['y']
        )

    _predictors = torch.ones(1, data.shape[1], 2)

    # shared --
    expected = {'lm1': torch.zeros(1), 'lm2': torch.zeros(1)}

    # share input:
    kf = _make_kf()
    for nm, proc in kf.processes.items():
        proc.get_measurement_matrix = check_input(proc.get_measurement_matrix, expected[nm])
    kf(data, X=_predictors * 0.)
    expected_call_count = len(expected)
    assert _counter['get_measurement_matrix'] >= expected_call_count

    # separate ---
    expected['lm2'] = torch.ones(1)
    # individual input:
    kf = _make_kf()
    for nm, proc in kf.processes.items():
        proc.get_measurement_matrix = check_input(proc.get_measurement_matrix, expected[nm])
    kf(data, lm1__X=_predictors * 0., lm2__X=_predictors)
    expected_call_count += len(expected)
    assert _counter['get_measurement_matrix'] >= expected_call_count

    # specific overrides general
    kf(data, X=_predictors * 0., lm2__X=_predictors)
    expected_call_count += len(expected)
    assert _counter['get_measurement_matrix'] >= expected_call_count

    # make sure check_input is being called:
    with pytest.raises(AssertionError) as exc_info:
        kf(data, X=_predictors * 0.)
    assert "false" in str(exc_info.value).lower()


@torch.no_grad()
def test_predictions(ndim: int = 2):
    data = torch.zeros((2, 5, ndim))
    kf = KalmanFilter(
        processes=[LocalLevel(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
        measures=[str(i) for i in range(ndim)]
    )
    pred = kf(data)
    assert len(tuple(pred)) == 2
    assert isinstance(np.asanyarray(pred), np.ndarray)
    means, covs = pred
    assert isinstance(means, torch.Tensor)
    assert isinstance(covs, torch.Tensor)

    with pytest.raises(ValueError):
        pred[1]

    with pytest.raises(ValueError):
        pred[(1,)]

    pred_group2 = pred[[1]]
    assert tuple(pred_group2.covs.shape) == (1, 5, ndim, ndim)
    assert (pred_group2.state_means == pred.state_means[1, :, :]).all()
    assert (pred_group2.state_covs == pred.state_covs[1, :, :, :]).all()

    pred_time3 = pred[:, [2]]
    assert tuple(pred_time3.covs.shape) == (2, 1, ndim, ndim)
    assert (pred_time3.state_means == pred.state_means[:, 2, :]).all()
    assert (pred_time3.state_covs == pred.state_covs[:, 2, :, :]).all()


@torch.no_grad()
def test_no_proc_variance():
    kf = KalmanFilter(processes=[LinearModel(id='lm', predictors=['x1', 'x2'])], measures=['y'])
    cov = kf.process_covariance({}, num_groups=1, num_times=1).squeeze()
    assert cov.shape[-1] == 2
    assert (cov == 0).all()


@pytest.mark.parametrize("dtype,ndim,compiled", [
    (torch.float64, 2, False),
    (torch.float64, 1, False)
])
@torch.no_grad()
def test_dtype(dtype: torch.dtype, ndim: int, compiled: bool):
    data = torch.zeros((2, 5, ndim), dtype=dtype)
    kf = KalmanFilter(
        processes=[LocalLevel(id=f'll{i}', measure=str(i)) for i in range(ndim)],
        measures=[str(i) for i in range(ndim)]
    )
    if compiled:
        kf = torch.jit.script(kf)
    kf.to(dtype=dtype)
    pred = kf(data)
    assert pred.means.dtype == dtype
    loss = pred.log_prob(data)
    assert loss.dtype == dtype


@pytest.mark.parametrize("measure_log_std", [0., 2.])
@torch.no_grad()
def test_initial_state_continuation(measure_log_std: float):
    """
    Filtering the first part of a series, then forecasting from ``get_state_at_times()`` on the rest, should match a
    single pass over the whole series.
    """
    torch.manual_seed(0)
    measures = ['y1', 'y2']
    kf = KalmanFilter(processes=[LocalTrend(id=f'trend_{m}', measure=m) for m in measures], measures=measures)
    kf.measure_covariance.cholesky_log_diag.fill_(measure_log_std)
    y = torch.randn((3, 20, len(measures))).cumsum(1) * 10
    split = 12

    full = kf(y)
    first = kf(y[:, :split], include_updates_in_output=True)
    state = first.get_state_at_times(split - 1)
    # backwards-compatible with the (mean, cov) tuple that used to be returned:
    mean, cov = state
    assert len(state) == 2 and state[0] is mean and state[1] is cov

    cont = kf(y[:, split:], initial_state=state)
    assert torch.allclose(cont.state_means, full.state_means[:, split:], atol=1e-4)
    assert torch.allclose(cont.state_covs, full.state_covs[:, split:], rtol=1e-4, atol=1e-4)
    # passing a plain tuple is equivalent:
    cont_tuple = kf(y[:, split:], initial_state=(mean, cov))
    assert torch.allclose(cont_tuple.state_covs, cont.state_covs)


@torch.no_grad()
def test_adaptive_scaling_with_empty_measure_cov():
    """
    With adaptive scaling, measures without a measure-variance (e.g. binary measures in the BinomialFilter) should get
    no scaling, and the ordering of measures shouldn't matter.
    """
    from torchcast.kalman_filter import BinomialFilter

    def make(measures):
        torch.manual_seed(0)
        return BinomialFilter(
            processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
            measures=measures,
            binary_measures=['visit'],
            adaptive_scaling=True
        )

    torch.manual_seed(1)
    y = torch.stack([(torch.rand(3, 15) > .5).float(), torch.randn(3, 15).cumsum(1) * 3], -1)  # visit, spend
    bf1 = make(['visit', 'spend'])
    bf1.adaptive_scaling.initialize(y.shape[1])
    # make an equivalent model with the measures (and so the state-elements) in the opposite order.
    # copy parameters by name (not buffers, which encode which measure is binary); state-covariances are position-based,
    # so make them diagonal and flip:
    bf2 = make(['spend', 'visit'])
    params2 = dict(bf2.named_parameters())
    for name, param in bf1.named_parameters():
        params2[name].copy_(param)
    for bf in (bf1, bf2):
        for cov in (bf.initial_covariance, bf.process_covariance):
            cov.cholesky_off_diag.zero_()
    for cov1, cov2 in [(bf1.initial_covariance, bf2.initial_covariance),
                       (bf1.process_covariance, bf2.process_covariance)]:
        cov2.cholesky_log_diag.copy_(cov1.cholesky_log_diag.flip(0))

    pred1 = bf1(y)
    pred2 = bf2(y.flip(-1))
    assert torch.allclose(pred1.state_means, pred2.state_means.flip(-1), atol=1e-5)
    assert torch.allclose(pred1.measure_covs, pred2.measure_covs.flip(-1).flip(-2), atol=1e-5)
    # scaling was actually applied to the gaussian measure:
    assert not torch.allclose(pred1.measure_covs[:, 1:, 1, 1], pred1.measure_covs[:, :1, 1, 1].expand(-1, 14))


@torch.no_grad()
def test_nonlinear_covs_warns_once():
    import warnings
    from torchcast.kalman_filter import BinomialFilter
    from torchcast.state_space import predictions

    torch.manual_seed(0)
    bf = BinomialFilter(processes=[LocalLevel(id='level')], measures=['visit'])
    bf.mc_sampling = 10
    pred = bf((torch.rand(2, 5, 1) > .5).float())
    predictions._warn_once.pop('cov', None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert pred.covs is None
        assert pred.covs is None
    assert len([w for w in caught if 'no closed-form covariance' in str(w.message)]) == 1


@pytest.mark.parametrize("config", ['sigmoid', 'saturated', 'saturated+sigmoid'])
def test_ekf_jacobian_matches_autograd(config: str):
    """
    The EKF linearization: the measurement-matrix returned by the MeasurementModel should be the jacobian of the
    measured-mean wrt the state (in particular, a measure-function's derivative is evaluated at its input).
    """
    from torchcast.process import SaturatedLinearModel
    from torchcast.internals.monte_carlo import FixedWhiteNoise

    torch.manual_seed(0)
    processes = [LocalLevel(id='level')]
    kwargs = {}
    if 'saturated' in config:
        processes.append(SaturatedLinearModel(id='slm', predictors=['a', 'b']))
        kwargs['X'] = torch.randn(4, 3, 2)
    kf = KalmanFilter(processes=processes, measures=['y'],
                      measure_funs={'y': 'sigmoid'} if 'sigmoid' in config else None)
    kf.mc_sampling = FixedWhiteNoise(10, random_state=np.random.RandomState(0))
    with torch.no_grad():
        pred = kf(torch.rand(4, 3, 1), **kwargs)
    mm = pred.measurement_model
    means = pred.state_means[:, 1].clone()
    _, measure_mat = mm(means, time=1)
    for g in range(means.shape[0]):
        def fun(state):
            full = means.clone()
            full[g] = state
            return mm(full, time=1)[0][g]

        jac = torch.autograd.functional.jacobian(fun, means[g].clone())
        assert torch.allclose(measure_mat[g], jac, atol=1e-5), (g, measure_mat[g], jac)


@torch.no_grad()
def test_sigmoid_legacy_jacobian():
    """The (deprecated) `legacy_jacobian` flag reproduces the pre-1.1.3 linearization: sigmoid'(sigmoid(z))."""
    from torchcast.internals.monte_carlo import FixedWhiteNoise

    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    kf.mc_sampling = FixedWhiteNoise(10, random_state=np.random.RandomState(0))
    pred = kf(torch.rand(4, 3, 1))
    mm = pred.measurement_model
    means = pred.state_means[:, 1]
    z = means[:, 0]  # (LocalLevel: the pre-sigmoid measured mean is the state)

    def deriv(x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    _, measure_mat = mm(means, time=1)
    assert torch.allclose(measure_mat[:, 0, 0], deriv(z), atol=1e-6)
    kf.measure_funs['y'].legacy_jacobian = True
    _, measure_mat_legacy = mm(means, time=1)
    assert torch.allclose(measure_mat_legacy[:, 0, 0], deriv(torch.sigmoid(z)), atol=1e-6)
    # the flag is per-model:
    kf2 = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    assert not kf2.measure_funs['y'].legacy_jacobian


def test_sigmoid_legacy_jacobian_unpickling():
    """
    Models pickled before v1.1.3 have Sigmoid measure-funs without a `legacy_jacobian` instance-attribute; on loading,
    they should keep their original (legacy) behavior. Models created (and pickled) with this version should not.
    """
    import pickle
    from torchcast.internals.batch_design import Sigmoid

    z = torch.tensor([[2.]])

    def jacobian(sigmoid) -> torch.Tensor:
        return sigmoid.adjust_measure_mat(torch.ones(1, 1), z)

    fixed = torch.sigmoid(z) * (1 - torch.sigmoid(z))
    legacy = torch.sigmoid(torch.sigmoid(z)) * (1 - torch.sigmoid(torch.sigmoid(z)))
    new = pickle.loads(pickle.dumps(Sigmoid()))
    assert new.legacy_jacobian is False and torch.allclose(jacobian(new), fixed)
    # simulate an object pickled by an older version (no `legacy_jacobian` attribute):
    old_style = Sigmoid()
    del old_style.legacy_jacobian
    old_loaded = pickle.loads(pickle.dumps(old_style))
    assert not hasattr(old_loaded, 'legacy_jacobian') and torch.allclose(jacobian(old_loaded), legacy)


@torch.no_grad()
def test_joseph_form_option():
    """`joseph_form=False` uses the simpler covariance update -- the same, in exact arithmetic."""
    from torchcast.kalman_filter import BinomialFilter

    torch.manual_seed(0)
    y = torch.randn(4, 20, 2).cumsum(1) * .3
    y[0, 3:6, 1] = float('nan')
    preds = {}
    for joseph_form in (True, False):
        torch.manual_seed(1)
        kf = KalmanFilter(
            processes=[LocalTrend(id=f'trend_{m}', measure=m) for m in ['a', 'b']],
            measures=['a', 'b'],
            joseph_form=joseph_form,
        )
        assert kf.joseph_form is joseph_form
        preds[joseph_form] = kf(y, n_step=2)
    assert torch.allclose(preds[True].state_means, preds[False].state_means, atol=1e-5)
    assert torch.allclose(preds[True].state_covs, preds[False].state_covs, atol=1e-5)
    assert torch.allclose(preds[True].log_prob(y), preds[False].log_prob(y), atol=1e-4)
    # the simple update's output is exactly symmetric:
    A = torch.randn(5, 4, 4)
    cov = A @ A.transpose(-1, -2) + torch.eye(4)
    H, R = torch.randn(5, 2, 4), torch.eye(2).expand(5, -1, -1)
    K = KalmanFilter._kalman_gain(cov=cov, H=H, R=R)
    new_cov = KalmanFilter._simple_covariance_update(cov=cov, K=K, H=H)
    assert torch.equal(new_cov, new_cov.transpose(-1, -2))
    assert torch.allclose(new_cov, KalmanFilter._covariance_update(cov=cov, K=K, H=H, R=R), atol=1e-5)

    # the binomial-filter passes it through:
    visit = (torch.rand(3, 15, 1) > .4).float()
    bf_preds = []
    for joseph_form in (True, False):
        torch.manual_seed(1)
        bf = BinomialFilter(processes=[LocalLevel(id='level')], measures=['visit'], joseph_form=joseph_form)
        assert bf.joseph_form is joseph_form
        bf_preds.append(bf(visit))
    assert torch.allclose(bf_preds[0].state_covs, bf_preds[1].state_covs, atol=1e-5)

    # the setting selects the update:
    def fail(*args, **kwargs):
        raise AssertionError("joseph-form update was called")

    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['a'], joseph_form=False)
    kf._covariance_update = fail
    kf(y[..., :1])
    kf.joseph_form = True
    with pytest.raises(AssertionError, match="joseph-form"):
        kf(y[..., :1])
    # older pickles (no attribute) use the joseph form:
    del kf.joseph_form
    with pytest.raises(AssertionError, match="joseph-form"):
        kf(y[..., :1])
