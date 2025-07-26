import copy
import itertools
from collections import defaultdict
from typing import Callable, Dict
from unittest import TestCase, expectedFailure

import torch
from parameterized import parameterized
from torch import Tensor

from torchcast.internals.utils import get_nan_groups

from torchcast.kalman_filter import KalmanFilter

import numpy as np
from filterpy.kalman import KalmanFilter as filterpy_KalmanFilter

from torchcast.process import LocalTrend, LinearModel, LocalLevel
from torchcast.process.process import Process


class TestKalmanFilter(TestCase):
    @parameterized.expand(itertools.product([1, 2, 3], [1, 2, 3]))
    @torch.no_grad()
    def test_nans(self, ndim: int = 3, n_step: int = 1):
        ntimes = 4 + n_step
        data = torch.ones((5, ntimes, ndim)) * 10
        data[0, 2, 0:(ndim - 1)] = float('nan')
        data[2, 2, 0] = float('nan')

        # test critical helper fun:
        get_nan_groups2 = torch.jit.script(get_nan_groups)
        nan_groups = {2}
        if ndim > 1:
            nan_groups.add(0)
        for t in range(ntimes):
            for group_idx, valid_idx in get_nan_groups2(torch.isnan(data[:, t])):
                if t == 2:
                    if valid_idx is None:
                        self.assertEqual(len(group_idx), data.shape[0] - len(nan_groups))
                        self.assertFalse(bool(set(group_idx.tolist()).intersection(nan_groups)))
                    else:
                        self.assertLess(len(valid_idx), ndim)
                        self.assertGreater(len(valid_idx), 0)
                        if len(valid_idx) == 1:
                            if ndim == 2:
                                self.assertSetEqual(set(valid_idx.tolist()), {1})
                                self.assertSetEqual(set(group_idx.tolist()), nan_groups)
                            else:
                                self.assertSetEqual(set(valid_idx.tolist()), {ndim - 1})
                                self.assertSetEqual(set(group_idx.tolist()), {0})
                        else:
                            self.assertSetEqual(set(valid_idx.tolist()), {1, 2})
                            self.assertSetEqual(set(group_idx.tolist()), {2})
                else:
                    self.assertIsNone(valid_idx)

        # test `update`
        # TODO: measure dim vs. state-dim

        # test integration:
        # TODO: make missing dim highly correlated with observed dims. upward trend in observed should get reflected in
        #       unobserved state
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)]
        )
#        kf = torch.jit.script(kf)
        obs_means, obs_covs = kf(data, n_step=n_step)
        self.assertFalse(torch.isnan(obs_means).any())
        self.assertFalse(torch.isnan(obs_covs).any())
        self.assertEqual(tuple(obs_means.shape), (5, ntimes, ndim))

    @torch.no_grad()
    def test_jit(self):
        from torchcast.state_space import Predictions

        # compile-able:
        h_module = SingleOutput()
        f_modules = torch.nn.ModuleDict()
        f_modules['position->position'] = SingleOutput()

        class ExProc(Process):
            def __init__(self, id: str, state_elements: list, h_module: torch.nn.Module, f_modules: dict):
                super().__init__(id=id, state_elements=state_elements)
                self.f_modules.update(f_modules)
                self.h_module = h_module

            def _build_h_mat(self, inputs: Dict[str, Tensor], num_groups: int, num_times: int) -> Tensor:
                return self.h_module()

        compilable = ExProc(id='compilable',
                            state_elements=['position'],
                            h_module=h_module,
                            f_modules=f_modules)

        torch_kf = KalmanFilter(
            processes=[compilable],
            measures=['y']
        )
        # runs:
        self.assertIsInstance(torch_kf(torch.tensor([[-5., 5., 1.]]).unsqueeze(-1)), Predictions)

        # not compile-able:
        not_compilable = ExProc(id='not_compilable',
                                state_elements=['position'],
                                h_module=lambda x=None: h_module(x),
                                f_modules=f_modules)
        torch_kf = KalmanFilter(
            processes=[not_compilable],
            measures=['y']
        )
        with self.assertRaises(RuntimeError) as cm:
            torch_kf = torch.jit.script(torch_kf)
        the_exception = cm.exception
        self.assertIn('failed to compile', str(the_exception))
        self.assertIn('TorchScript', str(the_exception))

        # but we can skip compilation:
        self.assertIsInstance(torch_kf(torch.tensor([[-5., 5., 1.]]).unsqueeze(-1)), Predictions)

    @torch.no_grad()
    def test_equations_decay(self):
        data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)
        num_times = data.shape[1]

        # make torch kf:
        torch_kf = KalmanFilter(
            processes=[LinearModel(id='lm', predictors=['x1', 'x2', 'x3'], fixed=False, decay=(.95, 1.))],
            measures=['y']
        )
        kwargs_per_process = torch_kf._parse_design_kwargs(
            input=data, out_timesteps=num_times, X=torch.randn(1, num_times, 3)
        )
        predict_kwargs, update_kwargs = torch_kf._build_design_mats(
            kwargs_per_process, num_groups=1, out_timesteps=num_times
        )
        F = predict_kwargs['F'][0][0]

        self.assertTrue((torch.diag(F) > .95).all())
        self.assertTrue((torch.diag(F) < 1.00).all())
        self.assertGreater(len(set(torch.diag(F).tolist())), 1)
        for r in range(F.shape[-1]):
            for c in range(F.shape[-1]):
                if r == c:
                    continue
                self.assertEqual(F[r, c], 0)

        # confirm decay works in forward pass
        # also tests that kf.forward works with `out_timesteps > input.shape[1]`
        pred = torch_kf(
            initial_state=torch_kf._prepare_initial_state(None, start_offsets=np.zeros(1)),
            X=torch.randn(1, num_times, 3),
            out_timesteps=num_times
        )
        for t in range(1, num_times):
            for i in range(3):
                self.assertLess(pred.state_means[:, t, i].abs(), pred.state_means[:, t - 1, i].abs())

    @torch.no_grad()
    def test_equations(self):
        data = torch.tensor([[-5., 5., 1., 0., 3.]]).unsqueeze(-1)
        num_times = data.shape[1]

        # make torch kf:
        torch_kf = KalmanFilter(
            processes=[LocalTrend(id='lt', decay_velocity=None, measure='y', velocity_multi=1.)],
            measures=['y']
        )
        expectedF = torch.tensor([[1., 1.], [0., 1.]])
        expectedH = torch.tensor([[1., 0.]])
        kwargs_per_process = torch_kf._parse_design_kwargs(input=data, out_timesteps=num_times)
        predict_kwargs, update_kwargs = torch_kf._build_design_mats(
            kwargs_per_process, num_groups=1, out_timesteps=num_times
        )
        F = predict_kwargs['F'][0][0]
        Q = predict_kwargs['Q'][0][0]
        H = update_kwargs['H'][0][0]
        R = update_kwargs['R'][0][0]
        assert torch.isclose(expectedF, F).all()
        assert torch.isclose(expectedH, H).all()

        # make filterpy kf:
        filter_kf = filterpy_KalmanFilter(dim_x=2, dim_z=1)
        filter_kf.x, filter_kf.P = torch_kf._prepare_initial_state(None)
        filter_kf.x = filter_kf.x.detach().numpy().T
        filter_kf.P = filter_kf.P.detach().numpy().squeeze(0)
        filter_kf.Q = Q.numpy()
        filter_kf.R = R.numpy()
        filter_kf.F = F.numpy()
        filter_kf.H = H.numpy()

        # compare:
        sb = torch_kf(data)

        #
        filter_kf.state_means = []
        filter_kf.state_covs = []
        for t in range(num_times):
            # 1step:
            filter_kf.predict()
            # update:
            filter_kf_copy = copy.deepcopy(filter_kf)
            filter_kf.update(data[:, t, :])
            # append:
            filter_kf.state_means.append(filter_kf_copy.x)
            filter_kf.state_covs.append(filter_kf_copy.P)

        assert np.isclose(sb.state_means.numpy().squeeze(), np.stack(filter_kf.state_means).squeeze(), rtol=1e-4).all()
        assert np.isclose(sb.state_covs.numpy().squeeze(), np.stack(filter_kf.state_covs).squeeze(), rtol=1e-4).all()

    @parameterized.expand([(1,), (2,), (3,)])
    @torch.no_grad()
    def test_equations_preds(self, n_step: int = 1):
        from torchcast.utils.data import TimeSeriesDataset
        from pandas import DataFrame

        class LinearModelFixed(LinearModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.no_icov_state_elements = self.state_elements

        kf = KalmanFilter(
            processes=[
                LinearModelFixed(id='lm', predictors=['x1', 'x2'])
            ],
            measures=['y']
        )
        kf._get_measure_scaling = lambda: torch.ones(2)

        kf.state_dict()['initial_mean'][:] = torch.tensor([1.5, -0.5])
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

        from pandas import Series

        if n_step == 0:
            with self.assertRaises(AssertionError):
                kf(y, X=X, n_step=n_step)
            return

        pred = kf(y, X=X, out_timesteps=X.shape[1], n_step=n_step)
        y_series = Series(y.squeeze().numpy())
        for shift in range(-2, 3):
            resid = y_series.shift(shift) - Series(pred.means.squeeze().numpy())
            if shift:
                # check there's no misalignment in internal n_step logic (i.e., realigning the input makes things worse)
                self.assertGreater((resid ** 2).mean(), 1.)
            else:
                self.assertLess((resid ** 2).mean(), .02)

    def test_keyword_dispatch(self):
        _counter = defaultdict(int)

        def check_input(func: Callable, expected: torch.Tensor) -> Callable:
            def outfunc(inputs, num_groups, num_times):
                x = inputs.get('X')
                _counter[func.__name__] += 1
                self.assertIsNotNone(x)
                _bool = (x == expected)
                if hasattr(_bool, 'all'):
                    _bool = _bool.all().item()
                self.assertTrue(_bool)
                return func(inputs, num_groups, num_times)

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
            proc._build_h_mat = check_input(proc._build_h_mat, expected[nm])
        kf(data, X=_predictors * 0.)
        expected_call_count = len(expected)
        self.assertGreaterEqual(_counter['_build_h_mat'], expected_call_count)

        # separate ---
        expected['lm2'] = torch.ones(1)
        # individual input:
        kf = _make_kf()
        for nm, proc in kf.processes.items():
            proc._build_h_mat = check_input(proc._build_h_mat, expected[nm])
        kf(data, lm1__X=_predictors * 0., lm2__X=_predictors)
        expected_call_count += len(expected)
        self.assertGreaterEqual(_counter['_build_h_mat'], expected_call_count)

        # specific overrides general
        kf(data, X=_predictors * 0., lm2__X=_predictors)
        expected_call_count += len(expected)
        self.assertGreaterEqual(_counter['_build_h_mat'], expected_call_count)

        # make sure check_input is being called:
        with self.assertRaises(AssertionError) as cm:
            kf(data, X=_predictors * 0.)
        self.assertEqual(str(cm.exception).lower(), "false is not true")

    def test_current_time(self):
        _state = {}

        def make_season(current_timestep: torch.Tensor):
            _state['call_counter'] += 1
            return current_timestep % 7

        class Season(Process):
            def __init__(self, id: str):
                super(Season, self).__init__(
                    id=id,
                    state_elements=['x'],
                )
                self.f_tensors['x->x'] = torch.ones(1)
                self.expected_kwargs = ['current_timestep']

            def _build_h_mat(self, inputs: Dict[str, Tensor], num_groups: int, num_times: int) -> Tensor:
                return make_season(inputs['current_timestep'])

        kf = KalmanFilter(
            processes=[Season(id='s1')],
            measures=['y']
        )
        kf._get_measure_scaling = lambda: torch.ones(1)
        data = torch.arange(7).view(1, -1, 1).to(torch.float32)
        for init_state in [0., 1.]:
            kf.state_dict()['initial_mean'][:] = torch.ones(1) * init_state
            _state['call_counter'] = 0
            pred = kf(data)
            # only needed to call once:
            self.assertEqual(_state['call_counter'], 1)

            # more suited to a season test but we'll check anyways:
            if init_state == 1.:
                self.assertTrue((pred.state_means == 1.).all())
            else:
                self.assertGreater(pred.state_means[:, -1], pred.state_means[:, 0])

    @torch.no_grad()
    def test_predictions(self, ndim: int = 2):
        data = torch.zeros((2, 5, ndim))
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'lm{i}', measure=str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)]
        )
        pred = kf(data)
        self.assertEqual(len(tuple(pred)), 2)
        self.assertIsInstance(np.asanyarray(pred), np.ndarray)
        means, covs = pred
        self.assertIsInstance(means, torch.Tensor)
        self.assertIsInstance(covs, torch.Tensor)

        with self.assertRaises(TypeError):
            pred[1]

        with self.assertRaises(TypeError):
            pred[(1,)]

        pred_group2 = pred[[1]]
        self.assertTupleEqual(tuple(pred_group2.covs.shape), (1, 5, ndim, ndim))
        self.assertTrue((pred_group2.state_means == pred.state_means[1, :, :]).all())
        self.assertTrue((pred_group2.state_covs == pred.state_covs[1, :, :, :]).all())

        pred_time3 = pred[:, [2]]
        self.assertTupleEqual(tuple(pred_time3.covs.shape), (2, 1, ndim, ndim))
        self.assertTrue((pred_time3.state_means == pred.state_means[:, 2, :]).all())
        self.assertTrue((pred_time3.state_covs == pred.state_covs[:, 2, :, :]).all())

    @torch.no_grad()
    def test_no_proc_variance(self):
        kf = KalmanFilter(processes=[LinearModel(id='lm', predictors=['x1', 'x2'])], measures=['y'])
        cov = kf.process_covariance({}, num_groups=1, num_times=1).squeeze()
        self.assertEqual(cov.shape[-1], 2)
        self.assertTrue((cov == 0).all())

    @parameterized.expand([
        (torch.float64, 2, False)
    ])
    @torch.no_grad()
    def test_dtype(self, dtype: torch.dtype, ndim: int = 2, compiled: bool = False):
        data = torch.zeros((2, 5, ndim), dtype=dtype)
        kf = KalmanFilter(
            processes=[LocalLevel(id=f'll{i}', measure=str(i)) for i in range(ndim)],
            measures=[str(i) for i in range(ndim)]
        )
        if compiled:
            kf = torch.jit.script(kf)
        kf.to(dtype=dtype)
        pred = kf(data)
        self.assertEqual(pred.means.dtype, dtype)
        loss = pred.log_prob(data)
        self.assertEqual(loss.dtype, dtype)
