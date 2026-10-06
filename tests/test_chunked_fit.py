import copy
import warnings

import numpy as np
import pytest
import torch

from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.covariance import Covariance
from torchcast.process import LocalLevel, LinearModel, Season
from torchcast.state_space import LossFun
from torchcast.utils import Stopping


NUM_GROUPS = 11


def _make_data(num_groups: int = NUM_GROUPS, num_times: int = 20):
    torch.manual_seed(42)
    X = torch.randn(num_groups, num_times, 2)
    y = torch.stack([
        X @ torch.tensor([1., -.5]) + torch.randn(num_groups, 1).cumsum(1),
        torch.randn(num_groups, num_times).cumsum(1),
    ], dim=-1)
    y = y + torch.randn_like(y)
    y[torch.rand_like(y) < .1] = float('nan')
    y[0, 15:] = float('nan')  # trailing missings
    start_offsets = np.array([np.datetime64('2020-01-06') + np.timedelta64(i, 'D') for i in range(num_groups)])
    weights = torch.rand(num_groups, num_times)
    return y, X, start_offsets, weights


def _make_model(variant: str = 'kf') -> KalmanFilter:
    """
    :param variant: 'kf'; 'adaptive' (adaptive-scaling, which has per-group state); 'sigmoid' (a measure-fun, so a
     monte-carlo log-prob using ``mc_sampling``, which is shared across groups); 'binomial' (BinomialFilter, binary y2);
     'group_cov' (measure-variance predicted from ``group_ids``, a covariance kwarg).
    """
    torch.manual_seed(1)
    processes = [
        LocalLevel(id='level1', measure='y1'),
        LinearModel(id='lm', measure='y1', predictors=['x1', 'x2']),
        LocalLevel(id='level2', measure='y2'),
        Season(id='season', measure='y2', period='7D', dt_unit='D', K=1),
    ]
    if variant == 'binomial':
        model = BinomialFilter(processes=processes, measures=['y1', 'y2'], binary_measures=['y2'])
        model.mc_sampling = 50  # set before copying, so copies share the draws
        return model
    measure_covariance = None
    if variant == 'group_cov':
        measure_covariance = Covariance.from_measures(
            ['y1', 'y2'],
            predict_variance=torch.nn.Sequential(torch.nn.Embedding(NUM_GROUPS, 2), torch.nn.Softplus()),
            expected_kwargs=['group_ids'],
        )
    model = KalmanFilter(
        processes=processes,
        measures=['y1', 'y2'],
        measure_covariance=measure_covariance,
        adaptive_scaling=variant == 'adaptive',
        measure_funs={'y2': 'sigmoid'} if variant == 'sigmoid' else None,
    )
    if variant == 'sigmoid':
        model.mc_sampling = 50
    return model


def _prepare_y(y: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == 'binomial':
        y = y.clone()
        y[..., 1] = torch.where(y[..., 1].isnan(), y[..., 1], (y[..., 1] > 0).float())
    return y


@pytest.mark.parametrize('chunk_size', [1, 3, 11, 50])
@pytest.mark.parametrize('use_weights', [False, True])
@pytest.mark.parametrize('variant', ['kf', 'adaptive', 'sigmoid', 'binomial', 'group_cov'])
def test_chunked_loss_and_grad(chunk_size: int, use_weights: bool, variant: str):
    y, X, start_offsets, weights = _make_data()
    y = _prepare_y(y, variant)
    kwargs = {'X': X, 'start_offsets': start_offsets}
    if variant == 'group_cov':
        kwargs['group_ids'] = torch.arange(NUM_GROUPS)
    if use_weights:
        kwargs['get_loss'] = LossFun(weights=weights)

    model = _make_model(variant)
    model_chunked = copy.deepcopy(model)

    # grads are zeroed at the end of fit, so capture them inside the closure via a hook on the optimizer:
    results = []
    for m, cs in [(model, None), (model_chunked, chunk_size)]:
        grads = {}
        optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.)
        optimizer.register_step_post_hook(
            lambda opt, *args: grads.update(g=torch.cat([p.grad.reshape(-1) for p in opt.param_groups[0]['params']]))
        )
        losses = []
        m.fit(
            y,
            optimizer=optimizer,
            stopping=Stopping(max_iter=1),
            verbose=0,
            set_initial_values=False,
            chunk_size=cs,
            callbacks=[losses.append],
            **kwargs
        )
        results.append((losses[0], grads['g']))

    (loss, grad), (loss_chunked, grad_chunked) = results
    assert loss == pytest.approx(loss_chunked, rel=1e-5)
    assert torch.allclose(grad, grad_chunked, rtol=1e-4, atol=1e-6)


def test_chunked_fit():
    y, X, start_offsets, _ = _make_data()
    model = _make_model()
    model_chunked = copy.deepcopy(model)
    stopping = {'max_iter': 5}
    model.fit(y, X=X, start_offsets=start_offsets, verbose=0, stopping=stopping)
    model_chunked.fit(y, X=X, start_offsets=start_offsets, verbose=0, stopping=stopping, chunk_size=4)
    for (nm, p1), p2 in zip(model.named_parameters(), model_chunked.parameters()):
        assert torch.allclose(p1, p2, rtol=1e-3, atol=1e-4), nm


def test_chunked_laplace(monkeypatch):
    from torchcast.state_space import state_space

    hessians = []

    def _hessian(*args, **kwargs):
        hessians.append(hessian(*args, **kwargs))
        return hessians[-1]

    hessian = state_space.hessian
    monkeypatch.setattr(state_space, 'hessian', _hessian)

    y, X, start_offsets, weights = _make_data()
    model = _make_model()
    kwargs = {'X': X, 'start_offsets': start_offsets, 'get_loss': LossFun(weights=weights, reduce='sum')}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # unfitted model: hessian isn't PD
        _, names = model.get_laplace_mvnorm(y, **kwargs)
        assert len(hessians) == 1
        _, names_chunked = model.get_laplace_mvnorm(y, chunk_size=4, **kwargs)
        assert len(hessians) == 1 + 3
    assert names == names_chunked
    assert torch.allclose(hessians[0], sum(hessians[1:]), rtol=1e-4, atol=1e-3)


def test_chunked_callable_kwargs():
    """
    callable_kwargs whose output is part of the graph: must be recomputed per chunk (each chunk's backward frees it)
    """
    y, X, start_offsets, _ = _make_data()
    model = _make_model()
    scale = torch.nn.Parameter(torch.ones(1))
    model.register_parameter('x_scale', scale)
    model.fit(
        y,
        start_offsets=start_offsets,
        callable_kwargs={'X': lambda: X * scale},
        verbose=0,
        stopping={'max_iter': 2},
        chunk_size=4
    )
    assert scale.grad is None  # zeroed (set to none) at the end of fit
    assert scale.item() != 1.


def test_subset_kwargs():
    num_groups, num_times = 6, 6  # equal, so shapes alone are ambiguous
    model = _make_model()
    model.measure_covariance.expected_kwargs = ['group_ids']
    X = torch.randn(num_groups, num_times, 2)
    kwargs = {
        'X': X,  # ProcessKwarg -> always split
        'lm__X': X,  # per-process override of the same
        'group_ids': torch.arange(num_groups),  # covariance kwarg -> split by shape
        'other': torch.arange(num_times),  # unknown -> never split, though its first dim equals num_groups
        'initial_state': (torch.zeros(num_groups, 5), torch.eye(5).unsqueeze(0)),  # cov broadcasts over groups
    }
    out = model._subset_kwargs(kwargs, slice(2, 4), num_groups)
    assert torch.equal(out['X'], X[2:4])
    assert torch.equal(out['lm__X'], X[2:4])
    assert torch.equal(out['group_ids'], torch.tensor([2, 3]))
    assert out['other'] is kwargs['other']
    assert out['initial_state'][0].shape == (2, 5)
    assert out['initial_state'][1] is kwargs['initial_state'][1]

    with pytest.raises(ValueError, match='`X`'):
        model._subset_kwargs({'X': X[:3]}, slice(0, 2), num_groups)
