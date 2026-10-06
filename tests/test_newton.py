import copy
import warnings

import pytest
import torch

from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalLevel
from torchcast.state_space import LossFun, NewtonResult
from torchcast.state_space.newton import saddle_free_step

from tests.test_chunked_fit import _make_data, _make_model


def _ar_data(num_groups: int = 30, num_times: int = 60, seed: int = 0) -> torch.Tensor:
    """
    random-walk + AR(1) + noise: the variances (and AR decay) are correlated, so LBFGS is slow along a ridge.
    """
    gen = torch.Generator().manual_seed(seed)
    rw = (torch.randn(num_groups, num_times, generator=gen) * .3).cumsum(1)
    ar = torch.zeros(num_groups, num_times)
    eps = torch.randn(num_groups, num_times, generator=gen) * .7
    for t in range(1, num_times):
        ar[:, t] = .9 * ar[:, t - 1] + eps[:, t]
    return (rw + ar + torch.randn(num_groups, num_times, generator=gen)).unsqueeze(-1)


def _ar_model() -> KalmanFilter:
    torch.manual_seed(0)
    return KalmanFilter(processes=[LocalLevel(id='rw'), LocalLevel(id='ar', decay=(.5, .99))], measures=['y'])


@pytest.mark.parametrize('max_step', [float('inf'), .1])
def test_saddle_free_step_is_descent(max_step: float):
    torch.manual_seed(0)
    for _ in range(20):
        A = torch.randn(6, 6, dtype=torch.float64)
        hess = (A + A.T) / 2
        evals, evecs = torch.linalg.eigh(hess)
        assert (evals < 0).any()
        grad = torch.randn(6, dtype=torch.float64)
        step, decrement = saddle_free_step(grad, evals, evecs, eig_floor=1e-6, max_step=max_step)
        assert (grad @ step).item() < 0
        assert step.abs().max().item() <= max_step + 1e-12
        assert decrement > 0


def test_newton_improves_on_lbfgs_ridge():
    y = _ar_data()
    model = _ar_model()
    model.fit(y, verbose=0, stopping={'abstol': 1e-3})  # loss-based stopping: stops early on the ridge
    lbfgs_loss = LossFun()(model(y), y).item()

    # (full convergence takes ~40 steps here, as the AR's initial variance heads to zero; but most of the gain is in
    # the first few)
    result = model.newton_refine(y, max_steps=5, verbose=False)
    loss_scale = y.shape[0] * y.shape[1]
    assert (lbfgs_loss - result.loss) * loss_scale > 2.  # log-likelihood units
    # losses never increase (beyond float noise):
    losses = [h['loss'] for h in result.history] + [result.loss]
    assert all(b <= a + 1e-6 for a, b in zip(losses, losses[1:]))


def test_newton_converges():
    torch.manual_seed(0)
    y = (torch.randn(20, 30) * .5).cumsum(1).unsqueeze(-1) + torch.randn(20, 30, 1)
    model = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'])
    model.fit(y, verbose=0, stopping={'abstol': 1e-2})
    result = model.newton_refine(y, max_steps=15, verbose=False)
    assert result.converged


def test_newton_chunked_matches_unchunked():
    y, X, start_offsets, weights = _make_data(num_times=10)
    model = _make_model()
    model.fit(y, X=X, start_offsets=start_offsets, verbose=0, stopping={'max_iter': 10})
    model_chunked = copy.deepcopy(model)

    kwargs = {'X': X, 'start_offsets': start_offsets, 'get_loss': LossFun(weights=weights)}
    newton_kwargs = {'max_steps': 2, 'decrement_tol': None, 'verbose': False}
    result = model.newton_refine(y, **newton_kwargs, **kwargs)
    result_chunked = model_chunked.newton_refine(y, chunk_size=4, hessian_chunk_size=3, **newton_kwargs, **kwargs)

    assert len(result.history) == len(result_chunked.history) == 2
    for h1, h2 in zip(result.history, result_chunked.history):
        assert h1['loss'] == pytest.approx(h2['loss'], rel=1e-5)
        assert h1['ls_scale'] == h2['ls_scale']
    assert torch.allclose(result.params, result_chunked.params, rtol=1e-3, atol=1e-3)
    assert torch.allclose(result.hessian, result_chunked.hessian, rtol=1e-2, atol=1e-4)


def test_newton_hessian_matches_laplace(monkeypatch):
    from torchcast.state_space import state_space

    y = _ar_data(num_groups=10, num_times=30)
    model = _ar_model()
    model.fit(y, verbose=0)
    result = model.newton_refine(y, max_steps=2, decrement_tol=None, verbose=False)

    hessians = []

    def _hessian(*args, **kwargs):
        hessians.append(hessian(*args, **kwargs))
        return hessians[-1]

    hessian = state_space.hessian
    monkeypatch.setattr(state_space, 'hessian', _hessian)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # not necessarily PD
        mvnorm, names = model.get_laplace_mvnorm(y)
    assert names == result.param_names
    assert torch.allclose(mvnorm.loc, result.params)
    assert torch.allclose(result.summed_hessian(), hessians[0], rtol=1e-4, atol=1e-3)


def test_newton_reuse_hessian():
    y = _ar_data(num_groups=10, num_times=30)
    model = _ar_model()
    model.fit(y, verbose=0, stopping={'max_iter': 3})
    result = model.newton_refine(y, max_steps=4, reuse_hessian=2, decrement_tol=None, verbose=False)
    assert [h['hessian_age'] for h in result.history][:3] == [0, 1, 2]
    # final hessian is at the final params, even though the last step may have used a reused one:
    fresh = copy.deepcopy(model).newton_refine(y, max_steps=0, verbose=False)
    assert torch.allclose(fresh.params, result.params)
    assert torch.allclose(fresh.hessian, result.hessian, rtol=1e-4, atol=1e-6)


def test_fit_newton_finish():
    y = _ar_data(num_groups=10, num_times=30)
    model = _ar_model()
    model.fit(y, verbose=0, stopping={'max_iter': 3}, newton_finish={'max_steps': 2, 'decrement_tol': None})
    assert isinstance(model.newton_result, NewtonResult)
    assert len(model.newton_result.history) == 2
    assert len(model.newton_result.weak_directions(num=2)) == 2


@pytest.mark.parametrize('variant', ['sigmoid', 'adaptive', 'binomial'])
def test_newton_end_to_end(variant: str):
    """
    monte-carlo log-prob (deterministic, via ``mc_sampling``), adaptive-scaling, binomial
    """
    from tests.test_chunked_fit import _prepare_y

    y, X, start_offsets, _ = _make_data(num_groups=6, num_times=10)
    y = _prepare_y(y, variant)
    model = _make_model(variant)
    model.fit(y, X=X, start_offsets=start_offsets, verbose=0, stopping={'max_iter': 3}, set_initial_values=False)
    result = model.newton_refine(y, X=X, start_offsets=start_offsets, max_steps=1, verbose=False)
    losses = [h['loss'] for h in result.history] + [result.loss]
    assert all(b <= a + 1e-5 for a, b in zip(losses, losses[1:]))
