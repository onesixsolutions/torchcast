"""
``StateSpaceModel.fit()``'s closure caches its last evaluation: L-BFGS with ``max_iter=1`` starts each ``step()`` by
evaluating the point the previous line search accepted, which it almost always evaluated last.
"""
import math

import pytest
import torch

from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LinearModel, LocalLevel
from torchcast.state_space.state_space import _OptimizerClosure
from torchcast.utils import Stopping

NUM_ITER = 8


def _make_kf(ndim: int = 2) -> KalmanFilter:
    return KalmanFilter(
        processes=[LocalLevel(id=f'll{i}', measure=str(i)) for i in range(ndim)] +
                  [LinearModel(id=f'lm{i}', predictors=['x1', 'x2', 'x3'], measure=str(i)) for i in range(ndim)],
        measures=[str(i) for i in range(ndim)]
    )


def _data(num_groups: int = 60, num_times: int = 20):
    torch.manual_seed(123)
    X = torch.randn((num_groups, num_times, 3))
    with torch.no_grad():
        y = torch.distributions.MultivariateNormal(*_make_kf().simulate(out_timesteps=num_times,
                                                                        num_groups=num_groups, X=X)).sample()
    return y, X


def _fit(y, X, cache: bool, monkeypatch) -> tuple[KalmanFilter, list, int]:
    if not cache:
        original = _OptimizerClosure.__call__

        def _no_cache(self):
            self._cache = None
            return original(self)

        monkeypatch.setattr(_OptimizerClosure, '__call__', _no_cache)
    torch.manual_seed(1)
    kf = _make_kf()
    forwards = []
    kf.register_forward_hook(lambda *args: forwards.append(1))
    losses = []
    # (abstol=0: never converges, so both fits run exactly NUM_ITER iterations)
    kf.fit(y, X=X, stopping=Stopping(abstol=0., max_iter=NUM_ITER), callbacks=[losses.append], verbose=0)
    monkeypatch.undo()
    return kf, losses, len(forwards)


def test_fit_caches_repeated_evaluations(monkeypatch):
    y, X = _data()
    kf_cached, losses_cached, forwards_cached = _fit(y, X, cache=True, monkeypatch=monkeypatch)
    kf_plain, losses_plain, forwards_plain = _fit(y, X, cache=False, monkeypatch=monkeypatch)

    # the same fit (the cache serves identical evaluations):
    for (name, p1), p2 in zip(kf_cached.named_parameters(), kf_plain.parameters()):
        torch.testing.assert_close(p1, p2, msg=name)
    assert losses_cached == losses_plain

    # with fewer forward passes: without the cache, each iteration also re-evaluates its starting point, and (in this
    # test) the loss after each step is a separate evaluation, so the uncached fit before this change did
    # ``forwards_plain - NUM_ITER``:
    forwards_before = forwards_plain - NUM_ITER
    print(f"\nforward passes in {NUM_ITER} iterations: {forwards_before} before, {forwards_cached} with the cache")
    assert forwards_cached <= forwards_before - (NUM_ITER - 2)


def test_fit_reports_the_current_iterates_loss(monkeypatch):
    y, X = _data()
    kf, losses, _ = _fit(y, X, cache=True, monkeypatch=monkeypatch)
    assert len(losses) == NUM_ITER
    with torch.no_grad():
        final_loss = -kf(y, X=X).log_prob(y).mean().item()
    # (the loss passed to callbacks and stopping is at the parameters the iteration ended on, not where it started):
    assert abs(losses[-1] - final_loss) < 1e-5
    assert all(later <= earlier + 1e-6 for earlier, later in zip(losses, losses[1:]))


@pytest.mark.parametrize('bad', ['nan', 'linalg_error'])
def test_bad_trial_point_backtracks(bad):
    # the first trial point of iteration 2's line search is unusable: a nan loss (as when a trial step overflows) or
    # invalid covariances. the line search backs off from it and the fit carries on (with an infinite loss, the line
    # search's interpolation made the next trial step nan and the fit failed)
    y, X = _data()
    torch.manual_seed(1)
    kf = _make_kf()
    calls = {'n': 0}

    def get_loss(pred, _y):
        calls['n'] += 1
        loss = -pred.log_prob(_y).mean()
        if calls['n'] == 3:
            if bad == 'linalg_error':
                raise torch.linalg.LinAlgError('simulated')
            return loss * float('nan')
        return loss

    losses = []
    kf.fit(y, X=X, get_loss=get_loss, stopping=Stopping(abstol=0., max_iter=4), callbacks=[losses.append], verbose=0)
    assert len(losses) == 4 and all(map(math.isfinite, losses)), losses
    assert all(later <= earlier + 1e-6 for earlier, later in zip(losses, losses[1:]))
    assert all(torch.isfinite(p).all() for p in kf.parameters())
