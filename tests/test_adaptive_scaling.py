import copy
import io
import math

import torch

from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalLevel
from torchcast.state_space.adaptive_scaling import EWMAdaptiveScaler


def _make_scaler(num_measures: int = 1, rho: float = .5, weight: float = .5) -> EWMAdaptiveScaler:
    scaler = EWMAdaptiveScaler(num_measures=num_measures)
    scaler.initialize(num_timesteps=100)
    with torch.no_grad():
        for r in scaler._rhos:
            r.raw[:] = math.log(rho / (1 - rho))  # logit, so `rho()` == rho
        scaler.weight[:] = weight
    scaler.reset()
    return scaler


@torch.no_grad()
def test_no_observations_multiplier_is_one():
    scaler = _make_scaler(num_measures=2)
    # group 0 is never observed; group 1 is always observed:
    skip_mask = torch.tensor([[True, True], [False, False]])
    for _ in range(5):
        multi = scaler(torch.full((2, 2), 3.), skip_mask)
        assert (multi[0] == 1.).all()
        assert (scaler._running[0] == 1.).all()
        assert (multi[1] != 1.).all()


@torch.no_grad()
def test_one_observation_shrinks_toward_one():
    scaler = _make_scaler(rho=.5)
    resid = 3.
    scaler(torch.tensor([[resid]]), torch.tensor([[False]]))
    running = scaler._running.item()
    assert 1. < running < resid ** 2

    tau = scaler._taus.exp().item()
    alpha = .5 * tau / (1 + tau)
    assert math.isclose(running, (1 - alpha) * 1. + alpha * resid ** 2, rel_tol=1e-5)

    # a later skipped observation leaves it unchanged:
    scaler(torch.tensor([[100.]]), torch.tensor([[True]]))
    assert math.isclose(scaler._running.item(), running, rel_tol=1e-6)


def _make_kf() -> KalmanFilter:
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='lvl', measure='y')], measures=['y'], adaptive_scaling=True)
    with torch.no_grad():
        kf.adaptive_scaling.weight[:] = .5
    kf.adaptive_scaling.initialize(num_timesteps=20)
    return kf


@torch.no_grad()
def test_unobserved_group_unscaled_in_model():
    kf = _make_kf()
    y = torch.randn(2, 20, 1) * 5
    y[1] = float('nan')

    # weight=0 means multiplier 1 regardless of residuals:
    kf_noscale = copy.deepcopy(kf)
    kf_noscale.adaptive_scaling.weight[:] = 0.

    pred = kf(y)
    pred_noscale = kf_noscale(y)
    assert torch.allclose(pred.covs[1], pred_noscale.covs[1])
    assert not torch.allclose(pred.covs[0], pred_noscale.covs[0])


@torch.no_grad()
def test_legacy_zero_init():
    # instances unpickled from older versions have no `_running_init` attribute, so fall back to the class-default:
    kf = _make_kf()
    del kf.adaptive_scaling.__dict__['_running_init']
    assert kf.adaptive_scaling._running_init == 0.
    buff = io.BytesIO()
    torch.save(kf, buff)
    buff.seek(0)
    kf_loaded = torch.load(buff, weights_only=False)
    assert kf_loaded.adaptive_scaling._running_init == 0.
    kf_loaded.adaptive_scaling.reset()
    kf_loaded.adaptive_scaling(torch.zeros(1, 1), torch.tensor([[True]]))
    assert math.isclose(kf_loaded.adaptive_scaling._running.item(), kf_loaded.adaptive_scaling.eps, rel_tol=1e-6)

    # new instances pickle the new behavior:
    buff = io.BytesIO()
    torch.save(_make_kf(), buff)
    buff.seek(0)
    assert torch.load(buff, weights_only=False).adaptive_scaling._running_init == 1.


@torch.no_grad()
def test_legacy_state_dict():
    # state-dicts from older versions have no extra-state:
    sd = _make_kf().state_dict()
    assert sd.pop('adaptive_scaling._extra_state') == {'running_init': 1.}
    kf = _make_kf()
    kf.load_state_dict(sd)  # strict
    assert kf.adaptive_scaling._running_init == 0.

    # round-trip of a new state-dict:
    kf.load_state_dict(_make_kf().state_dict())
    assert kf.adaptive_scaling._running_init == 1.
