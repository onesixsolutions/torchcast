import copy
import io
import math

import pytest
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


def _make_kf_separate(process_weight: float) -> KalmanFilter:
    torch.manual_seed(0)
    measures = ['y1', 'y2']
    kf = KalmanFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
        measures=measures,
        adaptive_scaling=EWMAdaptiveScaler(num_measures=2, separate_process_weight=True),
    )
    with torch.no_grad():
        kf.adaptive_scaling.weight.fill_(.8)
        kf.adaptive_scaling.process_weight.fill_(process_weight)
    kf.adaptive_scaling.initialize(num_timesteps=25)
    return kf


def _noisy_y() -> torch.Tensor:
    torch.manual_seed(1)
    # groups on very different scales, so the multipliers matter:
    y = torch.randn((3, 25, 2)).cumsum(1) * torch.tensor([.1, 1., 10.]).view(3, 1, 1)
    y[1, 5:8, 0] = float('nan')
    return y


def test_separate_process_weight_default_off():
    scaler = EWMAdaptiveScaler(num_measures=2)
    assert not scaler.has_process_multiplier
    assert 'process_weight' not in dict(scaler.named_parameters())
    with pytest.raises(RuntimeError, match="separate_process_weight"):
        scaler.process_multiplier()


@torch.no_grad()
def test_separate_process_weight_tied_matches_default():
    # with process_weight == weight, the separate version is the same model as the default:
    y = _noisy_y()
    kf_sep = _make_kf_separate(process_weight=.8)
    kf_default = KalmanFilter(processes=copy.deepcopy(list(kf_sep.processes.values())), measures=kf_sep.measures,
                              adaptive_scaling=True)
    kf_default.load_state_dict({k: v for k, v in kf_sep.state_dict().items() if 'process_weight' not in k})
    kf_default.adaptive_scaling._called_initialize = True
    pred_sep, pred_default = kf_sep(y, n_step=3), kf_default(y, n_step=3)
    assert torch.allclose(pred_sep.state_covs, pred_default.state_covs, rtol=1e-5, atol=1e-6)
    assert torch.allclose(pred_sep.covs, pred_default.covs, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_separate_process_weight_zero_leaves_process_cov_unscaled():
    y = _noisy_y()
    kf_zero, kf_tied = _make_kf_separate(process_weight=0.), _make_kf_separate(process_weight=.8)
    pred_zero, pred_tied = kf_zero(y), kf_tied(y)
    # the process-covariance isn't scaled with process_weight=0, so the state-covariances differ:
    assert not torch.allclose(pred_zero.state_covs, pred_tied.state_covs, rtol=1e-2)
    # ... and match those of a model whose adaptive multiplier never reaches the process-covariance (it's applied to
    # it only in `_predict_cov`):
    kf_check = copy.deepcopy(kf_tied)
    orig = kf_check._predict_cov
    kf_check._predict_cov = lambda cov, transition_mat, Q, scaling=None, mask=None: orig(
        cov, transition_mat, Q, scaling=None, mask=mask
    )
    assert torch.allclose(kf_check(y).state_covs, pred_zero.state_covs, rtol=1e-5, atol=1e-7)


def test_separate_process_weight_is_learned():
    y = _noisy_y()
    kf = _make_kf_separate(process_weight=.3)
    kf(y).log_prob(y).mean().backward()
    assert kf.adaptive_scaling.process_weight.grad is not None
    assert (kf.adaptive_scaling.process_weight.grad != 0).any()


@torch.no_grad()
def test_separate_process_weight_continuation_and_pickle():
    y = _noisy_y()
    kf = _make_kf_separate(process_weight=.2)
    split = 15
    full = kf(y, include_updates_in_output=True)
    state = kf(y[:, :split], include_updates_in_output=True).get_state_at_times(split - 1)
    assert state.scaling['multiplier'].shape[-1] == 2 * len(kf.measures)
    cont = kf(y[:, split:], initial_state=state)
    assert torch.allclose(cont.state_covs, full.state_covs[:, split:], rtol=1e-4, atol=1e-6)
    assert torch.allclose(cont.measure_covs, full.measure_covs[:, split:], rtol=1e-4, atol=1e-6)

    buff = io.BytesIO()
    torch.save(kf, buff)
    buff.seek(0)
    kf_loaded = torch.load(buff, weights_only=False)
    assert torch.allclose(kf_loaded(y).state_covs, kf(y).state_covs)
    kf_loaded.load_state_dict(kf.state_dict())  # strict

    # scalers created without the option (incl. those pickled before it existed) have no `process_weight`:
    assert not _make_kf().adaptive_scaling.has_process_multiplier
