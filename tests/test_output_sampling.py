"""
Tests for how sample-based outputs draw their noise: randomized quasi-monte-carlo (scrambled Sobol). (Accuracy vs.
ground-truth is tested in test_nonlinear.py; see ``benchmarks/output_sampling.py`` for a comparison with monte-carlo.)
"""
import math
from unittest import mock

import numpy as np
import pytest
import torch

from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalLevel
from torchcast.state_space import Predictions
from torchcast.state_space import predictions as predictions_module
from torchcast.mixture.component import MixtureComponent


def _sigmoid_predictions(mixture: bool = False, num_groups: int = 4) -> Predictions:
    torch.manual_seed(0)
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        measure_funs={'y': 'sigmoid'},
        mixture=[MixtureComponent(measure='y', offset_init=-.4, prob_init=.2, id='low')] if mixture else None,
    )
    with torch.no_grad():
        kf.measure_covariance.cholesky_log_diag.fill_(math.log(.05))
        kf.process_covariance.cholesky_log_diag.fill_(math.log(2.))
    kf.mc_sampling = 50
    y = torch.rand(num_groups, 6, 1) * .6 + .2
    with torch.no_grad():
        return kf(y, out_timesteps=10)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _monte_carlo():
    """The private debugging flag: plain monte-carlo draws, as before quasi-monte-carlo was added."""
    return mock.patch.object(predictions_module, '_QMC_OUTPUTS', False)


@torch.no_grad()
def test_reproducible_with_seed_or_generator():
    """The same seed (or generator) gives the same outputs; a different one, a different scramble."""
    pred = _sigmoid_predictions(mixture=True)
    kwargs = dict(use_map=False, num_samples=256)
    assert torch.equal(pred.get_means(**kwargs), pred.get_means(**kwargs))
    assert torch.equal(pred.get_means(**kwargs, generator=_gen(1)), pred.get_means(**kwargs, generator=_gen(1)))
    assert not torch.equal(pred.get_means(**kwargs, generator=_gen(1)), pred.get_means(**kwargs, generator=_gen(2)))

    df_kwargs = dict(conf=.9, use_map=False, num_samples=256, derived={'y2': lambda s: s['y'] ** 2})
    cols = ['mean', 'lower', 'upper']
    df1 = pred.to_dataframe(**df_kwargs, generator=_gen(1))[cols].values
    assert np.array_equal(df1, pred.to_dataframe(**df_kwargs, generator=_gen(1))[cols].values)
    assert not np.array_equal(df1, pred.to_dataframe(**df_kwargs, generator=_gen(2))[cols].values)

    # torch.manual_seed() sets the predictions' seed (when first needed):
    def fresh_means(seed: int) -> torch.Tensor:
        p = _sigmoid_predictions(mixture=True)
        torch.manual_seed(seed)
        return p.get_means(**kwargs)

    assert torch.equal(fresh_means(3), fresh_means(3))
    assert not torch.equal(fresh_means(3), fresh_means(4))


@torch.no_grad()
def test_qmc_draws_are_balanced_sobol():
    """
    The uniforms are a scrambled Sobol sequence: with 2 ** m points, each column has exactly one point in each of the
    2 ** m equal intervals. ``num_samples`` is rounded up to a power of 2, and the state's draws are standard-normal.
    """
    pred = _sigmoid_predictions(mixture=True)
    uniforms = pred._output_uniforms(256)
    em_dim = pred.measurement_model_flat.extended_measure_mat.shape[1]
    assert uniforms.shape == (256, em_dim + 1 + 2 * len(pred.measurement_model.measures))
    cells = torch.sort(torch.floor(uniforms * 256).long(), dim=0).values
    assert torch.equal(cells, torch.arange(256).unsqueeze(-1).expand_as(cells))

    noise = pred._output_state_noise(1000)
    assert noise.shape == (1024, em_dim)
    assert torch.allclose(noise, torch.special.ndtri(pred._output_uniforms(1024)[:, :em_dim]).float())
    assert torch.equal(pred.get_means(use_map=False, num_samples=1000), pred.get_means(use_map=False, num_samples=1024))

    with pytest.raises(ValueError, match="power of 2"):
        predictions_module._scrambled_sobol(1000, 2, seed=0)


@torch.no_grad()
def test_qmc_chunks_agree(monkeypatch):
    """Outputs computed in chunks of groups (for memory) are the same as all at once: the draws are shared."""
    pred = _sigmoid_predictions(mixture=True, num_groups=6)
    kwargs = dict(conf=.9, use_map=False, num_samples=256, derived={'y2': lambda s: s['y'] ** 2})
    df = pred.to_dataframe(**kwargs)
    df_states = pred.to_dataframe(type='states', num_samples=256)
    monkeypatch.setattr(predictions_module, '_MAX_OUTPUT_SAMPLE_ELEMENTS', 256 * 10 * 2 * 2)  # 2 groups per chunk
    cols = ['mean', 'lower', 'upper']
    assert np.allclose(pred.to_dataframe(**kwargs)[cols].values, df[cols].values)
    assert np.array_equal(pred.to_dataframe(type='states', num_samples=256)[cols].values, df_states[cols].values)


@torch.no_grad()
def test_monte_carlo_flag_matches_plain_monte_carlo():
    """The debugging flag: independent standard-normals from the predictions' seed (without rounding to 2 ** m)."""
    pred = _sigmoid_predictions()
    em_dim = pred.measurement_model_flat.extended_measure_mat.shape[1]
    white_noise = torch.randn((300, em_dim), generator=_gen(pred._get_output_seed()))
    samples = pred._get_measured_mean_samples(
        measurement_model=pred.measurement_model_flat,
        state_means=pred.state_means_flat,
        state_covs=pred.state_covs_flat,
        white_noise=white_noise,
    )
    expected = samples.mean(0).view(pred.num_groups, pred.num_timesteps, 1)
    with _monte_carlo():
        assert torch.allclose(pred.get_means(use_map=False, num_samples=300), expected)


@torch.no_grad()
def test_sample_is_unaffected():
    """``sample()`` is plain monte-carlo, unaffected by quasi-monte-carlo outputs."""
    pred = _sigmoid_predictions(mixture=True)
    s_qmc = pred.sample(64, shared_noise=False, generator=_gen(0))
    with _monte_carlo():
        s_mc = pred.sample(64, shared_noise=False, generator=_gen(0))
    assert torch.equal(s_qmc.observations, s_mc.observations)


@torch.no_grad()
def test_qmc_more_accurate_than_mc():
    """
    For a smooth function of the state (here, the mean of a sigmoid), QMC's error is much smaller than MC's at the same
    number of samples.
    """
    pred = _sigmoid_predictions()
    truth = pred.get_means(use_map=False, num_samples=2 ** 16)

    def rmse() -> float:
        errors = [pred.get_means(use_map=False, num_samples=256, generator=_gen(i)) - truth for i in range(10)]
        return torch.stack(errors).pow(2).mean().sqrt().item()

    qmc_rmse = rmse()
    with _monte_carlo():
        mc_rmse = rmse()
    assert mc_rmse > 5 * qmc_rmse
