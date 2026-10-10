"""
Tests for the nonlinear (EKF / monte-carlo) code paths: measure-functions (sigmoid), nonlinear processes
(SaturatedLinearModel), and the BinomialFilter. (The EKF jacobian -- vs. autograd -- is tested in
test_kalman_filter.py.) Where possible, results are compared to an independent ground-truth
(autograd jacobians; gauss-hermite quadrature of exact marginal likelihoods). Monte-carlo noise is pinned with seeded
`FixedWhiteNoise`, so these tests are deterministic.
"""
import math
from typing import Optional

import numpy as np
import pytest
import torch
from scipy import stats
from torch.distributions import Binomial

from torchcast.internals.monte_carlo import FixedWhiteNoise
from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.process import LocalLevel, SaturatedLinearModel
from torchcast.state_space.mixture import MixtureComponent, MixtureModel
from torchcast.state_space.transforms import LogTransform


def _white_noise(num_samples: int, seed: int = 0) -> FixedWhiteNoise:
    return FixedWhiteNoise(num_samples, random_state=np.random.RandomState(seed))


def _gauss_hermite_expectation(fun, mean: torch.Tensor, var: torch.Tensor, num_nodes: int = 200) -> torch.Tensor:
    """E[fun(Z)] for Z ~ N(mean, var), elementwise, with high-order quadrature."""
    x, w = np.polynomial.hermite.hermgauss(num_nodes)
    x = torch.as_tensor(x, dtype=torch.float64)
    w = torch.as_tensor(w / math.sqrt(math.pi), dtype=torch.float64)
    nodes = mean.double().unsqueeze(-1) + (2 * var.double()).sqrt().unsqueeze(-1) * x
    return (w * fun(nodes)).sum(-1)


def _linear_moments(pred, measure_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For a model whose only nonlinearity is a measure-function, the pre-function measured mean (``H @ state``) is
    gaussian: return its mean and variance for each group*time.
    """
    mm = pred.measurement_model_flat
    H = mm._get_linear_measure_mat(0)
    mean = (H @ pred.state_means_flat.unsqueeze(-1)).squeeze(-1)[:, measure_idx]
    var = (H @ pred.state_covs_flat @ H.transpose(-1, -2))[:, measure_idx, measure_idx]
    return mean, var


@torch.no_grad()
def test_saturated_measured_mean_formula():
    torch.manual_seed(0)
    process = SaturatedLinearModel(id='slm', predictors=['a', 'b'])
    X = torch.randn(5, 1, 2)
    cache = process.prepare_measurement_cache(X=X)
    mean = torch.randn(5, 3)  # coefficients for a, b; then the ceiling
    measured = process.get_measured_mean(mean, time=0, cache=cache)
    yhat = (X[:, 0] * mean[:, :2]).sum(-1)
    ceiling = mean[:, 2]
    # the measurement function should: equal yhat far below the ceiling, and approach the ceiling far above it:
    far_below = process.get_measured_mean(
        torch.cat([mean[:, :2], (yhat + 50).unsqueeze(-1)], -1), time=0, cache=cache
    )
    assert torch.allclose(far_below, yhat, atol=1e-3)
    far_above = process.get_measured_mean(
        torch.cat([mean[:, :2], (yhat - 50).unsqueeze(-1)], -1), time=0, cache=cache
    )
    assert torch.allclose(far_above, yhat - 50, atol=1e-2)
    # and never exceed min(yhat, ceiling) by much / never exceed yhat:
    assert (measured <= yhat + 1e-6).all()


@torch.no_grad()
def test_sigmoid_mc_log_prob_matches_quadrature():
    """
    With a sigmoid measure-function and gaussian likelihood, the marginal likelihood is
    ``E[N(y; sigmoid(Z), R)]`` over the (gaussian) pre-sigmoid measured mean ``Z``.
    """
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    kf.mc_sampling = _white_noise(20_000)
    y = torch.rand(3, 8, 1) * .6 + .2
    pred = kf(y)
    lp = pred.log_prob(y).reshape(-1).double()

    z_mean, z_var = _linear_moments(pred, 0)
    r = pred.measure_covs_flat[:, 0, 0].double()
    obs = y.reshape(-1).double()

    def density(z):
        mu = torch.sigmoid(z.clamp(-8, 8))
        return torch.exp(-.5 * (obs.unsqueeze(-1) - mu) ** 2 / r.unsqueeze(-1)) / (2 * math.pi * r.unsqueeze(-1)).sqrt()

    exact = _gauss_hermite_expectation(density, z_mean, z_var).log()
    assert torch.allclose(lp, exact, atol=.005)


@pytest.mark.parametrize("observed_counts", [True, False])
@torch.no_grad()
def test_binomial_mc_log_prob_matches_quadrature(observed_counts: bool):
    torch.manual_seed(0)
    num_obs = 5
    bf = BinomialFilter(
        processes=[LocalLevel(id='level')], measures=['y'], observed_counts=observed_counts,
        do_post_hoc_correction=False
    )
    bf.mc_sampling = _white_noise(20_000)
    counts = torch.randint(0, num_obs + 1, (3, 8, 1)).float()
    y = counts if observed_counts else counts / num_obs
    pred = bf(y, num_obs=num_obs)
    lp = pred.log_prob(y).reshape(-1).double()

    z_mean, z_var = _linear_moments(pred, 0)
    k = counts.reshape(-1).double()

    def lik(z):
        p = torch.sigmoid(z.clamp(-8, 8))
        return Binomial(total_count=num_obs, probs=p).log_prob(k.unsqueeze(-1)).exp()

    exact = _gauss_hermite_expectation(lik, z_mean, z_var).log()
    # (monte-carlo error with 20k samples is ~.003 here -- confirmed to shrink with more samples, i.e. not a bias)
    assert torch.allclose(lp, exact, atol=.015)
    assert abs((lp - exact).mean()) < .005


@torch.no_grad()
def test_binomial_counts_vs_proportions():
    """The same data as counts (observed_counts=True) or proportions (observed_counts=False) is the same model."""
    torch.manual_seed(0)
    num_obs = 4
    counts = torch.randint(0, num_obs + 1, (3, 10, 1)).float()
    counts[0, 4] = float('nan')

    preds = {}
    for observed_counts in (True, False):
        torch.manual_seed(1)
        bf = BinomialFilter(processes=[LocalLevel(id='level')], measures=['y'], observed_counts=observed_counts)
        bf.mc_sampling = _white_noise(500)
        y = counts if observed_counts else counts / num_obs
        preds[observed_counts] = (bf(y, num_obs=num_obs), y)
    (p_counts, y_counts), (p_props, y_props) = preds[True], preds[False]
    assert torch.allclose(p_counts.state_means, p_props.state_means)
    assert torch.allclose(p_counts.state_covs, p_props.state_covs)
    assert torch.allclose(p_counts.log_prob(y_counts), p_props.log_prob(y_props), atol=1e-5)
    df_counts = p_counts.to_dataframe(use_map=True)
    df_props = p_props.to_dataframe(use_map=True)
    assert np.allclose(df_counts['mean'].values, df_props['mean'].values)


@torch.no_grad()
def test_binomial_update_adds_binomial_variance():
    torch.manual_seed(0)
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend']],
        measures=['visit', 'spend'],
        binary_measures=['visit'],
        observed_counts=False,
        do_post_hoc_correction=False,
    )
    measured_mean = torch.tensor([[.2, 1.], [.7, 2.]])
    measure_cov = torch.diag_embed(torch.tensor([[0., .5], [0., .5]]))
    num_obs = torch.tensor([[1.], [3.]])
    input = torch.tensor([[1., 1.5], [0., 2.5]])
    _, mm_out, mcov_out = bf._prepare_update(
        input=input, measured_mean=measured_mean, measure_cov=measure_cov, num_obs=num_obs, binary_idx=[0]
    )
    expected = measured_mean[:, 0] * (1 - measured_mean[:, 0]) / num_obs[:, 0]
    assert torch.allclose(mcov_out[:, 0, 0], expected)
    assert torch.allclose(mcov_out[:, 1, 1], measure_cov[:, 1, 1])  # gaussian measure unchanged
    assert torch.allclose(mm_out, measured_mean)  # no post-hoc correction


@torch.no_grad()
def test_mc_predictions_match_quadrature():
    """
    For a sigmoid measure-function with gaussian likelihood, the monte-carlo mean (use_map=False) should be
    E[sigmoid(Z)], and the interval-bounds should be quantiles of sigmoid(Z) + noise.
    """
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    # small measurement-noise, so that the monte-carlo error in the mean is dominated by the (pinned) state samples:
    kf.measure_covariance.cholesky_log_diag.fill_(math.log(.05))
    kf.mc_sampling = _white_noise(20_000)
    y = torch.rand(3, 8, 1) * .6 + .2
    pred = kf(y)
    df = pred.to_dataframe(conf=.9, use_map=False)

    z_mean, z_var = _linear_moments(pred, 0)
    r = pred.measure_covs_flat[:, 0, 0].double()
    expected_mean = _gauss_hermite_expectation(lambda z: torch.sigmoid(z.clamp(-8, 8)), z_mean, z_var)
    assert np.allclose(df['mean'].values, expected_mean.numpy(), atol=.003)

    def cdf_at(q):
        q = torch.as_tensor(q, dtype=torch.float64).unsqueeze(-1)
        return _gauss_hermite_expectation(
            lambda z: torch.special.ndtr((q - torch.sigmoid(z.clamp(-8, 8))) / r.unsqueeze(-1).sqrt()), z_mean, z_var
        )

    # (~5 standard-errors for a sample-quantile's coverage with 20k samples)
    assert np.allclose(cdf_at(df['lower'].values).numpy(), .05, atol=.01)
    assert np.allclose(cdf_at(df['upper'].values).numpy(), .95, atol=.01)


@torch.no_grad()
def test_binomial_log_prob_with_missing_binary():
    """Rows where the binary measure is missing get the (exact) gaussian log-prob of the other measure."""
    torch.manual_seed(0)
    visit = (torch.rand(3, 10) > .4).float()
    spend = torch.randn(3, 10).cumsum(1) * .3 + 2
    visit[:, 5] = float('nan')
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend']],
        measures=['visit', 'spend'],
        binary_measures=['visit'],
    )
    bf.mc_sampling = _white_noise(500)
    y = torch.stack([visit, spend], -1)
    pred = bf(y)
    lp = pred.log_prob(y)[:, 5]

    H = pred.measurement_model_flat._get_linear_measure_mat(0)
    rows = torch.arange(pred.num_groups) * pred.num_timesteps + 5
    mean = (H @ pred.state_means_flat.unsqueeze(-1)).squeeze(-1)[rows, 1]
    var = (H @ pred.state_covs_flat @ H.transpose(-1, -2))[rows, 1, 1] + pred.measure_covs_flat[rows, 1, 1]
    expected = torch.distributions.Normal(mean, var.sqrt()).log_prob(spend[:, 5])
    assert torch.allclose(lp, expected, atol=1e-5)


def test_binomial_fit_recovers_probability():
    torch.manual_seed(0)
    true_prob = .25
    y = (torch.rand(30, 40, 1) < true_prob).float()
    bf = BinomialFilter(processes=[LocalLevel(id='level')], measures=['y'])
    bf.mc_sampling = _white_noise(100)
    bf.fit(y, stopping={'max_iter': 15}, verbose=0)
    with torch.no_grad():
        pred = bf(y)
        mean_prob = pred.means[:, -10:, 0].mean().item()
    assert abs(mean_prob - true_prob) < .05


# mixture components on measures with a nonlinear measurement --------------------------------------------------------

# the 'low' component: an offset from the standard regime's (sigmoid) mean, with extra noise
_COMPONENT_OFFSET, _COMPONENT_SD = -.4, .03
_LOW_VALUE = .1


def _sigmoid_mixture_kf(num_samples: int = 20_000, process_var_multi: Optional[float] = None) -> KalmanFilter:
    """
    A sigmoid measure-function with gaussian likelihood, plus a 'low' mixture component.

    :param process_var_multi: If set, the process-variance (relative to the measure-variance) -- e.g. large enough
     that the sigmoid's nonlinearity matters (vs. the linearized approximation).
    """
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        measure_funs={'y': 'sigmoid'},
        mixture=[MixtureComponent(measure='y', mean_init=_COMPONENT_OFFSET, prob_init=.2, id='low')],
    )
    with torch.no_grad():
        kf.measure_covariance.cholesky_log_diag.fill_(math.log(.05))
        if process_var_multi is not None:
            kf.process_covariance.cholesky_log_diag.fill_(math.log(process_var_multi))
        kf.mixture.components[0]._log_std.fill_(math.log(_COMPONENT_SD))
    kf.mc_sampling = _white_noise(num_samples)
    return kf


def _sigmoid_mixture_data(low: float = .2, high: float = .8) -> torch.Tensor:
    torch.manual_seed(0)
    y = torch.rand(3, 8, 1) * (high - low) + low
    y[torch.rand(3, 8) < .2] = _LOW_VALUE
    return y


@torch.no_grad()
def test_mixture_nonlinear_log_prob_matches_quadrature():
    """
    The marginal likelihood is ``P(standard) * E[N(y; sigmoid(Z), R)] + P(low) * E[N(y; sigmoid(Z) + mu, R + sigma^2)]``,
    using the regime-prior at each timestep.

    (With default state-uncertainty: the monte-carlo log-prob converges slowly when the measurement-noise is small
    relative to the spread of the sampled means, so this tests the mixture logic rather than monte-carlo efficiency.)
    """
    kf = _sigmoid_mixture_kf()
    y = _sigmoid_mixture_data()
    pred = kf(y)
    lp = pred.log_prob(y).reshape(-1).double()

    z_mean, z_var = _linear_moments(pred, 0)
    r = pred.measure_covs_flat[:, 0, 0].double()
    obs = y.reshape(-1).double()
    probs = pred.regime_probs.reshape(-1, 2).double()  # combos: (standard, low)

    def density(z, offset=0., var=r):
        mu = torch.sigmoid(z.clamp(-8, 8)) + offset
        return torch.exp(-.5 * (obs.unsqueeze(-1) - mu) ** 2 / var.unsqueeze(-1)) / (2 * math.pi * var.unsqueeze(-1)).sqrt()

    standard_lik = _gauss_hermite_expectation(density, z_mean, z_var)
    component_lik = _gauss_hermite_expectation(
        lambda z: density(z, offset=_COMPONENT_OFFSET, var=r + _COMPONENT_SD ** 2), z_mean, z_var
    )
    exact = (probs[:, 0] * standard_lik + probs[:, 1] * component_lik).log()
    # (monte-carlo error; confirmed to shrink at ~1/sqrt(num_samples), i.e. not a bias)
    assert torch.allclose(lp, exact, atol=.01)
    assert (lp - exact).abs().mean() < .003

    # and it's differentiable wrt the component's params:
    with torch.enable_grad():
        pred = kf(y)
        pred.log_prob(y).sum().backward()
    component = kf.mixture.components[0]
    for param in (component.mean, component._log_std, component.logit):
        assert param.grad is not None and param.grad.abs() > 0


@torch.no_grad()
def test_mixture_nonlinear_predictions_match_quadrature():
    """
    Means: ``E[sigmoid(Z)] + P(low) * mu`` (the component is an offset). Intervals: quantiles of the mixture of
    ``sigmoid(Z) + noise`` and ``sigmoid(Z) + mu + noise'`` (with the component's extra variance). With
    ``use_map=True``, ``E[sigmoid(Z)]`` is replaced by ``sigmoid(E[Z])``.
    """
    # (enough state-uncertainty that the nonlinearity matters -- the linearized mean/intervals fail this test)
    kf = _sigmoid_mixture_kf(process_var_multi=5.)
    y = _sigmoid_mixture_data(.7, .9)  # (away from sigmoid(0), where the sigmoid is ~linear)
    pred = kf(y, out_timesteps=14)  # (forecasts, so the state-uncertainty grows)

    z_mean, z_var = _linear_moments(pred, 0)
    r = pred.measure_covs_flat[:, 0, 0].double()
    probs = pred.regime_probs.reshape(-1, 2).double()
    standard_mean = _gauss_hermite_expectation(lambda z: torch.sigmoid(z.clamp(-8, 8)), z_mean, z_var)
    expected_mean = standard_mean + probs[:, 1] * _COMPONENT_OFFSET

    assert np.allclose(pred.means.reshape(-1).numpy(), expected_mean.numpy(), atol=.003)

    df = pred.to_dataframe(conf=.9, use_map=False)
    assert np.allclose(df['mean'].values, expected_mean.numpy(), atol=.003)

    def cdf_at(q):
        q = torch.as_tensor(q, dtype=torch.float64)
        standard = _gauss_hermite_expectation(
            lambda z: torch.special.ndtr((q.unsqueeze(-1) - torch.sigmoid(z.clamp(-8, 8))) / r.unsqueeze(-1).sqrt()),
            z_mean, z_var
        )
        component = _gauss_hermite_expectation(
            lambda z: torch.special.ndtr(
                (q.unsqueeze(-1) - torch.sigmoid(z.clamp(-8, 8)) - _COMPONENT_OFFSET) /
                (r.unsqueeze(-1) + _COMPONENT_SD ** 2).sqrt()
            ),
            z_mean, z_var
        )
        return probs[:, 0] * standard + probs[:, 1] * component

    # (~5 standard-errors for a sample-quantile's coverage with 20k samples)
    assert np.allclose(cdf_at(df['lower'].values).numpy(), .05, atol=.01)
    assert np.allclose(cdf_at(df['upper'].values).numpy(), .95, atol=.01)

    df_map = pred.to_dataframe(conf=.9, use_map=True)
    map_mean = torch.sigmoid(z_mean.double()) + probs[:, 1] * _COMPONENT_OFFSET
    assert np.allclose(df_map['mean'].values, map_mean.numpy(), atol=1e-5)
    assert np.allclose(df_map['lower'].values, df['lower'].values)

    # with a transform, the draws are back-transformed (so the mean is E[exp(Y)] under the mixture):
    df_exp = pred.to_dataframe(conf=.9, transform=LogTransform())
    standard_exp = _gauss_hermite_expectation(lambda z: torch.exp(torch.sigmoid(z.clamp(-8, 8))), z_mean, z_var)
    expected_exp = standard_exp * (
        probs[:, 0] * torch.exp(r / 2) +
        probs[:, 1] * torch.exp(_COMPONENT_OFFSET + (r + _COMPONENT_SD ** 2) / 2)
    )
    assert np.allclose(df_exp['mean'].values, expected_exp.numpy(), atol=.005)
    assert np.allclose(df_exp['lower'].values, np.exp(df['lower'].values), rtol=1e-5)

    # each regime's mean uses its own transform (`Transform.for_regime`): here, a component-transform whose noise is a
    # single node at -10 (standard-deviations), so its back-transformed mean is
    # E[exp(sigmoid(Z))] * exp(mu - 10 * sqrt(R + sigma^2)):
    from torchcast.state_space import RegimeTransform, SmearingTransform
    rt = RegimeTransform(LogTransform(), components={'low': SmearingTransform(LogTransform(), torch.tensor([-10.]))})
    df_rt = pred.to_dataframe(conf=.9, transform=rt)
    expected_rt = standard_exp * (
        probs[:, 0] * torch.exp(r / 2) +
        probs[:, 1] * torch.exp(_COMPONENT_OFFSET - 10 * (r + _COMPONENT_SD ** 2).sqrt())
    )
    assert np.allclose(df_rt['mean'].values, expected_rt.numpy(), atol=.005)


@torch.no_grad()
def test_mixture_nonlinear_get_mixture_warns():
    kf = _sigmoid_mixture_kf(num_samples=100)
    y = _sigmoid_mixture_data()
    pred = kf(y)
    with pytest.warns(UserWarning, match="linearized"):
        mixture = pred.get_mixture('y')
    # the regime-probabilities are exact:
    assert torch.allclose(mixture.probs, pred.regime_probs)
    with pytest.warns(UserWarning, match="linearized"):
        pred._get_regime_combos()


def test_binomial_mixture_with_nonlinear_process():
    """
    The motivating use-case: a binary 'visit' measure, plus a 'spend' measure (missing when there's no visit) with a
    saturating (nonlinear) process and a mixture component for low-spend visits.
    """
    torch.manual_seed(0)
    num_groups, num_times = 4, 15
    X = torch.randn(num_groups, num_times, 2)
    visit = (torch.rand(num_groups, num_times) > .4).float()
    spend = torch.randn(num_groups, num_times).cumsum(1) * .2 + 3.
    spend[torch.rand(num_groups, num_times) > .85] = -1.
    spend[visit == 0] = float('nan')
    y = torch.stack([visit, spend], -1)

    bf = BinomialFilter(
        processes=[
            LocalLevel(id='level_visit', measure='visit'),
            LocalLevel(id='level_spend', measure='spend'),
            SaturatedLinearModel(id='slm', predictors=['a', 'b'], measure='spend'),
        ],
        measures=['visit', 'spend'],
        binary_measures=['visit'],
        mixture=MixtureModel([MixtureComponent(measure='spend', mean_init=-1., prob_init=.1, id='quick')]),
    )
    bf.mc_sampling = _white_noise(50)
    pred = bf(y, X=X)
    lp = pred.log_prob(y)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    assert bf.mixture.components[0].mean.grad.abs() > 0

    with torch.no_grad():
        df = pred.to_dataframe(conf=.9, transform={'spend': LogTransform()})
    assert np.isfinite(df[['mean', 'lower', 'upper']].values).all()

    bf.zero_grad()
    bf.fit(y, X=X, stopping={'max_iter': 3}, verbose=0)


@torch.no_grad()
def test_mc_mvnorm_log_prob():
    from torchcast.state_space.predictions import _mc_mvnorm_log_prob

    torch.manual_seed(0)
    for d in (1, 3):
        A = torch.randn(5, d, d)
        cov = A @ A.transpose(-1, -2) + torch.eye(d)
        resid = torch.randn(7, 5, d)
        expected = torch.distributions.MultivariateNormal(torch.zeros(d), cov).log_prob(resid)
        assert torch.allclose(_mc_mvnorm_log_prob(resid, cov), expected, atol=1e-5)


@torch.no_grad()
def test_mixture_nonlinear_multivariate_negligible():
    """
    With two gaussian measures (one nonlinear, with a mixture component), a negligible component gives the same
    log-prob as no mixture -- incl. the joint (bivariate) monte-carlo likelihood of the standard regime.
    """
    torch.manual_seed(0)
    measures = ['a', 'b']
    X = torch.randn(3, 10, 2)
    y = torch.randn(3, 10, 2).cumsum(1) * .2
    y[0, 4, 1] = float('nan')

    def make(mixture):
        torch.manual_seed(1)
        kf = KalmanFilter(
            processes=[
                LocalLevel(id='level_a', measure='a'),
                LocalLevel(id='level_b', measure='b'),
                SaturatedLinearModel(id='slm', predictors=['x1', 'x2'], measure='b'),
            ],
            measures=measures,
            mixture=mixture,
        )
        kf.mc_sampling = _white_noise(200)
        return kf

    kf_mix = make([MixtureComponent(measure='b', mean_init=-3., prob_init=.1, id='low')])
    with torch.no_grad():
        kf_mix.mixture.components[0].logit.fill_(-40.)
    kf = make(None)
    kf.load_state_dict({k: v for k, v in kf_mix.state_dict().items() if not k.startswith('mixture.')})
    lp_mix = kf_mix(y, X=X).log_prob(y)
    lp = kf(y, X=X).log_prob(y)
    assert torch.allclose(lp_mix, lp, atol=1e-4)
