import math

import numpy as np
import pytest
import torch
from scipy import stats

from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.process import LocalLevel, LocalTrend, SaturatedLinearModel
from torchcast.state_space import MixtureComponent, LogTransform

# monte-carlo tests: sample sizes and tolerances are chosen so that each check is ~5 standard-errors


def _correlated_kf() -> KalmanFilter:
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalTrend(id=f'trend_{m}', measure=m) for m in 'ab'], measures=['a', 'b'])
    with torch.no_grad():
        kf.measure_covariance.cholesky_off_diag.fill_(1.5)  # strongly correlated measurement noise
    return kf


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@torch.no_grad()
def test_sample_moments():
    kf = _correlated_kf()
    pred = kf(torch.randn(3, 12, 2).cumsum(1))
    n = 20_000
    s = pred.sample(n, generator=_gen())
    assert s.observations.shape == (n, 3, 12, 2) and s.means.shape == (n, 3, 12, 2) and s.covs.shape == (n, 3, 12, 2, 2)
    mean, cov = pred
    sd = cov.diagonal(dim1=-2, dim2=-1).sqrt()
    assert ((s.observations.mean(0) - mean).abs() < 5 * sd / math.sqrt(n)).all()
    resid = s.observations - mean
    emp_cov = torch.einsum('sgtm,sgtk->gtmk', resid, resid) / n
    assert torch.allclose(emp_cov.diagonal(dim1=-2, dim2=-1), sd ** 2, rtol=5 * math.sqrt(2 / n))
    corr, emp_corr = cov[..., 0, 1] / (sd[..., 0] * sd[..., 1]), emp_cov[..., 0, 1] / (sd[..., 0] * sd[..., 1])
    assert (corr.abs() > .3).all()  # (the test is only meaningful if the measures are correlated)
    assert torch.allclose(emp_corr, corr, atol=5 / math.sqrt(n))

    # without observation noise, the conditional covariance is the measure-covariance:
    s2 = pred.sample(10, observation_noise=False, generator=_gen())
    assert s2.observations is None
    assert torch.allclose(s2.covs, pred.measure_covs.expand(10, -1, -1, -1, -1))
    assert torch.equal(s2['a'], s2.means[..., 0])
    # reproducible given a generator:
    assert torch.equal(pred.sample(5, generator=_gen(1)).observations, pred.sample(5, generator=_gen(1)).observations)


@torch.no_grad()
def test_sample_mixture():
    torch.manual_seed(0)
    kf = KalmanFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in 'ab'],
        measures=['a', 'b'],
        mixture=[MixtureComponent(measure='a', mean_init=-3., prob_init=.2, id='low')]
    )
    component = kf.mixture.components[0]
    pred = kf(torch.randn(2, 8, 2))
    n = 40_000
    s = pred.sample(n, generator=_gen())
    mix = pred.get_mixture('a')

    in_low = s.means[..., 0] == component.mean
    p = mix.probs[..., 1]
    assert ((in_low.float().mean(0) - p).abs() < 5 * (p * (1 - p) / n).sqrt()).all()
    # in the 'low' regime, 'a' has the component's variance and is uncorrelated with 'b':
    assert torch.allclose(s.covs[..., 0, 0][in_low], component.var.expand(int(in_low.sum())))
    assert (s.covs[..., 0, 1][in_low] == 0).all()
    # the observations follow the mixture:
    for q in (.05, .5, .95):
        emp_cdf_at_quantile = (s['a'] <= mix.quantile(q)).float().mean(0)
        assert ((emp_cdf_at_quantile - q).abs() < 5 * math.sqrt(q * (1 - q) / n)).all()


@torch.no_grad()
def test_sample_binomial():
    torch.manual_seed(1)
    visit = (torch.rand(3, 12) > .4).float()
    spend = torch.randn(3, 12).cumsum(1) * .2 + 3.
    spend[visit == 0] = float('nan')
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend']],
        measures=['visit', 'spend'],
        binary_measures=['visit'],
    )
    bf.mc_sampling = 100
    pred = bf(torch.stack([visit, spend], -1))
    n = 20_000
    s = pred.sample(n, generator=_gen())
    assert set(torch.unique(s['visit']).tolist()) <= {0., 1.}
    prob = s.means[..., 0]
    assert ((prob > 0) & (prob < 1)).all()
    # given the (sampled) probability, the variance is binomial:
    assert torch.allclose(s.covs[..., 0, 0], prob * (1 - prob))
    # draws are consistent with the probabilities:
    assert ((s['visit'].mean(0) - prob.mean(0)).abs() < 5 * .5 / math.sqrt(n)).all()


@torch.no_grad()
def test_derived_sum_of_correlated_measures():
    """
    For linear-gaussian measures, a + b is gaussian with variance var(a) + var(b) + 2 * cov(a, b): a derived quantity
    must account for the correlation.
    """
    kf = _correlated_kf()
    pred = kf(torch.randn(3, 12, 2).cumsum(1))
    conf = .9
    df = pred.to_dataframe(conf=conf, derived={'total': lambda s: s['a'] + s['b']})
    df_total = df.query("measure == 'total'")

    mean, cov = pred
    total_mean = (mean[..., 0] + mean[..., 1]).reshape(-1).numpy()
    total_sd = (cov[..., 0, 0] + cov[..., 1, 1] + 2 * cov[..., 0, 1]).sqrt().reshape(-1).numpy()
    n = 1000  # (the default `derived_num_samples`)
    assert np.all(np.abs(df_total['mean'].values - total_mean) < 5 * total_sd / math.sqrt(n))
    z = stats.norm.ppf(.95)
    # (the se of a sample-quantile of a normal at the 95th %ile is ~2.1 * sd / sqrt(n))
    for col, sign in (('lower', -1), ('upper', 1)):
        expected = total_mean + sign * z * total_sd
        assert np.all(np.abs(df_total[col].values - expected) < 5 * 2.2 * total_sd / math.sqrt(n))
    # ...which differs from what you'd get assuming independence:
    independent_sd = (cov[..., 0, 0] + cov[..., 1, 1]).sqrt().reshape(-1).numpy()
    assert np.all(np.abs(total_sd / independent_sd - 1) > .15)

    # deterministic across calls:
    df2 = pred.to_dataframe(conf=conf, derived={'total': lambda s: s['a'] + s['b']})
    assert df.equals(df2)


@torch.no_grad()
def test_derived_with_transform_and_actuals():
    from torchcast.utils import TimeSeriesDataset

    kf = _correlated_kf()
    y = torch.randn(2, 10, 2).cumsum(1) * .2
    y[0, 3, 1] = float('nan')
    dataset = TimeSeriesDataset(y, group_names=['g0', 'g1'], start_times=np.zeros(2, dtype='int'),
                                measures=[['a', 'b']], dt_unit=None)
    pred = kf(y)
    df = pred.to_dataframe(dataset, transform=LogTransform(), derived={'total': lambda s: s['a'] + s['b']})
    df_total = df.query("measure == 'total'")

    # derived functions get samples on the back-transformed scale, so the mean is a sum of lognormal means:
    mean, cov = pred
    var = cov.diagonal(dim1=-2, dim2=-1)
    expected = torch.exp(mean + var / 2).sum(-1)
    # the exact standard-error of the sample-mean of exp(a) + exp(b) (lognormal variances/covariance):
    lognormal_var = torch.exp(2 * mean + var) * (torch.exp(var) - 1)
    lognormal_cov = torch.exp(mean.sum(-1) + var.sum(-1) / 2) * (torch.exp(cov[..., 0, 1]) - 1)
    se = ((lognormal_var.sum(-1) + 2 * lognormal_cov) / 1000).sqrt()  # (the default `derived_num_samples`)
    assert np.all(np.abs(df_total['mean'].values - expected.reshape(-1).numpy()) < 5 * se.reshape(-1).numpy())
    # the function is also applied to the (back-transformed) actuals:
    expected_actual = y.exp().sum(-1).reshape(-1).numpy()
    assert np.allclose(df_total['actual'].values, expected_actual, equal_nan=True)
    assert np.isnan(df_total['actual'].values).sum() == 1

    with pytest.raises(ValueError, match="same as measures"):
        pred.to_dataframe(derived={'a': lambda s: s['a']})
    with pytest.raises(ValueError, match="returned shape"):
        pred.to_dataframe(derived={'bad': lambda s: s['a'].mean(0)})
    with pytest.raises(ValueError, match="conf=None"):
        pred.to_dataframe(derived={'total': lambda s: s['a'] + s['b']}, conf=None)
    # the sample-count is configurable:
    df_small = pred.to_dataframe(derived={'total': lambda s: s['a'] + s['b']}, derived_num_samples=50)
    assert not np.allclose(df_small.query("measure == 'total'")['mean'].values, df_total['mean'].values)


@torch.no_grad()
def test_sample_nonlinear_independent_rows():
    """
    For nonlinear models, samples are drawn via the monte-carlo machinery (sampling the linear measured-mean plus the
    nonlinear processes' states), but -- unlike the fixed monte-carlo noise, which is shared across rows -- with
    independent draws for each group/timestep. They should also match the monte-carlo predictive mean.
    """
    torch.manual_seed(0)
    kf = KalmanFilter(
        processes=[LocalLevel(id='level'), SaturatedLinearModel(id='slm', predictors=['x1', 'x2'])],
        measures=['y'],
    )
    kf.mc_sampling = 5000
    X = torch.randn(2, 6, 2)
    pred = kf(torch.randn(2, 6, 1), X=X)
    samples = pred.sample(5000, observation_noise=False, generator=_gen())
    means = samples.means[..., 0].reshape(5000, -1)  # (samples, rows)
    corr = np.corrcoef(means.T.numpy())
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert np.abs(off_diag).max() < .1  # (~5 standard-errors of a correlation with 5000 samples is .07)
    se = means.std(0) / math.sqrt(5000)
    assert ((means.mean(0) - pred.means.reshape(-1)).abs() < 5 * se + 1e-3).all()


@torch.no_grad()
def test_derived_actuals_missing_measure():
    """Measures that aren't in the dataset are all-nan when a derived function is applied to the actuals."""
    from torchcast.utils import TimeSeriesDataset

    kf = _correlated_kf()
    y = torch.randn(2, 5, 1)
    dataset = TimeSeriesDataset(y, group_names=['g0', 'g1'], start_times=np.zeros(2, dtype='int'),
                                measures=[['a']], dt_unit=None)
    pred = kf(torch.cat([y, torch.randn(2, 5, 1)], -1))
    with pytest.warns(UserWarning, match="not present in your dataset"):
        df = pred.to_dataframe(dataset, derived={'total': lambda s: s['a'] + s['b'], 'a2': lambda s: s['a'] * 2})
    assert df.query("measure == 'total'")['actual'].isna().all()
    assert np.allclose(df.query("measure == 'a2'")['actual'].values, 2 * y.reshape(-1).numpy())


@torch.no_grad()
def test_derived_binomial_spend():
    """
    The motivating example: expected weekly spend = visit * spend, where spend is only observed when there's a visit.
    """
    torch.manual_seed(1)
    visit = (torch.rand(3, 12) > .4).float()
    log_spend = torch.randn(3, 12).cumsum(1) * .2 + 3.
    log_spend[visit == 0] = float('nan')
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'log_spend']],
        measures=['visit', 'log_spend'],
        binary_measures=['visit'],
        mixture=[MixtureComponent(measure='log_spend', mean_init=1., prob_init=.1, id='quick')],
    )
    bf.mc_sampling = 100
    pred = bf(torch.stack([visit, log_spend], -1))
    df = pred.to_dataframe(
        transform={'log_spend': LogTransform()},
        derived={'weekly_spend': lambda s: s['visit'] * s['log_spend'].nan_to_num()},
        use_map=True,
    )
    df_ws = df.query("measure == 'weekly_spend'")
    assert np.isfinite(df_ws[['mean', 'lower', 'upper']].values).all()
    # with a meaningful chance of no visit, the lower bound is zero:
    assert (df_ws['lower'] == 0).any()


_MEMORY_SCRIPT = """
import resource, sys, torch
from torchcast.kalman_filter import KalmanFilter
from torchcast.process import LocalTrend, Season

def peak_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2 ** 20 if sys.platform == 'darwin' else rss / 2 ** 10  # bytes on macOS, KB on linux

torch.manual_seed(0)
measure_funs = {'y': 'sigmoid'} if sys.argv[1] == 'nonlinear' else None
kf = KalmanFilter(
    processes=[LocalTrend(id='trend'), Season(id='season', period=7, dt_unit=None, K=8, fixed=True)],
    measures=['y'],
    measure_funs=measure_funs,
)
kf.mc_sampling = 10
with torch.no_grad():
    pred = kf(torch.rand(20, 100, 1) * .5 + .25, start_offsets=[0] * 20)  # 2000 rows, state-rank 18
    before = peak_mb()
    pred.sample(500)
print(peak_mb() - before)
"""


@pytest.mark.parametrize("model", ['linear', 'nonlinear'])
def test_sample_memory(model: str):
    """
    Regression test: sampling shouldn't materialize (num_samples, num_rows, state_rank, state_rank) tensors (which a
    broadcasting matmul does). Here that would be 500 * 2000 * 18 * 18 floats ~= 1.3GB.
    """
    import subprocess
    import sys

    res = subprocess.run([sys.executable, '-c', _MEMORY_SCRIPT, model], capture_output=True, text=True, check=True)
    peak_increase_mb = float(res.stdout.strip().splitlines()[-1])
    assert peak_increase_mb < 300, f"sample() increased peak memory by {peak_increase_mb:.0f}MB"
