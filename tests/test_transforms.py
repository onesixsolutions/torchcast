import math

import numpy as np
import pytest
import torch
from scipy import stats

from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.process import LocalLevel
from torchcast.state_space import Transform, LogTransform, BoxCoxTransform, MixtureComponent


class _QuadratureLog(LogTransform):
    # forces the (default) gauss-hermite path, for comparison with the closed-form
    inverse_mean = Transform.inverse_mean


def test_inverse_mean():
    mean = torch.tensor([0., 1., 5., 5.], dtype=torch.float64)
    var = torch.tensor([.01, .25, 1., 4.], dtype=torch.float64)
    assert torch.allclose(_QuadratureLog().inverse_mean(mean, var), LogTransform().inverse_mean(mean, var), rtol=1e-10)
    assert torch.allclose(LogTransform().inverse_mean(mean, var), torch.exp(mean + var / 2))
    # base 10:
    t10 = LogTransform(base=10)
    assert torch.allclose(t10.inverse(torch.tensor(2.)), torch.tensor(100.))
    assert torch.allclose(t10.inverse_mean(mean, var), _QuadratureLog(base=10).inverse_mean(mean, var), rtol=1e-8)
    # box-cox with lambda=0 is log:
    assert torch.allclose(BoxCoxTransform(0).inverse_mean(mean, var), torch.exp(mean + var / 2))
    # box-cox with lambda=.5: inverse is (.5 * x + 1) ** 2, so the mean is exact: (.5 * mu + 1) ** 2 + .25 * var.
    # (except that the inverse is only defined for x > -2, and is clamped below that -- so allow a tiny difference for
    # the widest distribution)
    assert torch.allclose(BoxCoxTransform(.5).inverse_mean(mean, var), (.5 * mean + 1) ** 2 + .25 * var, rtol=1e-5)
    with pytest.raises(ValueError, match="non-negative"):
        BoxCoxTransform(-.1)


@torch.no_grad()
def test_to_dataframe_transform():
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id=f'level_{m}', measure=m) for m in 'ab'], measures=['a', 'b'])
    y = torch.randn(3, 10, 2).cumsum(1) * .3 + 2
    pred = kf(y)
    df = pred.to_dataframe(conf=.9)
    df_t = pred.to_dataframe(conf=.9, transform={'a': LogTransform()})
    mean, cov = pred

    a, a_t = df.query("measure == 'a'"), df_t.query("measure == 'a'")
    expected_mean = torch.exp(mean[..., 0] + cov[..., 0, 0] / 2).reshape(-1).numpy()
    assert np.allclose(a_t['mean'].values, expected_mean, rtol=1e-5)
    # quantiles pass through monotone transforms:
    assert np.allclose(a_t['lower'].values, np.exp(a['lower'].values), rtol=1e-5)
    assert np.allclose(a_t['upper'].values, np.exp(a['upper'].values), rtol=1e-5)
    # 'b' is untouched:
    assert np.allclose(df_t.query("measure == 'b'")['mean'].values, df.query("measure == 'b'")['mean'].values)
    # a single transform applies to all measures:
    df_all = pred.to_dataframe(conf=.9, transform=LogTransform())
    assert np.allclose(df_all.query("measure == 'b'")['lower'].values, np.exp(df.query("measure == 'b'")['lower']))

    with pytest.raises(ValueError, match="not in the model"):
        pred.to_dataframe(transform={'c': LogTransform()})
    with pytest.raises(ValueError, match="conf=None"):
        pred.to_dataframe(transform=LogTransform(), conf=None)
    with pytest.raises(ValueError, match="only supported"):
        pred.to_dataframe(type='states', transform=LogTransform())


@torch.no_grad()
def test_to_dataframe_transform_actuals():
    from torchcast.utils import TimeSeriesDataset

    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['a'])
    y = torch.randn(2, 8, 1)
    dataset = TimeSeriesDataset(y, group_names=['g0', 'g1'], start_times=np.zeros(2, dtype='int'), measures=[['a']],
                                dt_unit=None)
    df = kf(y).to_dataframe(dataset, transform=LogTransform())
    assert np.allclose(df['actual'].values, np.exp(y[..., 0].reshape(-1).numpy()), rtol=1e-5)


@torch.no_grad()
def test_to_dataframe_transform_mixture():
    torch.manual_seed(0)
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        mixture=[MixtureComponent(measure='y', mean_init=-3., prob_init=.1, id='low')]
    )
    y = torch.randn(3, 10, 1) * .3 + 2
    y[:, 4] = -3.
    pred = kf(y)
    df = pred.to_dataframe(conf=.9)
    df_t = pred.to_dataframe(conf=.9, transform=LogTransform())
    mix = pred.get_mixture('y')
    # back-transform each regime, then mix:
    expected = (mix.probs * torch.exp(mix.means + mix.vars / 2)).sum(-1).reshape(-1).numpy()
    assert np.allclose(df_t['mean'].values, expected, rtol=1e-5)
    # ...which is not the same as back-transforming the collapsed moments:
    collapsed = torch.exp(mix.mean() + mix.var() / 2).reshape(-1).numpy()
    assert (np.abs(expected / collapsed - 1) > .1).any()
    assert np.allclose(df_t['lower'].values, np.exp(df['lower'].values), rtol=1e-5)


@torch.no_grad()
def test_to_dataframe_transform_binomial():
    torch.manual_seed(1)
    visit = (torch.rand(3, 12) > .4).float()
    spend = torch.randn(3, 12).cumsum(1) * .2 + 3.
    spend[visit == 0] = float('nan')
    y = torch.stack([visit, spend], -1)
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend']],
        measures=['visit', 'spend'],
        binary_measures=['visit'],
    )
    # monte-carlo noise varies across processes, so use enough samples that rtol=.02 is ~5 standard-errors:
    bf.mc_sampling = 100_000
    pred = bf(y)
    df_t = pred.to_dataframe(transform={'spend': LogTransform()}, use_map=True)
    # 'spend' goes through the monte-carlo path here; compare to the closed-form for its (linear-gaussian) moments:
    measured_mean, system_cov = pred._measured_moments_flat()
    expected = torch.exp(measured_mean[:, 1] + system_cov[:, 1, 1] / 2).numpy()
    assert np.allclose(df_t.query("measure == 'spend'")['mean'].values, expected, rtol=.02)
    with pytest.raises(ValueError, match="binary measures"):
        pred.to_dataframe(transform=LogTransform())


@torch.no_grad()
def test_to_dataframe_transform_nonlinear_gaussian():
    """
    A transform can be combined with a nonlinear measurement (as long as the likelihood is gaussian): the model is
    `T(y) = g(state) + noise`, so predictions on the original scale are `inverse(g(state) + noise)`.
    """
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    kf.mc_sampling = 20_000
    y = torch.rand(2, 8, 1) * .5 + .25
    pred = kf(y)
    df = pred.to_dataframe(conf=.9, use_map=False)
    df_t = pred.to_dataframe(conf=.9, use_map=False, transform=LogTransform())
    # quantiles pass through the (monotone) back-transform:
    assert np.allclose(df_t['lower'].values, np.exp(df['lower'].values), rtol=1e-4)
    assert np.allclose(df_t['upper'].values, np.exp(df['upper'].values), rtol=1e-4)
    # the mean is the mean of the back-transformed samples, so exceeds the back-transformed mean (jensen's):
    assert (df_t['mean'].values > np.exp(df['mean'].values)).all()


@torch.no_grad()
def test_samples_to_dataframe():
    from torchcast.state_space.predictions import _quantile

    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['a'])
    pred = kf(torch.randn(2, 5, 1))
    samples = torch.randn(1000, 2, 5)
    df = pred.samples_to_dataframe({'thing': samples}, conf=.8, actuals={'thing': torch.zeros(2, 5)})
    assert set(df['measure']) == {'thing'}
    assert np.allclose(df['mean'].values, samples.mean(0).reshape(-1).numpy(), atol=1e-6)
    assert np.allclose(df['lower'].values, torch.quantile(samples, .1, dim=0).reshape(-1).numpy(), atol=1e-6)
    assert np.allclose(df['upper'].values, torch.quantile(samples, .9, dim=0).reshape(-1).numpy(), atol=1e-6)
    assert (df['actual'] == 0).all()
    for q in (.025, .5, .975):
        assert torch.allclose(_quantile(samples, q), torch.quantile(samples, q, dim=0), atol=1e-6)
    with pytest.raises(ValueError, match="shape"):
        pred.samples_to_dataframe({'thing': torch.randn(10, 2, 4)})


@torch.no_grad()
def test_bias_adjust():
    mean = torch.tensor([0., 1., 5.], dtype=torch.float64)
    var = torch.tensor([.25, 1., 4.], dtype=torch.float64)
    # 0 -> no adjustment (the median); .5 -> half the variance; None/1 -> the mean:
    assert torch.allclose(LogTransform(bias_adjust=0).inverse_mean(mean, var), mean.exp())
    assert torch.allclose(LogTransform(bias_adjust=.5).inverse_mean(mean, var), torch.exp(mean + .25 * var))
    assert torch.allclose(LogTransform(bias_adjust=1).inverse_mean(mean, var), LogTransform().inverse_mean(mean, var))
    # quadrature and closed-form agree:
    assert torch.allclose(_QuadratureLog(bias_adjust=.5).inverse_mean(mean, var), torch.exp(mean + .25 * var))
    assert torch.allclose(BoxCoxTransform(0, bias_adjust=.5).inverse_mean(mean, var), torch.exp(mean + .25 * var))
    assert torch.allclose(BoxCoxTransform(.5, bias_adjust=0).inverse_mean(mean, var), BoxCoxTransform(.5).inverse(mean))
    with pytest.raises(ValueError, match="between 0 and 1"):
        LogTransform(bias_adjust=1.5)

    # in to_dataframe: the mean changes, the intervals don't
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'])
    pred = kf(torch.randn(2, 6, 1))
    df_full = pred.to_dataframe(transform=LogTransform())
    df_none = pred.to_dataframe(transform=LogTransform(bias_adjust=0))
    assert np.allclose(df_none['mean'].values, np.exp(pred.to_dataframe()['mean'].values), rtol=1e-5)
    assert (df_none['mean'].values < df_full['mean'].values).all()
    assert np.allclose(df_none[['lower', 'upper']].values, df_full[['lower', 'upper']].values)


@torch.no_grad()
def test_bias_adjust_warns_for_monte_carlo():
    import warnings

    torch.manual_seed(1)
    y = torch.stack([(torch.rand(2, 8) > .4).float(), torch.randn(2, 8) * .3 + 2], -1)
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend']],
        measures=['visit', 'spend'],
        binary_measures=['visit'],
    )
    bf.mc_sampling = 50
    pred = bf(y)
    # 'spend' has a linear measured-mean, so it's closed-form (even though the model is nonlinear) and bias_adjust
    # applies:
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*bias_adjust.*')
        df = pred.to_dataframe(transform={'spend': LogTransform(bias_adjust=.5)}, use_map=True)
    mean, cov = pred._measured_moments_flat()  # (exact for 'spend')
    expected = torch.exp(mean[:, 1] + .5 * cov[:, 1, 1] / 2)
    assert np.allclose(df.query("measure == 'spend'")['mean'].values, expected.reshape(-1).numpy(), rtol=1e-5)

    # a measure with a nonlinear measured-mean uses monte-carlo, where it's ignored:
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    kf.mc_sampling = 50
    pred = kf(torch.rand(2, 8, 1))
    with pytest.warns(UserWarning, match="bias_adjust"):
        pred.to_dataframe(transform=LogTransform(bias_adjust=.5), use_map=True)
