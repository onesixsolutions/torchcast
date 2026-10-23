import math

import numpy as np
import pytest
import torch
from scipy import stats

from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.process import LocalLevel
from torchcast.state_space import Transform, LogTransform, BoxCoxTransform
from torchcast.mixture.component import MixtureComponent


class _QuadratureLog(LogTransform):
    # forces the (default) gauss-hermite path, for comparison with the closed-form
    _expected_inverse = Transform._expected_inverse


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

    # the quadrature is computed in chunks (to bound memory), which doesn't change the result -- incl. for inputs that
    # broadcast (here: (7, 4) means, (4,) variances):
    means = mean + torch.randn(7, 4, dtype=torch.float64)
    one_chunk = BoxCoxTransform(.5).inverse_mean(means, var)
    many_chunks = BoxCoxTransform(.5)
    many_chunks.max_quadrature_size = 3 * many_chunks.num_nodes  # (3 elements per chunk)
    assert torch.equal(many_chunks.inverse_mean(means, var), one_chunk)
    assert one_chunk.shape == (7, 4)


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
        mixture=[MixtureComponent(measure='y', offset_init=-3., prob_init=.1, id='low')]
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
    # `adjust_bias=False` ignores it (e.g. for monte-carlo means):
    unadjusted = LogTransform(bias_adjust=0).inverse_mean(mean, var, adjust_bias=False)
    assert torch.allclose(unadjusted, torch.exp(mean + var / 2))
    assert torch.allclose(_QuadratureLog(bias_adjust=0).inverse_mean(mean, var, adjust_bias=False),
                          torch.exp(mean + var / 2), rtol=1e-6)
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
def test_bias_adjust_monte_carlo():
    """
    `bias_adjust` also applies to monte-carlo means: it scales the variance of both the sampled state and the noise,
    so 0 gives the plug-in inverse(g(MAP)) (like `use_map=True`), and 1 the usual monte-carlo mean.
    """
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

    # a nonlinear measure's mean is monte-carlo (with use_map=False), where it scales the state's variance too:
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    kf.mc_sampling = 50
    pred = kf(torch.rand(2, 8, 1))

    def means(bias_adjust, use_map=False) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='.*bias_adjust.*')
            return pred.get_means(transform=LogTransform(bias_adjust=bias_adjust), use_map=use_map)[..., 0].numpy()

    plug_in = torch.exp(pred._measured_moments_flat()[0][:, 0]).view(2, 8).numpy()  # exp(sigmoid(MAP))
    assert np.allclose(means(0), plug_in, rtol=1e-6)
    assert np.allclose(means(0, use_map=True), plug_in, rtol=1e-6)
    assert np.array_equal(means(1), means(None))
    assert (means(0) < means(.5)).all() and (means(.5) < means(1)).all()
    # each measure's own `bias_adjust` (only its own deviations are scaled):
    kf2 = KalmanFilter(processes=[LocalLevel(id=f'level_{m}', measure=m) for m in 'ab'], measures=['a', 'b'],
                       measure_funs={'a': 'sigmoid', 'b': 'sigmoid'})
    kf2.mc_sampling = 50
    pred2 = kf2(torch.rand(2, 8, 2))
    both = pred2.get_means(transform={'a': LogTransform(bias_adjust=0), 'b': LogTransform()}, use_map=False)
    assert torch.allclose(both[..., 0], torch.exp(pred2._measured_moments_flat()[0][:, 0]).view(2, 8), rtol=1e-6)
    assert torch.equal(both[..., 1], pred2.get_means(transform=LogTransform(), use_map=False)[..., 1])
    # (`derived` back-transforms samples, so it's ignored there:)
    with pytest.warns(UserWarning, match="bias_adjust"):
        pred.to_dataframe(transform=LogTransform(bias_adjust=.5), use_map=False, derived={'z': lambda s: s['y']})


@torch.no_grad()
def test_smearing_transform():
    from torchcast.state_space import SmearingTransform

    torch.manual_seed(0)
    mean, var = torch.randn(4, 6), torch.rand(4, 6) + .1

    # with few residuals, they're used directly: the mean is the smearing average
    resids = torch.distributions.StudentT(3.).sample((50,))
    smear = SmearingTransform(LogTransform(), resids)
    expected = torch.exp(mean.unsqueeze(-1) + var.sqrt().unsqueeze(-1) * resids).mean(-1)
    assert torch.allclose(smear.inverse_mean(mean, var), expected, rtol=1e-5)
    assert torch.allclose(smear.inverse(mean), mean.exp())  # (intervals: unaffected)
    # weights:
    w = torch.rand(50)
    expected_w = (torch.exp(mean.unsqueeze(-1) + var.sqrt().unsqueeze(-1) * resids) * w / w.sum()).sum(-1)
    assert torch.allclose(SmearingTransform(LogTransform(), resids, weights=w).inverse_mean(mean, var), expected_w,
                          rtol=1e-5)

    # with many residuals, they're summarized by quantiles; gaussian residuals ~ the lognormal mean:
    resids = torch.randn(200_000)
    smear = SmearingTransform(LogTransform(), resids)
    assert smear.residuals.shape == (100,)
    assert torch.allclose(smear.inverse_mean(mean, var), LogTransform().inverse_mean(mean, var), rtol=.02)

    # the summary is the quantiles at the midpoints of equal-probability bins:
    smear = SmearingTransform(LogTransform(), torch.arange(1000.), num_nodes=10)
    assert torch.allclose(smear.residuals, torch.arange(10.) * 100 + 49.5)
    assert torch.allclose(smear.weights, torch.full((10,), .1))

    # bias_adjust comes from the base transform:
    assert torch.allclose(SmearingTransform(LogTransform(bias_adjust=0), resids).inverse_mean(mean, var), mean.exp())
    # non-finite residuals are dropped; validation:
    assert SmearingTransform(LogTransform(), torch.tensor([float('nan'), 1., -1.])).residuals.shape == (2,)
    with pytest.raises(ValueError, match="non-negative"):
        SmearingTransform(LogTransform(), torch.randn(5), weights=-torch.ones(5))
    with pytest.raises(ValueError, match="No"):
        SmearingTransform(LogTransform(), torch.tensor([float('nan')]))


@torch.no_grad()
def test_smearing_from_predictions():
    from torchcast.state_space import SmearingTransform

    torch.manual_seed(0)
    # heavy-tailed noise on the log-scale:
    y = (torch.randn(5, 40).cumsum(1) * .05 + torch.distributions.StudentT(3.).sample((5, 40)) * .3).unsqueeze(-1)
    kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'])
    pred = kf(y)
    smear = SmearingTransform.from_predictions(LogTransform(), pred, y, 'y', num_nodes=1000)
    mean, cov = pred
    resids = ((y - mean) / cov.diagonal(dim1=-2, dim2=-1).sqrt()).reshape(-1)
    assert torch.allclose(smear.residuals.sort().values, resids.sort().values, atol=1e-5)

    df = pred.to_dataframe(transform=smear, conf=.9)
    df_log = pred.to_dataframe(transform=LogTransform(), conf=.9)
    expected = smear.inverse_mean(mean[..., 0], cov[..., 0, 0])
    assert np.allclose(df['mean'].values, expected.reshape(-1).numpy(), rtol=1e-5)
    assert np.allclose(df[['lower', 'upper']].values, df_log[['lower', 'upper']].values)

    # nonlinear measures: not yet supported by `from_predictions()`
    kf_nl = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], measure_funs={'y': 'sigmoid'})
    kf_nl.mc_sampling = 20
    with pytest.raises(NotImplementedError, match="nonlinear"):
        SmearingTransform.from_predictions(LogTransform(), kf_nl(torch.rand(2, 5, 1)), torch.rand(2, 5, 1), 'y')
    # ...but a smearing transform can still be used with them (monte-carlo over the state, smearing for the noise):
    df_nl = kf_nl(torch.rand(2, 5, 1)).to_dataframe(transform=smear, use_map=False)
    assert np.isfinite(df_nl['mean'].values).all()


def _smearing_mixture_kf(component_sd: float) -> KalmanFilter:
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        mixture=[MixtureComponent(measure='y', offset_init=-4., prob_init=.1, id='low')],
    )
    with torch.no_grad():
        kf.mixture.components[0]._log_extra_std.fill_(math.log(component_sd))
    return kf


def _smearing_mixture_y() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    y = torch.randn(4, 30, 1).cumsum(1) * .05 + 3.
    is_low = torch.rand(4, 30, 1) < .15
    y[is_low] = y[is_low] - 4. + torch.randn(int(is_low.sum())) * .05  # 'quick visits': well below the usual level
    return y, is_low


@torch.no_grad()
def test_smearing_mixture():
    """
    For a mixture measure, each regime is smeared with its own residuals: each observation's standardized by that
    regime's moments, weighted by P(regime | observation).
    """
    from torchcast.state_space import SmearingTransform

    y, is_low = _smearing_mixture_y()
    pred = _smearing_mixture_kf(component_sd=.1)(y)
    mix = pred.get_mixture('y')
    smear = SmearingTransform.from_predictions(LogTransform(), pred, y, 'y', num_nodes=1000)
    assert set(smear.component_residuals) == {'low'}
    low = smear.component_residuals['low']

    # posterior regime-probabilities given each observation:
    obs = y[..., 0]
    lik = torch.distributions.Normal(mix.means, mix.vars.sqrt()).log_prob(obs.unsqueeze(-1)).exp()
    post = mix.probs * lik / (mix.probs * lik).sum(-1, keepdim=True)
    residuals = (obs.unsqueeze(-1) - mix.means) / mix.vars.sqrt()

    # standard regime: (fewer residuals than nodes, so they're used directly, in order)
    assert torch.allclose(smear.residuals, residuals[..., 0].reshape(-1), atol=1e-5)
    assert torch.allclose(smear.weights, (post[..., 0] / post[..., 0].sum()).reshape(-1), atol=1e-6)
    # ...where the outliers get ~no weight in total, though they're ~15% of observations:
    assert smear.weights[is_low.reshape(-1)].sum() < .05
    # the component's regime: standardized by its own moments, and weighted by P(low | y):
    assert torch.allclose(low.residuals, residuals[..., 1].reshape(-1), atol=1e-5)
    assert torch.allclose(low.weights, (post[..., 1] / post[..., 1].sum()).reshape(-1), atol=1e-6)
    # ...where the standard observations get ~no weight in total:
    assert low.weights[~is_low.reshape(-1)].sum() < .05

    # each regime's mean uses its own residuals:
    df = pred.to_dataframe(transform=smear)
    expected = (
        mix.probs[..., 0] * smear.inverse_mean(mix.means[..., 0], mix.vars[..., 0]) +
        mix.probs[..., 1] * low.inverse_mean(mix.means[..., 1], mix.vars[..., 1])
    )
    assert np.allclose(df['mean'].values, expected.reshape(-1).numpy(), rtol=1e-4)
    assert torch.equal(
        smear.inverse_mean(mix.means[..., 1], mix.vars[..., 1], regime='low'),
        low.inverse_mean(mix.means[..., 1], mix.vars[..., 1]),
    )
    # (a SmearingTransform without its own residuals for a component uses the standard ones:)
    single = SmearingTransform(LogTransform(), torch.randn(50))
    assert torch.equal(
        single.inverse_mean(mix.means[..., 1], mix.vars[..., 1], regime='low'),
        single.inverse_mean(mix.means[..., 1], mix.vars[..., 1]),
    )


@torch.no_grad()
def test_smearing_large_component_variance():
    """
    The motivating failure: a component with a large (learned) variance has a huge lognormal mean, which dominates the
    mixed mean. Smearing the component's regime with the residuals attributed to it keeps it near those observations.
    """
    from torchcast.state_space import SmearingTransform

    y, is_low = _smearing_mixture_y()
    pred = _smearing_mixture_kf(component_sd=3.)(y)
    mix = pred.get_mixture('y')
    lognormal_low = torch.exp(mix.means[..., 1] + mix.vars[..., 1] / 2)
    assert (lognormal_low[:, 10:] > 20).all()  # (vs. 'quick visits' of ~exp(-1), and typical visits of ~exp(3))

    smear = SmearingTransform.from_predictions(LogTransform(), pred, y, 'y')
    smeared_low = smear.inverse_mean(mix.means[..., 1], mix.vars[..., 1], regime='low')
    # (bounded by the observations; well above the quick visits' ~exp(-1) because this unfitted component is so broad
    # that the standard observations get substantial posterior weight in it)
    assert (smeared_low < y.exp().max()).all()
    assert (smeared_low < lognormal_low / 3).all()
    # vs. smearing every regime with the standard regime's residuals:
    df_smear = pred.to_dataframe(transform=smear)
    df_standard_only = pred.to_dataframe(transform=SmearingTransform(LogTransform(), smear.residuals, smear.weights))
    assert (df_smear['mean'].values < df_standard_only['mean'].values).all()


@torch.no_grad()
def test_smearing_from_predictions_missing():
    """Missing observations are skipped (incl. for mixture measures, whose residuals are weighted)."""
    from torchcast.state_space import SmearingTransform

    torch.manual_seed(0)
    y = torch.randn(2, 10, 1).cumsum(1) * .1 + 3.
    y[0, 3] = y[1, 7] = float('nan')
    for mixture in (None, [MixtureComponent(measure='y', offset_init=-1., prob_init=.1, id='low')]):
        kf = KalmanFilter(processes=[LocalLevel(id='level')], measures=['y'], mixture=mixture)
        smear = SmearingTransform.from_predictions(LogTransform(), kf(y), y, 'y')
        for t in [smear, *smear.component_residuals.values()]:
            assert t.residuals.shape == (18,)
            assert torch.isfinite(t.residuals).all() and torch.isfinite(t.weights).all()
        assert len(smear.component_residuals) == (0 if mixture is None else 1)


@pytest.mark.parametrize("use_map", [True, False])
@torch.no_grad()
def test_means_only(use_map: bool):
    """
    `to_dataframe(conf=False)` returns just the means -- the same as the default call's -- without any interval work;
    `get_means()` returns them as a tensor, back-transforming every measure (incl. a linear one in a nonlinear model).
    """
    torch.manual_seed(0)
    visit = (torch.rand(4, 12) > .3).float()
    spend = torch.randn(4, 12).cumsum(1) * .2 + 3.
    spend[visit == 0] = float('nan')
    y = torch.stack([visit, spend, spend * .5], -1)
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend', 'other']],
        measures=['visit', 'spend', 'other'],
        binary_measures=['visit'],
        mixture=[MixtureComponent(measure='spend', offset_init=-2., prob_init=.1, id='low')],
    )
    bf.mc_sampling = 200
    pred = bf(y)
    transform = {
        'spend': LogTransform(bias_adjust=.5),
        'other': LogTransform(),
    }

    df = pred.to_dataframe(transform=transform, use_map=use_map)

    def fail(*args, **kwargs):
        raise AssertionError("interval work was done")

    pred._get_intervals = pred.sample = fail
    df_means = pred.to_dataframe(transform=transform, use_map=use_map, conf=False)
    assert list(df_means.columns) == [c for c in df.columns if c not in ('lower', 'upper')]
    assert np.array_equal(df_means['mean'].values, df['mean'].values)

    means = pred.get_means(transform=transform, use_map=use_map)
    assert means.shape == (4, 12, 3)
    for j, m in enumerate(['visit', 'spend', 'other']):
        assert np.allclose(df.query(f"measure == '{m}'")['mean'].values, means[..., j].reshape(-1).numpy())
    # (back-transformed: e.g. 'other' -- linear, in a nonlinear model -- is on the original scale)
    assert (means[..., 2] > 1.5).all()



@torch.no_grad()
def test_means_only_derived():
    torch.manual_seed(0)
    kf = KalmanFilter(processes=[LocalLevel(id=f'level_{m}', measure=m) for m in 'ab'], measures=['a', 'b'])
    pred = kf(torch.randn(2, 8, 2))
    df = pred.to_dataframe(conf=False, derived={'total': lambda s: s['a'] + s['b']})
    assert 'lower' not in df.columns and not df['mean'].isna().any()
    assert set(df['measure']) == {'a', 'b', 'total'}
    with pytest.raises(ValueError, match="only supported for"):
        pred.to_dataframe(type='states', conf=False)


def test_transform_inverse_roundtrip():
    from torchcast.state_space import SmearingTransform

    x = torch.rand(20, dtype=torch.float64) * 10 + .1
    for t in [
        LogTransform(), LogTransform(base=10), BoxCoxTransform(0), BoxCoxTransform(.5),
        SmearingTransform(LogTransform(base=10), torch.randn(5)),
    ]:
        assert torch.allclose(t.inverse(t.transform(x)), x), t
    assert torch.allclose(LogTransform(base=10).transform(torch.tensor(100.)), torch.tensor(2.))


def test_dataset_from_dataframe_transform():
    """`TimeSeriesDataset.from_dataframe(transform=)` mirrors `Predictions.to_dataframe(transform=)`."""
    import pandas as pd
    from torchcast.utils import TimeSeriesDataset

    torch.manual_seed(0)
    df = pd.DataFrame({
        'group': np.repeat(['g0', 'g1'], 10),
        'time': np.tile(np.arange(10), 2),
        'a': np.random.RandomState(0).rand(20) * 10 + 1,
        'b': np.random.RandomState(1).rand(20) * 10 + 1,
        'x': np.random.RandomState(2).randn(20),
    })
    kwargs = dict(group_colname='group', time_colname='time', dt_unit=None)

    # a single transform applies to the time-series' measures -- not the predictors:
    cols = dict(y_colnames=['a', 'b'], X_colnames=['x'])
    ds = TimeSeriesDataset.from_dataframe(df, **cols, transform=LogTransform(), **kwargs)
    raw = TimeSeriesDataset.from_dataframe(df, **cols, **kwargs)
    # (transformed in float64, then cast)
    assert torch.allclose(ds.tensors[0], raw.tensors[0].log(), rtol=1e-4, atol=1e-6)
    assert torch.equal(ds.tensors[1], raw.tensors[1])
    # ... or all of `measure_colnames`; or a dict:
    ds = TimeSeriesDataset.from_dataframe(df, measure_colnames=['a', 'b'], transform={'b': LogTransform(base=10)}, **kwargs)
    raw = TimeSeriesDataset.from_dataframe(df, measure_colnames=['a', 'b'], **kwargs)
    assert torch.equal(ds.tensors[0][..., 0], raw.tensors[0][..., 0])
    assert torch.allclose(ds.tensors[0][..., 1], raw.tensors[0][..., 1].log10(), rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="not in the time-series"):
        TimeSeriesDataset.from_dataframe(
            df, y_colnames=['a'], X_colnames=['x'], transform={'x': LogTransform()}, **kwargs
        )
    # (the input dataframe is unchanged)
    assert (df['a'] > 1).all()

    # symmetric with `to_dataframe(transform=)`: the actuals come back on the original scale:
    transform = {'a': LogTransform(), 'b': BoxCoxTransform(.5)}
    ds = TimeSeriesDataset.from_dataframe(df, measure_colnames=['a', 'b'], transform=transform, **kwargs)
    kf = KalmanFilter(processes=[LocalLevel(id=f'level_{m}', measure=m) for m in 'ab'], measures=['a', 'b'])
    with torch.no_grad():
        out = kf(ds.tensors[0]).to_dataframe(ds, transform=transform)
    merged = out.merge(df, on=['group', 'time'])
    for m in 'ab':
        rows = merged.query(f"measure == '{m}'")
        assert np.allclose(rows['actual'].values, rows[m].values, rtol=1e-5)


def test_dataloader_from_dataframe_transform():
    """`TimeSeriesDataLoader.from_dataframe(transform=)` matches `TimeSeriesDataset.from_dataframe(transform=)`."""
    import pandas as pd
    from torchcast.utils import TimeSeriesDataset, TimeSeriesDataLoader

    rs = np.random.RandomState(0)
    df = pd.DataFrame({
        'group': np.repeat(['g0', 'g1', 'g2'], 10),
        'time': np.tile(np.arange(10), 3),
        'a': rs.rand(30) * 10 + 1,
        'b': rs.rand(30) * 10 + 1,
        'x': rs.randn(30),
    })
    kwargs = dict(group_colname='group', time_colname='time', dt_unit=None)
    transform = {'a': LogTransform(), 'b': BoxCoxTransform(.5)}
    expected = TimeSeriesDataset.from_dataframe(
        df, y_colnames=['a', 'b'], X_colnames=['x'], transform=transform, **kwargs
    )
    # (`X_colnames` as a function: predictors are never transformed)
    for X_colnames in (['x'], lambda d: d[['x']]):
        loader = TimeSeriesDataLoader.from_dataframe(
            df, y_colnames=['a', 'b'], X_colnames=X_colnames, transform=transform, batch_size=2, **kwargs
        )
        batches = list(loader)
        assert len(batches) == 2
        for batch in batches:
            idx = [list(expected.group_names).index(g) for g in batch.group_names]
            assert torch.equal(batch.tensors[0], expected.tensors[0][idx])
            assert torch.equal(batch.tensors[1], expected.tensors[1][idx])


def _mixture_preds(nonlinear: bool):
    torch.manual_seed(0)
    kf = KalmanFilter(
        processes=[LocalLevel(id='level', measure='y')],
        measures=['y'],
        measure_funs={'y': 'sigmoid'} if nonlinear else None,
        mixture=[MixtureComponent('y', offset_init=-1., prob_init=.3, id='low')],
    )
    with torch.no_grad():
        return kf(torch.rand(3, 10, 1) if nonlinear else torch.randn(3, 10, 1))


@torch.no_grad()
def test_bias_adjust_per_regime_linear():
    # e.g. the median of a low-outlier regime, the mean of the standard regime
    pred = _mixture_preds(nonlinear=False)
    means = pred.get_means(transform=LogTransform(bias_adjust={'low': 0.}))[..., 0]
    mixture = pred.get_mixture('y')
    standard = torch.exp(mixture.means[..., 0] + mixture.vars[..., 0] / 2)
    low = torch.exp(mixture.means[..., 1])
    assert torch.allclose(means, mixture.probs[..., 0] * standard + mixture.probs[..., 1] * low)


@torch.no_grad()
def test_bias_adjust_per_regime_nonlinear(monkeypatch):
    pred = _mixture_preds(nonlinear=True)
    measured_mean, _ = pred._measured_moments_flat()
    probs = pred._measure_regime_probs('y').reshape(-1, 2)
    offset = pred.mixture.components[0].offset

    def means(bias_adjust):
        return pred.get_means(transform=LogTransform(bias_adjust=bias_adjust), use_map=False, num_samples=50)[..., 0]

    # all regimes at bias_adjust=0: the plug-in inverse(MAP) per regime, mixed -- with no sampling
    with monkeypatch.context() as m:
        m.setattr(type(pred), '_get_output_samples', lambda *a, **k: pytest.fail("shouldn't sample"))
        all_zero = means({None: 0., 'low': 0.})
        with pytest.warns(UserWarning, match='group-names'):
            df = pred.to_dataframe(conf=False, transform=LogTransform(bias_adjust=0.), use_map=False)
    assert np.allclose(df['mean'].to_numpy(), all_zero.reshape(-1).numpy())
    expected = probs[:, 0] * torch.exp(measured_mean[:, 0]) + probs[:, 1] * torch.exp(measured_mean[:, 0] + offset)
    assert torch.allclose(all_zero.reshape(-1), expected)

    # each regime is handled separately: changing one regime's value changes only its term
    # (low at 0) - (both at 0) == (neither at 0) - (standard at 0): the standard regime's term, sampled minus plug-in
    assert torch.allclose(means({'low': 0.}) - all_zero, means(None) - means({None: 0.}), atol=1e-5)

    # nonzero values must be the same across a nonlinear measure's regimes
    with pytest.warns(UserWarning, match="must be 0 or the same"):
        means({None: .5, 'low': .3})


def test_bias_adjust_validation():
    with pytest.raises(ValueError):
        LogTransform(bias_adjust={'low': 1.5})
    assert LogTransform(bias_adjust={'low': 0.}).var_multi('low') == 0.
    assert LogTransform(bias_adjust={'low': 0.}).var_multi(None) == 1.
    assert LogTransform(bias_adjust=.5).var_multi('low') == .5
