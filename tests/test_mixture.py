import math

import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal, Normal

from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.process import LocalLevel, LinearModel
from torchcast.state_space.mixture import MixtureComponent, MixtureModel, RegimeTransition, StickyTransition


def _make_kf(measures, mixture_measures, **mixture_kwargs) -> KalmanFilter:
    torch.manual_seed(123)
    mixture = None
    if mixture_measures:
        mixture = MixtureModel(
            [MixtureComponent(measure=m, mean_init=-4., prob_init=.2, id=f'{m}_low') for m in mixture_measures],
            **mixture_kwargs
        )
    return KalmanFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
        measures=measures,
        mixture=mixture,
    )


def _make_y(num_groups: int = 3, num_times: int = 15, num_measures: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    y = torch.randn((num_groups, num_times, num_measures)).cumsum(1) * .3
    y[:, ::4, 0] -= 4.  # some low outliers on the first measure
    return y


def test_mixture_model_combos():
    components = [
        MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='a1'),
        MixtureComponent(measure='c', mean_init=0., prob_init=.2, id='c1'),
        MixtureComponent(measure='c', mean_init=0., prob_init=.2, id='c2'),
    ]
    rm = MixtureModel(components)
    assert rm.mixture_measures == ['a', 'c']
    assert rm.num_combos == 2 * 3
    assert rm.combos[0] == (None, None)
    base_probs = rm.base_probs()
    assert torch.isclose(base_probs.sum(), torch.tensor(1.))
    # marginal prob of 'a' being non-standard is its prob_init (it's the only component for 'a'):
    a_weird = torch.tensor([combo[0] is not None for combo in rm.combos])
    assert torch.isclose(base_probs[a_weird].sum(), torch.tensor(.1))

    # all observed: every combo is distinguishable
    effective, mapping = rm.effective_combos(['a', 'b', 'c'])
    assert len(effective) == 6
    assert effective[0] == ()
    # 'c' unobserved: combos collapse onto 'a' being standard or not
    effective, mapping = rm.effective_combos(['a', 'b'])
    assert len(effective) == 2
    assert (mapping == a_weird.long()).all()
    # measure-indices refer to the observed measures:
    effective, _ = rm.effective_combos(['b', 'c'])
    assert {i for eff in effective for i, _ in eff} == {1}


def test_mixture_model_validation():
    with pytest.raises(ValueError, match="not in `measures`"):
        KalmanFilter(
            processes=[LocalLevel(id='lvl')],
            measures=['a'],
            mixture=[MixtureComponent(measure='z', mean_init=0., prob_init=.1, id='z1')]
        )
    with pytest.raises(ValueError, match="unique ids"):
        MixtureModel([MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='x') for _ in range(2)])
    with pytest.raises(ValueError, match="combos"):
        MixtureModel(
            [MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='a1')],
            transition=StickyTransition(num_combos=3)
        )
    with pytest.raises(ValueError, match="not yet supported .* non-gaussian"):
        BinomialFilter(
            processes=[LocalLevel(id='lvl', measure='visit')],
            measures=['visit'],
            mixture=[MixtureComponent(measure='visit', mean_init=0., prob_init=.1, id='v1')]
        )
    # a list of components is shorthand for a MixtureModel:
    kf = KalmanFilter(
        processes=[LocalLevel(id='lvl')],
        measures=['a'],
        mixture=[MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='a1')]
    )
    assert isinstance(kf.mixture, MixtureModel) and isinstance(kf.mixture.transition, StickyTransition)


def test_sticky_transition():
    torch.manual_seed(1)
    transition = StickyTransition(num_combos=4)
    with torch.no_grad():
        transition._stay_logit.normal_()
    base_probs = torch.softmax(torch.randn(4), 0)
    T = transition.matrix(base_probs)
    assert torch.allclose(T.sum(-1), torch.ones(4))
    # base_probs is the stationary distribution:
    assert torch.allclose(base_probs @ T, base_probs, atol=1e-6)
    # forward matches the matrix form:
    posterior = torch.softmax(torch.randn(5, 4), -1)
    assert torch.allclose(transition(posterior, base_probs), posterior @ T, atol=1e-6)
    # no stickiness -> static mixture:
    with torch.no_grad():
        transition._stay_logit.fill_(-30.)
    assert torch.allclose(transition(posterior, base_probs), base_probs.expand(5, -1), atol=1e-6)
    assert torch.allclose(transition.initial(base_probs, 5), base_probs.expand(5, -1))


def test_custom_transition():
    class MyTransition(RegimeTransition):
        def initial(self, base_probs, num_groups):
            return base_probs.expand(num_groups, -1)

        def forward(self, posterior, base_probs):
            return posterior

    kf = _make_kf(['y'], ['y'], transition=MyTransition(num_combos=2))
    assert isinstance(kf.mixture.transition, MyTransition)


class StaticTransition(RegimeTransition):
    def initial(self, base_probs, num_groups):
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior, base_probs):
        return base_probs.expand(posterior.shape[0], -1)


def test_parameters_registered_and_get_grads():
    kf = _make_kf(['y1', 'y2'], ['y1'])
    names = {n for n, _ in kf.named_parameters()}
    for expected in ('mixture.components.0.mean', 'mixture.components.0._log_std',
                     'mixture.components.0.logit', 'mixture.transition._stay_logit'):
        assert expected in names
    assert any(k.startswith('mixture.') for k in kf.state_dict())

    y = _make_y()
    kf(y).log_prob(y).sum().backward()
    component = kf.mixture.components[0]
    for param in (component.mean, component._log_std, component.logit):
        assert param.grad is not None and param.grad.abs() > 0
    assert (kf.mixture.transition._stay_logit.grad.abs() > 0).all()


@pytest.mark.parametrize("measures,mixture_measures", [
    (['y1'], ['y1']),
    (['y1', 'y2'], ['y1']),
    (['y1', 'y2'], ['y1', 'y2']),
])
@torch.no_grad()
def test_negligible_mixture_matches_kf(measures, mixture_measures):
    kf_mix = _make_kf(measures, mixture_measures)
    for component in kf_mix.mixture.components:
        component.logit.fill_(-30.)
    kf = _make_kf(measures, [])
    kf.load_state_dict({k: v for k, v in kf_mix.state_dict().items() if not k.startswith('mixture.')})

    y = _make_y(num_measures=len(measures))
    y[0, 3:6, 0] = float('nan')
    y[1, 7, -1] = float('nan')
    pred_mix, pred = kf_mix(y), kf(y)
    assert torch.allclose(pred_mix.state_means, pred.state_means, atol=1e-5)
    assert torch.allclose(pred_mix.state_covs, pred.state_covs, atol=1e-5)
    assert torch.allclose(pred_mix.log_prob(y), pred.log_prob(y), atol=1e-4)


@torch.no_grad()
def test_update_step_univariate():
    kf = _make_kf(['y'], ['y'])
    component = kf.mixture.components[0]
    prob = .2

    mean = torch.tensor([[1.0]])
    cov = torch.tensor([[[.5]]])
    H = torch.tensor([[[1.0]]])
    R = torch.tensor([[[.3]]])
    for obs in (1.2, -3.5):
        input = torch.tensor([[obs]])
        state = kf._update_step(input=input, mean=mean, cov=cov, measured_mean=mean, measure_mat=H, measure_cov=R)

        # by hand: the standard regime is the usual update; the component's regime is an update with the residual
        # offset by the component's mean, and the component's variance added to the measurement-noise.
        S_n = cov + R
        K_n = cov / S_n
        mean_n = mean + K_n * (obs - mean)
        cov_n = (1 - K_n) * cov
        S_w = cov + R + component.var
        K_w = cov / S_w
        mean_w = mean + K_w * (obs - mean - component.mean)
        cov_w = (1 - K_w) * cov
        lik_n = Normal(mean, S_n.sqrt()).log_prob(input).exp()
        lik_w = Normal(mean + component.mean, S_w.sqrt()).log_prob(input).exp()
        w_w = prob * lik_w / (prob * lik_w + (1 - prob) * lik_n)
        w_n = 1 - w_w
        expected_mean = w_n * mean_n + w_w * mean_w
        expected_cov = w_n * (cov_n + (mean_n - expected_mean) ** 2) + w_w * (cov_w + (mean_w - expected_mean) ** 2)

        assert torch.allclose(state.regime_probs[:, 1], w_w.view(1), atol=1e-6)
        assert torch.allclose(state.mean, expected_mean.view(1, 1), atol=1e-6)
        assert torch.allclose(state.cov, expected_cov.view(1, 1, 1), atol=1e-6)


@torch.no_grad()
def test_update_step_unobserved_mixture_measure():
    kf = _make_kf(['y1', 'y2'], ['y1'])
    kf_plain = _make_kf(['y1', 'y2'], [])
    mean = torch.randn(4, 2)
    cov = torch.eye(2).expand(4, -1, -1) * .5
    # only 'y2' observed:
    kwargs = dict(
        input=torch.randn(4, 1),
        mean=mean,
        cov=cov,
        measured_mean=mean[:, [1]],
        measure_mat=torch.tensor([[[0., 1.]]]).expand(4, -1, -1),
        measure_cov=torch.full((4, 1, 1), .3),
    )
    state = kf._update_step(**kwargs, val_idx=torch.tensor([1]))
    expected_mean, expected_cov = kf_plain._update_step(**kwargs, val_idx=torch.tensor([1]))
    assert torch.allclose(state.mean, expected_mean)
    assert torch.allclose(state.cov, expected_cov)
    # the regime is unobserved, so the posterior is the prior:
    assert torch.allclose(state.regime_probs, kf.mixture.base_probs().expand(4, -1))


@torch.no_grad()
def test_log_prob_brute_force():
    measures = ['y1', 'y2']
    kf = _make_kf(measures, measures)
    y = _make_y(num_measures=2)
    y[0, 5, 1] = float('nan')
    pred = kf(y)
    lp = pred.log_prob(y)

    rm = kf.mixture
    H = torch.eye(2)
    R = kf.measure_covariance({}, num_groups=1, num_times=1)[0, 0]
    for g, t in [(0, 0), (1, 4), (2, 9), (0, 5)]:
        obs = y[g, t]
        observed = (~obs.isnan()).nonzero().view(-1).tolist()
        m = pred.state_means[g, t] @ H.T
        S = H @ pred.state_covs[g, t] @ H.T + R
        total = 0.
        for combo, prob in zip(rm.combos, pred.regime_probs[g, t]):
            # each measure in a component's regime: offset mean, and extra variance
            offset = torch.stack([torch.tensor(0.) if c is None else c.mean for c in combo])
            extra = torch.stack([torch.tensor(0.) if c is None else c.var for c in combo])
            m_c, S_c = m + offset, S + torch.diag(extra)
            lik = MultivariateNormal(m_c[observed], S_c[observed][:, observed]).log_prob(obs[observed]).exp()
            total += prob * lik
        assert math.isclose(lp[g, t].item(), math.log(total), rel_tol=1e-4)


@pytest.mark.parametrize("univariate_prob", [False, True])
def test_binomial_filter_with_mixture(univariate_prob: bool):
    """
    A binary measure (e.g. 'did they visit') plus a gaussian measure with a mixture component (e.g. 'log-spend'), where
    the gaussian measure is missing whenever the binary one is 0.
    """
    torch.manual_seed(1)
    num_groups, num_times = 4, 20
    visit = (torch.rand(num_groups, num_times) > .4).float()
    spend = torch.randn(num_groups, num_times).cumsum(1) * .2 + 3.
    spend[torch.rand(num_groups, num_times) > .85] = -1.  # quick visits
    spend[visit == 0] = float('nan')
    y = torch.stack([visit, spend], -1)

    measures = ['visit', 'spend']
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
        measures=measures,
        binary_measures=['visit'],
        mixture=MixtureModel(
            [MixtureComponent(measure='spend', mean_init=-1., prob_init=.1, id='quick')],
            univariate_prob=univariate_prob
        ),
    )
    bf.mc_sampling = 50
    pred = bf(y)
    lp = pred.log_prob(y)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    component = bf.mixture.components[0]
    for param in (component.mean, component._log_std, component.logit):
        assert param.grad is not None and param.grad.abs() > 0

    # end-to-end training:
    bf.zero_grad()
    bf.fit(y, stopping={'max_iter': 3}, verbose=0)


@torch.no_grad()
def test_binary_measures_excluded_from_responsibilities():
    """
    Binary measures don't contribute to regime-probabilities: with only a binary measure besides the mixture measure,
    scoring all measures is the same as scoring only the mixture measures (``univariate_prob=True``).
    """
    torch.manual_seed(1)
    visit = (torch.rand(3, 12) > .4).float()
    spend = torch.randn(3, 12).cumsum(1) * .2 + 3.
    spend[torch.rand(3, 12) > .8] = -1.
    spend[visit == 0] = float('nan')
    y = torch.stack([visit, spend], -1)

    preds = {}
    for univariate_prob in (False, True):
        torch.manual_seed(2)
        bf = BinomialFilter(
            processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['visit', 'spend']],
            measures=['visit', 'spend'],
            binary_measures=['visit'],
            mixture=MixtureModel(
                [MixtureComponent(measure='spend', mean_init=-1., prob_init=.1, id='quick')],
                univariate_prob=univariate_prob
            ),
        )
        bf.mc_sampling = 20
        preds[univariate_prob] = bf(y, include_updates_in_output=True)
    assert torch.allclose(preds[False].update_regime_probs, preds[True].update_regime_probs)
    assert torch.allclose(preds[False].state_means, preds[True].state_means)
    # (and responsibilities are actually informative here:)
    assert preds[False].update_regime_probs[..., 1].max() > .5


@torch.no_grad()
def test_no_stickiness_is_static():
    measures = ['y1', 'y2']
    kf_static = _make_kf(measures, measures, transition=StaticTransition(num_combos=4))
    kf_sticky = _make_kf(measures, measures)
    kf_sticky.load_state_dict(
        {k: v for k, v in kf_static.state_dict().items() if 'transition' not in k}, strict=False
    )
    kf_sticky.mixture.transition._stay_logit.fill_(-30.)
    y = _make_y(num_measures=2)
    y[0, 3:6, 0] = float('nan')
    pred_static, pred_sticky = kf_static(y), kf_sticky(y)
    base_probs = kf_static.mixture.base_probs()
    assert torch.allclose(pred_static.regime_probs, base_probs.expand_as(pred_static.regime_probs))
    assert torch.allclose(pred_sticky.regime_probs, pred_static.regime_probs, atol=1e-6)
    assert torch.allclose(pred_sticky.state_means, pred_static.state_means, atol=1e-5)
    assert torch.allclose(pred_sticky.log_prob(y), pred_static.log_prob(y), atol=1e-5)


@torch.no_grad()
def test_regime_probs_through_time():
    kf = _make_kf(['y'], ['y'])
    transition = kf.mixture.transition
    transition._stay_logit.fill_(2.)  # stay ~ .88
    base_probs = kf.mixture.base_probs()

    y = torch.zeros((2, 10, 1))
    y[0, 4] = -4.  # an outlier (at the component's mean) for group 0 only
    y[:, 7] = float('nan')
    pred = kf(y)
    priors = pred.regime_probs
    assert torch.allclose(priors[:, 0], base_probs.expand(2, -1))
    # after the outlier, group 0 is more likely to be in the 'low' regime -- both vs. the base-rate and vs. group 1:
    assert priors[0, 5, 1] > .5 > base_probs[1]
    assert priors[0, 5, 1] > priors[1, 5, 1]
    # ...and this decays back towards the base-rate as normal observations come in:
    assert priors[0, 6, 1] < priors[0, 5, 1]
    # when the observation is missing, the prior just evolves via the transition:
    assert torch.allclose(priors[:, 8], transition(priors[:, 7], base_probs), atol=1e-6)
    # forecasting past the data:
    pred_fcast = kf(y, out_timesteps=13)
    assert torch.allclose(pred_fcast.regime_probs[:, 11], transition(pred_fcast.regime_probs[:, 10], base_probs))
    # slicing keeps priors aligned:
    assert torch.allclose(pred[:, 4:6].regime_probs, priors[:, 4:6])


@pytest.mark.parametrize("n_step,every_step", [(2, True), (3, True), (3, False)])
@torch.no_grad()
def test_n_step_regime_probs(n_step: int, every_step: bool):
    """
    As in ``test_n_step_matches_nan_forecast``: an h-step prediction for t should match a 1-step prediction with the
    observations between t - h and t missing.
    """
    kf = _make_kf(['y1', 'y2'], ['y1'])
    kf.mixture.transition._stay_logit.fill_(1.)
    y = _make_y(num_measures=2)
    pred_n = kf(y, n_step=n_step, every_step=every_step)
    for t in range(y.shape[1]):
        h = min(t + 1, n_step) if every_step else (t % n_step) + 1
        y_nan = y.clone()
        y_nan[:, (t - h + 1):t] = float('nan')
        pred_1 = kf(y_nan, n_step=1)
        assert torch.allclose(pred_n.state_means[:, t], pred_1.state_means[:, t], atol=1e-5)
        assert torch.allclose(pred_n.state_covs[:, t], pred_1.state_covs[:, t], atol=1e-5)
        assert torch.allclose(pred_n.regime_probs[:, t], pred_1.regime_probs[:, t], atol=1e-6)
    # so the log-prob matches too:
    assert torch.allclose(pred_n.log_prob(y)[:, -1], kf(y_nan, n_step=1).log_prob(y)[:, -1], atol=1e-5)


@torch.no_grad()
def test_initial_state_continuation():
    kf = _make_kf(['y1', 'y2'], ['y1'])
    kf.mixture.transition._stay_logit.fill_(1.)
    y = _make_y(num_measures=2)
    y[:, 11, 0] = -4.  # an outlier right before the split, so the regime-probs at the split are informative
    split = 12

    full = kf(y)
    state = kf(y[:, :split], include_updates_in_output=True).get_state_at_times(split - 1)
    assert state.regime_probs.shape == (y.shape[0], kf.mixture.num_combos)
    cont = kf(y[:, split:], initial_state=state)
    assert torch.allclose(cont.state_means, full.state_means[:, split:], atol=1e-5)
    assert torch.allclose(cont.state_covs, full.state_covs[:, split:], atol=1e-5)
    assert torch.allclose(cont.regime_probs, full.regime_probs[:, split:], atol=1e-6)
    assert torch.allclose(cont.log_prob(y[:, split:]), full.log_prob(y)[:, split:], atol=1e-5)

    # a plain (mean, cov) tuple restarts regime-probs from the transition's initial distribution:
    cont_tuple = kf(y[:, split:], initial_state=tuple(state))
    assert torch.allclose(cont_tuple.regime_probs[:, 0], kf.mixture.base_probs().expand(y.shape[0], -1))

    # 'prediction'-type states also carry regime-probs:
    pred_state = full.get_state_at_times(split, type_='prediction')
    assert torch.allclose(pred_state.regime_probs, full.regime_probs[:, split])

    # simulate from the state:
    sim = kf.simulate(out_timesteps=5, initial_state=state, num_sims=2)
    assert sim.regime_probs.shape == (2 * y.shape[0], 5, kf.mixture.num_combos)

    # a model without mixture components rejects regime-probs:
    kf_plain = _make_kf(['y1', 'y2'], [])
    with pytest.raises(ValueError, match="no mixture components"):
        kf_plain(y[:, split:], initial_state=state)


def _make_kf_with_predictors(stay_logit: float = 1., coefs=(2., -.5)) -> KalmanFilter:
    """Like ``_make_kf(['y1', 'y2'], ['y1'])``, but the component's base-rate depends on predictors."""
    torch.manual_seed(123)
    kf = KalmanFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in ['y1', 'y2']],
        measures=['y1', 'y2'],
        mixture=[
            MixtureComponent(measure='y1', mean_init=-4., prob_init=.2, id='y1_low', predictors=['treated', 'other'])
        ],
    )
    with torch.no_grad():
        kf.mixture.components[0].coefs.copy_(torch.tensor(coefs))
        kf.mixture.transition._stay_logit.fill_(stay_logit)
    return kf


def _make_X(num_groups: int = 3, num_times: int = 15) -> torch.Tensor:
    torch.manual_seed(2)
    treated = (torch.rand(num_groups, num_times) > .5).float()
    return torch.stack([treated, torch.randn(num_groups, num_times)], -1)


@torch.no_grad()
def test_component_predictors_base_rates():
    y, X = _make_y(num_measures=2), _make_X()

    # zero coefficients: same as no predictors
    kf_zero = _make_kf_with_predictors(coefs=(0., 0.))
    kf_plain = _make_kf(['y1', 'y2'], ['y1'])
    kf_plain.load_state_dict({k: v for k, v in kf_zero.state_dict().items() if not k.endswith('coefs')})
    pred_zero, pred_plain = kf_zero(y, X=X), kf_plain(y)
    assert torch.allclose(pred_zero.regime_probs, pred_plain.regime_probs)
    assert torch.allclose(pred_zero.log_prob(y), pred_plain.log_prob(y))

    # no stickiness: the regime-prior at each timestep is just the base-rate given that timestep's predictors:
    kf = _make_kf_with_predictors(stay_logit=-30.)
    component = kf.mixture.components[0]
    pred = kf(y, X=X)
    expected = torch.sigmoid(component.logit + X @ component.coefs)
    assert torch.allclose(pred.regime_probs[..., 1], expected, atol=1e-6)
    # incl. when forecasting past the data:
    X_long = torch.cat([X, _make_X(num_times=4)], 1)
    pred_fcast = kf(y, X=X_long, out_timesteps=19)
    assert torch.allclose(pred_fcast.regime_probs[..., 1], torch.sigmoid(component.logit + X_long @ component.coefs),
                          atol=1e-6)
    # treated timesteps have more 'low' regime:
    treated = X[..., 0].bool()
    assert pred.regime_probs[..., 1][treated].mean() > 2 * pred.regime_probs[..., 1][~treated].mean()

    # with stickiness, the base-probs for the current predictors are still the stationary distribution:
    kf = _make_kf_with_predictors()
    base = kf.mixture.base_probs({0: X[:, 0]})  # (num_groups, num_combos)
    assert torch.allclose(kf.mixture.transition(base, base), base, atol=1e-6)
    assert torch.allclose(kf(y, X=X).regime_probs[:, 0], base, atol=1e-6)


@pytest.mark.parametrize("n_step,every_step", [(2, True), (3, True), (3, False)])
@torch.no_grad()
def test_component_predictors_n_step(n_step: int, every_step: bool):
    """As in ``test_n_step_regime_probs``, but with time-varying base-rates (catches off-by-one time-indexing)."""
    kf = _make_kf_with_predictors()
    y, X = _make_y(num_measures=2), _make_X()
    pred_n = kf(y, n_step=n_step, every_step=every_step, X=X)
    for t in range(y.shape[1]):
        h = min(t + 1, n_step) if every_step else (t % n_step) + 1
        y_nan = y.clone()
        y_nan[:, (t - h + 1):t] = float('nan')
        pred_1 = kf(y_nan, n_step=1, X=X)
        assert torch.allclose(pred_n.regime_probs[:, t], pred_1.regime_probs[:, t], atol=1e-6)
        assert torch.allclose(pred_n.state_means[:, t], pred_1.state_means[:, t], atol=1e-5)


@torch.no_grad()
def test_component_predictors_continuation():
    kf = _make_kf_with_predictors()
    y, X = _make_y(num_measures=2), _make_X()
    split = 9
    full = kf(y, X=X)
    state = kf(y[:, :split], X=X[:, :split], include_updates_in_output=True).get_state_at_times(split - 1)
    cont = kf(y[:, split:], X=X[:, split:], initial_state=state)
    assert torch.allclose(cont.regime_probs, full.regime_probs[:, split:], atol=1e-6)
    assert torch.allclose(cont.log_prob(y[:, split:]), full.log_prob(y)[:, split:], atol=1e-5)
    sim = kf.simulate(out_timesteps=5, initial_state=state, num_sims=2, X=X[:, split:split + 5])
    assert sim.regime_probs.shape == (2 * y.shape[0], 5, kf.mixture.num_combos)


def test_component_predictors_kwargs():
    y, X = _make_y(num_measures=2), _make_X()
    kf = _make_kf_with_predictors()
    with pytest.raises(TypeError, match="expected a `X`"):
        kf(y)
    with pytest.raises(ValueError, match="to have shape"):
        kf(y, X=X[..., :1])
    with pytest.raises(ValueError, match="timesteps"):
        kf(y, X=X[:, :5])
    with pytest.raises(ValueError, match="needs `X`"):
        kf.mixture.log_base_probs()
    with pytest.raises(ValueError, match="list of strings"):
        MixtureComponent(measure='y1', mean_init=0., prob_init=.1, id='x', predictors='treated')

    # a different `X` for a LinearModel, via the `{id}__X` override:
    torch.manual_seed(0)
    kf = KalmanFilter(
        processes=[LocalLevel(id='level', measure='y1'), LinearModel(id='lm', predictors=['a', 'b', 'c'], measure='y1')],
        measures=['y1'],
        mixture=[MixtureComponent(measure='y1', mean_init=-4., prob_init=.2, id='low', predictors=['treated'])],
    )
    X_lm = torch.randn(3, 15, 3)
    pred = kf(y[..., :1], X=X_lm, low__X=X[..., :1])
    lp = pred.log_prob(y[..., :1])
    lp.sum().backward()
    assert kf.mixture.components[0].coefs.grad.abs() > 0
    with pytest.raises(RuntimeError, match="Unexpected kwargs"):
        kf(y[..., :1], X=X_lm, low__X=X[..., :1], typo=1)


def test_component_predictors_fit():
    """A treatment that makes the 'low' regime more common: fitting recovers a positive coefficient."""
    torch.manual_seed(0)
    num_groups, num_times = 20, 30
    treated = (torch.arange(num_groups) % 2).float().view(-1, 1).expand(-1, num_times)
    y = torch.randn(num_groups, num_times).cumsum(1) * .1
    is_low = torch.rand(num_groups, num_times) < torch.where(treated.bool(), .4, .05)
    y[is_low] = -4. + torch.randn(int(is_low.sum())) * .3
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        mixture=[MixtureComponent(measure='y', mean_init=-3., prob_init=.1, id='low', predictors=['treated'])],
    )
    kf.fit(y.unsqueeze(-1), X=treated.unsqueeze(-1), stopping={'max_iter': 30}, verbose=0)
    component = kf.mixture.components[0]
    with torch.no_grad():
        p_untreated = torch.sigmoid(component.logit).item()
        p_treated = torch.sigmoid(component.logit + component.coefs[0]).item()
    assert .0 < p_untreated < .15
    assert .25 < p_treated < .6


def test_standard_probs():
    components = [
        MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='a1'),
        MixtureComponent(measure='c', mean_init=0., prob_init=.2, id='c1'),
        MixtureComponent(measure='c', mean_init=0., prob_init=.2, id='c2'),
    ]
    rm = MixtureModel(components)
    regime_probs = torch.softmax(torch.randn(4, rm.num_combos), -1)
    probs = rm.standard_probs(regime_probs, ['a', 'b', 'c'])
    for j, m in enumerate(['a', 'b', 'c']):
        if m == 'b':
            assert torch.allclose(probs[:, j], torch.ones(4))
            continue
        k = rm.mixture_measures.index(m)
        standard = torch.tensor([combo[k] is None for combo in rm.combos])
        assert torch.allclose(probs[:, j], regime_probs[:, standard].sum(-1))
    # at base-probs, matches the base-rate:
    assert torch.allclose(rm.standard_probs(rm.base_probs().unsqueeze(0), ['a'])[0], torch.tensor([.9]))


@torch.no_grad()
def test_ewma_scaler_weights():
    from torchcast.state_space.adaptive_scaling import EWMAdaptiveScaler

    torch.manual_seed(1)
    resids = [torch.randn(3, 2) for _ in range(5)]
    skip = torch.zeros(3, 2, dtype=torch.bool)

    def run(weights):
        torch.manual_seed(2)
        scaler = EWMAdaptiveScaler(num_measures=2)
        scaler.initialize(20)
        scaler.reset()
        return [scaler(r, skip, weights=weights) for r in resids], scaler

    unweighted, s0 = run(None)
    ones, s1 = run(torch.ones(3, 2))
    assert all(torch.allclose(a, b) for a, b in zip(unweighted, ones))

    # zero weight on the last residual == skipping it:
    torch.manual_seed(1)
    scaler = EWMAdaptiveScaler(num_measures=2)
    scaler.initialize(20)
    scaler.reset()
    for i, r in enumerate(resids):
        w = torch.ones(3, 2)
        w[0, 1] = 0. if i == 4 else 1.
        scaler(r, skip, weights=w)
    torch.manual_seed(1)
    scaler_skip = EWMAdaptiveScaler(num_measures=2)
    scaler_skip.initialize(20)
    scaler_skip.reset()
    for i, r in enumerate(resids):
        sk = skip.clone()
        sk[0, 1] = i == 4
        scaler_skip(r, sk)
    assert torch.allclose(scaler._running, scaler_skip._running)
    assert torch.allclose(scaler._time, scaler_skip._time.float())


@torch.no_grad()
def test_adaptive_scaling_ignores_explained_outliers():
    """
    A low outlier that's explained by the mixture component shouldn't inflate the adaptive scaling (whereas without the
    mixture, it does): for the scaling, it should be (almost) as if the observation were missing.
    """
    def make(with_mixture: bool):
        torch.manual_seed(0)
        kf = KalmanFilter(
            processes=[LocalLevel(id='level')],
            measures=['y'],
            adaptive_scaling=True,
            mixture=[MixtureComponent('y', mean_init=-6., prob_init=.05, id='low')] if with_mixture else None
        )
        kf.adaptive_scaling.initialize(30)
        kf.adaptive_scaling.weight.fill_(1.)
        if with_mixture:
            kf.mixture.components[0]._log_std.fill_(-1.)
        return kf

    torch.manual_seed(1)
    y = torch.randn(4, 30, 1) * .3
    y_outlier, y_missing = y.clone(), y.clone()
    y_outlier[:, 15] = -6.
    y_missing[:, 15] = float('nan')
    for with_mixture in (False, True):
        kf = make(with_mixture)
        # measure-variance at t=16 reflects the scaling after observing t=15:
        ratio = kf(y_outlier).measure_covs[:, 16, 0, 0] / kf(y_missing).measure_covs[:, 16, 0, 0]
        if with_mixture:
            assert torch.allclose(ratio, torch.ones_like(ratio), atol=.01)
        else:
            assert (ratio > 2).all()


def test_mixture_of_normals():
    from scipy import stats
    from torchcast.state_space.mixture import MixtureOfNormals

    # a single component is just a normal:
    single = MixtureOfNormals(['standard'], torch.ones(3, 1), torch.tensor([[0.], [1.], [-2.]]), torch.full((3, 1), 4.))
    for q in (.025, .5, .9):
        expected = torch.as_tensor(stats.norm.ppf(q, loc=[0., 1., -2.], scale=2.), dtype=torch.float32)
        assert torch.allclose(single.quantile(q), expected, atol=1e-4)

    # a skewed mixture:
    mix = MixtureOfNormals(
        ['standard', 'low'],
        probs=torch.tensor([[.9, .1]]),
        means=torch.tensor([[5., -5.]]),
        vars=torch.tensor([[.25, 1.]])
    )
    assert torch.allclose(mix.mean(), torch.tensor([4.]))
    assert torch.allclose(mix.var(), torch.tensor([.9 * (.25 + 25) + .1 * (1 + 25) - 16]))
    for q in (.05, .1, .5, .95):
        assert torch.allclose(mix.cdf(mix.quantile(q)), torch.tensor([q]), atol=1e-5)
    # the 5% quantile is in the low component, but the 50% is in the standard one:
    assert mix.quantile(.05) < -3 and mix.quantile(.5) > 4


@torch.no_grad()
def test_prediction_outputs():
    measures = ['y1', 'y2']
    kf = _make_kf(measures, ['y1'])
    y = _make_y(num_measures=2)
    y[0, 3:6, 0] = float('nan')
    pred = kf(y)
    G, T = y.shape[:2]

    # per-measure mixture:
    mix = pred.get_mixture('y1')
    assert mix.labels == ['standard', 'y1_low']
    assert mix.probs.shape == (G, T, 2)
    assert torch.allclose(mix.probs.sum(-1), torch.ones(G, T))
    with pytest.raises(ValueError, match="no mixture components"):
        pred.get_mixture('y2')

    # means/covs are the moments of the joint mixture; their marginals match the per-measure mixture:
    means, covs = pred
    assert torch.allclose(means[..., 0], mix.mean(), atol=1e-5)
    assert torch.allclose(covs[..., 0, 0], mix.var(), atol=1e-4)
    # 'y2' has no components, so its moments are the standard-regime ones:
    measured_mean, system_cov = pred._measured_moments_flat()
    assert torch.allclose(means[..., 1], measured_mean[:, 1].view(G, T), atol=1e-5)
    assert torch.allclose(covs[..., 1, 1], system_cov[:, 1, 1].view(G, T), atol=1e-5)

    # cross-covariance vs. sampling from the mixture, for one group/time:
    labels, probs, combo_means, combo_covs = pred._get_regime_combos()
    assert labels == [('standard',), ('y1_low',)]
    g, t = 1, 8
    torch.manual_seed(0)
    n = 400_000
    which = torch.multinomial(probs[g, t], n, replacement=True)
    samples = MultivariateNormal(combo_means[g, t], combo_covs[g, t]).sample((n,))[torch.arange(n), which]
    assert torch.allclose(samples.T.cov(), covs[g, t], atol=.02, rtol=.02)

    # dataframe: exact mixture quantiles for y1, gaussian for y2
    df = pred.to_dataframe(type='predictions', conf=.9)
    df_y1 = df.query("measure == 'y1'").sort_values(['group', 'time'])
    assert torch.allclose(torch.as_tensor(df_y1['lower'].values, dtype=torch.float32), mix.quantile(.05).reshape(-1),
                          atol=1e-4)
    assert (df['lower'] < df['mean']).all() and (df['mean'] < df['upper']).all()


@torch.no_grad()
def test_negligible_mixture_outputs_match_kf():
    kf_mix = _make_kf(['y1', 'y2'], ['y1'])
    kf_mix.mixture.components[0].logit.fill_(-30.)
    kf = _make_kf(['y1', 'y2'], [])
    kf.load_state_dict({k: v for k, v in kf_mix.state_dict().items() if not k.startswith('mixture.')})
    y = _make_y(num_measures=2)
    pred_mix, pred = kf_mix(y), kf(y)
    assert torch.allclose(pred_mix.means, pred.means, atol=1e-5)
    assert torch.allclose(pred_mix.covs, pred.covs, atol=1e-5)
    df_mix, df = pred_mix.to_dataframe(), pred.to_dataframe()
    for col in ('mean', 'lower', 'upper'):
        assert np.allclose(df_mix[col].values, df[col].values, atol=1e-3)


@torch.no_grad()
def test_binomial_prediction_outputs():
    torch.manual_seed(1)
    num_groups, num_times = 4, 20
    visit = (torch.rand(num_groups, num_times) > .4).float()
    spend = torch.randn(num_groups, num_times).cumsum(1) * .2 + 3.
    spend[visit == 0] = float('nan')
    y = torch.stack([visit, spend], -1)
    measures = ['visit', 'spend']
    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
        measures=measures,
        binary_measures=['visit'],
        mixture=[MixtureComponent(measure='spend', mean_init=-1., prob_init=.1, id='quick')],
    )
    bf.mc_sampling = 200
    pred = bf(y)
    means = pred.means
    assert torch.allclose(means[..., 1], pred.get_mixture('spend').mean(), atol=1e-5)
    assert ((means[..., 0] > 0) & (means[..., 0] < 1)).all()
    df = pred.to_dataframe(use_map=True)
    assert set(df['measure']) == {'visit', 'spend'}
    assert np.isfinite(df[['mean', 'lower', 'upper']].values).all()


@torch.no_grad()
def test_mixture_covs_warns_once():
    import warnings
    from torchcast.state_space import predictions

    kf = _make_kf(['y1', 'y2'], ['y1'])
    pred = kf(_make_y(num_measures=2))
    predictions._warn_once.pop('mixture_cov', None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = pred.means
        assert not [w for w in caught if 'mixture' in str(w.message)]
        mean, cov = pred  # (the common pattern the warning is for)
        _ = pred.covs
    assert len([w for w in caught if 'covariance of a (non-gaussian) mixture' in str(w.message)]) == 1


@torch.no_grad()
def test_components_are_offsets():
    """
    A component is an offset from each group's own (state-dependent) level: a group whose level is low isn't thereby
    in the 'low' regime -- only observations well below *its* level are.
    """
    torch.manual_seed(0)
    num_times = 30
    level = torch.tensor([3., -1.]).view(2, 1, 1)  # a typical group, and a group with a low level
    y = level + torch.randn(2, num_times, 1) * .1
    y[:, 20] -= 4.  # a 'quick visit' for both
    kf = KalmanFilter(
        processes=[LocalLevel(id='level')],
        measures=['y'],
        mixture=[MixtureComponent(measure='y', mean_init=-4., prob_init=.1, id='low')],
    )
    with torch.no_grad():
        kf.mixture.components[0]._log_std.fill_(math.log(.3))
        kf.measure_covariance.cholesky_log_diag.fill_(math.log(.1))
    pred = kf(y, include_updates_in_output=True)
    p_low = pred.update_regime_probs[..., 1]
    # once each group's level is learned, normal observations are standard for both. (Before that, the offset and the
    # level aren't identified: e.g. 'standard regime at level 3' vs. 'low regime at level 7' -- only the base-rate
    # distinguishes them -- so allow a burn-in.)
    normal = torch.ones(num_times, dtype=torch.bool)
    normal[:8] = normal[20] = False
    assert (p_low[:, normal] < .01).all()
    # ...and the quick visit is 'low' for both:
    assert (p_low[:, 20] > .99).all()
    # the quick visit barely moves the state (its offset is explained):
    assert torch.allclose(pred.update_means[:, 20], pred.update_means[:, 19], atol=.05)


@torch.no_grad()
def test_joseph_form_option():
    """
    `MixtureModel(joseph_form=False)` uses the simpler covariance update for the non-standard regime-combos -- the same,
    in exact arithmetic. The standard combo follows the model's `joseph_form`.
    """
    y = _make_y(num_measures=2)
    y[0, 3, 1] = float('nan')
    preds = {}
    for joseph_form in (True, False):
        kf = _make_kf(['y1', 'y2'], ['y1', 'y2'], joseph_form=joseph_form)
        assert kf.mixture.joseph_form is joseph_form
        preds[joseph_form] = kf(y, include_updates_in_output=True)
    assert MixtureModel([MixtureComponent(measure='y', mean_init=0., prob_init=.1, id='a')]).joseph_form
    for attr in ('update_means', 'update_covs', 'update_regime_probs'):
        assert torch.allclose(getattr(preds[True], attr), getattr(preds[False], attr), atol=1e-5)
    assert torch.allclose(preds[True].log_prob(y), preds[False].log_prob(y), atol=1e-5)

    # which update each combo uses: record the batch-size of each call. all mixture measures are observed here, so
    # every update-step is a mixture update, with 4 effective combos x 3 groups -- the first 3 rows are the standard
    # combo.
    kf = _make_kf(['y1', 'y2'], ['y1', 'y2'])
    calls = {}

    def record(name, fun):
        def wrapped(cov, *args, **kwargs):
            calls.setdefault(name, set()).add(cov.shape[0])
            return fun(cov, *args, **kwargs)
        return wrapped

    kf._covariance_update = record('joseph', KalmanFilter._covariance_update)
    kf._simple_covariance_update = record('simple', KalmanFilter._simple_covariance_update)
    expected = {
        # (model, mixture): calls
        (True, True): {'joseph': {12}},
        (True, False): {'joseph': {3}, 'simple': {9}},
        (False, True): {'joseph': {9}, 'simple': {3}},
        (False, False): {'simple': {12}},
    }
    for (model_joseph, mixture_joseph), expected_calls in expected.items():
        calls.clear()
        kf.joseph_form = model_joseph
        kf.mixture.joseph_form = mixture_joseph
        kf(_make_y(num_measures=2))
        assert calls == expected_calls, (model_joseph, mixture_joseph)
    # older pickles (no attributes) use the joseph form:
    calls.clear()
    del kf.joseph_form, kf.mixture.joseph_form
    kf(_make_y(num_measures=2))
    assert calls == {'joseph': {12}}
