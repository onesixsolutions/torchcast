import math

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal

from torchcast.kalman_filter import KalmanFilter, BinomialFilter
from torchcast.process import LocalLevel
from torchcast.state_space.mixture import MixtureComponent, RegimeModel, RegimeTransition, StickyTransition


def _make_kf(measures, mixture_measures, **kwargs) -> KalmanFilter:
    torch.manual_seed(123)
    return KalmanFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
        measures=measures,
        mixture_components=[
            MixtureComponent(measure=m, mean_init=-4., prob_init=.2, id=f'{m}_low') for m in mixture_measures
        ] or None,
        **kwargs
    )


def _make_y(num_groups: int = 3, num_times: int = 15, num_measures: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    y = torch.randn((num_groups, num_times, num_measures)).cumsum(1) * .3
    y[:, ::4, 0] -= 4.  # some low outliers on the first measure
    return y


def test_regime_model_combos():
    components = [
        MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='a1'),
        MixtureComponent(measure='c', mean_init=0., prob_init=.2, id='c1'),
        MixtureComponent(measure='c', mean_init=0., prob_init=.2, id='c2'),
    ]
    rm = RegimeModel(components, measures=['a', 'b', 'c'])
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


def test_regime_model_validation():
    with pytest.raises(ValueError, match="not in `measures`"):
        RegimeModel([MixtureComponent(measure='z', mean_init=0., prob_init=.1, id='z1')], measures=['a'])
    with pytest.raises(ValueError, match="unique ids"):
        RegimeModel(
            [MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='x') for _ in range(2)],
            measures=['a']
        )
    with pytest.raises(ValueError, match="combos"):
        RegimeModel(
            [MixtureComponent(measure='a', mean_init=0., prob_init=.1, id='a1')],
            measures=['a'],
            transition=StickyTransition(num_combos=3)
        )
    with pytest.raises(ValueError, match="no `mixture_components`"):
        KalmanFilter(processes=[LocalLevel(id='lvl')], measures=['a'], regime_transition=StickyTransition(2))
    with pytest.raises(ValueError, match="measure-function"):
        BinomialFilter(
            processes=[LocalLevel(id='lvl', measure='visit')],
            measures=['visit'],
            mixture_components=[MixtureComponent(measure='visit', mean_init=0., prob_init=.1, id='v1')]
        )


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

    kf = _make_kf(['y'], ['y'], regime_transition=MyTransition(num_combos=2))
    assert isinstance(kf.regime_model.transition, MyTransition)


class StaticTransition(RegimeTransition):
    def initial(self, base_probs, num_groups):
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior, base_probs):
        return base_probs.expand(posterior.shape[0], -1)


def test_parameters_registered_and_get_grads():
    kf = _make_kf(['y1', 'y2'], ['y1'])
    names = {n for n, _ in kf.named_parameters()}
    for expected in ('regime_model.components.0.mean', 'regime_model.components.0._log_std',
                     'regime_model.components.0.logit', 'regime_model.transition._stay_logit'):
        assert expected in names
    assert any(k.startswith('regime_model.') for k in kf.state_dict())

    y = _make_y()
    kf(y).log_prob(y).sum().backward()
    component = kf.regime_model.components[0]
    for param in (component.mean, component._log_std, component.logit):
        assert param.grad is not None and param.grad.abs() > 0
    assert (kf.regime_model.transition._stay_logit.grad.abs() > 0).all()


@pytest.mark.parametrize("measures,mixture_measures", [
    (['y1'], ['y1']),
    (['y1', 'y2'], ['y1']),
    (['y1', 'y2'], ['y1', 'y2']),
])
@torch.no_grad()
def test_negligible_mixture_matches_kf(measures, mixture_measures):
    kf_mix = _make_kf(measures, mixture_measures)
    for component in kf_mix.regime_model.components:
        component.logit.fill_(-30.)
    kf = _make_kf(measures, [])
    kf.load_state_dict({k: v for k, v in kf_mix.state_dict().items() if not k.startswith('regime_model.')})

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
    component = kf.regime_model.components[0]
    prob = .2

    mean = torch.tensor([[1.0]])
    cov = torch.tensor([[[.5]]])
    H = torch.tensor([[[1.0]]])
    R = torch.tensor([[[.3]]])
    for obs in (1.2, -3.5):
        input = torch.tensor([[obs]])
        state = kf._update_step(input=input, mean=mean, cov=cov, measured_mean=mean, measure_mat=H, measure_cov=R)

        # by hand:
        S = cov + R
        K = cov / S
        mean_n = mean + K * (obs - mean)
        cov_n = (1 - K) * cov
        lik_n = Normal(mean, S.sqrt()).log_prob(input).exp()
        lik_w = Normal(component.mean, component.var.sqrt()).log_prob(input).exp()
        w_w = prob * lik_w / (prob * lik_w + (1 - prob) * lik_n)
        w_n = 1 - w_w
        expected_mean = w_n * mean_n + w_w * mean
        expected_cov = w_n * (cov_n + (mean_n - expected_mean) ** 2) + w_w * (cov + (mean - expected_mean) ** 2)

        assert torch.allclose(state.regime_post[:, 1], w_w.view(1), atol=1e-6)
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
    assert torch.allclose(state.regime_post, kf.regime_model.base_probs().expand(4, -1))


@torch.no_grad()
def test_log_prob_brute_force():
    measures = ['y1', 'y2']
    kf = _make_kf(measures, measures)
    y = _make_y(num_measures=2)
    y[0, 5, 1] = float('nan')
    pred = kf(y)
    lp = pred.log_prob(y)

    rm = kf.regime_model
    H = torch.eye(2)
    R = kf.measure_covariance({}, num_groups=1, num_times=1)[0, 0]
    for g, t in [(0, 0), (1, 4), (2, 9), (0, 5)]:
        obs = y[g, t]
        observed = (~obs.isnan()).nonzero().view(-1).tolist()
        m = pred.state_means[g, t] @ H.T
        S = H @ pred.state_covs[g, t] @ H.T + R
        total = 0.
        for combo, prob in zip(rm.combos, pred.regime_priors[g, t]):
            normal = [i for i in observed if combo[i] is None]
            lik = 1.
            if normal:
                lik *= MultivariateNormal(m[normal], S[normal][:, normal]).log_prob(obs[normal]).exp()
            for i in observed:
                if combo[i] is not None:
                    lik *= Normal(combo[i].mean, combo[i].var.sqrt()).log_prob(obs[i]).exp()
            total += prob * lik
        assert math.isclose(lp[g, t].item(), math.log(total), rel_tol=1e-4)


@pytest.mark.parametrize("univariate_mixture_prob", [False, True])
def test_binomial_filter_with_mixture(univariate_mixture_prob: bool):
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
        mixture_components=[MixtureComponent(measure='spend', mean_init=-1., prob_init=.1, id='quick')],
        univariate_mixture_prob=univariate_mixture_prob,
    )
    bf.mc_sampling = 50
    pred = bf(y)
    lp = pred.log_prob(y)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    component = bf.regime_model.components[0]
    for param in (component.mean, component._log_std, component.logit):
        assert param.grad is not None and param.grad.abs() > 0

    # end-to-end training:
    bf.zero_grad()
    bf.fit(y, stopping={'max_iter': 3}, verbose=0)


@torch.no_grad()
def test_no_stickiness_is_static():
    measures = ['y1', 'y2']
    kf_static = _make_kf(measures, measures, regime_transition=StaticTransition(num_combos=4))
    kf_sticky = _make_kf(measures, measures)
    kf_sticky.load_state_dict(
        {k: v for k, v in kf_static.state_dict().items() if 'transition' not in k}, strict=False
    )
    kf_sticky.regime_model.transition._stay_logit.fill_(-30.)
    y = _make_y(num_measures=2)
    y[0, 3:6, 0] = float('nan')
    pred_static, pred_sticky = kf_static(y), kf_sticky(y)
    base_probs = kf_static.regime_model.base_probs()
    assert torch.allclose(pred_static.regime_priors, base_probs.expand_as(pred_static.regime_priors))
    assert torch.allclose(pred_sticky.regime_priors, pred_static.regime_priors, atol=1e-6)
    assert torch.allclose(pred_sticky.state_means, pred_static.state_means, atol=1e-5)
    assert torch.allclose(pred_sticky.log_prob(y), pred_static.log_prob(y), atol=1e-5)


@torch.no_grad()
def test_regime_priors_through_time():
    kf = _make_kf(['y'], ['y'])
    transition = kf.regime_model.transition
    transition._stay_logit.fill_(2.)  # stay ~ .88
    base_probs = kf.regime_model.base_probs()

    y = torch.zeros((2, 10, 1))
    y[0, 4] = -4.  # an outlier (at the component's mean) for group 0 only
    y[:, 7] = float('nan')
    pred = kf(y)
    priors = pred.regime_priors
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
    assert torch.allclose(pred_fcast.regime_priors[:, 11], transition(pred_fcast.regime_priors[:, 10], base_probs))
    # slicing keeps priors aligned:
    assert torch.allclose(pred[:, 4:6].regime_priors, priors[:, 4:6])


@pytest.mark.parametrize("n_step,every_step", [(2, True), (3, True), (3, False)])
@torch.no_grad()
def test_n_step_regime_priors(n_step: int, every_step: bool):
    """
    As in ``test_n_step_matches_nan_forecast``: an h-step prediction for t should match a 1-step prediction with the
    observations between t - h and t missing.
    """
    kf = _make_kf(['y1', 'y2'], ['y1'])
    kf.regime_model.transition._stay_logit.fill_(1.)
    y = _make_y(num_measures=2)
    pred_n = kf(y, n_step=n_step, every_step=every_step)
    for t in range(y.shape[1]):
        h = min(t + 1, n_step) if every_step else (t % n_step) + 1
        y_nan = y.clone()
        y_nan[:, (t - h + 1):t] = float('nan')
        pred_1 = kf(y_nan, n_step=1)
        assert torch.allclose(pred_n.state_means[:, t], pred_1.state_means[:, t], atol=1e-5)
        assert torch.allclose(pred_n.state_covs[:, t], pred_1.state_covs[:, t], atol=1e-5)
        assert torch.allclose(pred_n.regime_priors[:, t], pred_1.regime_priors[:, t], atol=1e-6)
    # so the log-prob matches too:
    assert torch.allclose(pred_n.log_prob(y)[:, -1], kf(y_nan, n_step=1).log_prob(y)[:, -1], atol=1e-5)
