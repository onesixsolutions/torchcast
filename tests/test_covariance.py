import copy
import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import pytest
import torch

from torchcast.covariance import Covariance, corr_cholesky_from_unconstrained, num_off_diag
from torchcast.kalman_filter import BinomialFilter, KalmanFilter
from torchcast.process import LocalLevel
from torchcast.internals.utils import as_timestep_list
from torchcast.process.utils import Identity, ProcessKwarg


@torch.no_grad()
def test_from_log_cholesky():
    module = Covariance(id='test', rank=3, init_diag_multi=0.1)

    module.state_dict()['cholesky_log_diag'][:] = torch.arange(1., 3.1)
    module.state_dict()['cholesky_off_diag'][:] = torch.arange(1., 3.1)

    expected = torch.tensor([[7.3891, 2.7183, 5.4366],
                             [2.7183, 55.5982, 24.1672],
                             [5.4366, 24.1672, 416.4288]])
    diff = (expected - module({}, num_groups=1, num_times=1)).abs()
    assert (diff < .0001).all()


@torch.no_grad()
def test_empty_idx():
    module = Covariance(id='test', rank=3, empty_idx=[0], init_diag_multi=0.1)
    cov = module({}, num_groups=1, num_times=1)
    cov = cov.squeeze()
    assert (cov[0, :] == 0).all()
    assert (cov[:, 0] == 0).all()
    assert (cov == cov.t()).all()


def _random_cov(rank: int, scales: torch.Tensor, near_singular: bool = False, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(rank, rank, generator=gen, dtype=torch.float64)
    if near_singular and rank > 1:
        A[1] = A[0] + .02 * A[1]  # first two elements almost perfectly correlated
    else:
        A = torch.cat([A, torch.eye(rank, dtype=torch.float64)], 1)
    C = A @ A.t()
    std = C.diagonal().sqrt()
    return C / std.outer(std) * scales.outer(scales)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-7):
    actual, expected = actual.double(), expected.double()
    assert actual.shape == expected.shape
    if not expected.numel():
        return
    assert (actual - expected).abs().max() <= rtol * expected.abs().max() + atol


@pytest.mark.parametrize('rank', [1, 2, 3, 5, 12])
@pytest.mark.parametrize('kind', ['standard', 'scales', 'near_singular'])
@torch.no_grad()
def test_sd_corr_round_trip(rank: int, kind: str):
    scales = torch.ones(rank, dtype=torch.float64)
    if kind == 'scales':
        scales = torch.logspace(-2, 1, rank, dtype=torch.float64)
    cov = _random_cov(rank, scales, near_singular=kind == 'near_singular', seed=rank)

    # matrix -> sd_corr -> matrix
    module = Covariance.from_matrix(cov, method='sd_corr')
    _assert_close(module({}, num_groups=1, num_times=1)[0, 0], cov)

    # log_cholesky -> sd_corr -> log_cholesky: the matrix is preserved to float32 precision...
    lc = Covariance.from_matrix(cov, method='log_cholesky')
    lc2 = lc.to_method('sd_corr').to_method('log_cholesky')
    assert lc2.method == 'log_cholesky'
    _assert_close(lc2({}, 1, 1), lc({}, 1, 1))
    # ... and the params exactly (in float64; in float32, the log of a tiny cholesky-diagonal amplifies rounding)
    lc = Covariance.from_matrix(cov, method='log_cholesky').double()
    lc2 = lc.to_method('sd_corr').to_method('log_cholesky')
    assert lc2.cholesky_log_diag.dtype == torch.float64
    _assert_close(lc2.cholesky_log_diag, lc.cholesky_log_diag, rtol=1e-9, atol=1e-9)
    _assert_close(lc2.cholesky_off_diag, lc.cholesky_off_diag, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('predict', [False, True])
@torch.no_grad()
def test_sd_corr_empty_idx(predict: bool):
    rank, empty_idx = 5, [0, 3]
    non_empty = [i for i in range(rank) if i not in empty_idx]
    mini = _random_cov(len(non_empty), torch.tensor([1., .1, 2.], dtype=torch.float64))
    cov = torch.zeros(rank, rank, dtype=torch.float64)
    cov[np.ix_(non_empty, non_empty)] = mini

    kwargs = {}
    if predict:
        kwargs = {'predict_std': Identity(), 'predictor_kwargs': ['multi']}
    lc = Covariance.from_matrix(cov, method='log_cholesky', empty_idx=empty_idx, **kwargs)
    sc = Covariance.from_matrix(cov, method='sd_corr', empty_idx=empty_idx, **kwargs)
    inputs = {'multi': torch.rand(2, 4, len(non_empty)) + .5} if predict else {}
    out_lc = lc(inputs, num_groups=2, num_times=4)
    out_sc = sc(inputs, num_groups=2, num_times=4)
    assert (out_sc[..., empty_idx, :] == 0).all() and (out_sc[..., :, empty_idx] == 0).all()
    _assert_close(out_sc, out_lc)
    _assert_close(lc.to_method('sd_corr')(inputs, 2, 4), out_lc)
    # from the param-rank block gives the same:
    _assert_close(Covariance.from_matrix(mini, method='sd_corr')({}, 1, 1)[0, 0], mini)

    with pytest.raises(ValueError, match='empty_idx'):
        Covariance.from_matrix(torch.eye(rank), method='sd_corr', empty_idx=empty_idx)


@torch.no_grad()
def test_sd_corr_correlations():
    rank = 6
    # random (incl. large) u -> unit diagonal
    u = torch.randn(num_off_diag(rank)) * 10
    L = corr_cholesky_from_unconstrained(u, rank)
    assert ((L @ L.t()).diagonal() - 1).abs().max() < 1e-6

    # zeroing an element's row *and* column -> exactly uncorrelated with all others
    module = Covariance(rank=rank, init_diag_multi=1., method='sd_corr')
    module.corr_unconstrained.normal_(0, 2)
    i = 3
    for j in range(rank):
        if j != i:
            module.corr_unconstrained[module.off_diag_idx(i, j)] = 0.
    cov = module({}, 1, 1)[0, 0]
    assert (cov[i, [j for j in range(rank) if j != i]] == 0).all()
    # (... but only zeroing the row is a zero *partial* correlation, not a zero marginal correlation)


@torch.no_grad()
def test_corr_cholesky():
    cov = _random_cov(4, torch.tensor([1., .1, 3., 1.], dtype=torch.float64))
    std = cov.diagonal().sqrt()
    L_expected = torch.linalg.cholesky(cov / std.outer(std))
    for method in ('log_cholesky', 'sd_corr'):
        _assert_close(Covariance.from_matrix(cov, method=method).corr_cholesky(), L_expected)


@torch.no_grad()
def test_sd_corr_index_helpers():
    module = Covariance(rank=5, empty_idx=[1], init_diag_multi=1., method='sd_corr')
    module.corr_unconstrained.zero_()
    module.log_std_devs.zero_()
    module.corr_unconstrained[module.off_diag_idx(0, 3)] = 1.
    module.log_std_devs[module.param_idx(4)] = math.log(2.)
    cov = module({}, 1, 1)[0, 0]
    expected = torch.eye(5)
    expected[1, 1] = 0
    expected[4, 4] = 4.
    expected[0, 3] = expected[3, 0] = math.tanh(1.)
    _assert_close(cov, expected)
    with pytest.raises(ValueError):
        module.param_idx(1)
    with pytest.raises(ValueError):
        module.off_diag_idx(2, 2)


@pytest.mark.parametrize('scale', [0., 1., 30.])
def test_sd_corr_derivatives(scale: float):
    rank = 4
    torch.manual_seed(0)
    u = (torch.randn(num_off_diag(rank), dtype=torch.float64) * scale).requires_grad_()
    log_std = torch.randn(rank, dtype=torch.float64, requires_grad=True)

    def fun(log_std, u):
        L = log_std.exp().unsqueeze(-1) * corr_cholesky_from_unconstrained(u, rank)
        return L @ L.t()

    if scale < 10:
        # exact second derivatives, incl. at u=0 (where formulas based on abs() get them wrong):
        assert torch.autograd.gradgradcheck(fun, (log_std, u))
    else:
        H = torch.autograd.functional.hessian(lambda ls, uu: fun(ls, uu).sum(), (log_std, u))
        assert all(torch.isfinite(h).all() for row in H for h in row)


def test_sd_corr_freezing_hooks():
    module = Covariance(rank=4, init_diag_multi=1., method='sd_corr')
    keep_std = torch.ones_like(module.log_std_devs)
    keep_std[module.param_idx(2)] = 0
    module.log_std_devs.register_hook(lambda g: g * keep_std)
    keep_corr = torch.ones_like(module.corr_unconstrained)
    keep_corr[module.off_diag_idx(2, 0)] = 0
    module.corr_unconstrained.register_hook(lambda g: g * keep_corr)
    (module({}, 1, 1) ** 2).sum().backward()
    assert module.log_std_devs.grad[module.param_idx(2)] == 0
    assert module.corr_unconstrained.grad[module.off_diag_idx(2, 0)] == 0
    assert (module.log_std_devs.grad != 0).sum() == 3
    assert (module.corr_unconstrained.grad != 0).sum() == num_off_diag(4) - 1


def test_state_dict_across_methods():
    lc = Covariance(rank=3, init_diag_multi=1., id='initial_covariance')
    lc2 = Covariance(rank=3, init_diag_multi=1.)
    lc2.load_state_dict(lc.state_dict())  # same method: unchanged
    assert torch.equal(lc2.cholesky_off_diag, lc.cholesky_off_diag)
    sc = lc.to_method('sd_corr')
    with pytest.raises(RuntimeError, match=r"to_method\('sd_corr'\)"):
        Covariance(rank=3, init_diag_multi=1., method='sd_corr').load_state_dict(lc.state_dict())
    with pytest.raises(RuntimeError, match=r"saved with method='sd_corr'"):
        lc2.load_state_dict(sc.state_dict())
    with pytest.warns(UserWarning, match='experimental'):
        lr = Covariance(rank=3, init_diag_multi=1., method='low_rank')
    with pytest.raises(RuntimeError, match=r"saved with method='low_rank'"):
        Covariance(rank=3, init_diag_multi=1., method='sd_corr').load_state_dict(lr.state_dict())


def test_sd_corr_conditioning():
    """
    The motivating case: elements with much smaller variance than the others, whose correlations are free. In
    log_cholesky params, the off-diagonal entries are in absolute units, so the NLL's hessian gets badly conditioned
    as the scales diverge; in sd_corr params, it's invariant to the scales.
    """
    rank = 6
    small_scales = torch.tensor([1., 1., 1., 1., .05, .05], dtype=torch.float64)
    corr = _random_cov(rank, torch.ones(rank, dtype=torch.float64), seed=0)
    y_std = torch.distributions.MultivariateNormal(torch.zeros(rank, dtype=torch.float64), corr).sample((2000,))

    hessians = {}
    for method in ('log_cholesky', 'sd_corr'):
        for scale_name, scales in [('equal', torch.ones(rank, dtype=torch.float64)), ('small', small_scales)]:
            cov = corr * scales.outer(scales)
            y = y_std * scales
            module = Covariance.from_matrix(cov, method=method).double()
            names, values = zip(*module.named_parameters())
            sizes = [v.numel() for v in values]

            def nll(flat: torch.Tensor) -> torch.Tensor:
                params = dict(zip(names, flat.split(sizes)))
                mcov = torch.func.functional_call(module, params, ({}, 1, 1))[0, 0]
                return -torch.distributions.MultivariateNormal(torch.zeros_like(mcov[0]), mcov).log_prob(y).mean()

            hessians[method, scale_name] = torch.autograd.functional.hessian(
                nll, torch.cat([v.detach() for v in values])
            )

    def cond(H: torch.Tensor) -> float:
        eig = torch.linalg.eigvalsh(H)
        assert eig.min() > 0
        return (eig.max() / eig.min()).item()

    assert torch.allclose(hessians['sd_corr', 'small'], hessians['sd_corr', 'equal'])
    assert cond(hessians['log_cholesky', 'small']) > 100 * cond(hessians['log_cholesky', 'equal'])
    assert cond(hessians['log_cholesky', 'small']) > 100 * cond(hessians['sd_corr', 'small'])


def test_sd_corr_fit():
    torch.manual_seed(0)
    measures = ['a', 'b']
    num_groups, num_times = 5, 60
    noise_cov = torch.tensor([[1., .7 * .2], [.7 * .2, .2 ** 2]])
    level = torch.randn(num_groups, 1, 2).cumsum(1) * .1 + torch.randn(num_groups, num_times, 2).cumsum(1) * .1
    y = level + torch.distributions.MultivariateNormal(torch.zeros(2), noise_cov).sample((num_groups, num_times))

    def make_kf(method: str) -> KalmanFilter:
        processes = [LocalLevel(id=f'level_{m}', measure=m) for m in measures]
        return KalmanFilter(
            processes=processes,
            measures=measures,
            measure_covariance=Covariance.from_measures(measures, method=method),
            process_covariance=Covariance.from_processes(processes, method=method),
            initial_covariance=Covariance.from_processes(processes, cov_type='initial', method=method),
        )

    losses = {}
    for method in ('log_cholesky', 'sd_corr'):
        kf = make_kf(method)
        kf.fit(y, verbose=0, stopping={'abstol': 1e-6})
        with torch.no_grad():
            losses[method] = -kf(y).log_prob(y).mean().item()
    assert abs(losses['sd_corr'] - losses['log_cholesky']) < 1e-3 * abs(losses['log_cholesky'])


def test_binomial_filter_accepts_covariance():
    processes = [LocalLevel(id='level_a', measure='a'), LocalLevel(id='level_b', measure='b')]
    bf = BinomialFilter(
        processes=processes,
        measures=['a', 'b'],
        binary_measures=['a'],
        measure_covariance=Covariance(rank=2, empty_idx=[0], init_diag_multi=1., method='sd_corr')
    )
    assert bf.measure_covariance.method == 'sd_corr'


class _Linear(torch.nn.Module):
    """A linear model of the predictors, zero-initialized (so it starts at the base covariance)."""

    def __init__(self, num_in: int, num_out: int, positive: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(num_in, num_out, bias=False)
        torch.nn.init.zeros_(self.linear.weight)
        self.positive = positive

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.linear(X)
        return out.exp() if self.positive else out


def _cov_with_u(module: Covariance, u: torch.Tensor, std_multi: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The covariance ``module`` would have with ``corr_unconstrained=u`` (and std-devs multiplied by ``std_multi``)."""
    out = module.to_method('sd_corr')
    out.predict_corr = out.predict_std = None
    out.predictor_kwargs = []
    with torch.no_grad():
        out.corr_unconstrained.copy_(u)
        if std_multi is not None:
            out.log_std_devs.add_(std_multi.log())
    return out.base_cov()


@torch.no_grad()
def test_predict_std():
    module = Covariance(rank=3, init_diag_multi=1., method='sd_corr', predict_std=Identity(), predictor_kwargs=['m'])
    m = torch.rand(2, 4, 3) + .5
    _assert_close(module({'m': m}, 2, 4), module.base_cov() * m.unsqueeze(-1) * m.unsqueeze(-2))
    with pytest.raises(RuntimeError, match='<0'):
        module({'m': -m}, 2, 4)
    with pytest.raises(ValueError, match='3 columns'):
        module({'m': m[..., :2]}, 2, 4)


@torch.no_grad()
def test_predict_corr():
    rank, empty_idx = 4, [1]
    module = Covariance(
        rank=rank, empty_idx=empty_idx, init_diag_multi=1., predict_corr=Identity(), predictor_kwargs=['d']
    )
    assert module.method == 'sd_corr'  # the default with predict_corr
    module.corr_unconstrained.normal_()
    delta = torch.randn(2, 4, num_off_diag(3))

    out = module({'d': delta}, 2, 4)
    assert out.shape == (2, 4, rank, rank)
    assert (out[..., 1, :] == 0).all() and (out[..., :, 1] == 0).all()
    for g, t in [(0, 0), (1, 3)]:
        _assert_close(out[g, t], _cov_with_u(module, module.corr_unconstrained + delta[g, t]))
    # variances are unchanged; zero adjustment is the base covariance:
    _assert_close(out.diagonal(dim1=-2, dim2=-1), module.base_cov().diagonal().expand(2, 4, -1))
    _assert_close(module({'d': torch.zeros_like(delta)}, 2, 4), module.base_cov().expand(2, 4, -1, -1))

    # group-only predictions don't vary over time, and ``batch_cov`` doesn't expand them:
    cov, used = module.batch_cov({'d': delta[:, 0]}, num_groups=2, num_times=4)
    assert cov.shape == (2, 1, rank, rank) and used == {'d'}
    _assert_close(cov[:, 0], out[:, 0])
    # without predictions:
    static, used = Covariance(rank=rank, init_diag_multi=1.).batch_cov({}, num_groups=2, num_times=4)
    assert static.shape == (1, 1, rank, rank) and used == set()


@torch.no_grad()
def test_predict_std_and_corr():
    class _Pick(torch.nn.Module):
        def __init__(self, i: int):
            super().__init__()
            self.i = i

        def forward(self, *args):
            return args[self.i]

    module = Covariance(
        rank=3, init_diag_multi=1., predict_std=_Pick(0), predict_corr=_Pick(1), predictor_kwargs=['m', 'd']
    )
    module.corr_unconstrained.normal_()
    m = torch.rand(2, 3) + .5  # by group
    d = torch.randn(2, 5, 3)  # by group and time
    out, _ = module.batch_cov({'m': m, 'd': d}, num_groups=2, num_times=5)
    assert out.shape == (2, 5, 3, 3)
    _assert_close(out[1, 2], _cov_with_u(module, module.corr_unconstrained + d[1, 2], std_multi=m[1]))


def test_predict_corr_requires_sd_corr():
    with pytest.raises(ValueError, match="requires ``method='sd_corr'``"):
        Covariance(rank=3, init_diag_multi=1., method='log_cholesky', predict_corr=Identity(), predictor_kwargs=['d'])
    # from_measures picks sd_corr, even with >5 measures (which would otherwise default to low_rank):
    measures = list('abcdefg')
    assert Covariance.from_measures(measures, predict_corr=Identity(), predictor_kwargs=['d']).method == 'sd_corr'
    module = Covariance.from_measures(['a', 'b'], predict_corr=Identity(), predictor_kwargs=['d'])
    with pytest.raises(ValueError, match="requires ``method='sd_corr'``"):
        module.to_method('log_cholesky')
    with pytest.raises(ValueError, match='predictor_kwargs'):
        Covariance(rank=3, init_diag_multi=1., predict_corr=Identity())
    with pytest.raises(ValueError, match='no ``predict_std`` or ``predict_corr``'):
        Covariance(rank=3, init_diag_multi=1., predictor_kwargs=['d'])


def test_as_timestep_list():
    x = torch.randn(1, 1, 3, 3)
    out = as_timestep_list(x, num_groups=4, num_times=5)
    assert len(out) == 5 and all(o is out[0] for o in out) and out[0].shape == (4, 3, 3)
    x = torch.randn(4, 5, 3, 3)
    out = as_timestep_list(x, num_groups=4, num_times=5)
    assert torch.equal(torch.stack(out, 1), x)
    with pytest.raises(ValueError):
        as_timestep_list(x, num_groups=4, num_times=6)


def test_model_predictor_kwargs():
    torch.manual_seed(0)
    measures = ['a', 'b']
    processes = [LocalLevel(id=f'level_{m}', measure=m) for m in measures]
    kf = KalmanFilter(
        processes=processes,
        measures=measures,
        measure_covariance=Covariance.from_measures(measures, predict_corr=_Linear(1, 1), predictor_kwargs=['X']),
    )
    y = torch.randn(3, 10, 2)
    X = torch.randn(3, 10, 1)
    with torch.no_grad():
        kf.measure_covariance.predict_corr.linear.weight.fill_(1.)
        pred = kf(y, X=X)
        # a kwarg named ``{id}__{name}`` overrides ``name``:
        pred2 = kf(y, measure_covariance__X=X)
        assert torch.equal(pred.means, pred2.means)
        with pytest.raises(TypeError, match='`X`'):
            kf(y)
        with pytest.raises(RuntimeError, match='Unexpected kwargs'):
            kf(y, X=X, Z=X)
        # time-varying measure-cov:
        R = kf.measure_covariance({'X': X}, 3, 10)
        assert not torch.allclose(R[0, 0], R[0, 1])
    # covariance kwargs are declared as group-indexed (for ``fit(chunk_size=...)``), unless marked otherwise:
    assert {'X', 'measure_covariance__X'} <= kf._group_kwarg_names()
    kf.measure_covariance.predictor_kwargs = [ProcessKwarg('X', is_group_time_tensor=False)]
    assert 'X' not in kf._group_kwarg_names()


def test_fit_time_varying_correlation():
    """
    The noise-correlation between two measures flips between two regimes; a ``predict_corr`` linear in the regime
    indicator recovers both.
    """
    torch.manual_seed(1)
    measures = ['a', 'b']
    num_groups, num_times = 10, 100
    regime = (torch.arange(num_times) // 25 % 2).float()  # alternating blocks of 25 timesteps
    corr = torch.where(regime.bool(), -.3, .8)
    std = torch.tensor([1., .5])
    z = torch.randn(num_groups, num_times, 2)
    noise = torch.stack([z[..., 0], corr * z[..., 0] + (1 - corr ** 2).sqrt() * z[..., 1]], -1) * std
    y = torch.randn(num_groups, 1, 2) + torch.randn(num_groups, num_times, 2).cumsum(1) * .05 + noise
    X = torch.stack([1 - regime, regime], -1).expand(num_groups, -1, -1)

    processes = [LocalLevel(id=f'level_{m}', measure=m) for m in measures]
    kf = KalmanFilter(
        processes=processes,
        measures=measures,
        measure_covariance=Covariance.from_measures(measures, predict_corr=_Linear(2, 1), predictor_kwargs=['X']),
    )
    kf.fit(y, X=X, verbose=0)
    with torch.no_grad():
        R = kf.measure_covariance({'X': X}, num_groups, num_times)[0]
    learned_corr = R[:, 0, 1] / (R[:, 0, 0] * R[:, 1, 1]).sqrt()
    assert abs(learned_corr[regime == 0].mean() - .8) < .1
    assert abs(learned_corr[regime == 1].mean() + .3) < .1


def test_deprecated_predict_variance():
    with pytest.warns(DeprecationWarning, match='predict_variance'):
        old = Covariance.from_measures(['a', 'b'], predict_variance=True)
    assert [k.name for k in old.predictor_kwargs] == ['measure_var_multi']
    with pytest.warns(DeprecationWarning, match='predict_variance'):
        old = Covariance.from_processes([LocalLevel(id='level', measure='a')], predict_variance=True)
    assert [k.name for k in old.predictor_kwargs] == ['process_var_multi']

    with pytest.warns(DeprecationWarning):
        old = Covariance.from_measures(['a', 'b'], predict_variance=Identity(), expected_kwargs=['m'])
    new = Covariance.from_measures(['a', 'b'], predict_std=Identity(), predictor_kwargs=['m'])
    new.load_state_dict(old.state_dict())
    m = torch.rand(2, 3, 2) + .5
    assert torch.equal(old({'m': m}, 2, 3), new({'m': m}, 2, 3))
    with pytest.warns(DeprecationWarning):
        assert old.expected_kwargs == ['m']


def test_load_older_version():
    new = Covariance.from_measures(
        ['a', 'b'], predict_std=_Linear(1, 2, positive=True), predictor_kwargs=['X']
    )
    with torch.no_grad():
        new.predict_std.linear.weight.normal_()

    # a state-dict saved by an older version (``var_predict_module``):
    old_state = {k.replace('predict_std.', 'var_predict_module.'): v for k, v in new.state_dict().items()}
    loaded = Covariance.from_measures(['a', 'b'], predict_std=_Linear(1, 2, positive=True), predictor_kwargs=['X'])
    loaded.load_state_dict(old_state)
    assert torch.equal(loaded.predict_std.linear.weight, new.predict_std.linear.weight)

    # an unpickled module from an older version:
    state = copy.copy(new.__dict__)
    state['_modules'] = OrderedDict(var_predict_module=new.predict_std)
    state['expected_kwargs'] = ['X']
    del state['predictor_kwargs']
    unpickled = Covariance.__new__(Covariance)
    unpickled.__setstate__(state)
    assert unpickled.predict_std is new.predict_std and unpickled.predict_corr is None
    assert [k.name for k in unpickled.predictor_kwargs] == ['X']
    X = torch.randn(2, 3, 1)
    assert torch.equal(unpickled({'X': X}, 2, 3), new({'X': X}, 2, 3))
