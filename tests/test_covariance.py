import math
import pickle
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pytest
import torch

from torchcast.covariance import Covariance, corr_cholesky_from_unconstrained, num_off_diag, cov2corr
from torchcast.kalman_filter import BinomialFilter, KalmanFilter
from torchcast.process import LocalLevel, LocalTrend, Season
from torchcast.process.utils import Identity


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
        kwargs = {'predict_variance': Identity(), 'expected_kwargs': ['multi']}
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
            module.corr_unconstrained[module.get_off_diag_idx(i, j)] = 0.
    cov = module({}, 1, 1)[0, 0]
    assert (cov[i, [j for j in range(rank) if j != i]] == 0).all()
    # (... but only zeroing the row is a zero *partial* correlation, not a zero marginal correlation)


@torch.no_grad()
def test_corr_cholesky():
    cov = _random_cov(4, torch.tensor([1., .1, 3., 1.], dtype=torch.float64))
    std = cov.diagonal().sqrt()
    L_expected = torch.linalg.cholesky(cov / std.outer(std))
    for method in ('log_cholesky', 'sd_corr'):
        _assert_close(Covariance.from_matrix(cov, method=method).get_corr_cholesky(), L_expected)


@torch.no_grad()
def test_sd_corr_index_helpers():
    module = Covariance(rank=5, empty_idx=[1], init_diag_multi=1., method='sd_corr')
    module.corr_unconstrained.zero_()
    module.log_std_devs.zero_()
    module.corr_unconstrained[module.get_off_diag_idx(0, 3)] = 1.
    module.log_std_devs[module.get_param_idx(4)] = math.log(2.)
    cov = module({}, 1, 1)[0, 0]
    expected = torch.eye(5)
    expected[1, 1] = 0
    expected[4, 4] = 4.
    expected[0, 3] = expected[3, 0] = math.tanh(1.)
    _assert_close(cov, expected)
    with pytest.raises(ValueError):
        module.get_param_idx(1)
    with pytest.raises(ValueError):
        module.get_off_diag_idx(2, 2)


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
    keep_std[module.get_param_idx(2)] = 0
    module.log_std_devs.register_hook(lambda g: g * keep_std)
    keep_corr = torch.ones_like(module.corr_unconstrained)
    keep_corr[module.get_off_diag_idx(2, 0)] = 0
    module.corr_unconstrained.register_hook(lambda g: g * keep_corr)
    (module({}, 1, 1) ** 2).sum().backward()
    assert module.log_std_devs.grad[module.get_param_idx(2)] == 0
    assert module.corr_unconstrained.grad[module.get_off_diag_idx(2, 0)] == 0
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
    lr = Covariance(rank=3, init_diag_multi=1., method='low_rank')
    with pytest.raises(RuntimeError, match=r"saved with method='low_rank1'"):  # (with the number of factors)
        Covariance(rank=3, init_diag_multi=1., method='sd_corr').load_state_dict(lr.state_dict())


def test_sd_corr_conditioning():
    """
    The motivating case: elements with much smaller variance than the others, whose correlations are free. In
    log_cholesky params, the off-diagonal entries are in absolute units, so the NLL's hessian gets badly conditioned
    as the scales diverge; in sd_corr params, it's invariant to the scales.
    """
    torch.manual_seed(0)
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
        do_post_hoc_correction='num_obs',
        processes=processes,
        measures=['a', 'b'],
        binary_measures=['a'],
        measure_covariance=Covariance(rank=2, empty_idx=[0], init_diag_multi=1., method='sd_corr')
    )
    assert bf.measure_covariance.method == 'sd_corr'


def test_corr_groups_from_processes():
    processes = [LocalLevel(id='a'), LocalTrend(id='b'), LocalLevel(id='c'), LocalLevel(id='d')]
    # elements: a.position, b.position, b.velocity, c.position, d.position
    cov = Covariance.from_processes(processes, corr_groups=[['b', 'c']])
    expected = torch.tensor([[1., 0., 0., 0., 1.],
                             [0., 1., 1., 1., 0.],
                             [0., 1., 1., 1., 0.],
                             [0., 1., 1., 1., 0.],
                             [1., 0., 0., 0., 1.]])  # 'a' and 'd' are unlisted, so in one group together
    assert torch.equal(cov.corr_mask, expected)


def test_corr_groups_from_measures():
    cov = Covariance.from_measures(['x', 'y', 'z'], corr_groups=[['x'], ['y']])
    assert torch.equal(cov.corr_mask, torch.eye(3))


@pytest.mark.parametrize('corr_groups,match', [
    ([['a', 'nope']], "unknown process id 'nope'"),
    ([['a'], ['a', 'b']], "'a' in more than one group"),
    (['a', 'b'], "list of lists"),
])
def test_corr_groups_errors(corr_groups: list, match: str):
    processes = [LocalLevel(id='a'), LocalLevel(id='b')]
    with pytest.raises((ValueError, TypeError), match=match):
        Covariance.from_processes(processes, corr_groups=corr_groups)
    with pytest.raises(TypeError, match="not both"):
        Covariance.from_processes(processes, corr_groups=[['a']], corr_mask=torch.eye(2))


@pytest.mark.parametrize('method', ['log_cholesky', 'sd_corr'])
@pytest.mark.parametrize('empty_idx', [(), (1,)])
def test_corr_mask(method: str, empty_idx: tuple):
    torch.manual_seed(0)
    rank = 5
    corr_mask = torch.tensor([[1., 0., 1., 0., 1.],
                              [0., 1., 0., 1., 0.],
                              [1., 0., 1., 0., 1.],
                              [0., 1., 0., 1., 0.],
                              [1., 0., 1., 0., 1.]])
    module = Covariance(rank=rank, init_diag_multi=1., method=method, empty_idx=list(empty_idx), corr_mask=corr_mask)
    unmasked = Covariance(rank=rank, init_diag_multi=1., method=method, empty_idx=list(empty_idx))
    # (the masked module has parameters only for pairs that aren't masked out; unpacked, they're the same factor:)
    diag_name, off_diag_name = ('cholesky_log_diag', 'cholesky_off_diag') if method == 'log_cholesky' \
        else ('log_std_devs', 'corr_unconstrained')
    assert getattr(module, off_diag_name).numel() < getattr(unmasked, off_diag_name).numel()
    with torch.no_grad():
        getattr(unmasked, diag_name).copy_(getattr(module, diag_name))
        getattr(unmasked, off_diag_name).copy_(module._get_off_diag())
    with torch.no_grad():
        cov = module({}, num_groups=1, num_times=1).squeeze()
        cov_unmasked = unmasked({}, num_groups=1, num_times=1).squeeze()
    # masked entries are zero, others (and empty rows/cols) as without the mask:
    assert torch.allclose(cov, cov_unmasked * corr_mask)
    assert torch.linalg.eigvalsh(cov).min() > -1e-6
    # no gradient flows through masked-out entries:
    module({}, num_groups=1, num_times=1)[0, 0, 0, 3].backward()
    assert all(p.grad is None or (p.grad == 0).all() for p in module.parameters())
    # the mask is carried through conversions and the state-dict:
    assert torch.allclose(module.to_method('sd_corr')({}, num_groups=1, num_times=1).squeeze(), cov, atol=1e-5)
    assert 'corr_mask' in module.state_dict()


def test_corr_mask_validation():
    with pytest.raises(ValueError, match="symmetric"):
        Covariance(rank=2, init_diag_multi=1., corr_mask=torch.tensor([[1., 1.], [0., 1.]]))
    with pytest.raises(ValueError, match="diagonal"):
        Covariance(rank=2, init_diag_multi=1., corr_mask=torch.tensor([[1., 0.], [0., 0.]]))
    with pytest.raises(ValueError, match="positive semi-definite"):
        # 0 and 2 uncorrelated, but both correlated with 1: not PSD
        Covariance(rank=3, init_diag_multi=1., corr_mask=torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]]))


def test_corr_mask_absent_buffer():
    # modules pickled before ``corr_mask`` existed don't have the buffer:
    module = Covariance(rank=3, init_diag_multi=1.)
    assert 'corr_mask' not in module._buffers
    module({}, num_groups=1, num_times=1)


def test_corr_groups_in_kalman_filter():
    processes = [LocalLevel(id='level', measure='y'), LocalLevel(id='transient', measure='y', decay=(.5, .95))]
    kf = KalmanFilter(
        processes=processes,
        measures=['y'],
        process_covariance=Covariance.from_processes(processes, corr_groups=[['transient']])
    )
    y = torch.randn(3, 20, 1).cumsum(1)
    kf(y).log_prob(y).mean().backward()
    pcov = kf.process_covariance({}, num_groups=1, num_times=1).squeeze()
    assert pcov[0, 1] == 0 and pcov[1, 0] == 0


def test_warn_correlated_levels():
    def make(**kwargs):
        processes = [
            LocalLevel(id='level', measure='y'),
            LocalLevel(id='transient', measure='y', decay=(.5, .95))
        ]
        # (``stationary``: so that the free-initial-variance warning doesn't fire here)
        kwargs['initial_covariance'] = Covariance.from_processes(
            processes, cov_type='initial', stationary=['transient']
        )
        if kwargs.pop('uncorrelated', False):
            kwargs['process_covariance'] = Covariance.from_processes(processes, corr_groups=[['transient']])
        return KalmanFilter(processes=processes, measures=['y'], **kwargs)

    with pytest.warns(UserWarning, match=r"corr_groups=\[\['transient'\]\]"):
        make()
    with pytest.warns(UserWarning, match="both decaying or both not"):
        KalmanFilter(processes=[LocalLevel(id='a'), LocalLevel(id='b')], measures=['y'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        make(uncorrelated=True)
        KalmanFilter(processes=[LocalLevel(id='a', measure='x'), LocalLevel(id='b', measure='y')], measures=['x', 'y'])



def _make_decaying_kf(stationary: Union[Sequence[str], dict] = (), decay=(.5, .95), **kwargs) -> KalmanFilter:
    # a random walk on each of two measures, and a decaying level on each (correlated with each other, not the walks)
    processes = [
        LocalLevel(id='level_a', measure='a'),
        LocalLevel(id='level_b', measure='b'),
        LocalLevel(id='transient_a', measure='a', decay=decay),
        LocalLevel(id='transient_b', measure='b', decay=decay)
    ]
    return KalmanFilter(
        processes=processes,
        measures=['a', 'b'],
        process_covariance=Covariance.from_processes(processes, corr_groups=[['transient_a', 'transient_b']]),
        initial_covariance=Covariance.from_processes(processes, cov_type='initial', stationary=stationary),
        **kwargs
    )


def _get_decay(kf: KalmanFilter, pid: str) -> torch.Tensor:
    return kf.processes[pid].state_elements['level'].transitions_to['level']().squeeze()


def test_stationary_initial_covariance():
    torch.manual_seed(0)
    kf = _make_decaying_kf(stationary=['transient_a', 'transient_b'])
    # no parameters for the stationary elements (their variances or any correlations):
    assert kf.initial_covariance.stationary_idx == [2, 3]
    assert kf.initial_covariance.empty_idx == {2, 3}
    assert kf.initial_covariance.param_rank == 2
    assert kf.initial_covariance.cholesky_off_diag.numel() == 1  # (just the two random walks)

    with torch.no_grad():
        kf.processes['transient_a'].state_elements['level'].transitions_to['level'].raw.fill_(.3)
        kf.processes['transient_b'].state_elements['level'].transitions_to['level'].raw.fill_(-.5)
        kf.process_covariance.cholesky_off_diag.fill_(.3)  # (correlated transients)
    decays = torch.stack([_get_decay(kf, 'transient_a'), _get_decay(kf, 'transient_b')])
    pcov = kf.process_covariance({}, num_groups=1, num_times=1).squeeze()
    scale = kf._get_measure_scaling().repeat(2)  # (measures a, b, a, b)
    init_cov = kf._default_initial_cov().squeeze(0) / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    # the stationary covariance, including the cross-covariance between the transients:
    expected = pcov[2:, 2:] / (1 - decays.unsqueeze(-1) * decays.unsqueeze(-2))
    assert expected[0, 1] != 0
    torch.testing.assert_close(init_cov[2:, 2:], expected)
    # zero with the random walks; the random walks' block is the free parameters:
    assert (init_cov[:2, 2:] == 0).all() and (init_cov[2:, :2] == 0).all()
    icov = kf.initial_covariance({}, num_groups=1, num_times=1).squeeze()
    torch.testing.assert_close(init_cov[:2, :2], icov[:2, :2])
    # it's what the filter would converge to, starting from it: ``F P F' + Q = P``
    torch.testing.assert_close(decays.unsqueeze(-1) * expected * decays.unsqueeze(-2) + pcov[2:, 2:], expected)

    # the decays and process-covariance get gradients through the initial covariance:
    init_cov[2:, 2:].sum().backward()
    assert kf.processes['transient_a'].state_elements['level'].transitions_to['level'].raw.grad.abs() > 0
    assert kf.process_covariance.cholesky_log_diag.grad.abs().sum() > 0

    # and in a forward pass:
    y = torch.randn(2, 10, 2)
    kf(y).log_prob(y).sum().backward()


def test_stationary_initial_covariance_errors():
    processes = [
        LocalLevel(id='level', measure='y'),
        LocalLevel(id='transient', measure='y', decay=(.5, .95)),
        LocalLevel(id='near_walk', measure='y', decay=(.5, 1.0)),
        LocalTrend(id='trend', measure='y', decay_position=(.5, .95))
    ]
    with pytest.raises(ValueError, match="unknown process ids"):
        Covariance.from_processes(processes, cov_type='initial', stationary=['nope'])
    with pytest.raises(ValueError, match="only for ``cov_type='initial'``"):
        Covariance.from_processes(processes, stationary=['transient'])
    with pytest.raises(TypeError, match="list of process ids"):
        Covariance.from_processes(processes, cov_type='initial', stationary='transient')
    with pytest.raises(TypeError, match="True.*False"):
        Covariance.from_processes(processes, cov_type='initial', stationary={'transient': 1})
    with pytest.raises(ValueError, match="unknown process ids"):
        Covariance.from_processes(processes, cov_type='initial', stationary={'nope': False})

    def make(stationary: Sequence[str]) -> KalmanFilter:
        icov = Covariance.from_processes(processes, cov_type='initial', stationary=stationary)
        return KalmanFilter(processes=processes, measures=['y'], initial_covariance=icov)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # (correlated levels)
        make(['transient'])
        with pytest.raises(ValueError, match="fixed at 1.0"):
            make(['level'])  # (no decay)
        with pytest.raises(ValueError, match="also transition into it"):
            make(['trend'])  # (velocity transitions into position)
        # a learned decay bounded at 1 is allowed (the likelihood keeps it below 1), but checked at forward:
        kf = make(['near_walk'])
        y = torch.randn(2, 5, 1)
        kf(y)
        with torch.no_grad():
            kf.processes['near_walk'].state_elements['level'].transitions_to['level'].raw.fill_(50.)
        with pytest.raises(RuntimeError, match="no stationary covariance unless it's less than 1"):
            kf(y)


def test_warn_free_decay_initial_variance():
    match = r"learned decay and a free initial variance: \['transient_a', 'transient_b'\]"
    with pytest.warns(UserWarning, match=match):
        _make_decaying_kf()
    with pytest.warns(UserWarning, match=r"stationary=\['transient_b'\]"):
        _make_decaying_kf(stationary=['transient_a'])
    with pytest.warns(UserWarning, match=match):
        _make_decaying_kf(decay=(.9, 1.0))  # a learned decay bounded at 1 counts too
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _make_decaying_kf(stationary=['transient_a', 'transient_b'])
        _make_decaying_kf(decay=.9)  # fixed decay: nothing to trade off
        _make_decaying_kf(decay=(.9, 1.0), stationary=['transient_a', 'transient_b'])
        KalmanFilter(processes=[LocalLevel(id='level', measure='y')], measures=['y'])  # no decay
        KalmanFilter(processes=[LocalTrend(id='trend', measure='y')], measures=['y'])  # velocity isn't level-like
    # keeping a free initial variance on purpose, as the warning suggests:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        kf = _make_decaying_kf(stationary={'transient_a': True, 'transient_b': False})
    assert kf.initial_covariance.stationary_idx == [2]
    assert kf.initial_covariance.nonstationary_idx == [3]
    assert kf.initial_covariance.param_rank == 3  # (transient_b keeps its parameters)
    with pytest.warns(UserWarning, match=r"free initial variance: \['transient_a'\]"):
        _make_decaying_kf(stationary={'transient_b': False})


def test_stationary_initial_covariance_default_unchanged():
    # by default the initial covariance keeps free parameters for the decaying levels:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kf = _make_decaying_kf()
    assert kf.initial_covariance.stationary_idx == []
    assert kf.initial_covariance.empty_idx == set()
    assert kf.initial_covariance.param_rank == 4
    icov = kf.initial_covariance({}, num_groups=1, num_times=1).squeeze()
    scale = kf._get_measure_scaling().repeat(2)
    torch.testing.assert_close(
        kf._default_initial_cov().squeeze(0), icov * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    )


def _make_masked_cov(method: str, corr_mask: bool = True) -> Covariance:
    # elements 0, 3 in one group, 1, 4 in another; 2 has no variance:
    groups = torch.tensor([0, 1, 0, 0, 1])
    mask = (groups.unsqueeze(-1) == groups.unsqueeze(-2)).to(torch.get_default_dtype())
    return Covariance(
        rank=5, init_diag_multi=.5, method=method, empty_idx=[2], corr_mask=mask if corr_mask else None
    )


def _get_param_names(method: str) -> tuple[str, str]:
    if method == 'log_cholesky':
        return 'cholesky_log_diag', 'cholesky_off_diag'
    return 'log_std_devs', 'corr_unconstrained'


@pytest.mark.parametrize('method', ['log_cholesky', 'sd_corr'])
def test_corr_mask_packed_params(method: str):
    torch.manual_seed(0)
    cov = _make_masked_cov(method)
    off_diag = getattr(cov, _get_param_names(method)[1])
    # 4 non-empty elements -> 6 pairs, of which 4 are masked out ((0 or 3) x (1 or 4)), leaving (0, 3) and (1, 4):
    assert off_diag.numel() == 2
    unpacked = cov._get_off_diag()
    assert unpacked.numel() == 6 and (unpacked != 0).sum() == 2

    # every parameter gets a gradient (none are redundant):
    mini_cov = cov._get_mini_cov()
    (mini_cov * torch.randn_like(mini_cov)).sum().backward()
    assert (off_diag.grad != 0).all()

    # indexing: masked-out pairs have no parameter; others index into the packed parameter
    with pytest.raises(ValueError, match="masked out"):
        cov.get_off_diag_idx(0, 1)
    assert sorted([cov.get_off_diag_idx(0, 3), cov.get_off_diag_idx(4, 1)]) == [0, 1]
    with torch.no_grad():
        before = cov._get_mini_cov()
        off_diag[cov.get_off_diag_idx(0, 3)] += 1.
        changed = (cov._get_mini_cov() - before).abs() > 0
    # (in the mini-cov, elements 0, 3 are rows 0, 2)
    assert changed[0, 2] and not changed[0, 1] and not changed[1, 3]

    # e.g. with ``corr_groups`` in a model: no off-diagonal parameters at all for two uncorrelated processes
    processes = [LocalLevel(id='level', measure='y'), LocalLevel(id='transient', measure='y', decay=(.5, .95))]
    pcov = Covariance.from_processes(processes, corr_groups=[['transient']], method=method)
    assert getattr(pcov, _get_param_names(method)[1]).numel() == 0


@pytest.mark.parametrize('method', ['log_cholesky', 'sd_corr'])
def test_corr_mask_set_matrix_rejects_masked_entries(method: str):
    # a non-zero entry for a masked-out pair can't be represented (and dropping its cholesky entry would also change
    # the entries within groups), so it raises
    cov = _make_masked_cov(method)
    idx = cov.non_empty_idx
    target = torch.eye(4, dtype=torch.float64) + .3 * (1 - cov.corr_mask[idx][:, idx].to(torch.float64))
    with pytest.raises(ValueError, match="masked out by ``corr_mask``"):
        cov.set_matrix_(target)


@pytest.mark.parametrize('method', ['log_cholesky', 'sd_corr'])
def test_corr_mask_representable_covs_unchanged(method: str):
    torch.manual_seed(1)
    cov = _make_masked_cov(method)
    idx = cov.non_empty_idx
    mini_mask = cov.corr_mask[idx][:, idx]
    A = torch.randn(4, 6, dtype=torch.float64)
    target = (A @ A.t() + torch.eye(4, dtype=torch.float64)) * mini_mask.to(torch.float64)
    cov.set_matrix_(target)
    torch.testing.assert_close(cov._get_mini_cov().detach().to(torch.float64), target, rtol=1e-5, atol=1e-6)
    # and conversions keep the mask and the packing:
    other = 'sd_corr' if method == 'log_cholesky' else 'log_cholesky'
    converted = cov.to_method(other)
    assert getattr(converted, _get_param_names(other)[1]).numel() == 2
    torch.testing.assert_close(converted._get_mini_cov(), cov._get_mini_cov(), rtol=1e-5, atol=1e-6)


def _make_low_rank(**kwargs) -> Covariance:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # (experimental)
        return Covariance(rank=8, init_diag_multi=.5, method='low_rank3', empty_idx=[3], **kwargs)


def test_low_rank_init_and_rotation():
    # init: about half the variance (``init_diag_multi ** 2``) from the factors, half from the diagonal
    torch.manual_seed(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        big = Covariance(rank=200, init_diag_multi=.5, method='low_rank4')
    lr_var = (big.lr_mat ** 2).sum(-1).mean()
    diag_var = (big.log_std_devs.exp() ** 2).mean()
    assert .4 < lr_var / .25 < .6 and .4 < diag_var / .25 < .6

    # the rotation leads with the rows whose loadings are the most independent, not ones that are nearly zero (like
    # seasonal elements with tiny variance) or redundant with one already chosen:
    cov = _make_low_rank()
    with torch.no_grad():
        cov.lr_mat.mul_(.001)
        cov.lr_mat[5] = torch.tensor([2., 0., 0.])
        cov.lr_mat[1] = torch.tensor([1.9, .1, 0.])  # (large, but nearly parallel to row 5)
        cov.lr_mat[3] = torch.tensor([0., 1., 0.])
        cov.lr_mat[6] = torch.tensor([.1, .1, .8])
        cov.lr_mat.copy_(cov.lr_mat @ torch.linalg.qr(torch.randn(3, 3))[0])  # (an arbitrary rotation)
    order = cov._get_lr_order()
    assert order[:3] == [5, 3, 6] and sorted(order) == list(range(7))

    # rotating to lower-trapezoidal in that order is exact, and doesn't change the order:
    before = cov._get_mini_cov().detach()
    redundant = cov._get_lr_redundant(order)
    assert redundant.sum() == 3 and redundant[5, 1] and redundant[5, 2] and redundant[3, 2]
    assert (cov.lr_mat[redundant] != 0).any()
    cov._rotate_low_rank_(order)
    assert (cov.lr_mat[redundant] == 0).all()
    torch.testing.assert_close(cov._get_mini_cov().detach(), before)
    assert cov._get_lr_order() == order

    # the canonical form is unique: equivalent loadings (differing by a rotation) give the same result, with a
    # non-negative diagonal
    other = _make_low_rank()
    with torch.no_grad():
        other.lr_mat.copy_(cov.lr_mat @ torch.linalg.qr(torch.randn(3, 3))[0])
    other._rotate_low_rank_(other._get_lr_order())
    torch.testing.assert_close(other.lr_mat, cov.lr_mat)
    assert (cov.lr_mat[order[:3], [0, 1, 2]] >= 0).all()

    with pytest.raises(ValueError, match="number of factors"):
        Covariance(rank=3, init_diag_multi=1., method='low_rank4')
    with pytest.raises(TypeError, match="low_rank"):
        Covariance(rank=3, init_diag_multi=1.)._rotate_low_rank_([0, 1, 2])


def test_low_rank_set_matrix_round_trip():
    torch.manual_seed(2)
    cov = _make_low_rank()
    W = torch.randn(7, 3, dtype=torch.float64)
    target = W @ W.t() + torch.diag(torch.rand(7, dtype=torch.float64) + .1)
    cov.set_matrix_(target)
    torch.testing.assert_close(cov._get_mini_cov().detach().to(torch.float64), target, rtol=1e-4, atol=1e-5)
    # and converting from another method:
    sd_corr = Covariance.from_matrix(target, method='sd_corr')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        low_rank = sd_corr.to_method('low_rank3')
    torch.testing.assert_close(low_rank._get_mini_cov(), sd_corr._get_mini_cov(), rtol=1e-4, atol=1e-5)


def _fit_low_rank_excess_loss(seed: int, old_init: bool) -> float:
    """Fit a 2-factor covariance to simulated data (20 L-BFGS iterations); return its excess loss over the truth."""
    torch.manual_seed(seed)
    p, k, n = 12, 2, 400
    W = torch.randn(p, k, dtype=torch.float64) * torch.tensor([2., 1.], dtype=torch.float64)
    true_cov = W @ W.t() + torch.diag(torch.rand(p, dtype=torch.float64) * .3 + .05)
    y = torch.distributions.MultivariateNormal(torch.zeros(p, dtype=torch.float64), true_cov).sample((n,))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cov = Covariance(rank=p, init_diag_multi=1., method=f'low_rank{k}').double()
    if old_init:
        with torch.no_grad():
            cov.lr_mat.copy_(.01 * torch.randn(p, k))
            cov.log_std_devs.copy_(.1 * torch.randn(p))

    def get_loss(c: torch.Tensor) -> torch.Tensor:
        return -torch.distributions.MultivariateNormal(torch.zeros(p, dtype=c.dtype), c).log_prob(y).mean()

    opt = torch.optim.LBFGS(cov.parameters(), max_iter=20, line_search_fn='strong_wolfe')

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = get_loss(cov._get_mini_cov())
        loss.backward()
        return loss

    opt.step(closure)
    return (get_loss(cov._get_mini_cov()) - get_loss(true_cov)).item()


def test_low_rank_init_recovers_factors():
    new = sorted(_fit_low_rank_excess_loss(seed, old_init=False) for seed in range(12))
    old = sorted(_fit_low_rank_excess_loss(seed, old_init=True) for seed in range(12))
    # the new init gets about as close as the true covariance (it can do better: it's fit to the sample)
    assert new[6] < .05
    # factor-loadings near zero start at a saddle-point, so with the same budget they do worse:
    assert old[6] > .3


def test_low_rank_old_state_dict_loads():
    # the parameterization is unchanged (only the init is new), so old state-dicts load as is:
    torch.manual_seed(3)
    old = _make_low_rank()
    with torch.no_grad():
        old.lr_mat.copy_(.01 * torch.randn_like(old.lr_mat))
    state_dict = old.state_dict()
    state_dict._metadata[''] = {'version': 1}
    new = _make_low_rank()
    new.load_state_dict(state_dict)
    torch.testing.assert_close(new._get_mini_cov(), old._get_mini_cov(), rtol=0, atol=0)


def test_low_rank_identified_after_rotation():
    # at the optimum of a gaussian likelihood, the hessian is singular along the rotations, and non-singular once
    # the rotated factor's redundant entries are left out
    torch.manual_seed(4)
    p, k = 6, 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cov = Covariance(rank=p, init_diag_multi=1., method=f'low_rank{k}').double()
    W = torch.randn(p, k, dtype=torch.float64)
    target = W @ W.t() + torch.diag(torch.rand(p, dtype=torch.float64) + .2)
    cov.set_matrix_(target)
    order = cov._get_lr_order()
    cov._rotate_low_rank_(order)
    lr0, std0 = cov.lr_mat.detach().clone(), cov.log_std_devs.detach().clone()

    def nll(flat: torch.Tensor) -> torch.Tensor:
        lr, log_std = flat[:p * k].reshape(p, k), flat[p * k:]
        c = lr @ lr.t() + torch.diag(log_std.exp() ** 2)
        return torch.logdet(c) + torch.trace(torch.linalg.solve(c, target))

    hess = torch.autograd.functional.hessian(nll, torch.cat([lr0.reshape(-1), std0]))
    assert torch.linalg.eigvalsh(hess).abs().min() < 1e-8
    keep = torch.cat([~cov._get_lr_redundant(order).reshape(-1), torch.ones(p, dtype=torch.bool)])
    assert torch.linalg.eigvalsh(hess[keep][:, keep]).min() > 1e-4


def test_laplace_leaves_out_low_rank_rotation():
    torch.manual_seed(5)
    processes = [LocalLevel(id=f'level_{m}', measure=m) for m in 'abc']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kf = KalmanFilter(
            processes=processes,
            measures=list('abc'),
            initial_covariance=Covariance.from_processes(processes, cov_type='initial', method='low_rank2')
        )
    y = torch.randn(2, 10, 3)
    with torch.no_grad():
        before = kf(y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # (an unfitted model's hessian needn't be positive-definite)
        mvnorm, names = kf.get_laplace_mvnorm(y)
    num_params = sum(p.numel() for p in kf.parameters())
    assert len(names) == mvnorm.loc.numel() == num_params - 1  # k * (k - 1) / 2 for k = 2
    lr_mat = kf.initial_covariance.lr_mat
    left_out = [i for i in range(lr_mat.numel()) if f'initial_covariance.lr_mat[{i}]' not in names]
    assert len(left_out) == 1 and lr_mat.reshape(-1)[left_out[0]] == 0  # (the entry the rotation set to zero)
    with torch.no_grad():
        after = kf(y)
    torch.testing.assert_close(after.means, before.means)
    torch.testing.assert_close(after.covs, before.covs)


def _make_seasonal_processes(num_measures: int) -> list:
    # 5 elements per measure (a level, and a season with 4)
    return [
        p for i in range(num_measures)
        for p in (
            LocalLevel(id=f'level{i}', measure=f'y{i}'),
            Season(id=f'season{i}', measure=f'y{i}', period=7, dt_unit=None, K=2)
        )
    ]


def test_default_methods():
    # initial covariance: low_rank from 15 elements with parameters
    big, small = _make_seasonal_processes(3), _make_seasonal_processes(2)
    cov = Covariance.from_processes(big, cov_type='initial')
    assert cov.method == 'low_rank' and cov.param_rank == 15 and cov.lr_mat.shape == (15, 3)
    assert Covariance.from_processes(small, cov_type='initial').method == 'log_cholesky'
    # not with ``corr_groups``, nor for the process-covariance, nor when ``method`` is passed:
    assert Covariance.from_processes(big, cov_type='initial', corr_groups=[['level0']]).method == 'log_cholesky'
    assert Covariance.from_processes(big).method == 'log_cholesky'
    explicit = Covariance.from_processes(big, cov_type='initial', method='log_cholesky')
    assert explicit.method == 'log_cholesky'
    # stationary elements don't count:
    decaying = big + [LocalLevel(id='transient', measure='y0', decay=(.5, .95))]
    assert Covariance.from_processes(decaying, cov_type='initial').method == 'low_rank'
    transients = [LocalLevel(id=f't{i}', measure=f'y{i}', decay=(.5, .95)) for i in range(3)]
    mixed = _make_seasonal_processes(3)[:-1] + transients + [LocalLevel(id='t3', measure='y0', decay=(.5, .95))]
    assert Covariance.from_processes(mixed, cov_type='initial').param_rank == 15
    assert Covariance.from_processes(mixed, cov_type='initial').method == 'low_rank'
    stationary = ['t0', 't1', 't2', 't3']
    assert Covariance.from_processes(mixed, cov_type='initial', stationary=stationary).method == 'log_cholesky'

    # measure covariance: always log_cholesky by default
    assert Covariance.from_measures([f'y{i}' for i in range(8)]).method == 'log_cholesky'

    # a default model:
    kf = KalmanFilter(processes=big, measures=['y0', 'y1', 'y2'])
    assert kf.initial_covariance.method == 'low_rank'
    assert kf.process_covariance.method == 'log_cholesky'


def test_older_default_state_dicts_raise_with_fix():
    torch.manual_seed(0)
    measures = [f'y{i}' for i in range(3)]

    def make(**kwargs) -> KalmanFilter:
        return KalmanFilter(processes=_make_seasonal_processes(3), measures=measures, **kwargs)

    # saved under the old defaults: a full initial covariance (and, with > 5 measures, a low_rank measure-covariance)
    old = make(
        initial_covariance=Covariance.from_processes(
            _make_seasonal_processes(3), cov_type='initial', method='log_cholesky'
        )
    )
    y = torch.randn(2, 12, 3)
    start_offsets = np.zeros(2, dtype='int')
    with torch.no_grad():
        old_pred = old(y, start_offsets=start_offsets)

    # a model built with the new defaults raises, saying how to load it:
    new = make()
    assert new.initial_covariance.method == 'low_rank'
    with pytest.raises(RuntimeError, match=r"from_processes\(\.\.\., method='log_cholesky'\).*torchcast 2\.0"):
        new.load_state_dict(old.state_dict())
    # following that, it loads with the same predictions:
    fixed = make(
        initial_covariance=Covariance.from_processes(
            _make_seasonal_processes(3), cov_type='initial', method='log_cholesky'
        )
    )
    fixed.load_state_dict(old.state_dict())
    with torch.no_grad():
        new_pred = fixed(y, start_offsets=start_offsets)
    torch.testing.assert_close(new_pred.means, old_pred.means)
    torch.testing.assert_close(new_pred.covs, old_pred.covs)

    # the message names the number of factors for low_rank:
    many = [f'y{i}' for i in range(7)]
    old_mcov = Covariance.from_measures(many, method='low_rank')
    with pytest.raises(RuntimeError, match=r"from_measures\(\.\.\., method='low_rank2'\)"):
        Covariance.from_measures(many).load_state_dict(old_mcov.state_dict())
    fixed_mcov = Covariance.from_measures(many, method='low_rank2')
    fixed_mcov.load_state_dict(old_mcov.state_dict())
    torch.testing.assert_close(fixed_mcov._get_mini_cov(), old_mcov._get_mini_cov())


def test_prepare_for_hessian_hook(monkeypatch):
    # ``get_laplace_mvnorm()`` calls ``prepare_for_hessian_()`` on any submodule that has it, and leaves out what it
    # returns -- nothing covariance-specific:
    torch.manual_seed(6)
    kf = KalmanFilter(processes=[LocalLevel(id='level', measure='y')], measures=['y'])
    calls = []

    def prepare_for_hessian_() -> dict:
        calls.append(True)
        return {'initial_mean': torch.ones(1, dtype=torch.bool)}

    monkeypatch.setattr(kf.processes['level'], 'prepare_for_hessian_', prepare_for_hessian_, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, names = kf.get_laplace_mvnorm(torch.randn(2, 10, 1))
    assert calls == [True]
    assert 'processes.level.initial_mean[0]' not in names
    assert len(names) == sum(p.numel() for p in kf.parameters()) - 1

    # a non-callable ``prepare_for_hessian_`` is a mistake, not something to skip:
    monkeypatch.setattr(kf.processes['level'], 'prepare_for_hessian_', {'initial_mean': True}, raising=False)
    with pytest.raises(TypeError, match="not callable"):
        kf.get_laplace_mvnorm(torch.randn(2, 10, 1))

    # other covariance methods leave everything in:
    assert Covariance(rank=3, init_diag_multi=1.).prepare_for_hessian_() == {}
    # "low_rank" with groups: only one of the groups' rotations can be pinned
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    corr_mask = (groups.unsqueeze(-1) == groups.unsqueeze(-2)).to(torch.get_default_dtype())
    masked = Covariance(rank=6, init_diag_multi=1., method='low_rank2', corr_mask=corr_mask)
    with pytest.warns(UserWarning, match="each group's factor-loadings can be rotated separately"):
        masked.prepare_for_hessian_()


def test_cov2corr():
    cov = _random_cov(4, torch.tensor([1., .1, 3., 2.], dtype=torch.float64))
    corr = cov2corr(cov)
    std = cov.diagonal().sqrt()
    torch.testing.assert_close(corr, cov / std.outer(std))
    torch.testing.assert_close(corr.diagonal(), torch.ones(4, dtype=torch.float64))
    # batched:
    torch.testing.assert_close(cov2corr(torch.stack([cov, 2 * cov])), torch.stack([corr, corr]))


def test_log_chol_to_chol_deprecated_alias():
    from torchcast.covariance import log_chol_to_chol
    log_diag, off_diag = torch.randn(3), torch.randn(3)
    with pytest.warns(FutureWarning, match="use ``torchcast.covariance.log_chol_to_chol"):
        L = Covariance.log_chol_to_chol(log_diag, off_diag)
    torch.testing.assert_close(L, log_chol_to_chol(log_diag, off_diag))


def test_covariance_base_shim():
    import torchcast.covariance.base as base
    with pytest.warns(FutureWarning, match="covariance.base.Covariance` is deprecated"):
        assert base.Covariance is Covariance
    # missing attributes raise AttributeError (so ``getattr(..., default)`` works), and dunders don't warn:
    assert getattr(base, 'does_not_exist', None) is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert getattr(base, '__sphinx_mock__', None) is None
        base.__file__


def test_covariance_in_the_wrong_slot_raises():
    processes = [LocalLevel(id='level', measure='y')]
    # e.g. forgetting ``cov_type='initial'``, which makes a process-covariance (different elements get parameters):
    with pytest.raises(ValueError, match=r"created as ``process_covariance``.*cov_type='initial'"):
        KalmanFilter(
            processes=processes, measures=['y'], initial_covariance=Covariance.from_processes(processes)
        )
    with pytest.raises(ValueError, match=r"created as ``initial_covariance``.*cov_type='process'"):
        KalmanFilter(
            processes=processes,
            measures=['y'],
            process_covariance=Covariance.from_processes(processes, cov_type='initial')
        )
    with pytest.raises(ValueError, match=r"used as ``measure_covariance``.*from_measures"):
        KalmanFilter(
            processes=processes, measures=['y'], measure_covariance=Covariance.from_processes(processes)
        )
    # a covariance with a custom id is renamed, with a warning, as before:
    custom = Covariance(rank=1, init_diag_multi=1., id='my_covariance')
    with pytest.warns(UserWarning, match="Id already set to my_covariance"):
        KalmanFilter(processes=processes, measures=['y'], initial_covariance=custom)
