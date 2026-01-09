import torch
from torchcast.covariance import Covariance


@torch.no_grad()
def test_from_log_cholesky():
    module = Covariance(id='test', rank=3)

    module.state_dict()['cholesky_log_diag'][:] = torch.arange(1., 3.1)
    module.state_dict()['cholesky_off_diag'][:] = torch.arange(1., 3.1)

    expected = torch.tensor([[7.3891, 2.7183, 5.4366],
                             [2.7183, 55.5982, 24.1672],
                             [5.4366, 24.1672, 416.4288]])
    diff = (expected - module({}, num_groups=1, num_times=1)).abs()
    assert (diff < .0001).all()


@torch.no_grad()
def test_empty_idx():
    module = Covariance(id='test', rank=3, empty_idx=[0])
    cov = module({}, num_groups=1, num_times=1)
    cov = cov.squeeze()
    assert (cov[0, :] == 0).all()
    assert (cov[:, 0] == 0).all()
    assert (cov == cov.t()).all()
