import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch():
    """
    ``Predictions`` draws the seed for its sample-based outputs (e.g. ``to_dataframe()`` intervals for nonlinear
    measures, ``derived``) from torch's global random state. Seed it, so that tests with monte-carlo tolerances are
    deterministic even if they don't seed it themselves.
    """
    torch.manual_seed(12345)
