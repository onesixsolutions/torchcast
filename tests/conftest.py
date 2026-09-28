import pytest


@pytest.fixture(autouse=True)
def _pin_derived_seed(monkeypatch):
    """
    ``Predictions`` draws the seed for its sample-based outputs (``to_dataframe(derived=...)``, intervals for
    nonlinear mixture measures) at import, so it differs between test-runs. Pin it, so that tests with monte-carlo
    tolerances are deterministic.
    """
    from torchcast.state_space import predictions

    monkeypatch.setattr(predictions, '_DERIVED_SEED', 12345)
