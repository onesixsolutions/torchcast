from typing import Collection

import torch

from torchcast.internals.utils import get_subclasses


class MeasureFun:
    _alias2cls = None
    aliases: Collection[str]

    def __call__(self, measured_mean: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(self, input_mean: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjust_measure_mat(self, measure_mat: torch.Tensor, measured_mean: torch.Tensor) -> torch.Tensor:
        """
        Apply the chain-rule for this measure-function to the measurement-matrix (the EKF linearization).

        :param measure_mat: The row(s) of the measurement matrix for this measure.
        :param measured_mean: The *input* to this measure-function, i.e. the measured-mean before it's applied.
        """
        raise NotImplementedError

    @classmethod
    def from_alias(cls, alias: str) -> 'MeasureFun':
        if cls._alias2cls is None:
            cls._alias2cls = {}
            for subcls in get_subclasses(MeasureFun):
                for a in subcls.aliases:
                    cls._alias2cls[a] = subcls
        klass = cls._alias2cls.get(alias, None)
        if not klass:
            raise ValueError(f"Unknown measure function alias: {alias}. Available aliases: {set(cls._alias2cls)}")
        return klass()


class Sigmoid(MeasureFun):
    """
    The sigmoid measure-function (e.g. for the binary measures of :class:`.BinomialFilter`).

    ``legacy_jacobian``: before v1.1.3, the EKF linearization of the sigmoid was (incorrectly) evaluated at the
    post-sigmoid value, i.e. ``sigmoid'(sigmoid(z))`` rather than ``sigmoid'(z)``. Setting
    ``my_model.measure_funs['my_measure'].legacy_jacobian = True`` reproduces that, for comparing results against the
    previous behavior. Sigmoid objects unpickled from an earlier version don't have this attribute, and keep the
    previous behavior. Will be removed in a future version.
    """
    aliases = ('sigmoid', 'ilogit', 'expit', 'inv_logit')

    def __init__(self):
        self.legacy_jacobian = False

    def __call__(self, measured_mean: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(measured_mean.clamp(-8, 8))

    def inverse_transform(self, input_mean: torch.Tensor) -> torch.Tensor:
        assert (0 <= input_mean <= 1).all()
        return torch.special.logit(input_mean, eps=1e-7)

    def adjust_measure_mat(self, measure_mat: torch.Tensor, measured_mean: torch.Tensor) -> torch.Tensor:
        # a missing attribute means this object was unpickled from a version before the fix: keep the old behavior.
        if getattr(self, 'legacy_jacobian', True):
            measured_mean = self(measured_mean)  # (see `legacy_jacobian` in the class docstring)
        measured_mean = measured_mean.clamp(-8, 8)
        numer = torch.exp(-measured_mean)
        denom = (torch.exp(-measured_mean) + 1) ** 2
        return measure_mat * (numer / denom).unsqueeze(-1)
