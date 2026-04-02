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
    aliases = ('sigmoid', 'ilogit', 'expit', 'inv_logit')

    def __call__(self, measured_mean: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(measured_mean.clamp(-8, 8))

    def inverse_transform(self, input_mean: torch.Tensor) -> torch.Tensor:
        assert (0 <= input_mean <= 1).all()
        return torch.special.logit(input_mean, eps=1e-7)

    def adjust_measure_mat(self, measure_mat: torch.Tensor, measured_mean: torch.Tensor) -> torch.Tensor:
        measured_mean = measured_mean.clamp(-8, 8)
        numer = torch.exp(-measured_mean)
        denom = (torch.exp(-measured_mean) + 1) ** 2
        return measure_mat * (numer / denom).unsqueeze(-1)
