import itertools
import math
from collections.abc import Sequence

import torch


class _MixtureComponentBase(torch.nn.Module):
    def __init__(self, measure: str, id: str):
        super().__init__()
        self.measure = measure
        self.id = id

    @property
    def is_null(self) -> bool:
        raise NotImplementedError

    @classmethod
    def traverse(cls,
                 mixture_components: Sequence['MixtureComponent'],
                 measures: Sequence[str]) -> Sequence[tuple[tuple['_MixtureComponentBase', ...], torch.Tensor]]:
        if not isinstance(measures, Sequence) or isinstance(measures, str):
            raise ValueError(f"`measures` must be a sequence, got {type(measures)}")

        mi_by_measure = {m: [NullMixtureComponent(m).to(mixture_components[0].logit)] for m in measures}
        for mi in mixture_components:
            if mi.measure not in measures:
                continue
            mi_by_measure[mi.measure].append(mi)
        for m, mis in mi_by_measure.items():
            _ids = [mi.id for mi in mis]
            if len(_ids) != len(set(_ids)):
                raise ValueError(f"Mixture components must have unique ids within a measure, but {m} got {_ids}")

        # per-column calculation: probabilities within each measure
        probs_by_measure = {}
        for measure, mis in mi_by_measure.items():
            logits = [mi.logit for mi in mis]
            probs_by_measure[measure] = torch.softmax(torch.stack(logits, dim=-1), dim=-1).unbind(-1)
        regime_combos = itertools.product(*[mi_by_measure[m] for m in measures])
        prob_combos = (math.prod(probs) for probs in itertools.product(*[probs_by_measure[m] for m in measures]))
        return list(zip(regime_combos, prob_combos))


class MixtureComponent(_MixtureComponentBase):

    def __init__(self,
                 measure: str,
                 mean_init: float,
                 prob_init: float,
                 id: str):
        super().__init__(measure=measure, id=id)

        mean_init = torch.as_tensor(mean_init, dtype=torch.get_default_dtype())
        self.mean = torch.nn.Parameter(mean_init)
        self._log_std = torch.nn.Parameter(torch.zeros_like(mean_init))

        assert 0 < prob_init < 1
        prob_init = torch.as_tensor([prob_init], dtype=torch.get_default_dtype())
        # raw logit relative to an implicit "main" reference logit fixed at 0.
        # exact calibration to `prob_init` only holds when this is the sole active
        # component -- with >1 components, softmax normalizes jointly, so actual
        # initial probs will differ slightly from prob_init (see `mixing_probs`)
        self.logit = torch.nn.Parameter(torch.log(prob_init) - torch.log1p(-prob_init))

    @property
    def is_null(self) -> bool:
        return False

    @property
    def var(self) -> torch.Tensor:
        return self._log_std.exp() ** 2


class NullMixtureComponent(_MixtureComponentBase):
    def __init__(self, measure: str):
        super().__init__(measure=measure, id='standard')
        self.logit = torch.zeros(1)

    @property
    def is_null(self) -> bool:
        return True

    def to(self, *args, **kwargs):
        self.logit = self.logit.to(*args, **kwargs)
        return self


class RegimeTransition(torch.nn.Module):
    def __init__(self, num_regimes: int, self_persist_init: float = 3.0):
        super().__init__()
        self.num_regimes = num_regimes
        # only column 0 ("main") is the fixed reference (implicit zero);
        # free_logits holds columns 1..K for every row -- (num_regimes, num_regimes - 1)
        init = torch.zeros(num_regimes, num_regimes - 1)
        init[0, :] = -self_persist_init  # row 0 (main): initially favors staying at col 0
        for i in range(1, num_regimes):
            init[i, i - 1] = self_persist_init  # row i: initially favors staying at col i
        self.free_logits = torch.nn.Parameter(init)

    @property
    def logits(self) -> torch.Tensor:
        zeros_col = torch.zeros(self.num_regimes, 1, device=self.free_logits.device, dtype=self.free_logits.dtype)
        return torch.cat([zeros_col, self.free_logits], dim=-1)  # (num_regimes, num_regimes)

    @property
    def matrix(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def predict(self, posterior_prev: torch.Tensor) -> torch.Tensor:
        return posterior_prev @ self.matrix
