import math
from typing import Optional, Sequence, Dict
import torch.nn.functional as F
from warnings import warn

import torch
from torch import jit, Tensor

from torchcast.process import Process


class SmoothingMatrix(torch.nn.Module):

    @classmethod
    def from_measures_and_processes(cls,
                                    measures: Sequence[str],
                                    processes: Sequence['Process'],
                                    method: str = 'full',
                                    **kwargs) -> 'SmoothingMatrix':
        """
        :param measures: List of measure-names.
        :param processes: A list of :class:`.Process` modules.
        :param method: Parameterization, currently supports 'full' and 'low_rank'.
        :param kwargs: Other kwargs to pass to ``__init__``
        :return: A :class:`.SmoothingMatrix` object that can be used in your :class:`.ExpSmoother`.
        """
        measures = list(measures)
        state_rank = 0
        fixed_states = []
        for p in processes:
            fixed_els = [nm for nm, se in p.state_elements.items() if not se.has_process_variance]
            for i, se in enumerate(p.state_elements):
                if se in fixed_els:
                    fixed_states.append(state_rank + i)
                # TODO: warn if time-varying H but some unfixed_els
            state_rank += len(p.state_elements)
        return cls(
            measure_rank=len(measures),
            state_rank=state_rank,
            fixed_states=fixed_states,
            method=method,
            **kwargs
        )

    def __init__(self,
                 measure_rank: int,
                 state_rank: int,
                 method: str = 'full',
                 fixed_states: Sequence[int] = (),
                 init_bias: int = -10,
                 id: Optional[str] = None):

        super().__init__()

        self.id = id

        self.state_rank = state_rank
        self.measure_rank = measure_rank
        self.full_states = [i for i in range(self.state_rank) if i not in fixed_states]

        self.init_bias = init_bias

        self.unconstrained_params: Optional[torch.nn.Parameter] = None
        self.lr1: Optional[torch.nn.Parameter] = None
        self.lr2: Optional[torch.nn.Parameter] = None
        if method == 'full':
            self.method = method
            self.unconstrained_params = torch.nn.Parameter(.1 * torch.randn(self.measure_rank * len(self.full_states)))
        elif method.startswith('low_rank'):
            if measure_rank == 1:
                warn("Using `method='low_rank'` with 1 measure")
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                ub = (len(self.full_states) * self.measure_rank) / (len(self.full_states) + self.measure_rank)
                low_rank = int(math.sqrt(ub))
            self.lr1 = torch.nn.Parameter(.1 * torch.randn(len(self.full_states), low_rank))
            self.lr2 = torch.nn.Parameter(.1 * torch.randn(low_rank, self.measure_rank))
        else:
            raise ValueError(f"Unrecognized method `{method}`")

        # todo: support prediction
        self.expected_kwargs = None

    def forward(self,
                inputs: Dict[str, Tensor],
                num_groups: int,
                num_times: int,
                _ignore_input: bool = False) -> Tensor:
        if len(inputs) > 0 and not _ignore_input:
            raise NotImplementedError

        if self.method == 'full':
            K = torch.zeros(
                (self.state_rank, self.measure_rank),
                dtype=self.unconstrained_params.dtype,
                device=self.unconstrained_params.device
            )
            k_full = torch.sigmoid(self.unconstrained_params + self.init_bias)
            K[self.full_states, :] = k_full.view(len(self.full_states), self.measure_rank)
        else:
            assert self.method == 'low_rank'
            assert self.lr1 is not None
            assert self.lr2 is not None
            # we want lr1 @ lr2 to be constrained to 0-1. this could be accomplished by applying sigmoid after matmul,
            # but there may be matmul-ordering optimizations to keeping lr1 and lr2 separate for `predict` or `update`
            # so we want to instead constrain lr1 and lr2 themselves
            # lr1: each row needs to sum to 0<=sum(row)<=1. softmax is traditionally overparameterized b/c sums to 1,
            #      but here it is *not* overparameterized b/c sums to <=1 (`* torch.sigmoid(l1.sum)` part)
            lr1 = torch.zeros((self.state_rank, self.lr1.shape[-1]), dtype=self.lr1.dtype, device=self.lr1.device)
            lr1[self.full_states] = \
                F.softmax(self.lr1, -1) * torch.sigmoid(self.lr1.sum(-1, keepdim=True) + self.init_bias / 2)
            # lr2: each element is 0<=el<=1
            lr2 = torch.sigmoid(self.lr2 + self.init_bias / 2)
            # "there may be matmul-ordering optimizations to keeping lr1 and lr2 separate" -- in theory, yes.
            # in practice haven't gotten to work -- may revisit, for now they are pre-multiplied
            K = lr1 @ lr2
        K = K.unsqueeze(0).expand(num_groups, -1, -1)
        K = K.unsqueeze(1).expand(-1, num_times, -1, -1)
        return K

    @jit.ignore
    def set_id(self, id: str) -> 'SmoothingMatrix':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self
