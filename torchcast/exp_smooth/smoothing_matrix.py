import math
from typing import List, Optional, Sequence, Dict
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
        from torchcast.covariance import get_state_element_names
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
            state_element_names=get_state_element_names(processes),
            measures=measures,
            **kwargs
        )

    def __init__(self,
                 measure_rank: int,
                 state_rank: int,
                 method: str = 'full',
                 fixed_states: Sequence[int] = (),
                 init_bias: int = -10,
                 state_element_names: Optional[Sequence[str]] = None,
                 measures: Optional[Sequence[str]] = None,
                 id: Optional[str] = None):
        """
        You should rarely call this directly: use :func:`SmoothingMatrix.from_measures_and_processes`, or pass a dict
        of its options as the :class:`.ExpSmoother`'s ``smoothing_matrix``.

        :param measure_rank: The number of measures (columns).
        :param state_rank: The number of state-elements (rows).
        :param method: Parameterization, currently supports 'full' and 'low_rank'.
        :param fixed_states: State-elements with no smoothing (no process-variance).
        :param init_bias: Added to the unconstrained parameters, so the gains start near zero.
        :param state_element_names: The names of the rows (see :func:`torchcast.covariance.get_state_element_names`),
         and ``measures`` those of the columns. Set by :func:`SmoothingMatrix.from_measures_and_processes`; if
         ``None``, the :class:`.ExpSmoother` sets them. The model raises if they don't match its own.
        :param measures: See ``state_element_names``.
        :param id: Deprecated, has no effect.
        """

        super().__init__()

        if id is not None:
            warn("``SmoothingMatrix(id=...)`` is deprecated, and has no effect.", FutureWarning, stacklevel=2)
        self.state_element_names: Optional[List[str]] = None
        self.measures: Optional[List[str]] = None
        if state_element_names is not None:
            if len(state_element_names) != state_rank:
                raise ValueError(
                    f"``state_element_names`` has {len(state_element_names)} names, expected {state_rank}."
                )
            self.state_element_names = list(state_element_names)
        if measures is not None:
            if len(measures) != measure_rank:
                raise ValueError(f"``measures`` has {len(measures)} names, expected {measure_rank}.")
            self.measures = list(measures)

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

    @property
    def id(self) -> str:
        return 'smoothing_matrix'

    def _bind(self, state_element_names: Sequence[str], measures: Sequence[str]) -> 'SmoothingMatrix':
        """
        Called by the :class:`.ExpSmoother` with the names of its state-elements and measures: raises if they don't
        match this module's, and sets them if they weren't set.
        """
        how = "SmoothingMatrix.from_measures_and_processes(measures, processes, ...)"
        for attr, expected, kind, source in [
            ('state_element_names', list(state_element_names), 'state-elements', 'processes'),
            ('measures', list(measures), 'measures', 'measures')
        ]:
            current = getattr(self, attr)
            rank = self.state_rank if attr == 'state_element_names' else self.measure_rank
            if current is None:
                if rank != len(expected):
                    raise ValueError(
                        f"The ``smoothing_matrix`` has {rank} {kind}, but the model has {len(expected)}: {expected}. "
                        f"Create it with ``{how}``."
                    )
            elif current != expected:
                raise ValueError(
                    f"The ``smoothing_matrix`` was created for the {kind} {current}, but the model's are {expected}. "
                    f"Create it with ``{how}``, with the model's {source}, in the same order."
                )
            setattr(self, attr, expected)
        return self

    @jit.ignore
    def set_id(self, id: str) -> 'SmoothingMatrix':
        """Deprecated, has no effect."""
        warn("``SmoothingMatrix.set_id()`` is deprecated, and has no effect.", FutureWarning, stacklevel=2)
        return self

    def __setstate__(self, state: dict):
        # pickled before ``id`` became a property, and before the element names existed:
        state.pop('id', None)
        state.setdefault('state_element_names', None)
        state.setdefault('measures', None)
        super().__setstate__(state)
