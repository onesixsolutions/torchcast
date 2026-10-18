"""
Chunked fitting: the loss, gradient and hessian of a model summed over chunks of groups (see
``StateSpaceModel.fit(chunk_size=...)``), so that only one chunk's computation-graph is in memory at a time.
"""
from typing import Any, Callable, Iterator, List, NamedTuple, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import torch

from torchcast.internals.hessian import hessian
from .state import StateTuple

if TYPE_CHECKING:
    from .state_space import StateSpaceModel


class _FitChunk(NamedTuple):
    group_slice: Union[slice, torch.Tensor]  # a slice, or indices, of the full data's groups
    y: torch.Tensor
    kwargs: dict
    get_loss: Callable
    weight: float


def _get_group_slices(num_groups: int, chunk_size: Optional[int]) -> List[slice]:
    if chunk_size is None or chunk_size >= num_groups:
        return [slice(None)]
    if chunk_size < 1:
        raise ValueError(f"`chunk_size` must be a positive integer, got {chunk_size}")
    return [slice(start, start + chunk_size) for start in range(0, num_groups, chunk_size)]


def _subset_groups(value: Any,
                   group_slice: Union[slice, torch.Tensor],
                   num_groups: int,
                   strict: bool = False,
                   name: str = '') -> Any:
    """
    Subset a forward-kwarg to a chunk of groups: tensors/arrays/lists whose first dimension is ``num_groups`` are
    sliced; tuples and ``StateTuple`` (e.g. ``initial_state``) and dicts are handled recursively; anything else
    (e.g. scalars) is passed as-is. With ``strict`` (for kwargs known to be group-indexed), a first dimension other
    than ``num_groups`` (or 1, for broadcasting) raises.
    """
    if isinstance(value, (torch.Tensor, np.ndarray, list)):
        size = len(value) if isinstance(value, list) else (value.shape[0] if value.ndim else None)
        if size == num_groups:
            if isinstance(group_slice, slice):
                return value[group_slice]
            if isinstance(value, list):
                return [value[i] for i in group_slice.tolist()]
            if isinstance(value, np.ndarray):
                return value[group_slice.cpu().numpy()]
            return value[group_slice.to(value.device)]
        if strict and size != 1:
            raise ValueError(f"Expected `{name}` to have first dimension {num_groups} (num_groups), got {size}.")
        return value
    if isinstance(value, StateTuple):  # e.g. ``initial_state`` with ``regime_probs`` / ``scaling``
        return StateTuple(*(
            None if v is None else _subset_groups(v, group_slice, num_groups, strict, name)
            for v in (value.mean, value.cov, value.regime_probs, value.scaling)
        ))
    if isinstance(value, tuple):
        return tuple(_subset_groups(v, group_slice, num_groups, strict, name) for v in value)
    if isinstance(value, dict):
        return {k: _subset_groups(v, group_slice, num_groups, strict, name) for k, v in value.items()}
    return value


def _subset_loss_fun(get_loss: Callable, group_slice: Union[slice, torch.Tensor], num_groups: int) -> Callable:
    from .state_space import LossFun  # (here, since state_space imports this module)

    if (isinstance(group_slice, slice) and group_slice == slice(None)) or not isinstance(get_loss, LossFun):
        return get_loss
    return get_loss.subset(group_slice, num_groups)


class _ChunkedObjective:
    """
    A model's loss, summed over chunks of groups (see ``fit(chunk_size=...)``), and its gradient/hessian w.r.t. the
    parameters that require grad. Only one chunk's computation-graph is in memory at a time.

    :param hessian_chunks: Chunks to use for the hessian, which needs more memory per group than the loss/gradient.
     Defaults to ``chunks``.
    :param hessian_subsampled: Whether ``hessian_chunks`` are only a subsample of groups.
    :param resample_hessian_chunks: Optionally, a function returning new ``hessian_chunks`` (a new subsample), called
     for each hessian.
    """

    def __init__(self,
                 ss_model: 'StateSpaceModel',
                 chunks: Sequence[_FitChunk],
                 num_groups: int,
                 callable_kwargs: Optional[dict[str, Callable]] = None,
                 hessian_chunks: Optional[Sequence[_FitChunk]] = None,
                 hessian_subsampled: bool = False,
                 resample_hessian_chunks: Optional[Callable[[], Sequence[_FitChunk]]] = None):
        self.ss_model = ss_model
        self.chunks = chunks
        self.hessian_chunks = hessian_chunks or chunks
        self.hessian_subsampled = hessian_subsampled
        self.resample_hessian_chunks = resample_hessian_chunks
        self.num_groups = num_groups
        self.callable_kwargs = callable_kwargs or {}

        self.params = []
        self.param_names = []
        for nm, par in ss_model.named_parameters():
            if not par.requires_grad:
                continue
            self.param_names.extend(f'{nm}[{i}]' for i in range(par.numel()))
            self.params.append(par)

    def chunk_losses(self, chunks: Optional[Sequence[_FitChunk]] = None) -> Iterator[torch.Tensor]:
        """
        Yields each chunk's (weighted) loss. Callers should finish with each (e.g. ``backward()``) before the next, so
        that only one graph is in memory.
        """
        for chunk in (self.chunks if chunks is None else chunks):
            kwargs = chunk.kwargs
            if self.callable_kwargs:
                # called per chunk: outputs may be part of the graph, which is freed by each chunk's backward()
                callable_kwargs = {k: v() for k, v in self.callable_kwargs.items()}
                kwargs = {**kwargs, **self.ss_model._subset_kwargs(callable_kwargs, chunk.group_slice, self.num_groups)}
            pred = self.ss_model(chunk.y, **kwargs)
            yield chunk.get_loss(pred, chunk.y) * chunk.weight

    @torch.no_grad()
    def loss(self) -> float:
        return sum(loss.item() for loss in self.chunk_losses())

    def loss_and_grad(self) -> tuple[float, torch.Tensor]:
        total_loss = 0.
        total_grad = 0.
        for loss in self.chunk_losses():
            total_grad = total_grad + self._flat_grad(loss)
            total_loss += loss.item()
        return total_loss, total_grad

    def loss_grad_hessian(self) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        The loss and gradient are always from all groups. If ``hessian_chunks`` are a subsample of groups, the hessian
        is from those (as an estimate of the full-data hessian, on the same mean-scale).
        """
        if self.resample_hessian_chunks is not None:
            self.hessian_chunks = self.resample_hessian_chunks()
        total_loss = 0.
        total_grad = 0.
        total_hess = 0.
        for loss in self.chunk_losses(self.hessian_chunks):
            loss = loss.squeeze()
            if not self.hessian_subsampled:
                total_grad = total_grad + self._flat_grad(loss, retain_graph=True)
                total_loss += loss.item()
            total_hess = total_hess + hessian(output=loss, inputs=self.params, allow_unused=True, progress=False)
            del loss
        if self.hessian_subsampled:
            total_loss, total_grad = self.loss_and_grad()
        return total_loss, total_grad, total_hess

    def _flat_grad(self, loss: torch.Tensor, retain_graph: bool = False) -> torch.Tensor:
        grads = torch.autograd.grad(loss, self.params, retain_graph=retain_graph, allow_unused=True)
        return torch.cat([
            (torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(self.params, grads)
        ]).detach()

    def get_vector(self) -> torch.Tensor:
        # matches the ordering of ``hessian()``, ``param_names``
        return torch.cat([p.reshape(-1) for p in self.params])

    @torch.no_grad()
    def set_vector(self, vector: torch.Tensor):
        offset = 0
        for p in self.params:
            p.copy_(vector[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
