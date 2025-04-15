import warnings
from typing import Optional

import torch


class Stopping:
    """
    Allows control of convergence/stopping for the `fit()` method in state-space models.

    :param abstol: The absolute tolerance.
    :param patience: How many iterations monitored metrics need to change less than ``abstol`` before we stop.
    :param monitor: What should be monitored? Can be 'loss', 'params' or 'loss+params'.
    :param max_iter: The maximum number of iterations after which training will be stopped regardless of convergence.
    :param optimizer: The optimizer, not required if ``monitor`` doesn't include 'params'.
    """
    def __init__(self,
                 abstol: float = .0001,
                 patience: int = 2,
                 monitor: str = 'loss',
                 max_iter: int = 200,
                 optimizer: Optional[torch.optim.Optimizer] = None):

        self.monitor = monitor
        self.optimizer = optimizer
        self.abstol = abstol
        self.values = []
        self.patience = patience
        self.max_iter = max_iter
        self._patience_counter = 0
        self.last_change = float('nan')

    @property
    def convergence(self) -> str:
        return '{:.4}/{}'.format(self.last_change, self.abstol)

    @classmethod
    def from_dict(cls, **kwargs) -> 'Stopping':
        # handle deprecated
        if 'tol' in kwargs:
            _msg = "Please specify `abstol` not `tol`"
            if 'abstol' in kwargs:
                raise TypeError(_msg)
            warnings.warn(_msg)
            kwargs['abstol'] = kwargs.pop('tol')
        return cls(**kwargs)

    @torch.inference_mode()
    def _get_new_values(self, loss: Optional[float]):
        flat_params = []
        if 'params' in self.monitor:
            assert self.optimizer is not None
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    flat_params.append(p.view(-1))
        if 'loss' in self.monitor:
            flat_params.append(torch.as_tensor([loss]))
        return torch.cat(flat_params)

    @torch.inference_mode()
    def __call__(self, loss: Optional[float]) -> bool:
        self.values.append(self._get_new_values(loss))
        if len(self.values) == 1:
            return False

        abs_changes = (self.values[-1] - self.values[-2]).abs()
        self.last_change = abs_changes.max()
        if self.last_change > self.abstol:
            self._patience_counter = 0
            return False

        self._patience_counter += 1
        converged = self._patience_counter >= self.patience
        if converged:
            return True

        if len(self.values) >= self.max_iter:
            warnings.warn(f"Max iters ({self.max_iter}) reached")
            return True

        return False