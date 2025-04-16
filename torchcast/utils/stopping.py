import warnings
from typing import Optional, Collection, Union, Callable, Iterable

import torch


class Stopping:
    """
    Allows control of convergence/stopping for the `fit()` method in state-space models.

    :param abstol: The absolute tolerance.
    :param patience: How many iterations monitored metrics need to change less than ``abstol`` before we stop.
    :param monitor_loss: Should loss be monitored as part of convergence checks?
    :param monitor_params: If ``True``, all parameters will be monitored. If a list of names, only those
     parameters will be monitored. Can also be a function that takes a param-name and returns true/false.
    :param max_iter: The maximum number of iterations after which training will be stopped regardless of convergence.
    :param module: The module whose parameters are being optimized. Required if ``monitor_params`` is not ``False``.
    """

    def __init__(self,
                 abstol: float = .0001,
                 patience: int = 2,
                 monitor_loss: bool = True,
                 monitor_params: Union[bool, Collection[str], callable] = False,
                 max_iter: int = 300,
                 module: Optional[torch.nn.Module] = None):

        if not monitor_loss and not monitor_params:
            raise ValueError("At least one of `monitor_loss` or `monitor_params` must be True")
        self.monitor_loss = monitor_loss
        self.monitor_params = monitor_params
        self.module = module
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
        flat = []

        if self.monitor_params:
            for p in self._get_module_params():
                flat.append(p.view(-1).cpu())

        if self.monitor_loss:
            flat.append(torch.as_tensor([loss]))

        return torch.cat(flat)

    def _get_module_params(self) -> Iterable[torch.Tensor]:
        assert self.module is not None
        module_parameters = {nm: p for nm, p in self.module.named_parameters() if p.requires_grad}

        assert self.monitor_params  # only called this method if truth-y
        if self.monitor_params is True:
            monitor_params = list(module_parameters)
        elif callable(self.monitor_params):
            monitor_params = [self.monitor_params(p) for p in module_parameters]
        else:  # a list:
            monitor_params = self.monitor_params

        for nm in monitor_params:
            yield module_parameters[nm]

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
