"""
These are simple training classes for PyTorch models, with specialized subclasses for torchcast's model-classes (i.e.,
when the data are too big for the ``StateSpaceModel.fit()`` method). Additionally, there is a
special class for training neural networks to embed complex seasonal patterns into lower dimensional embeddings.

While the classes in this module are helpful for quick development, they are not necessarily meant to replace more
sophisticated tools (e.g. PyTorch Lightning) in more complex settings.
"""
import warnings
from itertools import zip_longest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import torch.nn as nn
from typing import Generator, Union, Type, Sequence, Tuple, Dict, Optional, Callable

from tqdm.auto import tqdm

from torchcast.internals.utils import chunk_grouped_data
from torchcast.state_space import Predictions
from torchcast.utils import TimeSeriesDataset
from torchcast.utils.features import fourier_model_mat


class BaseTrainer:
    _device = None

    def __init__(self,
                 module: nn.Module,
                 optimizer: Union[Optimizer, Type[Optimizer]] = torch.optim.Adam):
        self.module = module
        if isinstance(optimizer, type):
            optimizer = optimizer([p for p in module.parameters() if p.requires_grad])
        self.optimizer = optimizer

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.epoch = 0

    def to(self, device) -> 'BaseTrainer':
        self.module.to(device)
        self._device = device
        return self

    def _get_closure(self, batch: any, forward_kwargs: dict) -> callable:
        raise NotImplementedError

    def _get_batch_numel(self, batch: any) -> int:
        raise NotImplementedError

    def __call__(self,
                 dataloader: DataLoader,
                 prog: bool = True,
                 forward_kwargs: dict = None) -> Generator[float, None, None]:
        forward_kwargs = forward_kwargs or {}
        if prog:
            prog = tqdm(total=len(dataloader))

        self.module.train()
        while True:
            self.epoch += 1
            epoch_loss = 0.
            n = 0
            try:
                for batch in dataloader:
                    closure = self._get_closure(batch, forward_kwargs)
                    loss = self.optimizer.step(closure)
                    batch_n = self._get_batch_numel(batch)
                    epoch_loss += (loss.item() * batch_n)
                    n += batch_n
                    if prog:
                        prog.update()
                        prog.set_description(f'Epoch {self.epoch}')
            except KeyboardInterrupt:
                if prog:
                    prog.close()
                break
            finally:
                prog.reset()

            yield epoch_loss / n

        if prog:
            prog.close()


class SimpleTrainer(BaseTrainer):
    """
    A simple trainer for a standard nn.Module (not a state-space model). Note: this is meant to be helpful for quick
    development, it's not meant to replace better tools (e.g. PyTorch Lightning) in more complex settings.

    Usage:

    .. code-block:: python

        dataloader = DataLoader(my_data, batch_size=32)
        trainer = SimpleTrainer(module=nn.Linear(10, 1))
        for loss in trainer(dataloader):
            # log the loss, early-stopping, etc.

    """
    _warned = False

    def __init__(self,
                 module: nn.Module,
                 optimizer: Union[Optimizer, Type[Optimizer]] = torch.optim.Adam,
                 loss_fn: callable = torch.nn.MSELoss()):
        self.loss_fn = loss_fn
        super().__init__(module=module, optimizer=optimizer)

    def _get_closure(self, batch: Dataset, forward_kwargs: dict) -> callable:
        inputs, targets, *_other = batch
        if len(_other) and not self._warned:
            warnings.warn("Ignoring additional tensors in batch.")
            self._warned = True
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        def closure():
            self.optimizer.zero_grad()
            outputs = self.module(inputs, **forward_kwargs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            return loss

        return closure

    def _get_batch_numel(self, batch: Dataset) -> int:
        if getattr(self.loss_fn, 'reduction', 'mean') != 'mean':
            raise RuntimeError("Can't handle loss functions with `reduction` other than 'mean'")
        return len(batch)


class StateSpaceTrainer(BaseTrainer):
    """
    A trainer for a :``StateSpaceModel``. This is for contexts in which the data are too large for
    ``StateSpaceModel.fit()`` to be practical. Rather than the base DataLoader, this class takes a
    :class:`torchcast.utils.TimeSeriesDataLoader`.

    Usage:

    .. code-block:: python

        from torchcast.kalman_filter import KalmanFilter
        from torchcast.utils import TimeSeriesDataLoader
        from torchcast.process import LocalTrend

        my_dl = TimeSeriesDataLoader.from_dataframe(my_df)
        my_model = KalmanFilter(processes=[LocalTrend(id='trend')])
        my_trainer = StateSpaceTrainer(module=my_model)
        for loss in my_trainer(my_dl, forward_kwargs={'n_step' : 14*7*24, 'every_step' : False}):
            # log the loss, early-stopping, etc.


    :param module: A ``StateSpaceModel`` instance (e.g. ``KalmanFilter`` or ``ExpSmoother``).
    :param dataset_to_kwargs: A callable that takes a :class:`torchcast.utils.TimeSeriesDataset` and returns a dictionary
     of keyword-arguments to pass to each call of the module's ``forward`` method. If left unspecified and the batch
     has a 2nd tensor, will pass ``tensor[1]`` as the ``X`` keyword.
    :param optimizer: An optimizer (or a class to instantiate an optimizer). Default is :class:`torch.optim.Adam`.
    """

    def __init__(self,
                 module: nn.Module,
                 dataset_to_kwargs: Optional[Union[Sequence[str], Callable[[TimeSeriesDataset], dict]]] = None,
                 optimizer: Union[Optimizer, Type[Optimizer]] = torch.optim.Adam):

        self.dataset_to_kwargs = dataset_to_kwargs
        super().__init__(module=module, optimizer=optimizer)

    def get_loss(self, pred: Predictions, y: torch.Tensor) -> torch.Tensor:
        return -pred.log_prob(y).mean()

    def _batch_to_args(self, batch: TimeSeriesDataset) -> Tuple[torch.Tensor, dict]:
        batch = batch.to(self._device)
        y = batch.tensors[0]

        if callable(self.dataset_to_kwargs):
            kwargs = self.dataset_to_kwargs(batch)
        else:
            if self.dataset_to_kwargs is None:
                self.dataset_to_kwargs = ['X'] if len(batch.tensors) > 1 else []

            kwargs = {}
            for i, (k, t) in enumerate(zip_longest(self.dataset_to_kwargs, batch.tensors[1:])):
                if k is None:
                    raise RuntimeError(
                        f"Found element-{i + 1} of the dataset.tensors, but `dataset_to_kwargs` doesn't have enough "
                        f"elements: {self.dataset_to_kwargs}"
                    )
                if t is None:
                    raise RuntimeError(
                        f"Found element-{i} of `dataset_to_kwargs`, but `dataset.tensors` doesn't have enough "
                        f"elements: {batch}"
                    )
                kwargs[k] = t
        return y, kwargs

    def _get_closure(self, batch: TimeSeriesDataset, forward_kwargs: dict) -> callable:

        def closure():
            # we call _batch_to_args from inside the closure in case `dataset_to_kwargs` is callable & involves grad.
            # only scenario this would matter is if optimizer is LBFGS (or another custom optimizer that calls closure
            # multiple times per step), in which case grad from callable would be lost after the first step.
            y, kwargs = self._batch_to_args(batch)
            kwargs.update(forward_kwargs)
            self.optimizer.zero_grad()
            pred = self.module(y, **kwargs)
            loss = self.get_loss(pred, y)
            loss.backward()
            return loss

        return closure

    def _get_batch_numel(self, batch: any) -> int:
        return batch.tensors[0].numel()


_warn_once = {}


def _default_getX(batch: TimeSeriesDataset) -> torch.Tensor:
    _, origX, *_other = batch.tensors
    if len(_other) and not _warn_once.get('warned', False):
        warnings.warn("Ignoring additional tensors in batch.")
        _warn_once['warned'] = True
    return origX


class ModelMatEmbeddingsTrainer(BaseTrainer):
    """
    This trainer is designed to train a :class:`torch.nn.Module` to take a model-matrix of predictors and place them in
    a lower-dimensional space such that a linear regression can effectively predict the target. This is useful in the
    context of state-space models, since those capture linear relationships.

    :param module: A ``torch.nn.Module`` that takes a model-matrix and returns a lower-dimensional embedding.
    :param loss_fn: A loss function. Default is :class:`torch.nn.MSELoss`.
    :param getX: A callable that takes a :class:`torchcast.utils.TimeSeriesDataset` and returns the original
     model-matrix tensor for passing to ``module``. If left unspecified, will assume the model-matrix is the 2nd
     tensor in ``batch.tensors``.
    """

    def __init__(self,
                 module: nn.Module,
                 loss_fn: callable = torch.nn.MSELoss(),
                 getX: Optional[Callable[[TimeSeriesDataset], torch.Tensor]] = None,
                 **kwargs):

        super().__init__(module=module, **kwargs)
        self.loss_fn = loss_fn
        self._warned = False
        self._getX = getX or _default_getX

    def get_loss(self, predictions: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(y)
        return self.loss_fn(predictions[~nan_mask], y[~nan_mask])

    def _getXy(self, batch: TimeSeriesDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self._device)
        y = batch.tensors[0]
        X = self._getX(batch)
        return X, y

    @staticmethod
    def _l2_solve(y: torch.Tensor,
                  X: torch.Tensor,
                  prior_precision: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Given tensors y,X in the format expected by ``StateSpaceModel`` -- 3D arrays with groups*times*measures --
        return the solutions to a linear-model for each group.

        :param y: A 3d tensor of time-series values.
        :param X: A 3d tensor of predictors.
        :param prior_precision: Optional. A penalty matrix.
        :return: The solution.
        """

        # handling nans requires flattening + scatter_add:
        num_groups, num_times, num_preds = X.shape
        X = X.view(-1, X.shape[-1])
        y = y.view(-1, y.shape[-1])
        is_valid = ~torch.isnan(y).squeeze()
        group_ids_broad = torch.repeat_interleave(torch.arange(num_groups, device=y.device), num_times)
        X = X[is_valid]
        y = y[is_valid]
        group_ids_broad = group_ids_broad[is_valid]

        # Xty:
        Xty_els = X * y
        Xty = (torch.zeros(num_groups, num_preds, dtype=y.dtype, device=y.device)
               .scatter_add(0, group_ids_broad.unsqueeze(-1).expand_as(Xty_els), Xty_els))

        # XtX:
        XtX = torch.stack([Xg.t() @ Xg for Xg, in chunk_grouped_data(X, group_ids=group_ids_broad.cpu())])
        XtXp = XtX
        if prior_precision is not None:
            if prior_precision.shape != (num_preds, num_preds):
                raise ValueError(f"prior_precision must have shape ({num_preds}, {num_preds}), but got shape "
                                 f"{prior_precision.shape} (did you remember bias?)")
            XtXp = XtXp + prior_precision

        return torch.linalg.solve(XtXp, Xty.unsqueeze(-1))

    @classmethod
    def _solve_and_predict(cls,
                           y: torch.Tensor,
                           X: torch.Tensor,
                           add_bias: bool = True,
                           prior_precision: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Given tensors y,X in the format expected by ``StateSpaceModel`` -- 3D arrays with groups*times*measures --
        solve the linear-model for each group, then generate predictions from these.

        :param y: A 3d tensor of time-series values.
        :param X: A 3d tensor of predictors.
        :param add_bias: Whether to add a bias-term.
        :param prior_precision: Optional. A penalty matrix.
        :return: A tensor with the same dimensions as ``y`` with the predictions.
        """
        if add_bias:
            X = torch.cat([torch.ones_like(X[..., :1]), X], -1)
        coefs = cls._l2_solve(y=y, X=X, prior_precision=prior_precision)
        return (coefs.transpose(-1, -2) * X).sum(-1).unsqueeze(-1)

    def _get_closure(self, batch: TimeSeriesDataset, forward_kwargs: dict) -> callable:
        X, y = self._getXy(batch)

        def closure():
            self.optimizer.zero_grad()
            emb = self.module(X, **forward_kwargs)
            outputs = self._solve_and_predict(y=y, X=emb)  # handles nans
            loss = self.get_loss(outputs, y)
            loss.backward()
            return loss

        return closure

    def _get_batch_numel(self, batch: any) -> int:
        return batch.tensors[0].numel()

    def predict(self, batch: TimeSeriesDataset) -> torch.Tensor:
        """
        Since this requires passing `y` it's not really useful genuine prediction, but is primarily for
        visualizing/sanity-checking outputs after/during training.
        """
        X, y = self._getXy(batch)
        emb = self.module(X)
        return self._solve_and_predict(y=y, X=emb)


class SeasonalEmbeddingsTrainer(ModelMatEmbeddingsTrainer):
    """
    This trainer teaches a module to embed complex seasonal patterns (e.g. cycles on the yearly, weekly, daily level)
    into a lower-dimensional space.
    """

    def __init__(self,
                 module: nn.Module,
                 yearly: int,
                 weekly: int,
                 daily: int,
                 other: Sequence[Tuple[np.timedelta64, int]] = (),
                 loss_fn: callable = torch.nn.MSELoss(),
                 **kwargs):

        super().__init__(module=module, loss_fn=loss_fn, **kwargs)

        self.weekly = weekly
        self.yearly = yearly
        self.daily = daily
        self.other = other

        self._warned = False

    def times_to_arrays(self, times: np.ndarray) -> Dict[str, torch.Tensor]:
        out = {}
        if self.yearly:
            out['yearly'] = torch.as_tensor(fourier_model_mat(times, K=self.yearly, period='yearly'))
        if self.weekly:
            out['weekly'] = torch.as_tensor(fourier_model_mat(times, K=self.weekly, period='weekly'))
        if self.daily:
            out['daily'] = torch.as_tensor(fourier_model_mat(times, K=self.daily, period='daily'))
        for period, k in self.other:
            out[str(period)] = torch.as_tensor(fourier_model_mat(times, K=k, period=period))
        return out

    def times_to_model_mat(self, times: np.ndarray) -> torch.Tensor:
        arrays = list(self.times_to_arrays(times).values())
        return torch.cat(arrays, -1)

    def _getXy(self, batch: TimeSeriesDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self._device)

        y, *_other = batch.tensors
        X = self.times_to_model_mat(batch.times()).to(dtype=y.dtype, device=self._device)

        if len(_other):
            X = torch.cat([X, _other[0]], -1)
        if len(_other) > 1 and not self._warned:
            warnings.warn("Ignoring additional tensors in batch.")
            self._warned = True
        return X, y
