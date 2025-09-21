from typing import List, Optional, Sequence, Union, TYPE_CHECKING, Callable
from warnings import warn

import numpy as np
import torch

from tqdm.auto import tqdm

from torchcast.internals.batch_design import TransitionModel, MeasurementModel, MeasureFun
from torchcast.internals.hessian import hessian
from torchcast.internals.utils import repeat, true1d_idx, get_nan_groups
from torchcast.covariance import Covariance
from torchcast.state_space.predictions import Predictions
from torchcast.state_space.adaptive_scaling import EWMAdaptiveScaler, AdaptiveScaler
from torchcast.process.regression import Process

if TYPE_CHECKING:
    from torchcast.utils.stopping import Stopping


class StateSpaceModel(torch.nn.Module):
    """
    Base-class for any :class:`torch.nn.Module` which generates predictions/forecasts using a state-space model.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    :param measure_funs: A dictionary mapping measure-names to measurement-functions. Currently only supports 'sigmoid'.
    :param adaptive_scaling: Experimental feature to adaptively scale the covariance as a function of residuals. This
     is useful if different groups have very different magnitudes.
    """

    def __init__(self,
                 processes: Sequence['Process'],
                 measures: Sequence[str],
                 measure_covariance: Optional[Covariance] = None,
                 measure_funs: Optional[dict[str, str]] = None,
                 adaptive_scaling: Union[bool, AdaptiveScaler] = False):
        super().__init__()

        # measures:
        assert isinstance(measures, (tuple, list)), "`measures` must be a list/tuple"
        self.measures = measures

        if measure_covariance is None:
            measure_covariance = Covariance.from_measures(measures)
        else:
            assert measure_covariance.rank == 1 or measure_covariance.rank == len(measures)
        self.measure_covariance = measure_covariance.set_id('measure_covariance')

        self.measure_funs = {}
        for m, alias in (measure_funs or {}).items():
            self.measure_funs[m] = MeasureFun.from_alias(alias)

        if adaptive_scaling is True:
            adaptive_scaling = EWMAdaptiveScaler(num_measures=self.measure_covariance.param_rank)
        self.adaptive_scaling = adaptive_scaling

        # processes:
        self.dt_unit = None
        self.processes = torch.nn.ModuleDict()
        for process in processes:
            if process.id in self.processes:
                raise ValueError(f"duplicate process id: {process.id}")
            self.processes[process.id] = process

            if process.has_measure:
                if process.measure not in measures:
                    raise ValueError(f"'{process.id}' has measure '{process.measure}' not in `measures`.")
            else:
                if len(measures) > 1:
                    raise ValueError(f"Must set measure for '{process.id}' since there are multiple measures.")
                process.measure = measures[0]

            if process.dt_unit:
                if self.dt_unit and self.dt_unit != process.dt_unit:
                    raise ValueError(
                        f"found multiple dt-units across processes: {self.dt_unit} and {process.dt_unit}"
                    )
                else:
                    self.dt_unit = process.dt_unit

    def forward(self,
                y: Optional[torch.Tensor] = None,
                n_step: Union[int, float] = 1,
                start_offsets: Optional[Sequence] = None,
                out_timesteps: Optional[Union[int, float]] = None,
                initial_state: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor, None] = None,
                every_step: bool = True,
                include_updates_in_output: bool = False,
                simulate: Optional[int] = None,
                prediction_kwargs: Optional[dict] = None,
                **kwargs) -> 'Predictions':
        """
        Generate n-step-ahead predictions from the model.

        :param y: A (group X time X measures) tensor. Optional if ``initial_state`` is specified.
        :param n_step: What is the horizon for the predictions output for each timepoint? Defaults to one-step-ahead
         predictions (i.e. n_step=1).
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``y``. If you passed ``dt_unit`` when constructing those processes, then you should pass an
         array of datetimes here. Otherwise you can pass an array of integers. Or leave ``None`` if there are no
         seasonal processes.
        :param out_timesteps: The number of timesteps to produce in the output. This is useful when passing a tensor
         of predictors that goes later in time than the `input` tensor -- you can specify ``out_timesteps=X.shape[1]``
         to get forecasts into this later time horizon.
        :param initial_state: The initial prediction for the state of the system. This is a tuple of mean, cov
         tensors you might extract from a previous call to forward (see ``include_updates_in_output`` below); you would
         have a ``Predictions`` object, which you can call :func:`get_state_at_times()` on. If left unset, will learn
         the initial state from the data. You can also pass a mean but not a cov, in situations where you want to
         predict the initial state mean but use the default cov.
        :param every_step: By default, ``n_step`` ahead predictions will be generated at every timestep. If
         ``every_step=False``, then these predictions will only be generated every `n_step` timesteps. For example,
         with hourly data, ``n_step=24`` and ``every_step=True``, each timepoint would be a forecast generated with
         data 24-hours in the past. But with ``every_step=False`` the first timestep would be 1-step-ahead, the 2nd
         would be 2-step-ahead, ... the 23rd would be 24-step-ahead, the 24th would be 1-step-ahead, etc. The advantage
         to ``every_step=False`` is speed: training data for long-range forecasts can be generated without requiring
         the model to produce and discard intermediate predictions every timestep.
        :param include_updates_in_output: If False, only the ``n_step`` ahead predictions are included in the output.
         This means that we cannot use this output to generate the ``initial_state`` for subsequent forward-passes. Set
         to True to allow this -- False by default to reduce memory.
        :param simulate: If specified, will generate `simulate` samples from the model.
        :param prediction_kwargs: A dictionary of kwargs to pass to initialize ``Predictions()``.
        :param kwargs: Further arguments passed to the `processes`. For example, the :class:`.LinearModel` expects an
         ``X`` argument for predictors.
        :return: A :class:`.Predictions` object with :func:`Predictions.log_prob()` and
         :func:`Predictions.to_dataframe()` methods.
        """

        if y is None:
            if out_timesteps is None:
                raise RuntimeError("If no y is passed, must specify `out_timesteps`")
        else:
            if not torch.is_floating_point(y):
                raise ValueError(f"Expected y to be a float tensor, got {y.dtype}")
            if torch.isinf(y).any():
                raise ValueError("y contains infinite values.")

        initial_state = self._prepare_initial_state(
            initial_state,
            start_offsets=start_offsets,
        )
        if simulate and simulate > 1:
            init_mean, init_cov = initial_state
            initial_state = repeat(init_mean, simulate, dim=0), repeat(init_cov, simulate, dim=0)
            if start_offsets is not None:  # need to repeat for passing to predictions.set_metadata
                start_offsets = repeat(np.asarray(start_offsets), simulate, dim=0)

            kwargs = {
                k: (repeat(v, simulate, dim=0) if isinstance(v, (torch.Tensor, np.ndarray)) else v)
                for k, v in kwargs.items()
            }

        if isinstance(n_step, float):
            if not n_step.is_integer():
                raise ValueError("`n_step` must be an int.")
            n_step = int(n_step)
        if isinstance(out_timesteps, float):
            if not out_timesteps.is_integer():
                raise ValueError("`out_timesteps` must be an int.")
            out_timesteps = int(out_timesteps)

        assert n_step > 0

        meanu, covu, inputs, num_groups, out_timesteps = self._standardize_input(
            y,
            initial_state,
            out_timesteps=out_timesteps
        )

        # # used by fit() to reduce unneeded computations:
        last_measured_per_group = kwargs.pop('last_measured_per_group', None)
        if last_measured_per_group is None:
            last_measured_per_group = torch.full((num_groups,), out_timesteps, dtype=torch.int, device=meanu.device)
        nan_groups = kwargs.pop('nan_groups', None)
        if nan_groups is None:
            nan_groups = [None] * out_timesteps
        # # /

        # todo: update Covariance class to make this less hacky:
        mcov_kwargs = {}
        if self.measure_covariance.expected_kwargs:
            mcov_kwargs = {k: kwargs[k] for k in self.measure_covariance.expected_kwargs}
        measure_covs = list(self.measure_covariance(mcov_kwargs, num_groups, out_timesteps).unbind(1))

        if self.adaptive_scaling:
            self.adaptive_scaling.reset()

        #
        predict_kwargs, update_kwargs, used_keys = self._parse_kwargs(
            num_groups=num_groups,
            num_timesteps=out_timesteps,
            measure_covs=measure_covs,
            **kwargs
        )
        used_keys.update(mcov_kwargs)

        transition_model = TransitionModel(
            processes=self.processes,
            measures=self.measures,
            num_groups=num_groups,
            num_timesteps=out_timesteps,
        )
        measurement_model = MeasurementModel(
            processes=self.processes,
            measures=self.measures,
            num_groups=num_groups,
            num_timesteps=out_timesteps,
            measure_funs=self.measure_funs,
            **kwargs
        )
        used_keys = used_keys.union(measurement_model.used_keys)
        unused_kwargs = set(kwargs) - used_keys
        if unused_kwargs:
            raise RuntimeError(f"Unexpected kwargs in {type(self).__name__}.forward(): {set(unused_kwargs)})")

        # first loop through to do predict -> update
        scaling1step = None
        scale1s = []
        meanus = []
        covus = []
        mean1s = []
        cov1s = []
        for t in range(out_timesteps):
            tmask = (t <= last_measured_per_group)
            mean1step, transition_mat = transition_model(meanu, time=t, mask=tmask)
            cov1step = self._predict_cov(
                cov=covu,
                transition_mat=transition_mat,
                **{k: v[t] for k, v in predict_kwargs.items()},
                scaling=scaling1step,
                mask=tmask
            )
            mean1s.append(mean1step)
            cov1s.append(cov1step)
            scale1s.append(scaling1step)

            if simulate:
                meanu = torch.distributions.MultivariateNormal(mean1step, cov1step, validate_args=False).sample()
                covu = torch.eye(meanu.shape[-1]).expand(num_groups, -1, -1) * 1e-6
            elif t < len(inputs):
                measured_mean, measure_mat = measurement_model(mean1step, time=t)
                measure_cov = self._apply_cov_scaling(measure_covs[t], scaling1step)
                meanu, covu = self._update_step_with_nans(
                    input=inputs[t],
                    mean=mean1step,
                    cov=cov1step,
                    measured_mean=measured_mean,
                    measure_mat=measure_mat,
                    measure_cov=measure_cov,
                    nan_groups=nan_groups[t],
                    **{k: v[t] for k, v in update_kwargs.items()}
                )
                scaling1step = self._get_scaling_multi(measured_mean, inputs[t])
            else:
                meanu, covu = mean1step, cov1step

            meanus.append(meanu)
            covus.append(covu)

        # 2nd loop to get n_step predicts:
        meanps = {}
        covps = {}
        for t1 in range(out_timesteps):
            # tu: time of update
            # t1: time of 1step
            tu = t1 - 1

            # - if every_step, we run this loop every iter
            # - if not every_step, we run this loop every nth iter
            if every_step or (t1 % n_step) == 0:
                meanp, covp, scaling = mean1s[t1], cov1s[t1], scale1s[t1]  # already had to generate h=1 above
                for h in range(1, n_step + 1):
                    tu_h = tu + h
                    if tu_h >= out_timesteps:
                        break
                    if h > 1:
                        tmask = (tu_h <= last_measured_per_group)
                        meanp, F = transition_model(meanp, time=tu_h, mask=tmask)
                        covp = self._predict_cov(
                            cov=covu,
                            transition_mat=F,
                            **{k: v[tu_h] for k, v in predict_kwargs.items()},
                            scaling=scaling,
                            mask=tmask
                        )
                    if tu_h not in meanps:
                        meanps[tu_h] = meanp
                        covps[tu_h] = covp
                        measure_covs[tu_h] = self._apply_cov_scaling(measure_covs[tu_h], scaling)
                    else:
                        raise NotImplementedError  # TODO: how do we hit this?

        preds = [meanps[t] for t in range(out_timesteps)], [covps[t] for t in range(out_timesteps)]

        if include_updates_in_output:
            updates = meanus, covus
        else:
            updates = None

        prediction_kwargs = prediction_kwargs or {}
        if 'mc_white_noise' not in prediction_kwargs and self.is_nonlinear:
            mc_samples = kwargs.pop('mc_samples', None)
            if mc_samples is None:
                warn("`mc_samples` not set, using 250 samples", UserWarning)
                mc_samples = 250
            prediction_kwargs['mc_white_noise'] = torch.randn(
                (mc_samples, self.state_rank + 1),  # TODO
                device=meanu.device,
                dtype=meanu.dtype
            )
        preds = self._generate_predictions(
            preds=preds,
            updates=updates,
            measure_covs=measure_covs,
            measurement_model=measurement_model,
            **prediction_kwargs
        )
        return preds.set_metadata(
            start_offsets=start_offsets if start_offsets is not None else np.zeros(num_groups, dtype='int'),
            dt_unit=self.dt_unit
        )

    def _apply_cov_scaling(self,
                           cov: torch.Tensor,
                           scaling: Optional[torch.Tensor],
                           is_process_cov: bool = False) -> torch.Tensor:

        if scaling is None:
            return cov
        assert scaling.shape[-1] == len(self.measures)

        if is_process_cov:
            assert cov.shape[-1] == self.state_rank
            if len(scaling.shape) == 1:
                scaling = scaling.unsqueeze(0)
            elif len(scaling.shape) != 2:
                raise ValueError(f"Expected scaling to be 1d or 2d, got {scaling.shape}")
            # for process cov, need to map from measures to states:
            scaling = torch.cat([
                scaling[..., [self.measures.index(process.measure)]].repeat(1, process.rank)
                for process in self.processes.values()
            ], dim=-1)
        else:
            # for measure-cov nothing extra to do
            assert cov.shape[-1] == len(self.measures)
        return cov * scaling.unsqueeze(-2) * scaling.unsqueeze(-1)

    def _get_scaling_multi(self,
                           measured_mean: torch.Tensor,
                           input: torch.Tensor) -> Optional[torch.Tensor]:

        if self.adaptive_scaling:
            nan_mask = input.isnan()
            resid = input.nan_to_num() - measured_mean
            multi = self.adaptive_scaling(resid, nan_mask)

            # Handle empty measures (those not in the covariance structure)
            multi_padded = torch.ones_like(resid)
            full_idx = [i for i in range(resid.shape[-1]) if i not in self.measure_covariance.empty_idx]
            multi_padded[..., full_idx] = multi[..., full_idx]
            return multi_padded
        else:
            return None

    def fit(self,
            y: torch.Tensor,
            optimizer: Union[torch.optim.Optimizer, Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer]] = None,
            stopping: Union['Stopping', dict] = None,
            verbose: int = 2,
            callbacks: Sequence[callable] = (),
            get_loss: Optional[callable] = None,
            callable_kwargs: Optional[dict[str, callable]] = None,
            set_initial_values: bool = True,
            **kwargs):
        """
        A high-level interface for fitting a state-space model when all the training data fits in memory. If your data
        does not fit in memory, consider :class:`torchcast.utils.training.StateSpaceTrainer` or tools like pytorch
        lightning.

        :param y: A tensor containing the batch of time-series(es), see :func:`StateSpaceModel.forward()`.
        :param optimizer: The optimizer to use. Can also pass a function which takes the parameters and returns an
         optimizer instance. Default is :class:`torch.optim.LBFGS` with ``(line_search_fn='strong_wolfe', max_iter=1)``.
        :param stopping: Controls stopping/convergence rules; should be a :class:`torchcast.utils.Stopping` instance, or
         a dict of keyword-args to one. Example: ``stopping={'abstol' : .001, 'monitor' : 'params'}``
        :param verbose: If True (default) will print the loss and epoch.
        :param callbacks: A list of functions that will be called at the end of each epoch, which take the current
         epoch's loss value.
        :param get_loss: A function that takes the ``Predictions` object and the input data and returns the loss.
         Default is ``lambda pred, y: -pred.log_prob(y).mean()``.
        :param set_initial_values: If True, will set the initial mean to sensible value given ``y``, which helps speed
         up training if the data are not centered. Set to False if you are resuming fit on a partially fitted model.
        :param kwargs: Further keyword-arguments passed to :func:`StateSpaceModel.forward()`; but see also
         ``callable_kwargs``.
        :param callable_kwargs: The kwargs passed to the forward pass are static, but sometimes you want to recompute
         them each iteration -- indeed, this is required in some cases by how pytorch's autograd works. The values in
         this dictionary are no-argument functions that will be called each iteration to recompute the corresponding
         arguments.
        :return: This ``StateSpaceModel`` instance.
        """

        if callable(optimizer):
            optimizer = optimizer([p for p in self.parameters() if p.requires_grad])
        elif optimizer is None:
            optimizer = torch.optim.LBFGS(
                [p for p in self.parameters() if p.requires_grad],
                # https://discuss.pytorch.org/t/unclear-purpose-of-max-iter-kwarg-in-the-lbfgs-optimizer/65695/4
                max_iter=1,
                line_search_fn='strong_wolfe'
            )

        if set_initial_values:
            self._set_initial_values(y, verbose=verbose > 1, **kwargs)
            if self.adaptive_scaling:
                self.adaptive_scaling.initialize(y.shape[1])

        _deprecated = {k: kwargs.pop(k) for k in ['tol', 'patience', 'max_iter'] if k in kwargs}
        _dmsg = f"The following are deprecated, use `stopping` arg instead:\n{set(_deprecated)}"
        if stopping is None:
            stopping = {}
            if _deprecated:
                warn(_dmsg, DeprecationWarning)
                stopping.update(_deprecated)
        elif _deprecated:
            raise ValueError(_dmsg)
        from torchcast.utils.stopping import Stopping
        if not isinstance(stopping, Stopping):
            stopping = Stopping.from_dict(**stopping)
        stopping.module = self

        prog = tqdm(disable=True)
        if verbose > 1:
            prog = tqdm()
        callable_kwargs = callable_kwargs or {}

        kwargs = self._prepare_fit_kwargs(y, **kwargs)

        if get_loss is None:
            # precompute nan-groups instead of doing it on each call to log_prob:
            nan_groups_flat = get_nan_groups(torch.isnan(y).reshape(-1, y.shape[-1]))
            get_loss = lambda _pred, _y: -_pred.log_prob(_y, nan_groups_flat=nan_groups_flat).mean()

        closure = _OptimizerClosure(
            ss_model=self,
            y=y,
            get_loss=get_loss,
            prog=prog,
            callable_kwargs=callable_kwargs,
            optimizer=optimizer,
            stopping=stopping,
            kwargs=kwargs,
        )

        train_loss = float('nan')
        for epoch in range(stopping.max_iter):
            try:
                prog.reset()
                prog.set_description(f"Epoch {epoch:,}; Loss {train_loss:.4}; Convergence {stopping.convergence}")
                train_loss = optimizer.step(closure).item()
                for callback in callbacks:
                    callback(train_loss)

                if stopping(train_loss):
                    break
            except KeyboardInterrupt:
                break
            finally:
                optimizer.zero_grad(set_to_none=True)

        return self

    @property
    def is_nonlinear(self) -> bool:
        return any(not p.linear_measurement for p in self.processes.values()) or self.measure_funs

    def _prepare_fit_kwargs(self,
                            y: torch.Tensor,
                            **kwargs) -> dict:
        # precompute nan-groups for forward pass
        isnan = torch.isnan(y)
        kwargs['nan_groups'] = [get_nan_groups(isnan_t) for isnan_t in isnan.unbind(1)]

        #
        prediction_kwargs = kwargs.pop('prediction_kwargs', None) or {}
        # monte-carlo for Predictions.log_prob:
        mc_samples = kwargs.pop('mc_samples', None)
        if self.is_nonlinear and not mc_samples:
            raise ValueError("Nonlinear state-space models require `mc_samples` to be set.")
        if mc_samples:
            if 'mc_white_noise' not in prediction_kwargs:
                prediction_kwargs['mc_white_noise'] = torch.randn(
                    (mc_samples, self.state_rank + 1),  # TODO
                    device=y.device,
                    dtype=y.dtype
                )
        kwargs['prediction_kwargs'] = prediction_kwargs

        # see `last_measured_per_group` in forward docstring
        # todo: duplicate code in ``TimeSeriesDataset.get_durations()``
        any_measured_bool = ~torch.isnan(y).all(2).cpu()
        kwargs['last_measured_per_group'] = torch.as_tensor(
            [np.max(true1d_idx(any_measured_bool[g]).numpy(), initial=0) for g in range(y.shape[0])],
            dtype=torch.int,
            device=y.device
        ) + 1

        return kwargs

    def _generate_predictions(self,
                              preds: tuple[list[torch.Tensor], list[torch.Tensor]],
                              updates: Optional[tuple[list[torch.Tensor], list[torch.Tensor]]],
                              measure_covs: torch.Tensor,
                              measurement_model: 'MeasurementModel',
                              nan_groups: Optional[List[Sequence[tuple[torch.Tensor, Optional[torch.Tensor]]]]] = None,
                              mc_white_noise: Optional[torch.Tensor] = None,
                              **kwargs
                              ) -> 'Predictions':
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unexpected kwargs: {set(kwargs)})")
        return Predictions(
            measurement_model=measurement_model,
            states=preds,
            measure_covs=measure_covs,
            updates=updates,
            mc_white_noise=mc_white_noise,
        )

    def _parse_kwargs(self,
                      num_groups: int,
                      num_timesteps: int,
                      measure_covs: Sequence[torch.Tensor],
                      **kwargs) -> tuple[dict[str, Sequence], dict[str, Sequence], set]:
        """
        Parse keyword arguments into:
        - predict-kwargs (e.g. K or Q)
        - update-kwargs (?)
        - used keys (the kwarg-keys that were used, so later we can confirm nothing was unused)
        """
        predict_kwargs = {}
        update_kwargs = {}

        return predict_kwargs, update_kwargs, set()

    def _standardize_input(self,
                           input: Optional[torch.Tensor],
                           initial_state: tuple[torch.Tensor, torch.Tensor],
                           out_timesteps: Optional[int] = None):

        meanu, covu = initial_state

        if input is None:
            if out_timesteps is None:
                raise RuntimeError("If `input` is None must pass `out_timesteps`")
            inputs = []

            num_groups = meanu.shape[0]

            if covu.shape[0] == 1:
                covu = repeat(covu, times=num_groups, dim=0)
        else:
            if len(input.shape) != 3:
                raise ValueError(f"Expected len(input.shape) == 3 (group,time,measure)")
            if input.shape[-1] != len(self.measures):
                raise ValueError(f"Expected input.shape[-1] == {len(self.measures)} (len(self.measures))")

            num_groups = input.shape[0]
            if meanu.shape[0] == 1:
                meanu = meanu.expand(num_groups, -1)
            if covu.shape[0] == 1:
                covu = covu.expand(num_groups, -1, -1)

            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)
        return meanu, covu, inputs, num_groups, out_timesteps

    def _predict_cov(self,
                     cov: torch.Tensor,
                     transition_mat: torch.Tensor,
                     scaling: torch.Tensor,
                     mask: Optional[torch.Tensor],
                     **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _update_step_with_nans(self,
                               input: torch.Tensor,
                               mean: torch.Tensor,
                               cov: torch.Tensor,
                               measured_mean: torch.Tensor,
                               measure_mat: torch.Tensor,
                               measure_cov: torch.Tensor,
                               nan_groups: Optional[Sequence[tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
                               **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if nan_groups is None:
            nan_groups = get_nan_groups(torch.isnan(input))
        if len(nan_groups) == 1:
            group_idx, masks = nan_groups[0]
            if len(group_idx) == len(input) and masks is None:
                # no nans, no masking:
                return self._update_step(
                    input=input,
                    mean=mean,
                    cov=cov,
                    measured_mean=measured_mean,
                    measure_mat=measure_mat,
                    measure_cov=measure_cov,
                    **kwargs
                )
        elif not len(nan_groups):
            # all nans, nothing to do:
            return mean, cov

        new_mean = mean.clone()
        new_cov = cov.clone()
        for groups, masks in nan_groups:
            masked = self._mask_mats(
                groups,
                masks,
                input=input,
                measured_mean=measured_mean,
                measure_mat=measure_mat,
                measure_cov=measure_cov,
                **kwargs
            )

            new_mean[groups], new_cov[groups] = self._update_step(
                mean=mean[groups],
                cov=cov[groups],
                **masked,
                **{k: v for k, v in kwargs.items() if k not in masked}
            )
        return new_mean, new_cov

    def _mask_mats(self,
                   groups: torch.Tensor,
                   masks: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                   **kwargs) -> dict[str, torch.Tensor]:
        out = {}
        if masks is None:
            for nm, mat in kwargs.items():
                out[nm] = mat[groups]
        else:
            val_idx, m1d, m2d = masks
            for nm, mat in kwargs.items():
                if nm in ('input', 'measured_mean', 'measure_mat'):
                    out[nm] = mat[m1d]
                elif nm == 'measure_cov':
                    out[nm] = mat[m2d]
        return out

    def _update_step(self,
                     input: torch.Tensor,
                     mean: torch.Tensor,
                     cov: torch.Tensor,
                     measured_mean: torch.Tensor,
                     measure_mat: torch.Tensor,
                     measure_cov: torch.Tensor,
                     **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _mean_update(mean: torch.Tensor, K: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
        return mean + (K @ resid.unsqueeze(-1)).squeeze(-1)

    def _set_initial_values(self, y: torch.Tensor, verbose: bool = True, **kwargs):
        intercept_proceses = {}
        for process in self.processes.values():
            if process.intercept_state_element:
                if process.measure in intercept_proceses:
                    warn(
                        f"Multiple processes have intercept_state_element for measure '{process.measure}'; "
                        f"will not set initial values."
                    )
                    return
                intercept_proceses[process.measure] = process

        for measure in self.measures:
            if measure not in intercept_proceses:
                warn(
                    f"No process has `intercept_state_element` for measure '{measure}'; "
                    f"will not set initial values."
                )
                return

        for measure, process in intercept_proceses.items():
            value = self._get_good_initial_value_from_y(y, measure, **kwargs)
            if verbose:
                print(
                    f"For measure {measure}, setting initial value by setting "
                    f"'{process.id}.{process.intercept_state_element}' to {value:.4f}"
                )
            process.update_intercept(value)

    @torch.no_grad()
    def _get_good_initial_value_from_y(self,
                                       y: torch.Tensor,
                                       measure: str,
                                       **kwargs) -> torch.Tensor:

        midx = self.measures.index(measure)
        mean = torch.nanmean(y[..., midx])
        measure_fun = self.measure_funs.get(measure, None)
        if measure_fun:
            return measure_fun.inverse_transform(mean)
        return mean

    def _prepare_initial_state(self,
                               initial_state: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor, None],
                               start_offsets: Optional[Sequence] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(initial_state, torch.Tensor):
            initial_state = (initial_state, None)

        if initial_state is None:
            init_mean = (p.get_initial_mean(start_offsets) for p in self.processes.values())
            init_mean = [m if len(m.shape) == 2 else m.expand(1, -1) for m in init_mean]
            ngroups = max(m.shape[0] for m in init_mean)
            init_mean = torch.cat([m.expand(ngroups, -1) for m in init_mean], -1)
            init_cov = self.initial_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[:, 0]
        else:
            # TODO: we don't call `get_initial_mean` when initial_state is passed...
            #   this makes sense in some contexts -- e.g. a seasonal process from a previous call to forward() --
            #   but it is also bad in some contexts -- e.g. we have a model that predicts initial state, but we still
            #   need seasonal processes to evolve it.
            init_mean, init_cov = initial_state
            if len(init_mean.shape) != 2:
                raise ValueError(
                    f"Expected ``init_mean`` to have two-dimensions for (num_groups, state_dim), got {init_mean.shape}"
                )
            if init_cov is None:
                init_cov = self.initial_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[:, 0]
            if len(init_cov.shape) != 3:
                raise ValueError(
                    f"Expected ``init_cov`` to be 3-D with (num_groups, state_dim, state_dim), got {init_cov.shape}"
                )

        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        measure_std = mcov.diagonal(dim1=-2, dim2=-1).sqrt()
        for idx in self.measure_covariance.empty_idx:
            measure_std[idx] = torch.ones_like(measure_std[idx])  # empty measures have no variance, so set to 1
        init_cov = self._apply_cov_scaling(init_cov, scaling=measure_std, is_process_cov=True)

        return init_mean, init_cov

    @property
    def state_rank(self) -> int:
        return sum(p.rank for p in self.processes.values())

    def get_laplace_mvnorm(self,
                           y: torch.Tensor,
                           get_loss: Optional[callable] = None,
                           **kwargs) -> tuple[torch.distributions.MultivariateNormal, List[str]]:
        """
        :param y: observed data
        :param get_loss: A function that takes the ``Predictions`` object and the input data and returns the loss; note
         that unlike in :func:`fit()`, this function should return the summed loss (not mean). Default is just
         ``-pred.log_prob(y).sum()``, but you can override (e.g. for weights).
        :param kwargs: Keyword-arguments to the forward pass.
        :return: The multivariate normal distribution for the Laplace approximation, and the corresponding names of the
         parameters.
        """
        if not get_loss:
            get_loss = lambda _pred, _y: -_pred.log_prob(_y).sum()

        kwargs = self._prepare_fit_kwargs(y, **kwargs)

        pred = self(y, **kwargs)
        loss = get_loss(pred, y)

        all_params = []
        all_param_names = []
        for nm, par in self.named_parameters():
            if not par.requires_grad:
                continue
            all_param_names.extend(f'{nm}[{i}]' for i in range(par.numel()))
            all_params.append(par)
        # TODO: any way to verify reshape(-1) matches internals of hessian?
        means = torch.cat([p.reshape(-1) for p in all_params])

        hess = hessian(output=loss.squeeze(), inputs=all_params, allow_unused=True, progress=False)

        # create mvnorm for laplace approx:
        with torch.no_grad():
            try:
                mvnorm = torch.distributions.MultivariateNormal(
                    means, precision_matrix=hess, validate_args=True
                )
            except (RuntimeError, ValueError) as e:
                warn(
                    f"Unable to get valid covariance from optimized parameters (see error below)."
                    f"If you haven't already, fit the model with ``monitor_params=True`` (see the ``stopping`` argument"
                    f" of ``fit()``)."
                    f"\n{str(e)}"
                )
                fake_cov = torch.diag(torch.diag(hess).pow(-1).clip(min=1E-5))
                mvnorm = torch.distributions.MultivariateNormal(means, covariance_matrix=fake_cov)

        return mvnorm, all_param_names

    @torch.no_grad()
    def simulate(self,
                 out_timesteps: int,
                 initial_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                 start_offsets: Optional[Sequence] = None,
                 num_sims: int = 1,
                 num_groups: Optional[int] = None,
                 **kwargs):
        """
        Generate simulated state-trajectories from your model.

        :param out_timesteps: The number of timesteps to generate in the output.
        :param initial_state: The initial state of the system: a tuple of `mean`, `cov`. Can be obtained from previous
         model-predictions by calling ``get_state_at_times()`` on the output predictions.
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``initial_state``. If you passed ``dt_unit`` when constructing those processes, then you should
         pass an array of datetimes here, otherwise an array of ints. If there are no seasonal processes you can omit.
        :param num_sims: The number of state-trajectories to simulate per group. The output will be laid out in blocks
         (e.g. if there are 10 groups, the first ten elements of the output are sim 1, the next 10 elements are sim 2,
         etc.). Tensors associated with this output can be reshaped with ``tensor.reshape(num_sims, num_groups, ...)``.
        :param num_groups: The number of groups; if `None` will be inferred from the shape of `initial_state` and/or
         ``start_offsets``.
        :param kwargs: Further arguments passed to the `processes`.
        :return: A :class:`.Predictions` object with zero state-covariance.
        """

        if num_groups is not None:
            if start_offsets is not None and len(start_offsets) != num_groups:
                raise ValueError("Expected `len(start_offsets) == num_groups` (or num_groups=None)")
            if isinstance(initial_state, torch.Tensor):
                initial_state = (initial_state, None)
            if initial_state is None:
                initial_state = self._prepare_initial_state(initial_state, start_offsets=start_offsets)
                initial_state = (repeat(x, times=num_groups, dim=0) for x in initial_state)
            elif len(initial_state[0]) != num_groups:
                raise ValueError("Expected `initial_state` to have first dimension equal to `num_groups`")

        return self(
            start_offsets=start_offsets,
            out_timesteps=out_timesteps,
            initial_state=initial_state,
            simulate=num_sims,
            **kwargs
        )


def default_get_loss(pred: 'Predictions', y: torch.Tensor, **kwargs) -> torch.Tensor:
    return -pred.log_prob(y, **kwargs).mean()


class _OptimizerClosure:

    def __init__(self,
                 ss_model: StateSpaceModel,
                 y: torch.Tensor,
                 optimizer: torch.optim.Optimizer,
                 prog: tqdm,
                 stopping: 'Stopping',
                 kwargs: dict,
                 callable_kwargs: dict[str, callable],
                 get_loss: callable):
        self.ss_model = ss_model
        self.y = y
        self.optimizer = optimizer
        self.prog = prog
        self.stopping = stopping
        self.kwargs = kwargs
        self.callable_kwargs = callable_kwargs
        self.get_loss = get_loss

    def __call__(self):
        self.optimizer.zero_grad()
        self.kwargs.update({k: v() for k, v in self.callable_kwargs.items()})
        pred = self.ss_model(self.y, **self.kwargs)
        loss = self.get_loss(pred, self.y)
        loss.backward()
        self.prog.update()
        self.prog.set_description(
            f"Epoch {self.stopping.epoch:,}; Loss {loss.item():.4}; Convergence {self.stopping.convergence}"
        )
        return loss
