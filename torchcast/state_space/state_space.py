from typing import List, Optional, Sequence, Union, TYPE_CHECKING, Callable
from warnings import warn

import numpy as np
import torch

from tqdm.auto import tqdm

from torchcast.internals.batch_design import TransitionModel, MeasurementModel
from torchcast.internals.hessian import hessian
from torchcast.internals.utils import repeat, true1d_idx, get_nan_groups, mask_mats
from torchcast.covariance import Covariance
from torchcast.state_space.predictions import Predictions
from torchcast.process.regression import Process

if TYPE_CHECKING:
    from torchcast.utils.stopping import Stopping


class StateSpaceModel(torch.nn.Module):
    """
    Base-class for any :class:`torch.nn.Module` which generates predictions/forecasts using a state-space model.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    """

    def __init__(self,
                 processes: Sequence['Process'],
                 measures: Optional[Sequence[str]],
                 measure_covariance: Optional[Covariance] = None,
        super().__init__()

        # measures:
        assert isinstance(measures, (tuple, list)), "`measures` must be a list/tuple"
        self.measures = measures

        if measure_covariance is None:
            measure_covariance = Covariance.from_measures(measures)
        else:
            assert measure_covariance.rank == 1 or measure_covariance.rank == len(measures)
        self.measure_covariance = measure_covariance.set_id('measure_covariance')
        self.adaptive_measure_var = adaptive_measure_var

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

    @torch.jit.ignore()
    def forward(self,
                y: Optional[torch.Tensor] = None,
                n_step: Union[int, float] = 1,
                start_offsets: Optional[Sequence] = None,
                out_timesteps: Optional[Union[int, float]] = None,
                initial_state: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor, None] = None,
                every_step: bool = True,
                include_updates_in_output: bool = False,
                simulate: Optional[int] = None,
                last_measured_per_group: Optional[torch.Tensor] = None,
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
        :param last_measured_per_group: This provides a method to reduce unused computations in training. On each call
         to forward in training, you can supply to this argument a tensor indicating the last measured timestep for
         each group in the batch (this can be computed with ``last_measured_per_group=batch.get_durations()``, where
         ``batch`` is a :class:`TimeSeriesDataset`). In this case, predictions will not be generated after the
         specified timestep for each group; these can be discarded in training because, without any measurements, they
         wouldn't have been used in loss calculations anyways. Naturally this should never be set for
         inference/forecasting. This will automatically be set when calling ``fit()``, but if you're instread using a
         custom training loop, you can pass this manually.
        :param simulate: If specified, will generate `simulate` samples from the model.
        :param prediction_kwargs: A dictionary of kwargs to pass to initialize ``Predictions()``. Unused for base
         class, but can be used by subclasses (e.g. ``BinomialFilter``).
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

        if last_measured_per_group is None:
            last_measured_per_group = torch.full((num_groups,), out_timesteps, dtype=torch.int, device=meanu.device)

        # todo: update Covariance class to make this less hacky:
        mcov_kwargs = {}
        if self.measure_covariance.expected_kwargs:
            mcov_kwargs = {k: kwargs[k] for k in self.measure_covariance.expected_kwargs}
        measure_covs = self.measure_covariance(mcov_kwargs, num_groups, out_timesteps).unbind(1)

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
            **kwargs
        )
        used_keys = used_keys.union(measurement_model.used_keys)
        unused_kwargs = set(kwargs) - used_keys
        if unused_kwargs:
            raise RuntimeError(f"Unexpected kwargs in {type(self).__name__}.forward(): {set(unused_kwargs)})")

        # first loop through to do predict -> update
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
                mask=tmask
            )
            mean1s.append(mean1step)
            cov1s.append(cov1step)

            if simulate:
                meanu = torch.distributions.MultivariateNormal(mean1step, cov1step, validate_args=False).sample()
                covu = torch.eye(meanu.shape[-1]).expand(num_groups, -1, -1) * 1e-6
            elif t < len(inputs):
                measured_mean, measure_mat = measurement_model(mean1step, time=t)
                meanu, covu = self._update_step_with_nans(
                    input=inputs[t],
                    mean=mean1step,
                    cov=cov1step,
                    measured_mean=measured_mean,
                    measure_mat=measure_mat,
                    measure_cov=measure_covs[t],
                    **{k: v[t] for k, v in update_kwargs.items()}
                )
            else:
                meanu, covu = mean1step, cov1step

            meanus.append(meanu)
            covus.append(covu)

        # 2nd loop to get n_step predicts:
        # idx: Dict[int, int] = {}
        meanps = {}
        covps = {}
        for t1 in range(out_timesteps):
            # tu: time of update
            # t1: time of 1step
            tu = t1 - 1

            # - if every_step, we run this loop every iter
            # - if not every_step, we run this loop every nth iter
            if every_step or (t1 % n_step) == 0:
                meanp, covp = mean1s[t1], cov1s[t1]  # already had to generate h=1 above
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
                            mask=tmask
                        )
                    if tu_h not in meanps:
                        # idx[tu + h] = tu
                        meanps[tu_h] = meanp
                        covps[tu_h] = covp

        preds = [meanps[t] for t in range(out_timesteps)], [covps[t] for t in range(out_timesteps)]

        if include_updates_in_output:
            updates = meanus, covus
        else:
            updates = None

        prediction_kwargs = prediction_kwargs or {}
        preds = self._generate_predictions(preds, updates, measure_covs, measurement_model, **prediction_kwargs)
        return preds.set_metadata(
            start_offsets=start_offsets if start_offsets is not None else np.zeros(num_groups, dtype='int'),
            dt_unit=self.dt_unit
        )

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
            self._set_initial_values(y, verbose=verbose > 1)

        if not get_loss:
            get_loss = lambda _pred, _y: -_pred.log_prob(_y).mean()

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

        def closure():
            optimizer.zero_grad()
            kwargs.update({k: v() for k, v in callable_kwargs.items()})
            pred = self(y, **kwargs)
            loss = get_loss(pred, y)
            loss.backward()
            prog.update()
            prog.set_description(f"Epoch {epoch:,}; Loss {loss.item():.4}; Convergence {stopping.convergence}")
            return loss

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

    def _prepare_fit_kwargs(self, y: torch.Tensor, **kwargs) -> dict:
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
                              **kwargs
                              ) -> 'Predictions':
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unexpected kwargs: {set(kwargs)})")
        return Predictions(
            measurement_model=measurement_model,
            states=preds,
            measure_covs=measure_covs,
            updates=updates,
            **kwargs
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
                     mask: torch.Tensor,
                     **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _update_mat_dims(self) -> dict:
        """
        subclasses can specify dimensionality of optional update matrices
        """
        return {}

    def _update_step_with_nans(self,
                               input: torch.Tensor,
                               mean: torch.Tensor,
                               cov: torch.Tensor,
                               measured_mean: torch.Tensor,
                               measure_mat: torch.Tensor,
                               measure_cov: torch.Tensor,
                               **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        isnan = torch.isnan(input)
        if isnan.all():
            return mean, cov
        if isnan.any():
            mats = [
                ('input', input, (-1,)),
                ('measured_mean', measured_mean, (-1,)),
                ('measure_mat', measure_mat, (-1,)),
                ('measure_cov', measure_cov, (-2, -1))
            ]
            for k, dim in self._update_mat_dims.items():
                mats.append((k, kwargs[k], dim))
            new_mean = mean.clone()
            new_cov = cov.clone()
            for groups, val_idx in get_nan_groups(isnan):
                masked = mask_mats(groups, val_idx, mats=mats)
                new_mean[groups], new_cov[groups] = self._update_step(mean=mean[groups], cov=cov[groups], **masked)
            return new_mean, new_cov
        else:
            return self._update_step(
                input=input,
                mean=mean,
                cov=cov,
                measured_mean=measured_mean,
                measure_mat=measure_mat,
                measure_cov=measure_cov,
                **kwargs
            )

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

    def _set_initial_values(self, y: torch.Tensor, verbose: bool = True):
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
            value = self._get_good_initial_value_from_y(y, measure)
            if verbose:
                print(
                    f"For measure {measure}, setting initial value by setting "
                    f"'{process.id}.{process.intercept_state_element}' to to {value:.4f}"
                )
            process.update_intercept(value)

    @torch.no_grad()
    def _get_good_initial_value_from_y(self, y: torch.Tensor, measure: str) -> torch.Tensor:
        # TODO: measure funs
        midx = self.measures.index(measure)
        return torch.nanmean(y[..., midx])

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

        return init_mean, init_cov

    @property
    def state_rank(self) -> int:
        return sum(p.rank for p in self.processes.values())

    def _get_measure_scaling(self) -> torch.Tensor:
        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        measure_var = mcov.diagonal(dim1=-2, dim2=-1).unbind()

        multi = [
            measure_var[self.measures.index(process.measure)].expand(process.rank).sqrt()
            for process in self.processes.values()
        ]
        multi = torch.cat(multi)
        if (multi <= 0).any():
            raise RuntimeError(f"measure-cov diagonal is not positive:{measure_var}")
        return multi

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
