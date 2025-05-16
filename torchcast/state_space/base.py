from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable, Callable, Union, Type, TYPE_CHECKING
from warnings import warn

import numpy as np
import torch
from torch import nn, Tensor
from tqdm.auto import tqdm

from torchcast.internals.hessian import hessian
from torchcast.internals.utils import get_owned_kwargs, repeat, identity, true1d_idx
from torchcast.covariance import Covariance
from torchcast.state_space.predictions import Predictions
from torchcast.state_space.ss_step import StateSpaceStep
from torchcast.process.regression import Process

if TYPE_CHECKING:
    from torchcast.utils.stopping import Stopping


class StateSpaceModel(nn.Module):
    """
    Base-class for any :class:`torch.nn.Module` which generates predictions/forecasts using a state-space model.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    """
    ss_step_cls: Type[StateSpaceStep]
    _dt_unit = None

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]],
                 measure_covariance: Covariance,
                 **kwargs):
        super().__init__()

        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        self._validate(processes, measures)

        self.measure_covariance = measure_covariance
        if self.measure_covariance:
            self.measure_covariance.set_id('measure_covariance')

        # measures:
        self.measures = measures
        self.measure_to_idx = {m: i for i, m in enumerate(self.measures)}

        # processes:
        self.processes = nn.ModuleDict()
        self.process_to_slice: Dict[str, Tuple[int, int]] = {}
        self.state_rank = 0
        for p in processes:
            assert p.measure, f"{p.id} does not have its `measure` set"
            self.processes[p.id] = p
            self.process_to_slice[p.id] = (self.state_rank, self.state_rank + len(p.state_elements))
            self.state_rank += len(p.state_elements)

        # the initial mean
        self.initial_mean = torch.nn.Parameter(.1 * torch.randn(self.state_rank))

        self.ss_step = self.ss_step_cls(**self._get_ss_step_kwargs(**kwargs))

    def _get_ss_step_kwargs(self) -> dict:
        # child classes can take keyword-arguments
        return {}

    def _infer_dt_unit(self) -> Optional[np.timedelta64]:
        dt_unit_ns = None
        proc_with_dt = ''
        for p in self.processes.values():
            if hasattr(p, 'dt_unit_ns'):
                if dt_unit_ns is None:
                    dt_unit_ns = p.dt_unit_ns
                    proc_with_dt = p.id
                elif p.dt_unit_ns != dt_unit_ns:
                    raise ValueError(
                        f"Found multiple processes with different dt_units:"
                        f"{proc_with_dt}: {dt_unit_ns}"
                        f"{p.id}: {p.dt_unit_ns}"
                    )
        if dt_unit_ns is not None:
            return np.timedelta64(dt_unit_ns, 'ns')  # todo: promote

    @property
    def dt_unit(self) -> Optional[np.timedelta64]:
        if self._dt_unit is None:
            return self._infer_dt_unit()
        return self._dt_unit

    @dt_unit.setter
    def dt_unit(self, value: np.timedelta64):
        inferred = self._infer_dt_unit()
        if inferred is not None and inferred != value:
            raise ValueError(f"Cannot set `dt_unit` to {value} when it is already inferred to be {inferred}")
        else:
            self._dt_unit = value

    @torch.jit.ignore()
    def fit(self,
            *args,
            optimizer: Union[torch.optim.Optimizer, Callable[[Sequence[torch.Tensor]], torch.optim.Optimizer]] = None,
            stopping: Union['Stopping', dict] = None,
            verbose: int = 2,
            callbacks: Sequence[Callable] = (),
            get_loss: Optional[Callable] = None,
            callable_kwargs: Optional[Dict[str, Callable]] = None,
            set_initial_values: bool = True,
            **kwargs):
        """
        A high-level interface for invoking the standard model-training boilerplate. This is helpful to common cases in
        which the number of parameters is moderate and the data fit in memory. For other cases you are encouraged to
        roll your own training loop.

        :param args: A tensor containing the batch of time-series(es), see :func:`StateSpaceModel.forward()`.
        :param optimizer: The optimizer to use. Can also pass a function which takes the parameters and returns an
         optimizer instance. Default is :class:`torch.optim.LBFGS` with ``(line_search_fn='strong_wolfe', max_iter=1)``.
        :param stopping: Controls stopping/convergence rules; should be a :class`torchcast.utils.Stopping` instance, or
         a dict of keyword-args to one. Example: ``stopping={'abstol' : .001, 'monitor' : 'params'}``
        :param verbose: If True (default) will print the loss and epoch; for :class:`torch.optim.LBFGS` optimizer
         (the default) this progress bar will tick within each epoch to track the calls to forward.
        :param callbacks: A list of functions that will be called at the end of each epoch, which take the current
         epoch's loss value.
        :param get_loss: A function that takes the ``Predictions` object and the input data and returns the loss.
         Default is ``lambda pred, y: -pred.log_prob(y).mean()``.
        :param set_initial_values: Will set ``initial_mean`` to sensible value given ``y``, which helps speed
         up training if the data are not centered.
        :param kwargs: Further keyword-arguments passed to :func:`StateSpaceModel.forward()`.
        :param callable_kwargs: The kwargs passed to the forward pass are static, but sometimes you want to recompute
         them each iteration. The values in this dictionary are functions that will be called each iteration to
         recompute the corresponding arguments.
        :return: This ``StateSpaceModel`` instance.
        """

        # this may change in the future
        assert len(args) == 1
        y = args[0]

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
            self.set_initial_values(y, verbose=verbose > 1, **kwargs)

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

    def _prepare_fit_kwargs(self, y: Tensor, **kwargs) -> dict:
        # see `last_measured_per_group` in forward docstring
        # todo: duplicate code in ``TimeSeriesDataset.get_durations()``
        any_measured_bool = ~torch.isnan(y).all(2).cpu()
        kwargs['last_measured_per_group'] = torch.as_tensor(
            [np.max(true1d_idx(any_measured_bool[g]).numpy(), initial=0) for g in range(y.shape[0])],
            dtype=torch.int,
            device=y.device
        ) + 1
        return kwargs

    def get_laplace_mvnorm(self,
                           y: torch.Tensor,
                           get_loss: Optional[callable] = None,
                           **kwargs) -> Tuple[torch.distributions.MultivariateNormal, List[str]]:
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

    @torch.jit.ignore()
    def set_initial_values(self,
                           y: Tensor,
                           ilinks: Optional[Dict[str, callable]] = None,
                           verbose: bool = True,
                           **kwargs):
        if 'initial_mean' not in self.state_dict():
            return

        if ilinks is None:
            ilinks = {k: identity for k in self.measures}

        assert len(self.measures) == y.shape[-1]

        hits = {m: [] for m in self.measures}
        for pid in self.processes:
            process = self.processes[pid]
            # have to use the name since `jit.script()` strips the class
            if (getattr(process, 'original_name', None) or type(process).__name__) in ('LocalLevel', 'LocalTrend'):
                if 'position->position' in (process.f_modules or {}):
                    continue
                assert process.measure

                hits[process.measure].append(process.id)
                se_idx = process.state_elements.index('position')
                measure_idx = list(self.measures).index(process.measure)
                with torch.no_grad():
                    t0 = y[..., measure_idx]
                    orig_mean = t0[~torch.isnan(t0) & ~torch.isinf(t0)].mean().item()
                    init_mean = ilinks[process.measure](orig_mean)
                    if verbose:
                        print(f"Initializing {pid}.position to {init_mean}")
                    self.state_dict()['initial_mean'][self.process_to_slice[pid][se_idx]] = init_mean

        for measure, procs in hits.items():
            if len(procs) > 1:
                warn(
                    f"For measure '{measure}', multiple processes ({procs}) track the overall level; consider adding "
                    f"`decay` to all but one."
                )
            elif not len(procs):
                warn(
                    f"For measure '{measure}', no processes track the overall level; consider centering data in "
                    f"preprocessing prior to training (if you haven't already)."
                )

    @staticmethod
    def _validate(processes: Sequence[Process], measures: Sequence[str]):
        if not hasattr(measures, '__getitem__'):
            warn(f"`measures` appears to be an unordered collection -- needs to be ordered")

        assert len(measures) == len(set(measures))

        process_names = set()
        for p in processes:
            if p.id in process_names:
                raise ValueError(f"There are multiple processes with id '{p.id}'.")
            else:
                process_names.add(p.id)
            if isinstance(p, torch.jit.RecursiveScriptModule):
                raise TypeError(
                    f"Processes should not be wrapped in `torch.jit.script` *before* being passed to `KalmanFilter`"
                )
            if p.measure:
                if p.measure not in measures:
                    raise ValueError(f"'{p.id}' has measure '{p.measure}' not in `measures`.")
            else:
                if len(measures) > 1:
                    raise ValueError(f"Must set measure for '{p.id}' since there are multiple measures.")
                p.measure = measures[0]

    @torch.jit.ignore()
    def design_modules(self) -> Iterable[Tuple[str, nn.Module]]:
        for pid in self.processes:
            yield pid, self.processes[pid]
        if self.measure_covariance:
            yield 'measure_covariance', self.measure_covariance

    @torch.jit.ignore()
    def forward(self,
                *args,
                n_step: Union[int, float] = 1,
                start_offsets: Optional[Sequence] = None,
                out_timesteps: Optional[Union[int, float]] = None,
                initial_state: Union[Tuple[Tensor, Tensor], Tensor, None] = None,
                every_step: bool = True,
                include_updates_in_output: bool = False,
                simulate: Optional[int] = None,
                last_measured_per_group: Optional[Tensor] = None,
                prediction_kwargs: Optional[dict] = None,
                **kwargs) -> Predictions:
        """
        Generate n-step-ahead predictions from the model.

        :param args: A (group X time X measures) tensor. Optional if ``initial_state`` is specified.
        :param n_step: What is the horizon for the predictions output for each timepoint? Defaults to one-step-ahead
         predictions (i.e. n_step=1).
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``input``. If you passed ``dt_unit`` when constructing those processes, then you should pass an
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
         inference/forecasting.
        :param simulate: If specified, will generate `simulate` samples from the model.
        :param prediction_kwargs: A dictionary of kwargs to pass to initialize ``Predictions()``. Unused for base
         class, but can be used by subclasses (e.g. ``BinomialFilter``).
        :param kwargs: Further arguments passed to the `processes`. For example, the :class:`.LinearModel` expects an
         ``X`` argument for predictors.
        :return: A :class:`.Predictions` object with :func:`Predictions.log_prob()` and
         :func:`Predictions.to_dataframe()` methods.
        """

        assert len(args) <= 1
        if len(args):
            input = args[0]
            if not torch.is_floating_point(input):
                raise ValueError(f"Expected input to be a float tensor, got {input.dtype}")
            if torch.isinf(input).any():
                raise ValueError("Input contains infinite values.")
        else:
            input = None

        if out_timesteps is None and input is None:
            raise RuntimeError("If no input is passed, must specify `out_timesteps`")

        initial_state = self._prepare_initial_state(
            initial_state,
            start_offsets=start_offsets,
        )
        if simulate and simulate > 1:
            init_mean, init_cov = initial_state
            initial_state = repeat(init_mean, simulate, dim=0), repeat(init_cov, simulate, dim=0)
            if start_offsets is not None:
                start_offsets = repeat(np.asarray(start_offsets), simulate, dim=0)
            kwargs = {k: (repeat(v, simulate, dim=0) if isinstance(v, (Tensor, np.ndarray)) else v)
                      for k, v in kwargs.items()}

        if isinstance(n_step, float):
            if not n_step.is_integer():
                raise ValueError("`n_step` must be an int.")
            n_step = int(n_step)
        if isinstance(out_timesteps, float):
            if not out_timesteps.is_integer():
                raise ValueError("`out_timesteps` must be an int.")
            out_timesteps = int(out_timesteps)

        preds, updates, design_mats = self._script_forward(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            every_step=every_step,
            out_timesteps=out_timesteps,
            kwargs_per_process=self._parse_design_kwargs(
                input,
                out_timesteps=out_timesteps or input.shape[1],
                **kwargs
            ),
            last_measured_per_group=last_measured_per_group,
            simulate=bool(simulate)
        )
        prediction_kwargs = prediction_kwargs or {}
        preds = self._generate_predictions(
            preds=preds,
            updates=updates if include_updates_in_output else None,
            **design_mats,
            **prediction_kwargs
        )
        # todo: if no start_offsets, are they allowed to supply dt_unit?
        return preds.set_metadata(
            start_offsets=start_offsets if start_offsets is not None else np.zeros(preds.means.shape[0], dtype='int'),
            dt_unit=self.dt_unit
        )

    @torch.jit.ignore
    def _generate_predictions(self,
                              preds: Tuple[List[Tensor], List[Tensor]],
                              updates: Optional[Tuple[List[Tensor], List[Tensor]]] = None,
                              **kwargs) -> 'Predictions':
        """
        StateSpace subclasses may pass subclasses of `Predictions` (e.g. for custom log-prob)
        """

        if updates is not None:
            kwargs.update(update_means=updates[0], update_covs=updates[1])
        preds = Predictions(
            *preds,
            R=kwargs.pop('R'),
            H=kwargs.pop('H'),
            model=self,
            **kwargs
        )
        return preds

    @torch.jit.ignore
    def _prepare_initial_state(self,
                               initial_state: Union[Tuple[Tensor, Tensor], Tensor, None],
                               start_offsets: Optional[Sequence] = None) -> Tuple[Tensor, Tensor]:
        if isinstance(initial_state, Tensor):
            initial_state = (initial_state, None)

        if initial_state is None:
            init_mean = self.initial_mean[None, :].clone()
            init_cov = self.initial_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[:, 0]
        else:
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

        measure_scaling = torch.diag_embed(self._get_measure_scaling().unsqueeze(0))
        init_cov = measure_scaling @ init_cov @ measure_scaling

        if start_offsets is not None:
            if init_mean.shape[0] == 1:
                init_mean = init_mean.expand(len(start_offsets), -1)
            elif init_mean.shape[0] != len(start_offsets):
                raise ValueError("Expected ``len(start_offets) == initial_state[0].shape[0]``")

            if initial_state is None:
                # seasonal processes need to offset the initial mean:
                # TODO: should also handle cov?
                init_mean_w_offset = []
                for pid in self.processes:
                    p = self.processes[pid]
                    _process_slice = slice(*self.process_to_slice[pid])
                    init_mean_w_offset.append(p.offset_initial_state(init_mean[:, _process_slice], start_offsets))
                init_mean = torch.cat(init_mean_w_offset, 1)
            else:
                # if they passed an initial_state, we assume it's from a previous call to forward, so already offset
                pass

        return init_mean, init_cov

    @torch.jit.export
    def _script_forward(self,
                        input: Optional[Tensor],
                        kwargs_per_process: Dict[str, Dict[str, Tensor]],
                        initial_state: Tuple[Tensor, Tensor],
                        n_step: int = 1,
                        out_timesteps: Optional[int] = None,
                        last_measured_per_group: Optional[Tensor] = None,
                        every_step: bool = True,
                        simulate: bool = False,
                        ) -> Tuple[
        Tuple[List[Tensor], List[Tensor]],
        Tuple[List[Tensor], List[Tensor]],
        Dict[str, List[Tensor]]
    ]:
        """
        :param input: A (group X time X measures) tensor. Optional if `initial_state` is specified.
        :param kwargs_per_process: Keyword-arguments to the Processes (e.g. X=model_matrix for LinearModel).
        :param initial_state: A (mean, cov) tuple to use as the initial state.
        :param n_step: What is the horizon for predictions? Defaults to one-step-ahead (i.e. n_step=1).
        :param out_timesteps: The number of timesteps in the output. Might be longer than input if forecasting.
        :param every_step: Experimental. When n_step>1, we can generate these n-step-ahead predictions at every
         timestep (e.g. 24-hour-ahead predictions every hour), in which case we'd save the 24-step-ahead prediction.
         Alternatively, we could generate 24-hour-ahead predictions at every 24th hour, in which case we'd save
         predictions 1-24. The former corresponds to every_step=True, the latter to every_step=False. If n_step=1
         (the default) then this option has no effect.
        :param simulate: If True, will simulate state-trajectories and return a ``Predictions`` object with zero state
         covariance.
        :param last_measured_per_group: See forward().
        """
        assert n_step > 0

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

        predict_kwargs, update_kwargs = self._build_design_mats(
            kwargs_per_process=kwargs_per_process,
            num_groups=num_groups,
            out_timesteps=out_timesteps
        )
        if last_measured_per_group is None:
            last_measured_per_group = torch.full((num_groups,), out_timesteps, dtype=torch.int, device=meanu.device)

        # first loop through to do predict -> update
        meanus: List[Tensor] = []
        covus: List[Tensor] = []
        mean1s: List[Tensor] = []
        cov1s: List[Tensor] = []
        for t in range(out_timesteps):
            mean1step, cov1step = self.ss_step.predict(
                meanu,
                covu,
                mask=(t <= last_measured_per_group),
                kwargs={k: v[t] for k, v in predict_kwargs.items()}
            )
            mean1s.append(mean1step)
            cov1s.append(cov1step)

            if simulate:
                meanu = torch.distributions.MultivariateNormal(mean1step, cov1step, validate_args=False).sample()
                covu = torch.eye(meanu.shape[-1]).expand(num_groups, -1, -1) * 1e-6
            elif t < len(inputs):
                update_kwargs_t = {k: v[t] for k, v in update_kwargs.items()}
                meanu, covu = self.ss_step.update(
                    inputs[t],
                    mean1step,
                    cov1step,
                    update_kwargs_t,
                )
            else:
                meanu, covu = mean1step, cov1step

            meanus.append(meanu)
            covus.append(covu)

        # 2nd loop to get n_step predicts:
        # idx: Dict[int, int] = {}
        meanps: Dict[int, Tensor] = {}
        covps: Dict[int, Tensor] = {}
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
                        meanp, covp = self.ss_step.predict(
                            meanp,
                            covp,
                            mask=(tu_h <= last_measured_per_group),
                            kwargs={k: v[tu_h] for k, v in predict_kwargs.items()},
                        )
                    if tu_h not in meanps:
                        # idx[tu + h] = tu
                        meanps[tu_h] = meanp
                        covps[tu_h] = covp

        preds = [meanps[t] for t in range(out_timesteps)], [covps[t] for t in range(out_timesteps)]
        updates = meanus, covus

        return preds, updates, update_kwargs

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        """
        Build the design matrices. Implemented by subclasses (partially implemented by
        ``_build_transition_and_measure_mats()`` and ``_build_measure_var_mats()``, but torchscript
        doesn't currently support ``super()``).

        :param static_kwargs: Keys are ids of ``design_modules()``, values are dictionaries of kwargs that only need
         to be passed once.
        :param time_varying_kwargs: Keys are ids of ``design_modules()``, values are dictionaries of kwargs, whose
         values are lists (passed for each timestep).
        :param num_groups: Number of groups.
        :param out_timesteps: Number of timesteps.
        :return: Two dictionaries: ``predict_kwargs`` (passed to ``ss_step.predict()``) and
         ``update_kwargs`` (passed to ``ss_step.update()``).
        """
        raise NotImplementedError

    def _build_transition_and_measure_mats(self,
                                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                                           num_groups: int,
                                           out_timesteps: int) -> Tuple[List[Tensor], List[Tensor]]:
        # todo: if F and/or H are not time-varying, cheaper to build mat for 1 timestep and return [mat]*out_timesteps
        ms = self._get_measure_scaling()

        Fs = torch.zeros(
            (num_groups, out_timesteps, self.state_rank, self.state_rank),
            dtype=ms.dtype,
            device=ms.device
        )
        Hs = torch.zeros(
            (num_groups, out_timesteps, len(self.measures), self.state_rank),
            dtype=ms.dtype,
            device=ms.device
        )

        for pid, process in self.processes.items():
            _process_slice = slice(*self.process_to_slice[pid])
            p_kwargs = kwargs_per_process.get(pid, {})
            pH, pF = process(inputs=p_kwargs, num_groups=num_groups, num_times=out_timesteps)
            Hs[:, :, self.measure_to_idx[process.measure], _process_slice] = pH
            Fs[:, :, _process_slice, _process_slice] = pF

        Fs = Fs.unbind(1)
        Hs = Hs.unbind(1)

        return Fs, Hs

    @torch.jit.ignore()
    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        kwargs_per_process = defaultdict(dict)
        unused = set(kwargs)
        # TODO: what/why is current_timestep
        kwargs.update(input=input, current_timestep=torch.tensor(list(range(out_timesteps))).view(1, -1, 1))
        for submodule_nm, submodule in self.design_modules():
            for found_key, key_name, value in get_owned_kwargs(submodule, kwargs):
                unused.discard(found_key)
                kwargs_per_process[submodule_nm][key_name] = value
        if unused:
            warn(f"There are unused keyword arguments:\n{unused}")
        return dict(kwargs_per_process)

    def _get_measure_scaling(self) -> Tensor:
        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        measure_var = mcov.diagonal(dim1=-2, dim2=-1)
        multi = torch.zeros(mcov.shape[0:-2] + (self.state_rank,), dtype=mcov.dtype, device=mcov.device)
        for pid, process in self.processes.items():
            pidx = self.process_to_slice[pid]
            multi[..., slice(*pidx)] = measure_var[..., self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
        assert (multi > 0).all()
        return multi

    def __repr__(self) -> str:
        return f'{type(self).__name__}' \
               f'(processes={repr(list(self.processes.values()))}, measures={repr(list(self.measures))})'

    @torch.no_grad()
    @torch.jit.ignore()
    def simulate(self,
                 out_timesteps: int,
                 initial_state: Optional[Tuple[Tensor, Tensor]] = None,
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
            if start_offsets is None:
                start_offsets = [0] * num_groups
            elif len(start_offsets) != num_groups:
                raise ValueError("Expected `len(start_offsets) == num_groups` (or num_groups=None)")

        return self(
            start_offsets=start_offsets,
            out_timesteps=out_timesteps,
            initial_state=initial_state,
            simulate=num_sims,
            **kwargs
        )
