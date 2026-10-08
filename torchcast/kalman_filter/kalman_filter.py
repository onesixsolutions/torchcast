"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts using the full kalman-filtering
algorithm (or optionally extended-kalman filtering, if any measure-funs or nonlinear processes are used).

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.
"""
from typing import Sequence

from torchcast.covariance import Covariance
from torchcast.internals.utils import update_tensor, mvnorm_log_prob
from torchcast.process import Process

from torchcast.state_space.mixture import MixtureComponent, MixtureModel
from torchcast.state_space.state_space import StateSpaceModel, StateTuple

from typing import Optional, Union

import torch


class KalmanFilter(StateSpaceModel):
    """
    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    :param process_covariance: A module created with ``Covariance.from_processes(processes, type='process')``.
    :param initial_covariance: A module created with ``Covariance.from_processes(measures, type='initial')``.
    :param measure_funs: A dictionary mapping measure-names to measurement-functions. Currently only supports 'sigmoid'.
    :param adaptive_scaling: Experimental feature to adaptively scale the covariance as a function of residuals. This
     is useful if different groups have very different magnitudes.
    :param joseph_form: If True (the default), the update-step uses the Joseph form of the covariance update,
     ``(I - K H) P (I - K H)' + K R K'``. This keeps the covariance positive semi-definite even with numerical error in
     the kalman gain ``K`` (which then only has a second-order effect). ``False`` uses the simpler ``P - K H P``
     (symmetrized): less memory during training (its intermediate results, kept for the backward pass, are smaller)
     and faster, but errors in ``K`` have a first-order effect, so the covariance can lose positive-definiteness --
     e.g. with float32, long series, near-zero process-variances, or very precise measurements.
    :param mixture: Experimental. A :class:`.MixtureModel` (or a list of :class:`.MixtureComponent` objects); see
     :class:`.StateSpaceModel`.
    """

    def __init__(self,
                 processes: Sequence['Process'],
                 measures: Sequence[str],
                 measure_covariance: Optional[Covariance] = None,
                 process_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None,
                 measure_funs: Optional[dict[str, str]] = None,
                 adaptive_scaling: bool = False,
                 joseph_form: bool = True,
                 mixture: Union[MixtureModel, Sequence[MixtureComponent], None] = None):

        if initial_covariance is None:
            initial_covariance = Covariance.from_processes(processes, cov_type='initial')

        if process_covariance is None:
            process_covariance = Covariance.from_processes(processes, cov_type='process')

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance,
            measure_funs=measure_funs,
            adaptive_scaling=adaptive_scaling,
            mixture=mixture,
        )
        self.process_covariance = process_covariance.set_id('process_covariance')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')
        self.joseph_form = joseph_form

    def _predict_cov(self,
                     cov: torch.Tensor,
                     transition_mat: torch.Tensor,
                     Q: torch.Tensor,
                     scaling: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None or mask.all():
            mask = slice(None)
        F = transition_mat[mask]
        Q = Q[mask]
        scaling = scaling[mask] if scaling is not None else None
        Q = self._apply_cov_scaling(Q, scaling, is_process_cov=True)

        new_cov = update_tensor(cov, new=(F @ cov[mask] @ F.permute(0, 2, 1) + Q), mask=mask)
        return new_cov

    def _update_step(self,
                     input: torch.Tensor,
                     mean: torch.Tensor,
                     cov: torch.Tensor,
                     measured_mean: torch.Tensor,
                     measure_mat: torch.Tensor,
                     measure_cov: torch.Tensor,
                     val_idx: Optional[torch.Tensor] = None,
                     regime_prior: Optional[torch.Tensor] = None,
                     **kwargs) -> 'StateTuple':
        input, measured_mean, measure_cov = self._prepare_update(
            input=input,
            measured_mean=measured_mean,
            measure_cov=measure_cov,
            **kwargs
        )
        if self.mixture is None:
            return self._kalman_update(
                input=input,
                mean=mean,
                cov=cov,
                measured_mean=measured_mean,
                measure_mat=measure_mat,
                measure_cov=measure_cov,
            )
        return self._mixture_update(
            input=input,
            mean=mean,
            cov=cov,
            measured_mean=measured_mean,
            measure_mat=measure_mat,
            measure_cov=measure_cov,
            measures=self.measures if val_idx is None else [self.measures[i] for i in val_idx.tolist()],
            regime_prior=regime_prior,
        )

    def _prepare_update(self,
                        input: torch.Tensor,
                        measured_mean: torch.Tensor,
                        measure_cov: torch.Tensor,
                        **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hook for subclasses to validate/adjust the inputs to the update step (e.g. ``BinomialFilter`` adjusts the
        measure-covariance). Called exactly once per update-step, before any mixture hypotheses are constructed; any
        update-kwargs from ``_parse_kwargs`` are consumed here.

        :return: A tuple of (possibly modified) ``input``, ``measured_mean``, ``measure_cov``.
        """
        if kwargs:
            raise TypeError(f"`{type(self).__name__}._update_step()` received unexpected kwargs: {list(kwargs)}")
        return input, measured_mean, measure_cov

    def _kalman_update(self,
                       input: torch.Tensor,
                       mean: torch.Tensor,
                       cov: torch.Tensor,
                       measured_mean: torch.Tensor,
                       measure_mat: torch.Tensor,
                       measure_cov: torch.Tensor,
                       measured_cov: Optional[torch.Tensor] = None,
                       system_cov: Optional[torch.Tensor] = None,
                       joseph: Optional[bool] = None) -> 'StateTuple':
        """
        The kalman-filter update equations. ``measured_cov`` (P @ H.T) and ``system_cov`` (H @ P @ H.T + R) can be
        passed if already computed. ``joseph=False`` uses the simpler covariance update ``P - K @ H @ P`` instead of
        the Joseph form: less memory and compute, but less numerically robust. Defaults to the model's
        ``joseph_form`` (the Joseph form, if not set).
        """
        resid = input - measured_mean
        if measured_cov is None:
            measured_cov = cov @ measure_mat.permute(0, 2, 1)
        if system_cov is None:
            system_cov = measure_mat @ measured_cov + measure_cov
        K = self._kalman_gain(measured_cov=measured_cov, system_cov=system_cov)
        new_mean = self._mean_update(mean=mean, K=K, resid=resid)
        if joseph is None:
            joseph = getattr(self, 'joseph_form', True)
        if joseph:
            new_cov = self._covariance_update(cov=cov, K=K, H=measure_mat, R=measure_cov)
        else:
            new_cov = self._simple_covariance_update(cov=cov, K=K, H=measure_mat)
        return StateTuple(new_mean, new_cov)

    def _mixture_update(self,
                        input: torch.Tensor,
                        mean: torch.Tensor,
                        cov: torch.Tensor,
                        measured_mean: torch.Tensor,
                        measure_mat: torch.Tensor,
                        measure_cov: torch.Tensor,
                        measures: Sequence[str],
                        regime_prior: Optional[torch.Tensor] = None) -> 'StateTuple':
        """
        Update step when there are mixture components. Each (effective) regime-combo implies a hypothesis: for
        measures in a non-standard regime, the measured-mean is offset by the component's mean, and the component's
        variance is added to the measurement-noise. Each hypothesis is a full kalman update; they're weighted by their
        posterior probability and collapsed via moment-matching.

        :param measures: The measures in ``input`` (i.e. excluding any dropped because they were nan).
        :param regime_prior: A ``(num_groups, num_combos)`` tensor with the prior probability of each combo. Defaults
         to the regime-model's base-probs.
        :return: The collapsed state, with ``regime_probs`` set to the ``(num_groups, num_combos)`` posterior.
        """
        num_groups = input.shape[0]
        if regime_prior is None:
            log_prior = self.mixture.log_base_probs().expand(num_groups, -1)
        else:
            log_prior = regime_prior.clamp_min(1e-30).log()

        effective, mapping = self.mixture.effective_combos(measures)
        if len(effective) == 1:
            # no mixture measures observed, so observations are uninformative about the regime:
            standard = self._kalman_update(
                input=input,
                mean=mean,
                cov=cov,
                measured_mean=measured_mean,
                measure_mat=measure_mat,
                measure_cov=measure_cov,
            )
            standard.regime_probs = log_prior.exp()
            return standard

        # which measures contribute to the responsibilities. non-gaussian measures (e.g. binary) never do: their
        # gaussian approximation is crude, and their log-prob is computed separately from the gaussian measures' (so
        # using their correlation here would be inconsistent with training).
        if self.mixture.univariate_prob:
            score_idx = [i for i, m in enumerate(measures) if m in self.mixture.mixture_measures]
        else:
            score_idx = [i for i, m in enumerate(measures) if m not in self._non_gaussian_measures]
        score_idx = torch.as_tensor(score_idx, dtype=torch.long, device=input.device)

        # each effective combo offsets the measured-mean and adds to the measurement-noise variance. all combos are
        # updated in one batch: stack them along the group dimension.
        num_effective = len(effective)
        offsets = [self.mixture.effective_offsets(eff, len(measures), like=input) for eff in effective]
        shift = torch.stack([o[0] for o in offsets]).unsqueeze(1)  # (num_effective, 1, num_measures)
        extra_cov = torch.diag_embed(torch.stack([o[1] for o in offsets])).unsqueeze(1)  # (num_effective, 1, M, M)

        def batched(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0).expand(num_effective, *x.shape)

        measured_cov = cov @ measure_mat.permute(0, 2, 1)
        system_cov = batched(measure_mat @ measured_cov + measure_cov) + extra_cov  # (num_effective, G, M, M)
        input_e = batched(input) - shift
        flat = lambda x: x.reshape(num_effective * num_groups, *x.shape[2:])  # noqa: E731
        update_kwargs = dict(
            input=flat(input_e),
            mean=flat(batched(mean)),
            cov=flat(batched(cov)),
            measured_mean=flat(batched(measured_mean)),
            measure_mat=flat(batched(measure_mat)),
            measure_cov=flat(batched(measure_cov) + extra_cov),
            measured_cov=flat(batched(measured_cov)),
            system_cov=flat(system_cov),
        )
        # the standard combo (the first) uses the model's covariance-update; the others, the mixture's:
        standard_joseph = getattr(self, 'joseph_form', True)
        mixture_joseph = getattr(self.mixture, 'joseph_form', True)  # (missing in older pickles)
        if standard_joseph == mixture_joseph:
            states = self._kalman_update(**update_kwargs, joseph=standard_joseph)
        else:
            standard = self._kalman_update(**{k: v[:num_groups] for k, v in update_kwargs.items()},
                                           joseph=standard_joseph)
            others = self._kalman_update(**{k: v[num_groups:] for k, v in update_kwargs.items()},
                                         joseph=mixture_joseph)
            states = StateTuple(torch.cat([standard.mean, others.mean]), torch.cat([standard.cov, others.cov]))
        resid = (input_e - measured_mean)[..., score_idx]
        log_liks = mvnorm_log_prob(
            resid=flat(resid),
            cov=flat(system_cov[..., score_idx.unsqueeze(-1), score_idx.unsqueeze(0)])
        ).view(num_effective, num_groups).T  # (G, num_effective)

        # posterior over the full combo-table; combos that are indistinguishable given `measures` share a likelihood
        log_post = log_prior + log_liks[:, mapping]
        regime_post = torch.softmax(log_post, -1)
        return self._mix_updates(
            means=states.mean.view(num_effective, num_groups, mean.shape[-1]),
            covs=states.cov.view(num_effective, num_groups, *cov.shape[1:]),
            regime_post=regime_post,
            mapping=mapping,
        )

    @staticmethod
    def _mix_updates(means: torch.Tensor,
                     covs: torch.Tensor,
                     regime_post: torch.Tensor,
                     mapping: torch.Tensor) -> 'StateTuple':
        """
        Collapse a mixture of gaussian states into a single gaussian via moment-matching.

        :param means: A ``(num_effective, num_groups, state_rank)`` tensor: the (effective-combo) state-means to mix.
        :param covs: A ``(num_effective, num_groups, state_rank, state_rank)`` tensor: their covariances.
        :param regime_post: A ``(num_groups, num_combos)`` tensor of posterior probabilities over the full combo-table.
        :param mapping: A ``(num_combos,)`` tensor mapping each combo to its index in ``means`` (combos that are
         indistinguishable given the observed measures share a state).
        :return: The collapsed state, with ``regime_probs`` set to ``regime_post``.
        """
        # the weight of each state is the total probability of the combos that map to it:
        to = {'dtype': regime_post.dtype, 'device': regime_post.device}
        weights = torch.zeros((regime_post.shape[0], means.shape[0]), **to)
        weights = weights.index_add(1, mapping.to(regime_post.device), regime_post)  # (G, num_effective)
        new_mean = torch.einsum('ge,egs->gs', weights, means)
        diff = means - new_mean
        new_cov = torch.einsum('ge,egij->gij', weights, covs + diff.unsqueeze(-1) * diff.unsqueeze(-2))
        return StateTuple(mean=new_mean, cov=new_cov, regime_probs=regime_post)

    @staticmethod
    def _covariance_update(cov: torch.Tensor, K: torch.Tensor, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
        ikh = I - K @ H
        return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)

    @staticmethod
    def _simple_covariance_update(cov: torch.Tensor, K: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        new_cov = cov - K @ (H @ cov)
        return .5 * (new_cov + new_cov.permute(0, 2, 1))  # (symmetric in exact arithmetic)

    @staticmethod
    def _kalman_gain(measured_cov: torch.Tensor, system_cov: torch.Tensor) -> torch.Tensor:
        A = system_cov.permute(0, 2, 1)
        B = measured_cov.permute(0, 2, 1)
        Kt = torch.linalg.solve(A, B)
        K = Kt.permute(0, 2, 1)
        return K

    def _parse_kwargs(self,
                      num_groups: int,
                      num_timesteps: int,
                      measure_covs: Sequence[torch.Tensor],
                      **kwargs) -> tuple[dict[str, Sequence], dict[str, Sequence], set]:
        predict_kwargs, update_kwargs, used_keys = super()._parse_kwargs(
            num_groups=num_groups,
            num_timesteps=num_timesteps,
            measure_covs=measure_covs,
            **kwargs
        )

        # process-variance:
        pcov_kwargs = {}
        if self.process_covariance.expected_kwargs:
            pcov_kwargs = {k: kwargs[k] for k in self.process_covariance.expected_kwargs}
        used_keys |= set(pcov_kwargs)

        measure_scaling = self._get_measure_scaling()

        # todo: instead of branching here, clean up Covariance.forward():
        if pcov_kwargs:
            pcov_raw = self.process_covariance(pcov_kwargs, num_groups=num_groups, num_times=num_timesteps)
            Qs = self._apply_cov_scaling(pcov_raw, scaling=measure_scaling, is_process_cov=True)
            predict_kwargs['Q'] = Qs.unbind(1)
        else:
            # faster if not time-varying
            pcov_raw = self.process_covariance(pcov_kwargs, num_groups=num_groups, num_times=1).squeeze(1)
            Qs = self._apply_cov_scaling(pcov_raw, scaling=measure_scaling, is_process_cov=True)
            predict_kwargs['Q'] = [Qs] * num_timesteps

        return predict_kwargs, update_kwargs, used_keys


def main(num_groups: int = 50, num_timesteps: int = 100, bias: float = -2, prop_common: float = 0.5):
    from torchcast.process import LocalLevel
    from torchcast.utils import TimeSeriesDataset
    import pandas as pd
    from plotnine import geom_line, aes, ggtitle, theme
    torch.manual_seed(1234)

    measures = ['dim1']
    latent_common = torch.cumsum(.05 * torch.randn((num_groups, num_timesteps, 1)), dim=1)
    latent_ind = torch.cumsum(.05 * torch.randn((num_groups, num_timesteps, len(measures))), dim=1)
    assert 0 <= prop_common <= 1
    latent = (
            (1 - prop_common) * latent_ind  # per-measure trajectories
            + prop_common * latent_common.expand(num_groups, num_timesteps, len(measures))  # cross-measure traj
            + bias  # global bias
            + torch.randn((num_groups, 1, len(measures)))  # group-level starting-points
    )

    y = []
    for i, m in enumerate(measures):
        y.append(torch.distributions.Normal(loc=latent[..., i], scale=.5).sample())
        y[-1][torch.randn((num_groups, num_timesteps)) > 1.5] -= 5  # some random outliers
        # y[-1][torch.randn((num_groups, num_timesteps)) > 1.5] = float('nan')  # some random missings
    y = torch.stack(y, dim=-1)
    # first tensor in dataset is observed
    # second tensor is ground truth
    dataset = TimeSeriesDataset(
        y,
        latent,
        group_names=[f'group_{i}' for i in range(num_groups)],
        start_times=[pd.Timestamp('2023-01-01')] * num_groups,
        measures=[measures, [x.replace('dim', 'latent') for x in measures]],
        dt_unit='D'
    )

    bf = KalmanFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures]
        #          + [Season(id=f'season_{m}', measure=m, dt_unit='D', period=7, K=2) for m in measures]
        ,
        measures=measures,
        mixture=[
            MixtureComponent(measure='dim1', mean_init=-5, prob_init=0.1, id='dim1_low'),
        ]
    )

    y = dataset.tensors[0]
    bf.fit(y, start_offsets=dataset.start_offsets,
           stopping={'monitor_params': True},
           )
    _kwargs = {}
    preds = bf(
        dataset.tensors[0],
        start_offsets=dataset.start_offsets,
        **_kwargs,
    )
    df_preds = preds.to_dataframe(dataset)
    df_latent = (dataset.to_dataframe()
                 .drop(columns=measures)
                 .melt(id_vars=['group', 'time'], var_name='measure', value_name='latent')
                 .assign(measure=lambda _df: _df['measure'].str.replace('latent', 'dim')))

    df_plot = df_preds.merge(df_latent, how='left', on=['group', 'time', 'measure'])
    # group.drop_duplicates().sample(5)
    for g, _df in df_plot.query("group.isin(['group_2','group_41'])").groupby('group'):
        (
                preds.plot(_df)
                + geom_line(aes(y='latent'), color='purple')
                + ggtitle(g)
                + theme(figure_size=(6, 4))
        ).show()
    print(bf.state_dict())


if __name__ == '__main__':
    main()
