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
                       system_cov: Optional[torch.Tensor] = None) -> 'StateTuple':
        """
        The kalman-filter update equations. ``measured_cov`` (P @ H.T) and ``system_cov`` (H @ P @ H.T + R) can be
        passed if already computed.
        """
        resid = input - measured_mean
        if measured_cov is None:
            measured_cov = cov @ measure_mat.permute(0, 2, 1)
        if system_cov is None:
            system_cov = measure_mat @ measured_cov + measure_cov
        K = self._kalman_gain(measured_cov=measured_cov, system_cov=system_cov)
        new_mean = self._mean_update(mean=mean, K=K, resid=resid)
        new_cov = self._covariance_update(cov=cov, K=K, H=measure_mat, R=measure_cov)
        return StateTuple(new_mean, new_cov, resid=resid, system_cov=system_cov)

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
        Update step when there are mixture components. Each (effective) regime-combo implies a hypothesis: measures in
        a non-standard regime are dropped from the kalman update, and instead scored against their component. The
        hypotheses are weighted by their posterior probability and collapsed via moment-matching.

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

        measured_cov = cov @ measure_mat.permute(0, 2, 1)
        system_cov = measure_mat @ measured_cov + measure_cov
        standard = self._kalman_update(
            input=input,
            mean=mean,
            cov=cov,
            measured_mean=measured_mean,
            measure_mat=measure_mat,
            measure_cov=measure_cov,
            measured_cov=measured_cov,
            system_cov=system_cov,
        )
        if len(effective) == 1:
            # no mixture measures observed, so observations are uninformative about the regime:
            standard.regime_probs = log_prior.exp()
            return standard

        # which measures contribute to the responsibilities:
        if self.mixture.univariate_prob:
            score_idx = {i for i, m in enumerate(measures) if m in self.mixture.mixture_measures}
        else:
            score_idx = set(range(len(measures)))

        states_by_weird = {(): standard}
        states = []
        log_liks = []
        for eff in effective:
            weird_idx = tuple(i for i, _ in eff)
            normal_idx = [i for i in range(len(measures)) if i not in weird_idx]
            if weird_idx not in states_by_weird:
                if normal_idx:
                    idx = torch.as_tensor(normal_idx, dtype=torch.long, device=input.device)
                    idx2d = (slice(None), idx.unsqueeze(-1), idx.unsqueeze(0))
                    states_by_weird[weird_idx] = self._kalman_update(
                        input=input[:, idx],
                        mean=mean,
                        cov=cov,
                        measured_mean=measured_mean[:, idx],
                        measure_mat=measure_mat[:, idx],
                        measure_cov=measure_cov[idx2d],
                        measured_cov=measured_cov[:, :, idx],
                        system_cov=system_cov[idx2d],
                    )
                else:
                    # all measures in a non-standard regime: no update
                    states_by_weird[weird_idx] = StateTuple(mean, cov)
            states.append(states_by_weird[weird_idx])

            log_lik = torch.zeros(num_groups, dtype=input.dtype, device=input.device)
            scored_normal = [i for i in normal_idx if i in score_idx]
            if scored_normal:
                idx = torch.as_tensor(scored_normal, dtype=torch.long, device=input.device)
                log_lik = log_lik + mvnorm_log_prob(
                    resid=input[:, idx] - measured_mean[:, idx],
                    cov=system_cov[:, idx.unsqueeze(-1), idx.unsqueeze(0)]
                )
            for i, component in eff:
                log_lik = log_lik + mvnorm_log_prob(
                    resid=(input[:, i] - component.mean).unsqueeze(-1),
                    cov=component.var.expand(num_groups, 1, 1)
                )
            log_liks.append(log_lik)

        # posterior over the full combo-table; combos that are indistinguishable given `measures` share a likelihood
        log_post = log_prior + torch.stack(log_liks, -1)[:, mapping]
        regime_post = torch.softmax(log_post, -1)
        weights = torch.zeros((num_groups, len(effective)), dtype=input.dtype, device=input.device)
        weights = weights.index_add(1, mapping.to(input.device), regime_post)

        new_state = self._mix_updates(states, weights)
        new_state.regime_probs = regime_post
        return new_state

    @staticmethod
    def _mix_updates(states: Sequence['StateTuple'], weights: torch.Tensor) -> 'StateTuple':
        """
        Collapse a mixture of gaussian states into a single gaussian via moment-matching.

        :param states: The states to mix.
        :param weights: A ``(num_groups, len(states))`` tensor of mixture-weights, summing to 1 along the last dim.
        """
        weights = weights.unbind(-1)
        new_mean = sum(w.unsqueeze(-1) * s.mean for w, s in zip(weights, states))
        new_cov = 0
        for w, s in zip(weights, states):
            diff = (s.mean - new_mean).unsqueeze(-1)
            new_cov = new_cov + w.view(-1, 1, 1) * (s.cov + diff @ diff.permute(0, 2, 1))
        return StateTuple(mean=new_mean, cov=new_cov)

    @staticmethod
    def _covariance_update(cov: torch.Tensor, K: torch.Tensor, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
        ikh = I - K @ H
        return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)

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
