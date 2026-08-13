"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts using the full kalman-filtering
algorithm (or optionally extended-kalman filtering, if any measure-funs or nonlinear processes are used).

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.
"""
from typing import Sequence

from torchcast.covariance import Covariance
from torchcast.internals.utils import update_tensor
from torchcast.process import Process

from torchcast.state_space.mixture import MixtureComponent
from torchcast.state_space.state_space import StateSpaceModel, StateTuple

from typing import Optional

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
    """

    def __init__(self,
                 processes: Sequence['Process'],
                 measures: Sequence[str],
                 measure_covariance: Optional[Covariance] = None,
                 process_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None,
                 measure_funs: Optional[dict[str, str]] = None,
                 adaptive_scaling: bool = False,
                 mixture_components: Optional[Sequence[MixtureComponent]] = None,
                 univariate_mixture_prob: Optional[bool] = None):

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
            mixture_components=mixture_components,
        )
        self.process_covariance = process_covariance.set_id('process_covariance')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')

        self._univar_mix_idx = None
        mix_measures = set(m.measure for m in self.mixture_components)
        if len(mix_measures) == 1:
            mix_measure = mix_measures.pop()
            if len(measures) == 1 or univariate_mixture_prob:
                self._univar_mix_idx = next(i for i, m in enumerate(self.measures) if m == mix_measure)
        elif univariate_mixture_prob:
            raise ValueError(f"Cannot set `univariate_mixture_prob` if multiple mixture measures:\n{mix_measures}")

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
                     measured_cov: Optional[torch.Tensor] = None,
                     system_cov: Optional[torch.Tensor] = None,
                     recursive_call: bool = False,
                     **kwargs) -> 'StateTuple':

        resid = input - measured_mean
        if measured_cov is None:
            measured_cov = cov @ measure_mat.permute(0, 2, 1)
        if system_cov is None:
            system_cov = measure_mat @ measured_cov + measure_cov
        K = self._kalman_gain(measured_cov=measured_cov, system_cov=system_cov)
        new_mean = self._mean_update(mean=mean, K=K, resid=resid)
        new_cov = self._covariance_update(cov=cov, K=K, H=measure_mat, R=measure_cov)
        new_state = StateTuple(new_mean, new_cov, resid, system_cov)

        if self.mixture_components and not recursive_call:
            if val_idx is None:
                measures = self.measures
            else:
                measures = [self.measures[i] for i in val_idx.tolist()]
            return self._mixture_update(
                input=input,
                mean=mean,
                cov=cov,
                measured_mean=measured_mean,
                measure_mat=measure_mat,
                measure_cov=measure_cov,
                measured_cov=measured_cov,
                system_cov=system_cov,
                new_state=new_state,
                measures=measures,
                **kwargs
            )

        if kwargs:
            raise TypeError(f"`{type(self).__name__}._update_step()` received unexpected kwargs: {list(kwargs)}")
        return new_state

    def _mixture_update(self,
                        input: torch.Tensor,
                        mean: torch.Tensor,
                        cov: torch.Tensor,
                        measured_mean: torch.Tensor,
                        measure_mat: torch.Tensor,
                        measure_cov: torch.Tensor,
                        measured_cov: torch.Tensor,
                        system_cov: torch.Tensor,
                        new_state: StateTuple,
                        measures: Sequence[str],
                        **kwargs) -> 'StateTuple':
        if kwargs:
            raise TypeError(f"`{type(self).__name__}._update_step()` received unexpected kwargs: {list(kwargs)}")

        all_standard_state = new_state
        if self._univar_mix_idx is not None:
            all_standard_state = new_state.subset_measure(self._univar_mix_idx)

        # normal_idx = None
        partial_states = {}
        to_mix = []
        mix_probs = []
        for i, (regime_settings, prob) in enumerate(MixtureComponent.traverse(self.mixture_components, measures)):
            # for each measure, we are either in the standard regime or one of the mixtures
            # mask of which measures are standard:
            norm_mask = tuple(mi.is_null for mi in regime_settings)

            # the probability associated with this specific combination of dimensions being 'weird':
            mix_probs.append(prob)

            # all standard:
            if all(norm_mask):
                to_mix.append(all_standard_state)
                # normal_idx = i
                continue

            # calculate resid and var for 'weird' dim(s):
            norm_mask_t = torch.tensor(list(norm_mask), dtype=torch.bool)
            this_resid = input[:, ~norm_mask_t] - torch.as_tensor([m.mean for m in regime_settings if not m.is_null])
            this_var = torch.as_tensor([m.var for m in regime_settings if not m.is_null]).unsqueeze(0)

            # create a partial state that drops the 'weird' dim. only once per dim, cache into partial_states:
            if norm_mask_t.any():
                if norm_mask not in partial_states:
                    mask_idx = norm_mask_t.nonzero(as_tuple=True)[0]
                    partial_states[norm_mask] = self._update_step(
                        input=input[:, mask_idx],
                        mean=mean,
                        cov=cov,
                        measured_mean=measured_mean[:, mask_idx],
                        measure_mat=measure_mat[:, mask_idx],
                        measure_cov=measure_cov[:, mask_idx.unsqueeze(-1), mask_idx.unsqueeze(0)],
                        system_cov=system_cov[:, mask_idx.unsqueeze(-1), mask_idx.unsqueeze(0)],
                        measured_cov=measured_cov[:, :, mask_idx],
                        recursive_call=True
                    )
                partial_state = partial_states[norm_mask]
            else:
                partial_state = StateTuple(
                    mean,
                    cov,
                    resid=this_resid.unsqueeze(-1),
                    system_cov=torch.diag_embed(this_var.expand(mean.shape[0], -1))
                )

            if self._univar_mix_idx is None:
                # insert the 'weird' measured means (and vars) into the full state, so we'll be computing a full
                # multivariate log-prob for mixing probs
                to_mix.append(self._insert_univariate_states(
                    state=partial_state,
                    mask=norm_mask_t,
                    resid=this_resid,
                    vars=this_var,
                ))
            else:
                # just compute the log prob of this_resid and this_var, and use that as the mixing prob.
                # if it's a univariate filter, then this is equivalent to the `if` above, but cheaper; or the user
                # might have indicated at init that they are comfortable using this univariate approximation.
                to_mix.append(StateTuple(
                    mean=partial_state.mean,
                    cov=partial_state.cov,
                    resid=this_resid.unsqueeze(-1),
                    system_cov=torch.diag_embed(this_var.expand(mean.shape[0], -1))
                ))

        # assert normal_idx is not None
        # normal_lp = to_mix[normal_idx].log_prob + torch.log(mix_probs[normal_idx])
        # # TODO: this needs to be exposed outside of `update` so that adaptive_var can account for it.

        return self._mix_updates(to_mix, mixture_probs=mix_probs)

    @staticmethod
    def _mix_updates(states: Sequence['StateTuple'],
                     mixture_probs: Sequence[torch.Tensor],
                     ) -> 'StateTuple':
        assert len(states) == len(mixture_probs)
        if len(states) == 1:
            return states[0]

        log_liks = [state.log_prob + torch.log(p) for state, p in zip(states, mixture_probs)]
        log_norm = torch.logsumexp(torch.stack(log_liks), dim=0)
        weights = [torch.exp(ll - log_norm).view(-1, 1, 1) for ll in log_liks]

        new_mean = sum(w.squeeze(-1) * s.mean for w, s in zip(weights, states))
        diffs = [(s.mean - new_mean).unsqueeze(-1) for s in states]
        new_cov = sum(w * (s.cov + d @ d.permute(0, 2, 1)) for w, s, d in zip(weights, states, diffs))

        return StateTuple(mean=new_mean, cov=new_cov)

    @classmethod
    def _insert_univariate_states(cls,
                                  state: 'StateTuple',
                                  mask: torch.Tensor,  # (d,) bool -- True = normal (kept in `state`), False = weird
                                  resid: torch.Tensor,  # weird dims' residuals, (batch, num_weird[, 1])
                                  vars: torch.Tensor,  # weird dims' variances, (num_weird,) or (batch, num_weird)
                                  ) -> 'StateTuple':
        d = mask.shape[0]
        batch = resid.shape[0]

        normal_idx = mask.nonzero(as_tuple=True)[0]
        weird_idx = (~mask).nonzero(as_tuple=True)[0]

        state_resid = state.resid.reshape(batch, -1)  # (batch, num_normal)
        resid_flat = resid.reshape(batch, -1)  # (batch, num_weird)
        vars_flat = vars.to(resid.dtype) if vars.dim() > 1 else vars.to(resid.dtype).expand(batch, -1)

        full_resid = torch.zeros(batch, d, device=resid.device, dtype=resid.dtype)
        full_resid[:, normal_idx] = state_resid
        full_resid[:, weird_idx] = resid_flat

        full_cov = torch.zeros(batch, d, d, device=resid.device, dtype=resid.dtype)
        full_cov[:, normal_idx.unsqueeze(-1), normal_idx.unsqueeze(0)] = state.system_cov
        full_cov[:, weird_idx, weird_idx] = vars_flat

        return StateTuple(mean=state.mean, cov=state.cov, resid=full_resid, system_cov=full_cov)

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
        mixture_components=[
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
