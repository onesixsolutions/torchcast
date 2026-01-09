from math import log

import torch
from torch.distributions import Binomial
from typing import Sequence, TYPE_CHECKING, Optional, Union

from torchcast.covariance import Covariance
from torchcast.kalman_filter import KalmanFilter
from torchcast.state_space import Predictions
from torchcast.internals.batch_design import MeasurementModel, Sigmoid

if TYPE_CHECKING:
    from torchcast.process import Process


class BinomialFilter(KalmanFilter):
    def __init__(self,
                 processes: Sequence['Process'],
                 measures: Optional[Sequence[str]],
                 binary_measures: Optional[Sequence[str]] = None,
                 observed_counts: Optional[bool] = None,
                 measure_covariance: Optional[Covariance] = None,
                 process_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None,
                 adaptive_scaling: bool = False):

        if binary_measures is None:
            binary_measures = list(measures)

        measure_covariance = self._validate_measure_cov(
            measures=measures,
            binary_measures=binary_measures,
            measure_covariance=measure_covariance
        )
        self.observed_counts = observed_counts
        self.binary_measures = binary_measures

        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=measure_covariance,
            initial_covariance=initial_covariance,
            adaptive_scaling=adaptive_scaling,
            measure_funs={m: 'ilogit' for m in binary_measures},
        )

    @classmethod
    def _validate_measure_cov(cls,
                              measures: Sequence[str],
                              binary_measures: Sequence[str],
                              measure_covariance: Optional[Covariance]) -> Covariance:
        if isinstance(measures, str):
            raise ValueError(f"`measures` should be a list of strings not a string.")
        if isinstance(binary_measures, str):
            raise ValueError(f"`binary_measures` should be a list of strings not a string.")

        unexpected = set(binary_measures) - set(measures)
        if unexpected:
            raise RuntimeError(f"Some `binary_measures` not in `measures`: {unexpected}")

        mcov_empty_idx = [i for i, m in enumerate(measures) if m in binary_measures]
        if measure_covariance is None:
            measure_covariance = {}
        if isinstance(measure_covariance, dict):
            measure_covariance['id'] = 'measure_covariance'
            measure_covariance['rank'] = len(measures)
            measure_covariance['empty_idx'] = mcov_empty_idx

        if isinstance(measure_covariance, Covariance):  # todo: we should be able to eliminate this mess
            if set(measure_covariance.empty_idx) != set(mcov_empty_idx):
                raise ValueError(
                    f"Expected ``empty_idx`` to correspond to binary measures (i.e. {mcov_empty_idx}) but they did not "
                    f"(i.e. got {measure_covariance.empty_idx}). To resolve this, you could instead supply for the "
                    f"`measure_covariance` argument the keyword arguments for initializing a ``Covariance`` object "
                    f"(rather than passing the ``Covariance`` itself), then {cls.__name__} will figure out the "
                    f"empty-idx for you."
                )
        else:
            measure_covariance = Covariance(**measure_covariance)

        return measure_covariance

    def _generate_predictions(self,
                              preds: tuple[list[torch.Tensor], list[torch.Tensor]],
                              updates: Optional[tuple[list[torch.Tensor], list[torch.Tensor]]],
                              measure_covs: torch.Tensor,
                              measurement_model: 'MeasurementModel',
                              num_obs: Sequence[torch.Tensor],
                              observed_counts: bool,
                              **kwargs
                              ) -> 'Predictions':
        if kwargs:
            raise TypeError(f"{type(self).__name__} got unexpected kwargs: {set(kwargs)})")
        return BinomialPredictions(
            measurement_model=measurement_model,
            states=preds,
            measure_covs=measure_covs,
            updates=updates,
            mc_white_noise=self.mc_sampling if self.is_nonlinear else None,
            num_obs=num_obs,
            observed_counts=observed_counts,
        )

    def _mask_mats(self,
                   groups: torch.Tensor,
                   masks: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                   binary_idx: Optional[Sequence[int]] = None,
                   **kwargs) -> dict:
        out = super()._mask_mats(groups, masks, **kwargs)
        if masks is None or binary_idx is None:
            return out
        val_idx = masks[0]
        out['binary_idx'] = torch.isin(val_idx, torch.as_tensor(binary_idx)).nonzero().squeeze(-1)
        _binary_subset_idx = torch.tensor([i1 for i1, i2 in enumerate(binary_idx) if i2 in val_idx], dtype=torch.long)
        m1d = torch.meshgrid(groups, _binary_subset_idx, indexing='ij')
        out['num_obs'] = kwargs['num_obs'][m1d]
        return out

    @torch.no_grad()
    def _get_good_initial_value_from_y(self,
                                       y: torch.Tensor,
                                       measure: str,
                                       num_obs: Optional[torch.Tensor] = None,
                                       **kwargs) -> torch.Tensor:
        if measure in self.binary_measures and self.observed_counts:
            if num_obs is None:
                raise ValueError("num_obs should be passed because observed_counts=True")
            y = y.clone()
            midx = self.measures.index(measure)
            if not isinstance(num_obs, int):
                num_obs = num_obs[..., self.binary_measures.index(measure)]
            y[..., midx] = y[..., midx] / num_obs

        return super()._get_good_initial_value_from_y(
            y=y,
            measure=measure,
            **kwargs
        )

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
        if 'num_obs' in kwargs:
            num_obs = kwargs.pop('num_obs')
            if isinstance(num_obs, int):
                _nobs = [torch.ones(num_groups, len(self.binary_measures), device=measure_covs[0].device) * num_obs]
                update_kwargs['num_obs'] = _nobs * num_timesteps
            else:
                update_kwargs['num_obs'] = num_obs.unbind(1)
            used_keys.add('num_obs')
        return predict_kwargs, update_kwargs, used_keys

    def _update_step_with_nans(self,
                               input: torch.Tensor,
                               mean: torch.Tensor,
                               cov: torch.Tensor,
                               measured_mean: torch.Tensor,
                               measure_mat: torch.Tensor,
                               measure_cov: torch.Tensor,
                               **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._update_step_with_nans(
            input=input,
            mean=mean,
            cov=cov,
            measured_mean=measured_mean,
            measure_mat=measure_mat,
            measure_cov=measure_cov,
            binary_idx=[i for i, m in enumerate(self.measures) if m in self.binary_measures],
            **kwargs
        )

    def _update_step(self,
                     input: torch.Tensor,
                     mean: torch.Tensor,
                     cov: torch.Tensor,
                     measured_mean: torch.Tensor,
                     measure_mat: torch.Tensor,
                     measure_cov: torch.Tensor,
                     num_obs: Optional[torch.Tensor],
                     binary_idx: Sequence[int],
                     **kwargs) -> tuple[torch.Tensor, torch.Tensor]:

        # validate input:
        if (input[:, binary_idx] < 0).any():
            raise ValueError("BinomialFilter does not support negative inputs.")

        # validate num_obs, use to normalize input if observed_counts=True:
        if self.observed_counts is None:
            if num_obs is not None and (num_obs != 1).any():
                raise ValueError(
                    "If `num_obs` is supplied, must specify whether observed values are counts (observed_counts=True) "
                    "or proportions (observed_counts=False)."
                )
        elif self.observed_counts:
            if num_obs is None:
                raise ValueError("num_obs should be passed because observed_counts=True")
            input = input.clone()
            input[:, binary_idx] = input[:, binary_idx] / num_obs
            if (input[:, binary_idx] > 1).any():
                raise ValueError("Some inputs are > num_obs")

        # adjust measure-cov based on binomial identity relationship:
        bin_measure_cov = torch.zeros_like(measure_cov)
        binary_measured_mean = measured_mean[..., binary_idx]
        # mean of binomial target is n*p, variance is n*p*(1-p)
        # mean of target that is `binom_target / n` is p, variance is p*(1-p)/n (scaling a RV by N scales var by N**2)
        bin_measure_cov[..., binary_idx, binary_idx] = (
                binary_measured_mean * (1 - binary_measured_mean) / num_obs
        )
        measure_cov = measure_cov + bin_measure_cov

        return super()._update_step(
            input=input,
            mean=mean,
            cov=cov,
            measured_mean=measured_mean,
            measure_mat=measure_mat,
            measure_cov=measure_cov,
            **kwargs
        )

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

        prediction_kwargs = prediction_kwargs or {}
        prediction_kwargs['observed_counts'] = self.observed_counts
        if 'num_obs' not in kwargs:
            kwargs['num_obs'] = 1
        prediction_kwargs['num_obs'] = kwargs['num_obs']

        return super().forward(
            y=y,
            n_step=n_step,
            start_offsets=start_offsets,
            out_timesteps=out_timesteps,
            initial_state=initial_state,
            every_step=every_step,
            include_updates_in_output=include_updates_in_output,
            simulate=simulate,
            last_measured_per_group=last_measured_per_group,
            prediction_kwargs=prediction_kwargs,
            **kwargs
        )


class BinomialPredictions(Predictions):
    def __init__(self,
                 measurement_model: 'MeasurementModel',
                 states: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
                 measure_covs: Union[Sequence[torch.Tensor], torch.Tensor],
                 num_obs: Sequence[torch.Tensor],
                 observed_counts: Optional[bool] = None,
                 **kwargs):

        super().__init__(
            measurement_model=measurement_model,
            states=states,
            measure_covs=measure_covs,
            **kwargs
        )

        self.observed_counts = observed_counts
        self.binary_measures = [m for m, f in measurement_model.measure_funs.items() if isinstance(f, Sigmoid)]
        if isinstance(num_obs, int):
            num_obs = torch.full(
                (self.num_groups, self.num_timesteps, len(self.binary_measures)),
                fill_value=num_obs,
                device=measurement_model.device
            )
        self.num_obs = num_obs if isinstance(num_obs, torch.Tensor) else torch.stack(num_obs, 1)

    def _getitem_helper(self, item: tuple) -> dict:
        out = super()._getitem_helper(item)
        out['observed_counts'] = self.observed_counts
        out['num_obs'] = self.num_obs[item]
        return out

    def _get_log_prob_kwargs(self, groups: torch.Tensor, valid_idx: Optional[torch.Tensor]) -> dict:
        out = super()._get_log_prob_kwargs(groups, valid_idx)

        if self.binary_measures:
            num_obs = self.num_obs.view(-1, len(self.binary_measures))[groups]
            if valid_idx is not None:
                _valid_measures = [m for i, m in enumerate(self.measurement_model.measures) if i in valid_idx]
                valid_binary_idx = [i for i, m in enumerate(self.binary_measures) if m in _valid_measures]
                num_obs = num_obs[:, valid_binary_idx]
            out['num_obs'] = num_obs

        return out

    def _log_prob(self,
                  obs: torch.Tensor,
                  state_means: torch.Tensor,
                  state_covs: torch.Tensor,
                  measure_cov: torch.Tensor,
                  measurement_model: 'MeasurementModel',
                  num_obs: Optional[torch.Tensor] = None,
                  **kwargs) -> torch.Tensor:
        if kwargs:
            raise TypeError(f"`_log_prob()` does not accept additional keyword arguments, got {set(kwargs)}")

        binary_idx = [i for i, m in enumerate(measurement_model.measures) if m in self.binary_measures]
        gauss_idx = [i for i, m in enumerate(measurement_model.measures) if m not in self.binary_measures]
        gaussian_measures = [m for m in measurement_model.measures if m not in self.binary_measures]
        group_idx = torch.arange(obs.shape[0], dtype=torch.long)

        if gauss_idx:
            gauss_idx = torch.as_tensor(gauss_idx, dtype=torch.long)
            mask2d = torch.meshgrid(group_idx, gauss_idx, gauss_idx, indexing='ij')
            gaussian_lp = super()._log_prob(
                obs=obs[..., gauss_idx],
                state_means=state_means,
                state_covs=state_covs,
                measure_cov=measure_cov[mask2d],
                measurement_model=measurement_model.subset(measures=gaussian_measures),
                **kwargs
            )
        else:
            gaussian_lp = 0

        if len(binary_idx):
            if num_obs is None:
                raise RuntimeError("num_obs should be set because there are binary measures")
            mmean_samples = self._get_measured_mean_samples(
                measurement_model=measurement_model.subset(measures=self.binary_measures),
                state_means=state_means,
                state_covs=state_covs,
            )
            binom = Binomial(total_count=num_obs.unsqueeze(0), probs=mmean_samples, validate_args=False)
            _obs = obs[..., binary_idx]
            if not self.observed_counts:  # multiply by total_count b/c `obs` are props, but Binomial expects counts:
                _obs = _obs * num_obs
            mc_log_probs = binom.log_prob(_obs.unsqueeze(0))
            binary_lp = torch.sum(torch.logsumexp(mc_log_probs, dim=0), -1) - log(mc_log_probs.shape[0])
        else:
            binary_lp = 0

        return gaussian_lp + binary_lp


def main(num_groups: int = 50, num_timesteps: int = 365, bias: float = -1, prop_common: float = 1.):
    from torchcast.process import LocalLevel, Season
    from torchcast.utils import TimeSeriesDataset
    from scipy.special import expit
    import pandas as pd
    from plotnine import geom_line, aes, ggtitle
    torch.manual_seed(1234)

    TOTAL_COUNT = 4
    measures = ['dim1', 'dim2']
    binary_measures = ['dim2']
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
        if m in binary_measures:
            y.append(torch.distributions.Binomial(logits=latent[..., i], total_count=TOTAL_COUNT).sample())
            y[-1] /= TOTAL_COUNT
            y[-1][:, int(num_timesteps * .7):] = float('nan')
        else:
            y.append(torch.distributions.Normal(loc=latent[..., i], scale=.5).sample())
        y[-1][torch.randn((num_groups, num_timesteps)) > 1.5] = float('nan')  # some random missings
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

    bf = BinomialFilter(
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures]
                  + [Season(id=f'season_{m}', measure=m, dt_unit='D', period=7, K=2) for m in measures],
        measures=measures,
        binary_measures=binary_measures,
        observed_counts=False
    )

    y = dataset.tensors[0]
    bf.fit(y, start_offsets=dataset.start_offsets)
    _kwargs = {}
    # if TOTAL_COUNT != 1:
    #     _kwargs['num_obs'] = TOTAL_COUNT
    preds = bf(
        dataset.tensors[0],
        start_offsets=dataset.start_offsets,
        **_kwargs,
    )
    df_preds = preds.to_dataframe(dataset)
    if bf.observed_counts:
        df_preds.loc[df_preds['measure'].isin(binary_measures), ['mean', 'lower', 'upper']] *= TOTAL_COUNT
    df_latent = (dataset.to_dataframe()
                 .drop(columns=measures)
                 .melt(id_vars=['group', 'time'], var_name='measure', value_name='latent')
                 .assign(measure=lambda _df: _df['measure'].str.replace('latent', 'dim')))
    _is_binary = df_latent['measure'].isin(binary_measures)
    df_latent.loc[_is_binary, 'latent'] = expit(df_latent.loc[_is_binary, 'latent'])

    df_plot = df_preds.merge(df_latent, how='left', on=['group', 'time', 'measure'])
    for g, _df in df_plot.query("group.isin(group.drop_duplicates().sample(5))").groupby('group'):
        (
                preds.plot(_df)
                + geom_line(aes(y='latent'), color='purple')
                + ggtitle(g)
        ).show()
    # preds._white_noise = torch.zeros((1, len(binary_measures)))
    # print(preds.log_prob(y).mean())


if __name__ == '__main__':
    main()
