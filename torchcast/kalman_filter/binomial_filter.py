from typing import Tuple, Optional, Sequence, List, TYPE_CHECKING, Union, Dict

import numpy as np
import pandas as pd

from scipy.special import expit, logit

import torch
from torch import Tensor
from torch.distributions import Binomial

from torchcast.covariance import Covariance
from torchcast.covariance.util import mini_cov_mask
from torchcast.internals.utils import identity, class_or_instancemethod
from torchcast.kalman_filter import KalmanFilter
from torchcast.kalman_filter.ekf import EKFStep, EKFPredictions
from torchcast.process import Process
from torchcast.state_space.predictions import conf2bounds

if TYPE_CHECKING:
    from pandas import DataFrame


class BinomialStep(EKFStep):
    def __init__(self, binary_idx: Optional[Sequence[int]] = None, observed_counts: Optional[bool] = None):
        super().__init__()
        self.binary_idx = binary_idx
        self.observed_counts = observed_counts

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:

        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = list(range(input.shape[-1]))
        if (input[:, binary_idx] < 0).any():
            raise ValueError("BinomialFilter does not support negative inputs.")

        if self.observed_counts is None:
            if 'num_obs' in kwargs and (kwargs['num_obs'] != 1).any():
                raise ValueError(
                    "If `num_obs` is supplied, must specify whether observed values are counts (observed_counts=True) "
                    "or proportions (observed_counts=False)."
                )
        elif self.observed_counts:
            if 'num_obs' not in kwargs:
                raise ValueError("num_obs should be passed because observed_counts=True")
            input = input.clone()
            input[:, binary_idx] = input[:, binary_idx] / kwargs['num_obs']
            if (input[:, binary_idx] > 1).any():
                raise ValueError("Some inputs are > num_obs")

        return super()._update(
            input=input,
            mean=mean,
            cov=cov,
            kwargs=kwargs
        )

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor],
                   kwargs_dims: Optional[Dict[str, int]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        masked_input, new_kwargs = super()._mask_mats(
            groups=groups,
            val_idx=val_idx,
            input=input,
            kwargs=kwargs
        )

        if self.binary_idx is not None and val_idx is not None:
            new_kwargs['binary_idx'] = torch.as_tensor(
                [new_idx for new_idx, og_idx in enumerate(val_idx) if og_idx in self.binary_idx],
                dtype=torch.int,
                device=input.device
            )
        if 'num_obs' in kwargs:
            new_kwargs['num_obs'] = kwargs['num_obs'][groups]
            if val_idx is not None:
                _keep_idx = [i for i, bidx in enumerate(self.binary_idx) if bidx in val_idx]
                new_kwargs['num_obs'] = new_kwargs['num_obs'][:, _keep_idx]

        return masked_input, new_kwargs

    def _adjust_h(self, mean: Tensor, H: Tensor, kwargs: Dict[str, Tensor]) -> Tensor:
        """
        >>> import sympy
        >>> from sympy import exp, Matrix
        >>>
        >>>  def full_like(x, value):
        >>>     nrow, ncol = x.shape
        >>>     return sympy.Matrix([[value]*ncol]*nrow)
        >>>
        >>> def ones_like(x):
        >>>     return full_like(x, 1)
        >>>
        >>> def sigmoid(x):
        >>>     return (ones_like(x) + exp(-x)) ** -1
        >>>
        >>> sympy.init_printing(use_latex='mathjax')
        >>>
        >>> x1, x2, x3 = sympy.symbols("x, x', y")
        >>>
        >>> H = sympy.Matrix([[1, 0, 1]])
        >>>
        >>> state = sympy.Matrix([x1, x2, x3])
        >>> mmean_orig = H @ state
        >>> measured_mean = sigmoid(mmean_orig)
        >>>
        >>> J = measured_mean.jacobian(state)
        >>>
        >>> def jacob(h_dot_state):
        >>>     numer = exp(-h_dot_state)
        >>>     denom = (exp(-h_dot_state) + ones_like(h_dot_state)) ** 2
        >>>     return numer * denom ** -1  # sympy being weird, numer/denom doesn't work
        >>>
        >>> sympy.matrices.dense.matrix_multiply_elementwise(
        >>>     full_like(H, jacob(mmean_orig)),
        >>>     H
        >>> )
        """
        # this assert should not fail, b/c all processes only support a single measure
        assert not ((H != 0).sum(-2) > 1).any(), "BinomialFilter does not support the provided measurement-matrix"
        all_idx = list(range(H.shape[-1]))
        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = all_idx
        h_dot_state = (H @ mean.unsqueeze(-1)).squeeze(-1)
        adjustment = torch.ones_like(h_dot_state)
        binary_h_dot_state = h_dot_state[..., binary_idx].clamp(-10, 10)
        numer = torch.exp(-binary_h_dot_state)
        denom = (torch.exp(-binary_h_dot_state) + 1) ** 2
        adjustment[..., binary_idx] = numer / denom
        return H * adjustment.unsqueeze(-1)

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor], kwargs: dict[str, Tensor]) -> Tensor:
        all_idx = list(range(R.shape[-1]))
        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = all_idx
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        # binomial variance:
        newR = torch.zeros_like(R)
        binary_measured_mean = measured_mean[..., binary_idx]
        # mean of binomial target is n*p, variance is n*p*(1-p)
        # mean of target that is `binom_target / n` is p, variance is p*(1-p)/n (scaling a RV by N  scales var by N**2)
        num_obs = kwargs.get('num_obs', 1)
        newR[..., binary_idx, binary_idx] = (
                binary_measured_mean * (1 - binary_measured_mean) / num_obs
        )

        if gaussian_idx:
            # for gaussian measures, would be much more readable code if we just looped over the gaussian dims
            # and modified newR in-place. but now that newR has grad, don't want to modify in-place.
            # so we use masking trick
            gaussian_idx = torch.as_tensor(gaussian_idx)
            gaussian_cidx = torch.meshgrid(torch.arange(newR.shape[0]), gaussian_idx, gaussian_idx, indexing='ij')
            expand_mask = mini_cov_mask(rank=len(all_idx), empty_idx=binary_idx, device=R.device)  # todo: cache?
            gaussianR = expand_mask @ R[gaussian_cidx] @ expand_mask.transpose(-1, -2)
            newR = newR + gaussianR

        return newR

    def _adjust_measurement(self, x: Tensor, kwargs: dict[str, Tensor]) -> Tensor:
        all_idx = list(range(x.shape[-1]))
        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = all_idx
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        # again some awkwardness due to avoiding in-place on tensors with grad
        binary_out = torch.zeros_like(x)
        binary_out[..., binary_idx] = torch.sigmoid(x[..., binary_idx].clamp(-8, 8))
        gaussian_out = torch.zeros_like(x)
        gaussian_out[..., gaussian_idx] = x[..., gaussian_idx]
        return binary_out + gaussian_out


class BinomialFilter(KalmanFilter):
    ss_step_cls = BinomialStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 binary_measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Union[Covariance, dict]] = None,
                 initial_covariance: Optional[Covariance] = None,
                 observed_counts: Optional[bool] = None,
                 **kwargs):

        if isinstance(measures, str):
            raise ValueError(f"`measures` should be a list of strings not a string.")
        if isinstance(binary_measures, str):
            raise ValueError(f"`binary_measures` should be a list of strings not a string.")

        if binary_measures is None:
            binary_measures = list(measures)
        unexpected = set(binary_measures) - set(measures)
        if unexpected:
            raise RuntimeError(f"Some `binary_measures` not in `measures`: {unexpected}")
        self.binary_measures = measures if binary_measures is None else binary_measures

        mcov_empty_idx = [i for i, m in enumerate(measures) if m in self.binary_measures]
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
                    f"(rather than passing the ``Covariance`` itself), then {type(self).__name__} will figure out the "
                    f"empty-idx for you."
                )
        else:
            measure_covariance = Covariance(**measure_covariance)

        self.observed_counts = observed_counts

        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=measure_covariance,
            initial_covariance=initial_covariance,
            **kwargs
        )

    def _get_ss_step_kwargs(self) -> dict:
        return {
            'binary_idx': [idx for idx, m in enumerate(self.measures) if m in self.binary_measures],
            'observed_counts': self.observed_counts
        }

    def _get_measure_scaling(self) -> Tensor:
        # TODO: less code duplication?
        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        measure_var = mcov.diagonal(dim1=-2, dim2=-1)
        multi = torch.zeros(mcov.shape[0:-2] + (self.state_rank,), dtype=mcov.dtype, device=mcov.device)
        for pid, process in self.processes.items():
            pidx = self.process_to_slice[pid]
            if process.measure in self.binary_measures:
                multi[..., slice(*pidx)] = 1.0
            else:
                multi[..., slice(*pidx)] = measure_var[..., self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
        assert (multi > 0).all()
        return multi

    @torch.jit.ignore
    def _generate_predictions(self,
                              preds: Tuple[List[Tensor], List[Tensor]],
                              updates: Optional[Tuple[List[Tensor], List[Tensor]]] = None,
                              **kwargs) -> 'BinomialPredictions':
        if updates is not None:
            kwargs.update(update_means=updates[0], update_covs=updates[1])
        preds = BinomialPredictions(
            *preds,
            R=kwargs.pop('R'),
            H=kwargs.pop('H'),
            model=self,
            binary_measures=self.binary_measures,
            observed_counts=self.observed_counts,
            **kwargs
        )
        return preds

    @torch.jit.ignore()
    def set_initial_values(self,
                           y: Tensor,
                           ilinks: Optional[dict[str, callable]] = None,
                           verbose: bool = True,
                           num_obs: Optional[Tensor] = None,
                           **kwargs):
        y = y.clone()
        if self.observed_counts:
            if num_obs is None:
                raise RuntimeError("num_obs should be passed because observed_counts=True")
            binary_idx = [i for i, m in enumerate(self.measures) if m in self.binary_measures]
            y[..., binary_idx] /= num_obs
        ilinks = {m: (logit if m in self.binary_measures else identity) for m in self.measures}
        return super().set_initial_values(y=y, ilinks=ilinks, verbose=verbose)

    @torch.jit.ignore()
    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        num_obs = kwargs.pop('num_obs', None)
        kwargs_per_process = super()._parse_design_kwargs(input=input, out_timesteps=out_timesteps, **kwargs)
        if num_obs is None:
            if self.observed_counts:
                raise RuntimeError("num_obs should be passed because observed_counts=True")
            kwargs_per_process['num_obs'] = 1
        else:
            kwargs_per_process['num_obs'] = num_obs
        return kwargs_per_process

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        num_obs = kwargs_per_process.pop('num_obs', None)
        predict_kwargs, update_kwargs = super()._build_design_mats(
            kwargs_per_process=kwargs_per_process,
            num_groups=num_groups,
            out_timesteps=out_timesteps
        )
        if isinstance(num_obs, int):
            device = next(iter(self.parameters())).device
            _nobs = [torch.ones(num_groups, len(self.binary_measures), device=device) * num_obs]
            update_kwargs['num_obs'] = _nobs * out_timesteps
        elif num_obs is not None:
            update_kwargs['num_obs'] = num_obs.unbind(1)
        return predict_kwargs, update_kwargs

    def fit(self,
            *args,
            mc_samples: int,
            **kwargs):
        """
        :param mc_samples: Number of samples to draw for MC approximation to binomial likelihood.
        :param kwargs: Additional keyword arguments, see `func:torchcast.kalman_filter.KalmanFilter.fit`.
        """
        device = next(iter(self.parameters())).device
        kwargs['prediction_kwargs'] = kwargs.get('prediction_kwargs', {})
        if mc_samples:
            wn = torch.randn((mc_samples, len(self.binary_measures)), device=device)
        else:
            wn = torch.zeros((1, len(self.binary_measures)), device=device)
        kwargs['prediction_kwargs']['white_noise'] = wn
        return super().fit(*args, **kwargs)


class BinomialPredictions(EKFPredictions):

    def __init__(self,
                 *args,
                 binary_measures: Sequence[str],
                 observed_counts: Optional[bool] = None,
                 num_obs: List[Tensor],
                 white_noise: Optional[Tensor] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_obs = num_obs if isinstance(num_obs, torch.Tensor) else torch.stack(num_obs, 1)
        self.binary_measures = binary_measures
        self.observed_counts = observed_counts
        self._white_noise = white_noise

    @property
    def white_noise(self) -> Tensor:
        if self._white_noise is None:
            # todo: explain more
            raise RuntimeError("Cannot access `white_noise` attribute because it was not passed at init.")
        return self._white_noise

    def _get_log_prob_kwargs(self, groups: Tensor, valid_idx: Tensor) -> dict:
        white_noise = self.white_noise

        num_obs = None
        if self.binary_measures:
            num_obs = self.num_obs.view(-1, len(self.binary_measures))[groups]

        if valid_idx is not None:
            valid_measures = [m for i, m in enumerate(self.measures) if i in valid_idx]
            valid_binary_idx = [i for i, m in enumerate(self.binary_measures) if m in valid_measures]
            if num_obs is not None:
                num_obs = num_obs[:, valid_binary_idx]
            white_noise = white_noise[:, valid_binary_idx]
        out = {
            'measures': [m for i, m in enumerate(self.measures) if valid_idx is None or i in valid_idx],
            'white_noise': white_noise
        }
        if num_obs is not None:
            out['num_obs'] = num_obs

        return out

    # noinspection PyMethodOverriding
    def _log_prob(self,
                  obs: Tensor,
                  state_means: Tensor,
                  state_covs: Tensor,
                  R: Tensor,
                  H: Tensor,
                  white_noise: Tensor,
                  measures: Sequence[str],
                  num_obs: Optional[Tensor] = None,
                  **kwargs) -> Tensor:
        means, covs = self.observe(state_means, state_covs, R, H)
        group_idx = torch.arange(covs.shape[0], dtype=torch.int)
        binary_idx = torch.as_tensor([i for i, m in enumerate(measures) if m in self.binary_measures], dtype=torch.int)
        gauss_idx = torch.as_tensor([i for i in range(covs.shape[-1]) if i not in binary_idx], dtype=torch.int)
        if len(gauss_idx):
            gauss_cidx = torch.meshgrid(group_idx, gauss_idx, gauss_idx, indexing='ij')
            gauss = torch.distributions.MultivariateNormal(means[..., gauss_idx], covs[gauss_cidx], validate_args=False)
            gauss_lp = gauss.log_prob(obs[..., gauss_idx])
        else:
            gauss_lp = 0

        if len(binary_idx):
            if num_obs is None:
                raise RuntimeError("num_obs should be set because there are binary-measures")
            binary_cidx = torch.meshgrid(group_idx, binary_idx, binary_idx, indexing='ij')
            chol = torch.linalg.cholesky(covs[binary_cidx])
            corr_white_noise = chol.unsqueeze(0) @ white_noise.view(-1, 1, len(binary_idx), 1)
            mc_samples = corr_white_noise.squeeze(-1) + means[..., binary_idx].unsqueeze(0)
            binom = Binomial(total_count=num_obs.unsqueeze(0), logits=mc_samples, validate_args=False)
            _obs = obs[..., binary_idx]
            if not self.observed_counts:  # multiply by total_count b/c `obs` are props, but Binomial expects counts:
                _obs = _obs * binom.total_count
            # we don't want log_prob(x).mean(0), we want prob(x).mean(0).log()
            # this is a numerically stable way to do that:
            mc_log_probs = binom.log_prob(_obs)
            mc_marginal = torch.logsumexp(mc_log_probs, dim=0) - np.log(mc_samples.shape[0])
            binary_lp = mc_marginal.sum(-1)
        else:
            binary_lp = 0

        return gauss_lp + binary_lp

    @class_or_instancemethod
    def _adjust_measured_mean(cls,
                              x: Union[Tensor, np.ndarray, pd.Series],
                              measure: str,
                              std: Optional[Union[Tensor, np.ndarray, pd.Series]] = None,
                              conf: float = .95) -> Union[Tensor, pd.DataFrame]:
        if isinstance(cls, type):
            raise RuntimeError(f"Cannot call this method as a classmethod for {cls.__name__}, only instance method.")

        x_index = None
        if hasattr(x, 'to_numpy'):
            x_index = x.index if hasattr(x, 'index') else None
            x = x.to_numpy()

        if std is None:
            x = torch.as_tensor(x)
            if measure in cls.binary_measures:
                return torch.sigmoid(x)
            return x

        with torch.no_grad():
            if hasattr(std, 'to_numpy'):
                std = std.to_numpy()

            # todo: this is only state-uncertainty, should we also include measurement-uncertainty: p*(1-p)?
            lower, upper = conf2bounds(x, std, conf=conf)
            out = pd.DataFrame({'lower': lower, 'upper': upper}, index=None if x_index is None else x_index)

            if measure in cls.binary_measures:
                # if len(x.shape) > 1:
                #     raise NotImplementedError("TODO")
                # grid = np.linspace(0, 1, 100, endpoint=False) + 0.5 / 100
                # points = stats.norm(x[:, None], std[:, None]).ppf(grid[None, :])
                # out['mean'] = expit(points).mean(1)
                out['mean'] = expit(x)
                out['lower'] = expit(out['lower'])
                out['upper'] = expit(out['upper'])
            else:
                out['mean'] = x

            return out

    def _getitem_helper(self, item: tuple) -> dict:
        kwargs = super()._getitem_helper(item)
        kwargs['num_obs'] = self.num_obs[item]
        kwargs['white_noise'] = self._white_noise
        kwargs['binary_measures'] = self.binary_measures
        kwargs['observed_counts'] = self.observed_counts
        return kwargs


def main(num_groups: int = 100, num_timesteps: int = 100, bias: float = -1, prop_common: float = 1.):
    from torchcast.process import LocalLevel
    from torchcast.utils import TimeSeriesDataset
    import pandas as pd
    from plotnine import geom_line, aes, ggtitle
    torch.manual_seed(1234)

    total_count = 20
    measures = ['dim1', 'dim2']
    binary_measures = ['dim1']
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
            y.append(torch.distributions.Binomial(logits=latent[..., i], total_count=total_count).sample())
            # y[-1] /= total_count
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
        processes=[LocalLevel(id=f'level_{m}', measure=m) for m in measures],
        measures=measures,
        binary_measures=binary_measures,
        observed_counts=True
    )

    y = dataset.tensors[0]
    bf.fit(y,
           num_obs=total_count,
           mc_samples=64)
    preds = bf(
        dataset.tensors[0], num_obs=total_count
    )
    df_preds = preds.to_dataframe(dataset)
    df_preds.loc[df_preds['measure'].isin(binary_measures), ['mean', 'lower', 'upper']] *= total_count
    df_latent = (dataset.to_dataframe()
                 .drop(columns=measures)
                 .melt(id_vars=['group', 'time'], var_name='measure', value_name='latent')
                 .assign(measure=lambda _df: _df['measure'].str.replace('latent', 'dim')))
    _is_binary = df_latent['measure'].isin(binary_measures)
    df_latent.loc[_is_binary, 'latent'] = expit(df_latent.loc[_is_binary, 'latent']) * total_count

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
