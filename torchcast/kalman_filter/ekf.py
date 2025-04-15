from typing import Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torchcast.internals.utils import class_or_instancemethod
from torchcast.state_space import Predictions
from torchcast.utils import TimeSeriesDataset
from torchcast.kalman_filter.kalman_filter import KalmanStep


class EKFStep(KalmanStep):
    """
    Implements update for the extended kalman-filter. Currently limited:

    1. Does not implement any special ``predict``, so no custom functions for transition (only measurement).
    2. Assumes the custom measure function takes the form ``custom_fun(H @ state)``.

    This means that currently, this is primarily useful for approximating non-gaussian measures (e.g. poisson,
     binomial) via a link function.
    """

    def _adjust_h(self, mean: Tensor, H: Tensor, kwargs: Dict[str, Tensor]) -> Tensor:
        return H

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor], kwargs: Dict[str, Tensor]) -> Tensor:
        assert R is not None
        return R

    def _adjust_measurement(self, x: Tensor, kwargs: Dict[str, Tensor]) -> Tensor:
        return x

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        orig_H = kwargs['H']
        h_dot_state = (orig_H @ mean.unsqueeze(-1)).squeeze(-1)
        aux_kwargs = {k: v for k, v in kwargs.items() if k not in ('H', 'R')}
        kwargs['measured_mean'] = self._adjust_measurement(h_dot_state, kwargs)
        kwargs['H'] = self._adjust_h(mean, orig_H, kwargs)
        kwargs['R'] = self._adjust_r(kwargs['measured_mean'], kwargs.get('R', None), aux_kwargs)

        return super()._update(
            input=input,
            mean=mean,
            cov=cov,
            kwargs=kwargs
        )


class EKFPredictions(Predictions):
    @classmethod
    def _adjust_measured_mean(cls,
                              x: Union[Tensor, np.ndarray, pd.Series],
                              measure: str,
                              std: Optional[Union[Tensor, np.ndarray, pd.Series]] = None,
                              conf: float = .95) -> Union[Tensor, pd.DataFrame]:
        """
        In our EKF, the measured-mean is ``custom_fun(H @ state)``.

        - If only ``x`` (``= H @ state``) is passed, this method should apply the custom fun -- supporting x that is
          either a tensor (with grad), a numpy array, or a pandas series.
        - If both ``x`` and ``std`` is passed, this method should return a dataframe with mean, lower, upper columns.
          The mean column should have any bias-correction applied, and the lower/upper should be conf% confidence
          bounds (e.g. for plotting).
        """
        raise NotImplementedError

    def _log_prob(self,
                  obs: Tensor,
                  state_means: Tensor,
                  state_covs: Tensor,
                  R: Tensor,
                  H: Tensor,
                  **kwargs) -> Tensor:
        raise NotImplementedError

    def __array__(self) -> np.ndarray:
        with torch.no_grad():
            stds = torch.diagonal(self.covs, dim1=-1, dim2=-2).sqrt()
            out = []
            for i, m in enumerate(self.measures):
                out.append(self._adjust_measured_mean(self.means[..., i], std=stds[..., i], measure=m))
            return torch.stack(out, -1).numpy()

    def to_dataframe(self,
                     dataset: Optional[TimeSeriesDataset] = None,
                     type: str = 'predictions',
                     group_colname: Optional[str] = None,
                     time_colname: Optional[str] = None,
                     conf: Optional[float] = .95) -> pd.DataFrame:
        df = super().to_dataframe(
            dataset=dataset,
            type=type,
            group_colname=group_colname,
            time_colname=time_colname,
            conf=None
        )
        for m, _df in df.groupby('measure'):
            df.loc[_df.index, ['mean', 'lower', 'upper']] = self._adjust_measured_mean(
                _df['mean'],
                std=_df['std'],
                measure=m,
                conf=conf
            )
        return df.drop(columns=['std'])

    @class_or_instancemethod
    def plot(cls,
             df: Optional[pd.DataFrame] = None,
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs) -> pd.DataFrame:

        if df is not None and 'upper' not in df.columns and 'std' in df.columns:
            df = df.copy()
            for m, _df in df.groupby('measure'):
                df.loc[_df.index, ['mean', 'lower', 'upper']] = cls._adjust_measured_mean(
                    _df['mean'],
                    std=_df['std'],
                    measure=m,
                )
            df = df.drop(columns=['std'])

        return super().plot(
            df=df,
            group_colname=group_colname,
            time_colname=time_colname,
            max_num_groups=max_num_groups,
            split_dt=split_dt,
            **kwargs
        )
