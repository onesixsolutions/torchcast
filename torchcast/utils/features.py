import math
from typing import Union, Optional, TYPE_CHECKING
import numpy as np
import torch

if TYPE_CHECKING:
    from pandas import DataFrame, Series


def add_season_features(data: 'DataFrame',
                        K: int,
                        period: Union[np.timedelta64, str],
                        time_colname: Optional[str] = None) -> 'DataFrame':
    """
    Add season features to `data` by taking a date[time]-column and passing it through a fourier-transform.

    :param data: A dataframe with a date[time] column.
    :param K: The degrees of freedom for the fourier transform. Higher K means more flexible seasons can be captured.
    :param period: Either a np.timedelta64, or one of {'weekly','yearly','daily'}
    :param time_colname: The name of the date[time] column. Default is to try and guess with the following (in order):
     'datetime', 'date', 'timestamp', 'time'.
    :return: A copy of the original dataframe, now with K*2 additional columns capturing the seasonal pattern.
    """
    from pandas import concat

    if time_colname is None:
        try:
            time_colname = next(col for col in ('datetime', 'date', 'timestamp', 'time', 'dt') if col in data.columns)
        except StopIteration:
            raise ValueError("Unable to guess `time_colname`, please pass")
    df_season = fourier_model_mat(data[time_colname].values, K=K, period=period, output_fmt='dataframe')
    already = df_season.columns.isin(data.columns)
    if already.all():
        return data.copy(deep=False)
    elif already.any():
        raise RuntimeError(
            f"Some, but not all, of the following columns are already in `data`:\n{df_season.columns.tolist()}"
        )
    df_season.index = data.index
    return concat([data, df_season], axis=1)


def fourier_model_mat(datetimes: Union[np.ndarray, 'Series'],
                      K: int,
                      period: Union[np.timedelta64, str],
                      output_fmt: str = 'float64') -> np.ndarray:
    """
    :param datetimes: An array of datetimes.
    :param K: The expansion integer.
    :param period: Either a np.timedelta64, or one of {'weekly','yearly','daily'}
    :param output_fmt: A numpy dtype, or 'dataframe' to output a dataframe.
    :return: A numpy array (or dataframe) with the expanded fourier series.
    """
    # parse period:
    name = 'fourier'
    if isinstance(period, str):
        name = period
        if period == 'weekly':
            period = np.timedelta64(7, 'D')
        elif period == 'yearly':
            period = np.timedelta64(int(365.25 * 24), 'h')
        elif period == 'daily':
            period = np.timedelta64(24, 'h')
        else:
            raise ValueError("Unrecognized `period`.")

    orig_index = None
    if not isinstance(datetimes, np.ndarray) and isinstance(getattr(datetimes, 'values', None), np.ndarray):
        orig_index = getattr(datetimes, 'index', None)
        datetimes = datetimes.values

    output_dataframe = (output_fmt.lower() == 'dataframe')
    if output_dataframe:
        if len(datetimes.shape) == 2 and datetimes.shape[1] == 1:
            datetimes = datetimes.squeeze(-1)
        output_fmt = 'float64'

    period_int = int(period / np.timedelta64(1, 'ns'))
    time_int = (datetimes.astype("datetime64[ns]") - np.datetime64(0, 'ns')).astype('int64')

    # fourier matrix:
    out_shape = tuple(datetimes.shape) + (K * 2,)
    out = np.empty(out_shape, dtype=output_fmt)
    columns = []
    for idx in range(K):
        k = idx + 1
        for is_cos in range(2):
            val = 2. * np.pi * k * time_int / period_int
            out[..., idx * 2 + is_cos] = np.sin(val) if is_cos == 0 else np.cos(val)
            columns.append(f"{name}_K{k}_{'cos' if is_cos else 'sin'}")

    if output_dataframe:
        if len(out_shape) > 2:
            raise ValueError("Cannot output dataframe when input is 2+D array.")
        from pandas import DataFrame
        out = DataFrame(out, columns=columns, index=orig_index)

    return out


def fourier_tensor(time: torch.Tensor, seasonal_period: float, K: int) -> torch.Tensor:
    """
    Given an N-dimensional tensor, create an N+2 dimensional tensor with the 2nd to last dimension corresponding to the
    Ks and the last dimension corresponding to sin/cos.
    """
    out = torch.empty(time.shape + (K, 2))
    for idx in range(K):
        k = idx + 1
        for sincos in range(2):
            val = 2. * math.pi * k * time / seasonal_period
            out[..., idx, sincos] = torch.sin(val) if sincos == 0 else torch.cos(val)
    return out
