from typing import Sequence, TYPE_CHECKING

import numpy as np
from platformdirs import user_cache_path

if TYPE_CHECKING:
    from pandas import DataFrame


def load_air_quality_data(dt_unit: str,
                          outage_prob: float = 0.,
                          outage_persistence: float = .9,
                          outage_corr: float = -.6,
                          outage_seed: int = 123) -> 'DataFrame':
    """

    This dataset comes from the
    [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).
     It includes data on air pollutants and weather from 12 sites.

     Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-
     based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management,
     Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17).

    :param dt_unit: 'daily' or 'weekly'
    :param outage_prob: To simulate sensor outages: the probability that each of the PM2.5 and PM10 sensors is down
     (its value missing) at a given station and time. Outages come in runs (see ``outage_persistence``). Default 0 (no
     outages).
    :param outage_persistence: How persistent outages are, between 0 and 1: the autocorrelation of the latent AR(1)
     process that, when it exceeds a threshold, puts a sensor down. Higher means longer outages.
    :param outage_corr: The correlation of the two sensors' latent processes: negative values make it rarer for both
     to be down at once.
    :param outage_seed: Seed for the outages.
    :return: A dataframe
    """
    if dt_unit.lower().startswith('d'):
        dt_unit = 'daily'
    elif dt_unit.lower().startswith('w'):
        dt_unit = 'weekly'
    else:
        raise ValueError(f"Unrecognized dt_unit {dt_unit}.")
    try:
        from pandas import read_csv
    except ImportError:
        raise RuntimeError("Requires `pandas` package.")

    cache_path = user_cache_path(appname='torchcast', ensure_exists=True)

    time_col = 'date' if dt_unit == 'daily' else 'week'
    try:
        df = read_csv(cache_path / f'aq-{dt_unit}.csv', parse_dates=[time_col])
    except FileNotFoundError:
        print('loading from gh...')
        read_csv(
            f'https://raw.githubusercontent.com/onesixsolutions/torchcast/c675f04bb244a73155fbd98c110590bd735808bc/docs/examples/aq_{dt_unit}.csv',
            parse_dates=[time_col]
        ).to_csv(cache_path / f'aq-{dt_unit}.csv', index=False)
        df = read_csv(cache_path / f'aq-{dt_unit}.csv', parse_dates=[time_col])
    if outage_prob:
        df = _add_sensor_outages(
            df,
            cols=('PM2p5', 'PM10'),
            prob=outage_prob,
            persistence=outage_persistence,
            corr=outage_corr,
            seed=outage_seed,
            time_col=time_col,
        )
    return df


def _add_sensor_outages(df: 'DataFrame',
                        cols: Sequence[str],
                        prob: float,
                        persistence: float,
                        corr: float,
                        seed: int,
                        group_col: str = 'station',
                        time_col: str = 'date') -> 'DataFrame':
    """
    Sets ``cols`` to NaN during simulated outages: per group, a latent AR(1) process per column,
    ``z_t = persistence * z_{t-1} + sqrt(1 - persistence^2) * eps_t`` with ``eps ~ N(0, [[1, corr], [corr, 1]])`` (so
    each ``z`` is standard normal), and the column is missing where ``z`` exceeds its ``1 - prob`` quantile.
    """
    from scipy import stats

    if not 0 < prob < 1:
        raise ValueError(f"`outage_prob` should be between 0 and 1, got {prob}.")
    if len(cols) != 2:
        raise ValueError("Expected two columns.")
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(np.array([[1., corr], [corr, 1.]]))
    df = df.sort_values([group_col, time_col]).reset_index(drop=True)
    z = np.empty((len(df), 2))
    start = 0
    for n in df.groupby(group_col, sort=True).size():  # (in the same order as the sorted rows)
        eps = rng.standard_normal((n, 2)) @ chol.T
        z[start] = eps[0]
        for t in range(start + 1, start + n):
            z[t] = persistence * z[t - 1] + np.sqrt(1 - persistence ** 2) * eps[t - start]
        start += n
    outage = z > stats.norm.ppf(1 - prob)
    for j, col in enumerate(cols):
        df.loc[outage[:, j], col] = np.nan
    return df
