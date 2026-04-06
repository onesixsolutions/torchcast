from platformdirs import user_cache_path


def load_air_quality_data(dt_unit: str) -> 'DataFrame':
    """

    This dataset comes from the
    [UCI Machine Learning Data Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).
     It includes data on air pollutants and weather from 12 sites.

     Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-
     based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management,
     Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17).

    :param dt_unit: 'daily' or 'weekly'
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

    try:
        return read_csv(
            cache_path / f'aq-{dt_unit}.csv',
            parse_dates=['date' if dt_unit == 'daily' else 'week']
        )
    except FileNotFoundError:
        print('loading from gh...')
        read_csv(
            f'https://raw.githubusercontent.com/onesixsolutions/torchcast/c675f04bb244a73155fbd98c110590bd735808bc/docs/examples/aq_{dt_unit}.csv',
            parse_dates=['date' if dt_unit == 'daily' else 'week']
        ).to_csv(cache_path / f'aq-{dt_unit}.csv', index=False)
    return load_air_quality_data(dt_unit)
