from warnings import warn

import numpy as np
import torch

from torchcast.utils.data import TimeSeriesDataset


def test_time_series_dataset():
    values = torch.randn((3, 39, 2))

    batch = TimeSeriesDataset(
        values,
        group_names=['one', 'two', 'three'],
        start_times=[0, 0, 0],
        measures=[['y1', 'y2']],
        dt_unit=None
    )
    try:
        import pandas as pd
    except ImportError:
        warn("Not testing TimeSeriesDataset.to_dataframe, pandas not installed.")
        return
    df1 = batch.to_dataframe()

    df2 = pd.concat([
        pd.DataFrame(values[i].numpy(), columns=batch.all_measures).assign(group=group, time=batch.times()[0])
        for i, group in enumerate(batch.group_names)
    ])
    assert (df1 == df2).all().all()


def test_pad_x(num_times: int = 10):
    from pandas import DataFrame
    df = DataFrame({'x1': np.random.randn(num_times), 'x2': np.random.randn(num_times)})
    df['y'] = 1.5 * df['x1'] + -.5 * df['x2'] + .1 * np.random.randn(num_times)
    df['time'] = df.index.values
    df['group'] = '1'
    dataset1 = TimeSeriesDataset.from_dataframe(
        dataframe=df,
        group_colname='group',
        time_colname='time',
        dt_unit=None,
        X_colnames=['x1', 'x2'],
        y_colnames=['y']
    )
    dataset2 = TimeSeriesDataset.from_dataframe(
        dataframe=df,
        group_colname='group',
        time_colname='time',
        dt_unit=None,
        X_colnames=['x1', 'x2'],
        y_colnames=['y'],
        pad_X=None
    )
    assert not torch.isnan(dataset1.tensors[1]).any()
    assert not torch.isnan(dataset2.tensors[1]).any()
    assert (dataset1.tensors[1] == dataset2.tensors[1]).all()
