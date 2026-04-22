import numpy as np
import torch

from torchcast.utils.data import TimeSeriesDataset
import pandas as pd


def test_time_series_dataset():
    values = torch.randn((3, 39, 2))

    batch = TimeSeriesDataset(
        values,
        group_names=['one', 'two', 'three'],
        start_times=[0, 0, 0],
        measures=[['y1', 'y2']],
        dt_unit=None
    )

    df1 = batch.to_dataframe()

    df2 = pd.concat([
        pd.DataFrame(values[i].numpy(), columns=batch.all_measures).assign(group=group, time=batch.times()[0])
        for i, group in enumerate(batch.group_names)
    ])
    assert (df1 == df2).all().all()


def test_pad_x(num_times: int = 10):
    df = pd.DataFrame({'x1': np.random.randn(num_times), 'x2': np.random.randn(num_times)})
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


def test_standardize():
    y = torch.randn((3, 20, 1))
    X = torch.randn((3, 20, 2)) * 5 + 10
    ds = TimeSeriesDataset(
        y, X,
        group_names=['a', 'b', 'c'],
        start_times=[0, 0, 0],
        measures=[['y'], ['x1', 'x2']],
        dt_unit=None
    )

    # standardizing self: X tensor should have ~0 mean and ~1 std; y tensor unchanged
    ds_std = ds.standardize(which=(1,))
    assert ds_std.tensors[1].mean().abs() < 1e-5  # mean g2g
    assert abs(ds_std.tensors[1].std().item() - 1.0) < 0.05  # std-dev g2g
    assert torch.allclose(ds_std.tensors[0], y)  # first tensor unaffect ('which' arg)

    # standardizing a separate dataset:
    Xtrain = torch.as_tensor([[-1, 0, 1]], dtype=torch.float)
    Xtrain = torch.stack([Xtrain, Xtrain + 1], -1)
    ds_train = TimeSeriesDataset(
        Xtrain,
        group_names=['a'],
        start_times=[0],
        measures=[['x1', 'x2']],
        dt_unit=None
    )
    ds_val = TimeSeriesDataset(
        Xtrain * 2 + 1,
        group_names=['a'],
        start_times=[0],
        measures=[['x1', 'x2']],
        dt_unit=None
    )
    ds_val_std = ds_train.standardize(ds_val, which=(0,))
    Xval_std = ds_val_std.tensors[0]
    assert torch.allclose(Xval_std[:, :, 0].mean(), torch.as_tensor(1.))
    assert torch.allclose(Xval_std[:, :, 0].std(), torch.as_tensor(2.))
    assert torch.allclose(Xval_std[:, :, 1].mean(), torch.as_tensor(2.))
    assert torch.allclose(Xval_std[:, :, 1].std(), torch.as_tensor(2.))

# def test_different_behavior():
#     Xtrain = torch.as_tensor([[-1, 0, 1]], dtype=torch.float)
#     Xtrain = torch.stack([Xtrain, Xtrain + 1], -1)
#     torch_result = Xtrain.std(dim=(0,1))
#     np_result = Xtrain.numpy().std(axis=(0, 1), ddof=1)
#     print(torch_result)
#     print(np_result)
