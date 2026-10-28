import warnings

import numpy as np
import pytest
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
    # reset_index so comparison is value-based, not index-label-based
    pd.testing.assert_frame_equal(
        df1.sort_values(['group', 'time']).reset_index(drop=True),
        df2.sort_values(['group', 'time']).reset_index(drop=True),
    )


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


def _make_times(group_names, T):
    return np.arange(len(group_names) * T, dtype=float).reshape(len(group_names), T)


def _sorted(df, group_colname='group', time_colname='time'):
    return df.sort_values([group_colname, time_colname]).reset_index(drop=True)


@pytest.mark.parametrize("num_groups,num_times,num_measures", [(3, 5, 2), (2, 3, 2)])
def test_tensor_to_dataframe_values_roundtrip(num_groups, num_times, num_measures):
    data = np.arange(num_groups * num_times * num_measures, dtype=np.float32).reshape(
        num_groups, num_times, num_measures
    )
    tensor = torch.as_tensor(data)
    group_names = [chr(ord('a') + g) for g in range(num_groups)]
    measures = [f'y{m + 1}' for m in range(num_measures)]
    times = _make_times(group_names, num_times)
    df = TimeSeriesDataset.tensor_to_dataframe(
        tensor, times, group_names, 'group', 'time', measures
    )
    assert list(df.columns) == measures + ['group', 'time']
    assert len(df) == num_groups * num_times
    assert set(df['group']) == set(group_names)
    for grp_idx, name in enumerate(group_names):
        rows = _sorted(df[df['group'] == name])
        np.testing.assert_allclose(rows[measures].values, data[grp_idx])


def test_tensor_to_dataframe_trailing_nan_trimmed():
    G, T, M = 2, 6, 2
    tensor = torch.zeros(G, T, M)
    tensor[0, 4:, :] = float('nan')  # group 0 valid through t=3 (end_idx=4)
    tensor[1, 3:, :] = float('nan')  # group 1 valid through t=2 (end_idx=3)
    times = _make_times(['a', 'b'], T)
    df = TimeSeriesDataset.tensor_to_dataframe(
        tensor, times, ['a', 'b'], 'group', 'time', ['y1', 'y2']
    )
    assert len(df[df['group'] == 'a']) == 4
    assert len(df[df['group'] == 'b']) == 3


def test_tensor_to_dataframe_interior_nan_preserved():
    G, T, M = 1, 5, 2
    tensor = torch.zeros(G, T, M)
    tensor[0, 2, :] = float('nan')  # interior NaN — should NOT trim
    times = _make_times(['a'], T)
    df = TimeSeriesDataset.tensor_to_dataframe(
        tensor, times, ['a'], 'group', 'time', ['y1', 'y2']
    )
    assert len(df) == T
    assert np.isnan(df.iloc[2]['y1'])


def test_tensor_to_dataframe_all_nan_group_warns():
    G, T, M = 3, 4, 1
    tensor = torch.zeros(G, T, M)
    tensor[1, :, :] = float('nan')  # group 'b' is all NaN
    times = _make_times(['a', 'b', 'c'], T)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = TimeSeriesDataset.tensor_to_dataframe(
            tensor, times, ['a', 'b', 'c'], 'group', 'time', ['y']
        )
    assert any('b' in str(warning.message) for warning in w)
    assert set(df['group']) == {'a', 'c'}


def test_tensor_to_dataframe_all_groups_nan_returns_empty():
    G, T, M = 2, 3, 2
    tensor = torch.full((G, T, M), float('nan'))
    times = _make_times(['a', 'b'], T)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = TimeSeriesDataset.tensor_to_dataframe(
            tensor, times, ['a', 'b'], 'group', 'time', ['y1', 'y2']
        )
    assert len(df) == 0
    assert list(df.columns) == ['y1', 'y2', 'group', 'time']


def test_tensor_to_dataframe_times_assigned_correctly():
    G, T, M = 2, 4, 1
    tensor = torch.zeros(G, T, M)
    times = np.array([[10., 20., 30., 40.], [100., 200., 300., 400.]])
    df = TimeSeriesDataset.tensor_to_dataframe(
        tensor, times, ['a', 'b'], 'group', 'time', ['y']
    )
    np.testing.assert_array_equal(
        _sorted(df[df['group'] == 'a'])['time'].values, [10., 20., 30., 40.]
    )
    np.testing.assert_array_equal(
        _sorted(df[df['group'] == 'b'])['time'].values, [100., 200., 300., 400.]
    )


