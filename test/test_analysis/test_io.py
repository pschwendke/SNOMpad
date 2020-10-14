from trion.analysis.io import (
    export_csv, export_npz
)
import pytest
import numpy as np
import pandas as pd


def test_csv(tmp_path):
    pth = tmp_path / "test.csv"
    data = np.random.randn(100, 3)
    header = ["tap_x", "tap_y", "sig_A"]
    export_csv(pth, data, header)
    loaded = pd.read_csv(pth)
    loaded_data = loaded.to_numpy()
    loaded_cols = loaded.columns
    assert np.allclose(loaded_data, data)
    assert all(loaded_cols == header)


def test_npz(tmp_path):
    pth = tmp_path / "test.npz"
    data = np.random.randn(100, 3)
    header = ["tap_x", "tap_y", "sig_A"]
    export_npz(pth, data, header)
    loaded = np.load(pth)
    loaded_cols = list(loaded.keys())
    assert isinstance(loaded_cols, list)
    assert all([isinstance(v, str) for v in loaded_cols])
    loaded_data = list(loaded.values())
    loaded_data = np.vstack(loaded_data).T
    assert np.allclose(loaded_data, data)
    assert loaded_cols == header

