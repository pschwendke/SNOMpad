import pytest
import numpy as np
import pandas as pd

from trion.analysis.demod import bin_midpoints, shd_binning, shd_ft, shd


@pytest.mark.parametrize("n_bins", [4, 7, 16, 32])
def test_bin_midpoints(n_bins, lo=-np.pi, hi=np.pi):
    # generate the bin edges
    edges = np.linspace(lo, hi, n_bins+1)
    ref = 0.5*(edges[1:]+edges[:-1])
    centers = bin_midpoints(n_bins, lo, hi)
    assert centers.ndim == 1
    assert centers.shape == (n_bins,)
    assert np.allclose(centers, ref)


# loading some test data (210806_pshet_run02.npy)
test_data = pd.read_pickle('test_data.pkl')
test_data_shd_binned = pd.read_pickle('test_data_shd_binned.pkl')
test_data_shd_ft = pd.read_pickle('test_data_shd_ft.pkl')
test_data_shd = pd.read_pickle('test_data_shd.pkl')


@pytest.mark.parametrize('df', [test_data.copy()])
def test_shd_binning(df: pd.DataFrame, tap_nbins: int = 32):
    """Perform sHD binned average on `df`.
    Returns
    -------
        Dataframe containing the average per bins. Row index contains the bin
        number for tapping. Columns indicate the signal type.
    """
    # simply comparison of return to sample data
    assert shd_binning((df)).equals(test_data_shd_binned)


@pytest.mark.parametrize('avg', [test_data_shd.copy()])
def shd_ft(avg: pd.DataFrame):
    """Perform Fourier analysis for sHD demodulation.
    Returns
    -------
        Fourier components. Rows indicate tapping order `n`, columns indicate
        signal type.
    """
    # simply comparison of return to sample data
    assert shd_ft(avg).equals(test_data_shd_ft)


@pytest.mark.parametrize('df', [test_data.copy()])
def shd(df: pd.DataFrame, tap_nbins: int = 32):
    # simply comparison of return to sample data
    assert shd(df).equals(test_data_shd)