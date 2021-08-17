import pytest
import numpy as np
import pandas as pd

from trion.analysis.demod import bin_midpoints, shd_binning


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
test_file = np.load('test_data.npz')
data = pd.DataFrame.from_dict({channel: test_file[channel] for channel in test_file.files})


@pytest.mark.parametrize('df,tap_nbins', [(data, 32),
                                          (data.drop(columns='sig_b'), 32)])
def test_shd_binning(df: pd.DataFrame, tap_nbins: int):
    """Perform sHD binned average on `df`.
    Returns
    -------
        Dataframe containing the average per bins. Row index contains the bin
        number for tapping. Columns indicate the signal type.
    """
    returned_df = shd_binning(df.copy(), tap_nbins)
    assert returned_df.shape == (tap_nbins, sum(['sig' in names for names in df.columns]))
