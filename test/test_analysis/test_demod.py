import pytest
import numpy as np

from trion.analysis.demod import bin_index, bin_midpoints, shd_binning, shd_ft, shd, pshet_binning, pshet_ft, pshet

# TODO: Do we need duplicate tests for shd_ft and shd?
#  benchmarking for binning functions
#  dealing with missing bins
#  test passing pshet data to shd functions (which one?)
#  check all tests are 'online'


@pytest.mark.parametrize("n_bins", [4, 7, 16, 32])
def test_bin_midpoints(n_bins, lo=-np.pi, hi=np.pi):
    # generate the bin edges
    edges = np.linspace(lo, hi, n_bins+1)
    ref = 0.5 * (edges[1:]+edges[:-1])
    centers = bin_midpoints(n_bins, lo, hi)
    assert centers.ndim == 1
    assert centers.shape == (n_bins,)
    assert np.allclose(centers, ref)


def test_shd_binning(shd_data_points):
    """ Unit test for shd_binning. Testing binning and shape of returned DataFrame.
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    binned_data = shd_binning(test_data.copy())

    # data should be unchanged by binning
    assert binned_data['sig_a'].equals(test_data['sig_a'])

    # tap_n as index, signals as columns
    assert binned_data.shape[0] == tap_nbins
    assert binned_data.index.name == 'tap_n'
    assert all('sig' in binned_data.columns[i] for i in range(binned_data.shape[1]))


def test_shd_binning_shuffled(shd_data_points):
    """ Test that shd_binning returns a DataFrame ordered with respect to tap_n
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    shuffled_data = shd_binning(test_data.sample(frac=1), tap_nbins)

    assert shuffled_data.shape[0] == tap_nbins
    assert all(shuffled_data.index[i] < shuffled_data.index[i + 1] for i in range(len(shuffled_data) - 1))


@pytest.mark.parametrize('drops', [0, [0, 5], [3, 5, 8], [5, -1]])
def test_shd_binning_empty(shd_data_points, drops):
    """ Test the binning of data containing missing bins.
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    perforated_data = shd_binning(test_data.drop(test_data.index[drops]).copy(), tap_nbins)

    # the shape should be unchanged
    assert perforated_data.shape[0] == tap_nbins
    # Nan should be inserted in missing bins
    # TODO check for nans
    # assert all(np.isnan(perforated_data[drops]))


def test_shd_ft_shape(shd_data_points):
    """ Test the shape of the DataFrame returned by shd_ft.
    This test assumes that shd_binning is working as expected.
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    ft_data = shd_ft(shd_binning(test_data.copy(), tap_nbins))

    assert ft_data.shape[0] == tap_nbins // 2 + 1
    assert all('sig' in ft_data.columns[i] for i in range(ft_data.shape[1]))


def test_shd_ft_harmonics(shd_data_points):
    """ Retrieve amplitudes and phases of harmonics with shd_ft.
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    ft_data = shd_ft(shd_binning(test_data.copy(), tap_nbins))

    for n in range(tap_nharm):
        initial_phase = np.angle(tap_harm_amps[n])
        initial_amp = np.abs(tap_harm_amps[n])
        returned_phase = np.arctan2(np.imag(ft_data['sig_a'][n]), np.real(ft_data['sig_a'][n]))
        returned_amp = np.abs(ft_data['sig_a'][n])
        # np.fft.rfft only returns positive amplitudes
        if n > 0:
            returned_amp *= 2
        # data point in the middle of bins => phase is shifted by half a bin
        returned_phase -= n * np.pi / tap_nbins

        assert np.isclose(initial_phase, (returned_phase + np.pi) % np.pi)
        assert np.isclose(initial_amp, returned_amp)


@pytest.mark.parametrize('drops', [0, [0, 5], [3, 5, 8], [5, -1]])
def test_shd_ft_empty(shd_data_points, drops):
    """ Missing bins or bins filled with nans should be caught by shd_ft.
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params

    # TODO update handling of missing and nan filled bins
    with pytest.raises(NotImplementedError):
        shd_ft(shd_binning(test_data.drop(test_data.index[drops]).copy(), tap_nbins))


def test_pshet_binning(pshet_data_points):
    """ Unit test for pshet_binning. Only sig_a is tested.
    """
    # retrieve parameters and test data from fixture
    params, test_data = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    binned_data = pshet_binning(test_data.copy(), tap_nbins, ref_nbins)

    # test shape of returned DataFrame
    assert binned_data.shape[0] == tap_nbins
    assert binned_data.shape[1] == 2 * ref_nbins
    assert binned_data.index.name == 'tap_n'
    assert binned_data.columns.names[1] == 'ref_n'
    assert all('sig' in binned_data.columns[i][0] for i in range(binned_data.shape[1]))

    # test if every data point in test_data matches the value in its position in the binned_data DataFrame
    all_true = False
    for i in range(tap_nbins * ref_nbins):
        tap_n = bin_index(np.arctan2(test_data['tap_y'][i], test_data['tap_x'][i]), tap_nbins)
        ref_n = bin_index(np.arctan2(test_data['ref_y'][i], test_data['ref_x'][i]), ref_nbins)
        if binned_data['sig_a', ref_n][tap_n] == test_data['sig_a'][i]:
            all_true = True
        else:
            all_true = False
            break
    assert all_true


def test_pshet_binning_shuffled(pshet_data_points):
    """ Test that pshet_binning returns a DataFrame ordered with respect to tap_n and ref_n
    """
    # retrieve parameters and test data from fixture
    params, test_data = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    shuffled_data = pshet_binning(test_data.sample(frac=1), tap_nbins, ref_nbins)

    # test shape of returned DataFrame
    assert shuffled_data.shape[0] == tap_nbins
    assert shuffled_data.shape[1] == 2 * ref_nbins
    assert shuffled_data.index.name == 'tap_n'
    assert shuffled_data.columns.names[1] == 'ref_n'
    assert all('sig' in shuffled_data.columns[i][0] for i in range(shuffled_data.shape[1]))

    #  mixed data should be ordered with respect to tap_n and ref_n
    assert all(shuffled_data.index[i] < shuffled_data.index[i + 1] for i in range(shuffled_data.shape[0] - 1))
    channels = shuffled_data.columns.get_level_values(0).drop_duplicates()
    for ch in channels:
        assert all(shuffled_data[ch].columns[i] < shuffled_data[ch].columns[i + 1]
                   for i in range(int(shuffled_data.shape[1] / len(channels) - 1)))


@pytest.mark.parametrize('drops', [0, [0, 32], [31, 32, 100], [123, -1]])
def test_pshet_binning_empty(pshet_data_points, drops):
    """ Test the binning of data containing missing bins.
    """
    # retrieve parameters and test data from fixture
    params, test_data = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    perforated_data = pshet_binning(test_data.drop(test_data.index[drops]).copy(), tap_nbins, ref_nbins)

    # the shape should be unchanged
    assert perforated_data.shape[0] == tap_nbins
    assert perforated_data.shape[1] == 2 * ref_nbins

    # TODO check for inserted nans
    # Nan should be inserted in missing bins
    # tap_n = bin_index(np.arctan2(test_data['tap_y'][test_data.index[drops]],
    #                             test_data['tap_x'][test_data.index[drops]]), tap_nbins)
    # ref_n = bin_index(np.arctan2(test_data['ref_y'][test_data.index[drops]],
    #                             test_data['ref_x'][test_data.index[drops]]), ref_nbins)
    # assert all(np.isnan(perforated_data['sig_a', ref_n][tap_n]))    # does this work ?


def test_pshet_ft_shape(pshet_data_points):
    """ Test the shape of the DataFrame returned by pshet_ft.
    This test assumes that pshet_binning is working as expected.
    """
    # retrieve parameters and test data from fixture
    params, test_data = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    ft_data = pshet_ft(pshet_binning(test_data.copy(), tap_nbins, ref_nbins))

    # test shape of returned arrays
    assert all('sig' in key for key in ft_data.keys())
    for array in ft_data.values():
        assert array.shape[0] == tap_nbins
        assert array.shape[1] == ref_nbins / 2 + 1


def test_pshet_ft_harmonics(pshet_data_points):
    """ Retrieve amplitudes and phases of harmonics with phset_ft"""
    # retrieve parameters and test data from fixture
    params, test_data = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    ft_data = pshet_ft(pshet_binning(test_data.copy(), tap_nbins, ref_nbins))
    
    # signals, e.g. 'sig_a' are keys of ft_data
    for key in ft_data.keys():
        # tapping harmonics
        for n in range(tap_nharm):
            initial_tap_phase = np.angle(tap_harm_amps[n])
            initial_tap_amp = np.abs(tap_harm_amps[n])
            returned_tap_phase = np.arctan2(np.imag(ft_data[key][n, 0]), np.real(ft_data[key][n, 0]))
            returned_tap_amp = np.abs(ft_data['sig_a'][n, 0])
            # np.fft.rfft only returns positive amplitudes
            if n > 0:
                returned_tap_amp *= 2
            # data point in the middle of bins => phase is shifted by half a bin
            returned_tap_phase -= n * np.pi / tap_nbins

            assert np.isclose(initial_tap_phase, (returned_tap_phase + np.pi) % np.pi)
            assert np.isclose(initial_tap_amp, returned_tap_amp)

        # pshet harmonics
        for m in range(ref_nharm):
            initial_ref_phase = np.angle(ref_harm_amps[m])
            initial_ref_amp = np.abs(ref_harm_amps[m])
            returned_ref_phase = np.arctan2(np.imag(ft_data[key][0][m]), np.real(ft_data[key][0][m]))
            returned_ref_amp = np.abs(ft_data[key][0][m])
            # np.fft.rfft only returns positive amplitudes
            if m > 0:
                returned_ref_amp *= 2
            # data point in the middle of bins => phase is shifted by half a bin
            returned_ref_phase -= m * np.pi / ref_nbins

            assert np.isclose(initial_ref_phase, (returned_ref_phase + np.pi) % np.pi)
            assert np.isclose(initial_ref_amp, returned_ref_amp)
        
        
@pytest.mark.parametrize('drops', [0, [0, 32], [31, 32, 100], [123, -1]])
def test_pshet_ft_empty(pshet_data_points, drops):
    """ Missing bins or bins filled with nans should be caught by pshet_ft.
    """
    # retrieve parameters and test data from fixture
    params, test_data = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params

    # TODO update handling of missing and nan filled bins
    with pytest.raises(NotImplementedError):
        pshet_binning(test_data.drop(test_data.index[drops]).copy(), tap_nbins, ref_nbins)
