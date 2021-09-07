import pytest
import numpy as np
import pandas as pd

from trion.analysis.demod import bin_index, bin_midpoints, shd_binning, shd_ft, shd, pshet_binning, pshet_ft, pshet


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
    """ Test the binning of data containing empty bins.
    """
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    perforated_data = shd_binning(test_data.drop(test_data.index[drops]).copy(), tap_nbins)

    # the shape should be unchanged
    assert perforated_data.shape[0] == tap_nbins
    # Nan should be inserted in missing bins
    assert all(np.isnan(perforated_data[drops]))    # does this work ?


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
    # retrieve parameters and test data from fixture
    params, test_data = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params

    # TODO update handling of missing and nan filled bins
    # passing missing or nan filled bins to _ft should DO SOMETHING SPECIFIC LATER
    with pytest.raises(NotImplementedError):
        shd_ft(shd_binning(test_data.drop(test_data.index[drops]).copy(), tap_nbins))


# TODO: Do we need duplicate tests for shd_ft and shd?


if False:
    @pytest.mark.parametrize('tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points', pshet_params)
    def asdf_pshet_binning(tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points):
        """Perform pshet binned average on `df`.
        Returns
            Dataframe containing the average per bins. Row index contains the bin
            number for tapping. The column index is a `Multindex` where level 0 is
            the signal name as a string (ex: `"sig_a"`), and level 1 is the
            reference bin number. Therefore, the histogram for `sig_a` can be
            accessed via `avg["sig_a"]`.
        """
        df = single_data_points(tap_nbins=tap, ref_nbins=ref, tap_harm=tap_harm, pshet_harm=pshet_harm, zero_and_pi=zpi,
                                sparse=sp, shuffled=sh, mod=modulation, n_points=points)
        binned_data = pshet_binning(df.copy(), tap, ref)

        # test if every data point in df matches the value in its position in the binned_data DataFrame
        if points == 0:
            all_true = False
            for i in range(len(df)):
                try:
                    tap_n = bin_index(np.arctan2(df['tap_y'][i], df['tap_x'][i]), tap)
                    ref_n = bin_index(np.arctan2(df['ref_y'][i], df['ref_x'][i]), ref)
                    if binned_data['sig_a', ref_n][tap_n] == df['sig_a'][i]:
                        all_true = True
                    else:
                        all_true = False
                        break
                except KeyError:    # in sparse sample data
                    pass
            assert all_true

        #  mixed data should be ordered
        assert binned_data.index.name == 'tap_n'
        assert binned_data.columns.names[1] == 'ref_n'
        assert all(binned_data.index[i] < binned_data.index[i + 1] for i in range(binned_data.shape[0] - 1))
        channels = binned_data.columns.get_level_values(0).drop_duplicates()
        for ch in channels:
            assert all(binned_data[ch].columns[i] < binned_data[ch].columns[i + 1]
                       for i in range(int(binned_data.shape[1] / len(channels) - 1)))

        # test shape of returned DataFrame
        if not sp:
            assert binned_data.shape[0] == tap
            assert binned_data.shape[1] == 2 * ref
        assert all('sig' in binned_data.columns[i][0] for i in range(binned_data.shape[1]))


    @pytest.mark.parametrize('tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points', pshet_params)
    def asdf_pshet_ft(tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points):
        """Fourier transform an averaged pshet dataframe. Returns a dict with channel name as key ('sig_a' etc) and
        np.array as value containing tapping and pshet mod on axis 0 and 1 respectively."""
        df = single_data_points(tap_nbins=tap, ref_nbins=ref, tap_harm=tap_harm, pshet_harm=pshet_harm, zero_and_pi=zpi,
                                sparse=sp, shuffled=sh, mod=modulation, n_points=points)
        ft_data = pshet_ft(pshet_binning(df.copy(), tap, ref))

        for key in ft_data.keys():
            # amplitude of pshet harmonics
            pshet_harm_amps = np.abs(ft_data[key][0][1:pshet_harm + 1])
            # 0 order has roughly twice the amplitude (real fft)
            assert round(np.real(ft_data[key][0, 0]) / pshet_harm_amps.mean()) == 2
            # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
            assert (pshet_harm_amps.std() < pshet_harm_amps.mean() / 10)
            # following amplitude is at least 1 order of magnitude smaller
            assert ft_data[key][0, pshet_harm + 1] < pshet_harm_amps.mean() / 10

            # phase of pshet harmonics
            if not sp:    # sparse data sets produce bigger errors. This shows more in the harmonics
                for n in range(pshet_harm + 1):
                    phase = (np.arctan2(np.imag(ft_data[key][0][n]), np.real(ft_data[key][0][n])) + n * np.pi) % np.pi
                    # phase correction
                    if n > 0:
                        phase -= n * phase_offset
                        if points > 0:
                            phase -= n * np.pi / ref
                    assert round(abs((n - phase) / np.pi), 1) % 1 == 0    # tolerating relatively big errors here ...

            # amplitude of tapping harmonics
            tap_harm_amps = np.abs(ft_data['sig_a'][1:tap_harm + 1, 0])
            # 0 order has roughly twice the amplitude (real fft)
            assert round(np.real(ft_data[key][0, 0]) / tap_harm_amps.mean()) == 2
            # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
            assert (tap_harm_amps.std() < tap_harm_amps.mean() / 10)
            # following amplitude is at least 1 order of magnitude smaller
            assert ft_data[key][0, tap_harm + 1] < tap_harm_amps.mean() / 10

            # phase of tapping harmonics
            if not sp:  # sparse data sets produce bigger errors. This shows more in the harmonics
                for n in range(tap_harm + 1):
                    phase = (np.arctan2(np.imag(ft_data[key][n, 0]), np.real(ft_data[key][n, 0])) + n * np.pi) % np.pi
                    # phase correction
                    if n > 0:
                        phase -= n * phase_offset
                        if points > 0:
                            phase -= n * np.pi / tap
                    assert round(abs((n - phase) / np.pi), 1) % 1 == 0    # tolerating relatively big errors here ...

        # test shape of returned arrays
        if not sp:
            for array in ft_data.values():
                assert array.shape[0] == tap
                assert array.shape[1] == ref / 2 + 1
        assert all('sig' in key for key in ft_data.keys())


    @pytest.mark.parametrize('tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points', pshet_params)
    def asdf_pshet(tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points):
        """ simple sequence of pshet_binning and pshet_ft.
        The difference to test_pshet_ft is that ft_data is created with pshet. """
        df = single_data_points(tap_nbins=tap, ref_nbins=ref, tap_harm=tap_harm, pshet_harm=pshet_harm, zero_and_pi=zpi,
                                sparse=sp, shuffled=sh, mod=modulation, n_points=points)
        ft_data = pshet(df.copy(), tap, ref)

