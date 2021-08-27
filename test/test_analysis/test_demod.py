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


# Creating test data:
# data points in the middle of bins are wrongly binned, when some are shifted to 0 or pi, when zero_and_pi = True
# instead, data points are located just inside the bin boundary, defined by the offset
phase_offset = 0.0001


def single_data_points(tap_nbins: int = 32, ref_nbins: int = 16, tap_harm: int = 2, pshet_harm: int = 2,
                       zero_and_pi: bool = False, sparse: bool = False, shuffled: bool = False,
                       mod: str = 'shd', n_points: int = 0) -> pd.DataFrame:
    """ Creates one data point per bin, or randomly sampled modulation phase space.
    Multiple harmonics of tapping and pshet modulation can be created. Nth harmonics are added with a phase of n.

    Parameters
    ----------
    tap_nbins:
        number of bins for tapping modulation
    ref_nbins:
        number of bins for pshet modulation
    tap_harm:
        number of harmonics of tapping modulation in test signal, including DC as 0
    pshet_harm:
        number of harmonics of tapping modulation in test signal, including DC as 0
    zero_and_pi:
        if True, tap_phase = [0, -np.pi, np.pi] is included.
    sparse:
        drops half of the data points
    shuffled:
        mixes the order of data points
    mod:
        'shd' returns one data point for every tapping bin. 'pshet' returns one data point for every combination of
        tapping and pshet modulation bin.
    n_points:
        number of data points each for randomly sampling tapping and reference phase space.
        If 0, then one data point per bin is returned.

    Returns
    -------
    data_shd or data_pshet:
        DataFrame containing all data points as rows and channels as columns.
    """

    # tap phase
    tap_phase = np.arange(-np.pi, np.pi, 2 * np.pi / tap_nbins)
    tap_phase += phase_offset
    if n_points > 0:
        tap_phase = np.random.uniform(-np.pi, np.pi,  n_points)
    if zero_and_pi:
        tap_phase[abs(tap_phase) == abs(tap_phase).min()] = 0
        tap_phase[abs(tap_phase) == abs(tap_phase - np.pi).min()] = np.pi
        tap_phase[abs(tap_phase) == abs(tap_phase + np.pi).min()] = -np.pi
    tap_x = np.cos(tap_phase)
    tap_y = np.sin(tap_phase)

    # to use _harm as index:
    tap_harm += 1
    pshet_harm += 1

    if mod == 'shd':
        # tap modulated signal
        sig_shd = np.zeros(tap_phase.shape)
        for n in range(tap_harm):
            sig_shd += np.cos(n * tap_phase + n)
        data_shd = pd.DataFrame(np.array([sig_shd, tap_x, tap_y]).T, columns=['sig_a', 'tap_x', 'tap_y'])

        if shuffled:
            data_shd = data_shd.sample(frac=1)
        if sparse:
            data_shd = data_shd.drop(np.arange(0, tap_nbins, 2))

        return data_shd

    elif mod == 'pshet':
        # ref phase
        ref_phase = np.arange(-np.pi, np.pi, 2 * np.pi / ref_nbins)
        ref_phase += phase_offset
        if n_points > 0:
            ref_phase = np.random.uniform(-np.pi, np.pi, n_points)
        if zero_and_pi:
            ref_phase[abs(ref_phase) == abs(ref_phase).min()] = 0
            ref_phase[abs(ref_phase) == abs(ref_phase - np.pi).min()] = np.pi
            ref_phase[abs(ref_phase) == abs(ref_phase + np.pi).min()] = -np.pi
        ref_x = np.cos(ref_phase)
        ref_y = np.sin(ref_phase)

        # tap and ref modulated signal: one data point for each point in tap-ref phase space
        sig_pshet = np.zeros((len(tap_phase) * len(ref_phase), 6))
        for i in range(len(tap_phase)):
            for j in range(len(ref_phase)):
                row = i * len(ref_phase) + j
                data_point_tap = sum(np.cos(n * tap_phase[i] + n) for n in range(tap_harm))
                data_point_ref = sum(np.cos(n * ref_phase[j] + n) for n in range(pshet_harm))
                data_point = data_point_tap * data_point_ref
                sig_pshet[row] = [data_point, data_point, tap_x[i], tap_y[i], ref_x[j], ref_y[j]]    
        data_pshet = pd.DataFrame(sig_pshet, columns=['sig_a', 'sig_b', 'tap_x', 'tap_y', 'ref_x', 'ref_y'])

        if shuffled:
            data_pshet = data_pshet.sample(frac=1)
        if sparse:
            data_pshet = data_pshet.drop(np.arange(0, tap_nbins * ref_nbins, 2))

        return data_pshet

    else:
        raise ValueError("mod has to be either 'shd' or 'pshet'")


# parameters for creating test data with function single_data_points
shd_params = [(32, 2, False, False, False, 'shd', 0),
              (32, 2, False, True, False, 'shd', 0),
              (32, 2, False, False, True, 'shd', 0),
              (16, 2, False, False, False, 'shd', 0),
              (500, 2, False, False, False, 'shd', 0),
              (64, 8, False, False, False, 'shd', 0),
              (32, 2, True, False, False, 'shd', 0),
              (32, 2, False, False, False, 'pshet', 0),
              (32, 2, False, False, False, 'shd', 200_000)]


@pytest.mark.parametrize('tap, harm, zpi, sp, sh, modulation, points', shd_params)
def test_shd_binning(tap, harm, zpi, sp, sh, modulation, points):
    """Perform sHD binned average on `df`.
    Returns
        Dataframe containing the average per bins. Row index contains the bin
        number for tapping. Columns indicate the signal type.
    """
    df = single_data_points(tap_nbins=tap, tap_harm=harm, zero_and_pi=zpi,
                            sparse=sp, shuffled=sh, mod=modulation, n_points=points)
    binned_data = shd_binning(df.copy(), tap)

    # shd test data with one data point per bin should be unchanged by binning
    if modulation == 'shd' and points == 0:
        assert binned_data['sig_a'].reset_index(drop=True).equals(df['sig_a'].sort_index().reset_index(drop=True))

    # test that data is ordered with respect to tap_n
    assert binned_data.index.name == 'tap_n'
    assert all(binned_data.index[i] < binned_data.index[i + 1] for i in range(len(binned_data) - 1))

    # test for one row per bin, if test data is not sparse
    if not sp:
        assert binned_data.shape[0] == tap
    # all columns should be signals, i.e. 'sig_a', 'sig_b'
    assert all('sig' in binned_data.columns[i] for i in range(binned_data.shape[1]))


@pytest.mark.parametrize('tap, harm, zpi, sp, sh, modulation, points', shd_params)
def test_shd_ft(tap, harm, zpi, sp, sh, modulation, points):
    """Perform Fourier analysis for sHD demodulation.
    Returns
        Fourier components. Rows indicate tapping order `n`, columns indicate
        signal type.
    """
    df = single_data_points(tap_nbins=tap, tap_harm=harm, zero_and_pi=zpi,
                            sparse=sp, shuffled=sh, mod=modulation, n_points=points)
    ft_data = shd_ft(shd_binning(df.copy(), tap))

    # retrieved harmonics are rounded and compared
    if modulation == 'shd':
        # amplitude of harmonics
        harm_amps = np.abs(ft_data['sig_a'][1:harm + 1])
        # 0 order has roughly twice the amplitude (real fft)
        assert round(np.real(ft_data['sig_a'][0]) / harm_amps.mean()) == 2
        # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
        assert (harm_amps.std() < harm_amps.mean() / 10)
        # following amplitude is at least 1 order of magnitude smaller
        assert ft_data['sig_a'][harm + 1] < harm_amps.mean() / 10

        # phase of nth harmonic
        if not sp:    # phases in sparse data sets are more difficult (higher error)
            for n in range(harm + 1):
                phase = (np.arctan2(np.imag(ft_data['sig_a'][n]), np.real(ft_data['sig_a'][n])) + n * np.pi) % np.pi
                # phase correction
                if n > 0:
                    phase -= n * phase_offset
                    if points > 0:
                        phase -= n * np.pi / tap
                assert round((n - phase) / np.pi, 3) % 1 == 0

    # test shape of returned DataFrame
    if not sp:
        assert ft_data.shape[0] == tap // 2 + 1
    assert all('sig' in ft_data.columns[i] for i in range(ft_data.shape[1]))


@pytest.mark.parametrize('tap, harm, zpi, sp, sh, modulation, points', shd_params)
def test_shd(tap, harm, zpi, sp, sh, modulation, points):
    """ simple sequence of shd_binning and shd_ft.
    The difference to test_shd_ft is that ft_data is created with shd. """
    df = single_data_points(tap_nbins=tap, tap_harm=harm, zero_and_pi=zpi,
                            sparse=sp, shuffled=sh, mod=modulation, n_points=points)
    ft_data = shd(df.copy(), tap)

    # retrieved harmonics are rounded and compared
    if modulation == 'shd':
        # amplitude of harmonics
        harm_amps = np.abs(ft_data['sig_a'][1:harm + 1])
        # 0 order has roughly twice the amplitude (real fft)
        assert round(np.real(ft_data['sig_a'][0]) / harm_amps.mean()) == 2
        # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
        assert (harm_amps.std() < harm_amps.mean() / 10)
        # following amplitude is at least 1 order of magnitude smaller
        assert ft_data['sig_a'][harm + 1] < harm_amps.mean() / 10

        # phase of nth harmonic
        if not sp:    # phases in sparse data sets are more difficult (higher error)
            for n in range(harm + 1):
                phase = (np.arctan2(np.imag(ft_data['sig_a'][n]), np.real(ft_data['sig_a'][n])) + n * np.pi) % np.pi
                # phase correction
                if n > 0:
                    phase -= n * phase_offset
                    if points > 0:
                        phase -= n * np.pi / tap
                assert round((n - phase) / np.pi, 3) % 1 == 0

    # test shape of returned DataFrame
    if not sp:
        assert ft_data.shape[0] == tap // 2 + 1
    assert all('sig' in ft_data.columns[i] for i in range(ft_data.shape[1]))


# parameters for creating test data with function single_data_points
pshet_params = [(32, 16, 3, 2, False, False, False, 'pshet', 0),
                (32, 16, 3, 2, False, True, False, 'pshet', 0),
                (32, 16, 3, 2, False, False, True, 'pshet', 0),
                (11, 16, 3, 2, False, False, False, 'pshet', 0),
                (500, 16, 3, 2, False, False, False, 'pshet', 0),
                (32, 8, 3, 2, False, False, False, 'pshet', 0),
                (32, 500, 3, 2, False, False, False, 'pshet', 0),
                (64, 64, 5, 4, False, False, False, 'pshet', 0),
                (32, 16, 3, 2, True, False, False, 'pshet', 0),
                (32, 16, 3, 2, False, False, False, 'pshet', 500)]


@pytest.mark.parametrize('tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points', pshet_params)
def test_pshet_binning(tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points):
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
def test_pshet_ft(tap, ref, tap_harm, pshet_harm, zpi, sp, sh, modulation, points):
    """Fourier transform an averaged pshet dataframe. Returns a dict with channel name as key ('sig_a' etc) and
    np.array as value containing tapping and pshet mod on axis 0 and 1 respectively."""
    df = single_data_points(tap_nbins=tap, ref_nbins=ref, tap_harm=tap_harm, pshet_harm=pshet_harm, zero_and_pi=zpi,
                            sparse=sp, shuffled=sh, mod=modulation, n_points=points)
    ft_data = pshet_ft(pshet_binning(df.copy(), tap, ref))

    # retrieved pshet harmonics are rounded and compared
    for key in ft_data.keys():
        # amplitude of harmonics
        pshet_harm_amps = abs(np.real(ft_data[key][0][1:pshet_harm + 1]))
        # 0 order has roughly twice the amplitude (real fft)
        assert round(np.real(ft_data[key][0, 0]) / pshet_harm_amps.mean()) == 2
        # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
        assert (pshet_harm_amps.std() < pshet_harm_amps.mean() / 10)
        # following amplitude is at least 1 order of magnitude smaller
        assert ft_data[key][0, pshet_harm + 1] < pshet_harm_amps.mean() / 10

    # retrieved tapping harmonics are rounded and compared
    for key in ft_data.keys():
        # amplitude of harmonics
        tap_harm_amps = abs(np.round(np.real(ft_data['sig_a'][1:tap_harm + 1, 0])))
        # 0 order has roughly twice the amplitude (real fft)
        assert round(np.real(ft_data[key][0, 0])) // tap_harm_amps.mean() == 2
        # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
        assert (tap_harm_amps.std() < tap_harm_amps.mean() / 10)
        # following amplitude is at least 1 order of magnitude smaller
        assert ft_data[key][0, tap_harm + 1] < tap_harm_amps.mean() / 10

    # TODO test for phases

    # test shape of returned arrays
    if not sp:
        for array in ft_data.values():
            assert array.shape[0] == tap
            assert array.shape[1] == ref / 2 + 1
    assert all('sig' in key for key in ft_data.keys())


'''
@pytest.mark.parametrize('tap, ref, tapharm, pshetharm, zpi, sp, sh, modulation, points', pshet_params)
def test_pshet(tap, ref, tapharm, pshetharm, zpi, sp, sh, modulation, points):
    """ simple sequence of pshet_binning and pshet_ft.
    The difference to test_pshet_ft is that ft_data is created with pshet. """
    # TODO keep synchronized to test_pshet_ft
    df = single_data_points(tap_nbins=tap, ref_nbins=ref, tap_harm=tapharm, pshet_harm=pshetharm, zero_and_pi=zpi,
                            sparse=sp, shuffled=sh, mod=modulation, n_points=points)
    ft_data = pshet(df.copy(), tap, ref)

    # retrieved harmonics are rounded and compared
    for key in ft_data.keys():
        # amplitude of harmonics
        harm_amps = abs(np.round(np.real(ft_data[key][0][1:harm + 1])))
        # 0 order has roughly twice the amplitude (real fft)
        assert round(np.real(ft_data[key][0, 0]) / harm_amps.mean()) == 2
        # amplitudes are roughly the same (standard deviation < 1/10 amplitude)
        assert (harm_amps.std() < harm_amps.mean() / 10)
        # following amplitude is at least 1 order of magnitude smaller
        assert ft_data[key][0, harm + 1] < harm_amps.mean() / 10

    # test shape of returned arrays
    if not sp:
        for array in ft_data.values():
            assert array.shape[0] == tap
            assert array.shape[1] == ref / 2 + 1
    assert all('sig' in key for key in ft_data.keys())
'''
