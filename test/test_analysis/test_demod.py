import pytest
import numpy as np
from scipy.special import jv

from trion.analysis.demod import phase_offset, phased_ft, shd_binning, shd_ft, shd
from trion.analysis.demod import pshet_binning, pshet_ft, pshet_coefficients, pshet
from trion.analysis.signals import Signals, all_detector_signals

# some parameters to create test data
# all parameters are real to only test for retrieval of amplitudes, not phases
shd_parameters = [
    [32, 3, [1, 1, 1]],
    [32, 4, [1, 1, 2, .5]]
]
pshet_parameters = [
    [32, 3, [1, 1, 1], 16, 3, [1, 1, 1]],
    [64, 4, [1, 1, 2, .5], 64, 4, [1, 2, 1, .5]]
]
np.random.seed(2108312109)
npoints = [10, 1000, 10_000, 50_000, 75_000, 100_000]


def bin_index(phi, n_bins: int):
    """
    Compute the phase bin index.
    """
    lo = -np.pi
    step = 2*np.pi/n_bins
    return (phi - lo)//step


def test_phase_offset_1d():
    """ Unit test for phase_offset(). Test the phase of a shifted cosine wave
    """
    npts = 100
    phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)

    # without phase
    phase_in = 0
    sig = np.cos(phi + phase_in)
    phase_out = phase_offset(sig)
    assert np.isclose(phase_out, phase_in)

    # with phase
    phase_in = 1
    sig = np.cos(phi + phase_in)
    phase_out = phase_offset(sig)
    assert np.isclose(phase_out, phase_in)


def test_phase_offset_2d():
    """ Unit test for phase_offset(). Test the phase of shifted cosine waves
    """
    npts = 100
    phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)

    # without phases
    phase_in_1 = 0
    phase_in_2 = 0
    sig = np.cos(phi + phase_in_1)[np.newaxis, :] + np.cos(phi + phase_in_2)[:, np.newaxis]
    phase_out_1 = phase_offset(sig, axis=-1)
    phase_out_2 = phase_offset(sig, axis=-2)
    assert np.isclose(phase_out_1, phase_in_1)
    assert np.isclose(phase_out_2, phase_in_2)

    # with phases
    phase_in_1 = 1
    phase_in_2 = 2
    sig = np.cos(phi + phase_in_1)[np.newaxis, :] + np.cos(phi + phase_in_2)[:, np.newaxis]
    phase_out_1 = phase_offset(sig, axis=-1)
    phase_out_2 = phase_offset(sig, axis=-2)
    assert np.isclose(phase_out_1, phase_in_1)
    assert np.isclose(phase_out_2, phase_in_2)


def test_phased_ft_shape_1d():
    """ Unit test for phased_ft(). fft should be done on right axis on 1D arrays.
    """
    npts = 100
    phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    sig = np.cos(phi)
    ft = phased_ft(sig)
    assert np.isclose(ft[1], 0.5)


def test_phased_ft_shape_2d():
    """ Unit test for phased_ft(). fft should be done on right axis on 2D arrays.
    """
    npts = 100
    phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    sig = np.cos(phi)[np.newaxis, :] + np.cos(phi)[:, np.newaxis]
    # axis = -1
    ft = phased_ft(sig, axis=-1)
    assert np.isclose(ft[:, 1], 0.5).all()
    assert np.isclose(ft[1, 2:], 0).all()
    # axis = -2
    ft = phased_ft(sig, axis=-2)
    assert np.isclose(ft[2:, 1], 0).all()
    assert np.isclose(ft[1, :], 0.5).all()


def test_phased_ft_float():
    """ Unit test for phased_ft(). Transformed 1D and 2D arrays should be rotated by given angle.
    The 2D case is tested on axis = -1
    """
    npts = 100
    phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    # phase = 90° => real part goes to zero
    phase = np.pi / 2
    sig = np.cos(phi + phase)[np.newaxis, :] + np.zeros(npts)[:, np.newaxis]
    ft = phased_ft(sig, axis=-1)
    assert np.isclose(ft, 0).all()
    # now with correction as function argument
    sig = np.cos(phi + phase)[np.newaxis, :] + np.zeros(npts)[:, np.newaxis]
    ft = phased_ft(sig, axis=-1, correction=phase)
    assert np.isclose(ft[:, 1], 0.5).all()


def test_phased_ft_fft():
    """ Unit test for phased_ft(). Transformed 1D and 2D arrays should phased corrected using phase_offset().
    The 2D case is tested on axis = -1
    """
    npts = 1000
    phi = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    phase = 1
    sig = np.cos(phi + phase)[np.newaxis, :] + np.zeros(npts)[:, np.newaxis]
    ft = phased_ft(sig, axis=-1, correction='fft')
    assert np.isclose(ft[:, 1], 0.5).all()


@pytest.mark.parametrize('shd_data_points', shd_parameters, indirect=['shd_data_points'])
def test_shd_binning(shd_data_points):
    """ Unit test for shd_binning(). Testing binning and shape of returned DataFrame.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params

    binned_data = shd_binning(data, signals, tap_nbins)

    # data should be unchanged by binning
    assert np.allclose(binned_data, data[:, signals.index(Signals.sig_a)])
    assert binned_data.shape[-1] == tap_nbins


@pytest.mark.parametrize('shd_data_points', shd_parameters, indirect=['shd_data_points'])
def test_shd_binning_shuffled(shd_data_points):
    """ Test that shd_binning returns a DataFrame ordered with respect to tap_n
    """
    # retrieve parameters and test data from fixture
    params, data, signals = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params

    rand_data = data.copy()
    np.random.shuffle(rand_data)
    rand_binned = shd_binning(rand_data, signals, tap_nbins)

    assert np.allclose(rand_binned, data[:, signals.index(Signals.sig_a)])
    assert rand_binned.shape[-1] == tap_nbins


@pytest.mark.parametrize('drops', [0, [0, 5], [3, 5, 8], [5, -1]])
@pytest.mark.parametrize('shd_data_points', shd_parameters, indirect=['shd_data_points'])
def test_shd_binning_empty(shd_data_points, drops):
    """ Test the binning of data containing missing bins.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params

    perforated_data = shd_binning(np.delete(data, drops, axis=0), signals, tap_nbins)

    # the shape should be unchanged
    assert perforated_data.shape[-1] == tap_nbins
    # Nan should be inserted in missing bins
    assert np.isnan(perforated_data[:, drops]).all()


@pytest.mark.parametrize('shd_data_points', shd_parameters, indirect=['shd_data_points'])
def test_shd_ft_shape(shd_data_points):
    """ Test the shape of the DataFrame returned by shd_ft.
    This test assumes that shd_binning is working as expected.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params

    binned = shd_binning(data, signals, tap_nbins)
    ft_data = shd_ft(binned)

    assert ft_data.shape[0] == tap_nbins // 2 + 1


@pytest.mark.parametrize('shd_data_points', shd_parameters, indirect=['shd_data_points'])
def test_shd_ft_harmonics(shd_data_points):
    """ Retrieve amplitudes and phases of harmonics with shd_ft.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    binned = shd_binning(data, signals, tap_nbins)
    ft_data = shd_ft(binned)

    for n in range(tap_nharm):
        initial_amp = tap_harm_amps[n]
        returned_amp = ft_data[n][0]
        if n > 0:
            returned_amp *= 2
        assert np.isclose(initial_amp, returned_amp)


@pytest.mark.parametrize('drops', [0, [0, 5], [3, 5, 8], [5, -1]])
@pytest.mark.parametrize('shd_data_points', shd_parameters, indirect=['shd_data_points'])
def test_shd_empty(shd_data_points, drops):
    """ Missing bins or bins filled with nans should be caught by shd.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = shd_data_points
    tap_nbins, tap_nharm, tap_harm_amps = params
    data = np.delete(data, drops, axis=0)

    with pytest.raises(ValueError):
        shd(data, signals, tap_nbins)


@pytest.mark.parametrize('pshet_data_points', pshet_parameters, indirect=['pshet_data_points'])
def test_pshet_binning(pshet_data_points):
    """ Unit test for pshet_binning. Only sig_a is tested.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    binned_data = pshet_binning(data, signals, tap_nbins, ref_nbins)

    # test shape of returned DataFrame
    assert binned_data.shape[-1] == tap_nbins
    assert binned_data.shape[-2] == ref_nbins

    # test if every data point in test_data matches the value in its position in the binned_data DataFrame
    for i in range(tap_nbins * ref_nbins):
        tap_n = int(bin_index(np.arctan2(data[i, signals.index(Signals.tap_y)],
                                         data[i, signals.index(Signals.tap_x)]), tap_nbins))
        ref_n = int(bin_index(np.arctan2(data[i, signals.index(Signals.ref_y)],
                                         data[i, signals.index(Signals.ref_x)]), ref_nbins))
        # pytes bug: This test sometimes fails where the values are off by the sign. This is not reproducible.
        # should be 0.929514217098255 ==  0.929514217098255
        # and -0.7439935916898542 == -0.7439935916898542
        # for i == 0 (when it fails, it does from the start. sig_a and sig_b as returned from binning are swapped
        assert binned_data[0, ref_n, tap_n] == data[i, signals.index(Signals.sig_a)]


@pytest.mark.parametrize('pshet_data_points', pshet_parameters, indirect=['pshet_data_points'])
def test_pshet_binning_shuffled(pshet_data_points):
    """ Test that pshet_binning returns a DataFrame ordered with respect to tap_n and ref_n
    """
    # retrieve parameters and test data from fixture
    params, data, signals = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params

    rand_data = data.copy()
    np.random.shuffle(rand_data)
    shuffled_data = pshet_binning(rand_data, signals, tap_nbins, ref_nbins)

    # test shape of returned DataFrame
    assert shuffled_data.shape[-1] == tap_nbins
    assert shuffled_data.shape[-2] == ref_nbins

    # test if every data point in test_data matches the value in its position in the binned_data DataFrame
    for i in range(tap_nbins * ref_nbins):
        tap_n = int(bin_index(np.arctan2(data[i, signals.index(Signals.tap_y)],
                                         data[i, signals.index(Signals.tap_x)]), tap_nbins))
        ref_n = int(bin_index(np.arctan2(data[i, signals.index(Signals.ref_y)],
                                         data[i, signals.index(Signals.ref_x)]), ref_nbins))
        # This test sometimes fails where the values are off by the sign. This is not reproducible.
        assert shuffled_data[0, ref_n, tap_n] == data[i, signals.index(Signals.sig_a)]


@pytest.mark.parametrize('drops', [[0], [31, 32, 100], [123, -1]])
@pytest.mark.parametrize('pshet_data_points', pshet_parameters, indirect=['pshet_data_points'])
def test_pshet_binning_empty(pshet_data_points, drops):
    """ Test the binning of data containing missing bins.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params

    perforated_data = pshet_binning(np.delete(data, drops, axis=0), signals, tap_nbins, ref_nbins)

    # the shape should be unchanged
    assert perforated_data.shape[-1] == tap_nbins
    assert perforated_data.shape[-2] == ref_nbins

    # select dropped data points and assert that they are all nans in binned data frame
    tap_n = bin_index(np.arctan2(data[drops, signals.index(Signals.tap_y)],
                                 data[drops, signals.index(Signals.tap_x)]), tap_nbins)
    ref_n = bin_index(np.arctan2(data[drops, signals.index(Signals.ref_y)],
                                 data[drops, signals.index(Signals.ref_x)]), ref_nbins)
    for i, _ in enumerate(drops):
        assert all(np.isnan(perforated_data[:, int(ref_n[i]), int(tap_n[i])]))


@pytest.mark.parametrize('pshet_data_points', pshet_parameters, indirect=['pshet_data_points'])
def test_pshet_ft_shape(pshet_data_points):
    """ Test the shape of the DataFrame returned by pshet_ft.
    This test assumes that pshet_binning is working as expected.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    binned = pshet_binning(data, signals, tap_nbins, ref_nbins)
    ft_data = pshet_ft(binned)

    # test shape of returned arrays
    detector_signals = [s for s in signals if s in all_detector_signals]
    assert ft_data.shape[0] == len(detector_signals)
    assert ft_data.shape[1] == ref_nbins // 2 + 1
    assert ft_data.shape[2] == tap_nbins // 2 + 1


@pytest.mark.parametrize('pshet_data_points', pshet_parameters, indirect=['pshet_data_points'])
def test_pshet_ft_harmonics(pshet_data_points):
    """ Retrieve amplitudes and phases of harmonics with phset_ft"""
    # retrieve parameters and test data from fixture
    params, data, signals = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params
    binned = pshet_binning(data, signals, tap_nbins, ref_nbins)
    ft_data = pshet_ft(binned)

    # both signals are actually the same, because of pytest..
    for single_signal in ft_data:
        # tapping harmonics
        for n in range(tap_nharm):
            initial_tap_amp = tap_harm_amps[n]
            returned_tap_amp = single_signal[0, n]
            if n > 0:
                returned_tap_amp *= 2
            assert np.isclose(initial_tap_amp, np.real(returned_tap_amp))

        # pshet harmonics
        for m in range(ref_nharm):
            initial_ref_amp = ref_harm_amps[m]
            returned_ref_amp = single_signal[m, 0]
            if m > 0:
                returned_ref_amp *= 2
            assert np.isclose(initial_ref_amp, returned_ref_amp)
        
        
@pytest.mark.parametrize('drops', [0, [31, 32, 100], [123, -1]])
@pytest.mark.parametrize('pshet_data_points', pshet_parameters, indirect=['pshet_data_points'])
def test_pshet_empty(pshet_data_points, drops):
    """ Missing bins or bins filled with nans should be caught by pshet.
    """
    # retrieve parameters and test data from fixture
    params, data, signals = pshet_data_points
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = params

    with pytest.raises(ValueError):
        pshet(np.delete(data, drops, axis=0), signals, tap_nbins, ref_nbins)


def test_pshet_coeff():
    """ Check that pshet_coeff returns col_1 + 1j * col_2 per signal for arrays filled with random complex.
    """
    # generate some complex test data:
    gamma = 2.63
    random_sig_a = np.random.random_sample((10, 10)) * np.exp(1j)
    random_sig_b = np.random.random_sample((10, 10)) * np.exp(1j)
    test_signal = np.concatenate((random_sig_a[np.newaxis, :, :], random_sig_b[np.newaxis, :, :]), axis=0)
    test_data = pshet_coefficients(test_signal, gamma)

    # pshet coefficients should return col_1 + 1j * col_2
    coeffs_a = np.abs(random_sig_a[1, :]) / jv(1, gamma) + 1j * np.abs(random_sig_a[2, :]) / jv(2, gamma)
    coeffs_b = np.abs(random_sig_b[1, :]) / jv(1, gamma) + 1j * np.abs(random_sig_b[2, :]) / jv(2, gamma)
    assert np.allclose(test_data[:, 0], coeffs_a)
    assert np.allclose(test_data[:, 1], coeffs_b)


# BENCHMARKING ##############################
@pytest.mark.parametrize('noise_data', filter(lambda n: n > 100, npoints), indirect=['noise_data'])
def test_shd_binning_benchmark(benchmark, noise_data):
    """ Benchmarks the speed of shd binning for different lengths of random data sets"""
    tap_nbins = 32
    data,  signals = noise_data
    benchmark(shd_binning, data, signals, tap_nbins)


@pytest.mark.parametrize('noise_data', filter(lambda n: n > 1000, npoints), indirect=['noise_data'])
def test_pshet_binning_benchmark(benchmark, noise_data):
    """ Benchmarks the speed of pshet binning for different lengths of random data sets"""
    tap_nbins = 32
    ref_nbins = 16
    data, signals = noise_data
    benchmark(pshet_binning, data, signals, tap_nbins, ref_nbins)
