import numpy as np
from scipy.special import jv
from scipy.stats import binned_statistic_2d
from lmfit import Parameters, minimize

from snompad.utility.signals import Signals
from .demod_utils import kernel_interpolation_1d, kernel_interpolation_2d, pshet_obj_func, corrected_fft, sort_chopped


def pshet_phases(data: np.ndarray, signals: list, normalize=False) -> np.ndarray:
    """ Mainly calculates tap_p and ref_p. If normalize == True, sig_b is used for normalization.
    """
    if normalize:
        signal = data[:, signals.index(Signals.sig_a)] / data[:, signals.index(Signals.sig_b)]
    else:
        signal = data[:, signals.index(Signals.sig_a)]
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    ref_p = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    sig_and_phase = np.vstack([signal, tap_p, ref_p]).T
    return sig_and_phase


def pshet_binning(sig_and_phase: np.ndarray, tap_res: int = 64, ref_res: int = 64) -> np.ndarray:
    """ Performs 2D binning on sig_a onto tap_p, ref_p domain.

    PARAMETERS
    ----------
    sig_and_phase: np.ndarray
        data array with sig_a, tap_p, and ref_p on axis=1 and samples on axis=0
    tap_res: float
        number of tapping bins
    ref_res: float
        number of reference bins

    RETURNS
    -------
    binned: np.ndarray
        average signals for each bin between -pi, pi.
        tapping bins on axis=1, reference bins on axis=0.
    """
    returns = binned_statistic_2d(x=sig_and_phase[:, 1], y=sig_and_phase[:, 2], values=sig_and_phase[:, 0],
                                  statistic='mean', bins=[tap_res, ref_res],
                                  range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    binned = returns.statistic
    return binned.T


def pshet_kernel_average(sig_and_phase: np.ndarray, tap_res: int = 64, ref_res: int = 64) -> np.ndarray:
    """ Averages and interpolates signal onto even grid using kernel smoothening
    """
    tap_grid = np.linspace(-np.pi, np.pi, tap_res, endpoint=False) + np.pi / tap_res
    ref_grid = np.linspace(-np.pi, np.pi, ref_res, endpoint=False) + np.pi / ref_res
    binned = kernel_interpolation_2d(sig_and_phase[:, 0], sig_and_phase[:, 1], sig_and_phase[:, 2], tap_grid, ref_grid)
    return binned


def pshet_binned_kernel(sig_and_phase: np.ndarray, tap_res: int = 64, ref_res: int = 64) -> np.ndarray:
    """ Reference phase is divided into intervals, each on which 1D kernel interpolation is applied on the tap axis.
    """
    ref_bounds = np.linspace(-np.pi, np.pi, ref_res + 1, endpoint=True)
    ref_bins = []
    for i in range(ref_res):
        idx = np.logical_and(ref_bounds[i] < sig_and_phase[:, 2], sig_and_phase[:, 2] < ref_bounds[i + 1])
        b = np.vstack([sig_and_phase[idx, 0], sig_and_phase[idx, 1]])
        ref_bins.append(b)

    tap_grid = np.linspace(-np.pi, np.pi, tap_res, endpoint=False) + np.pi / tap_res
    binned = []
    for b in ref_bins:
        binned.append(kernel_interpolation_1d(signal=b[0], x_sig=b[1], x_grid=tap_grid))
    return np.array(binned)


def pshet_fitting(tap_ft: np.ndarray, max_order: int):
    """ Fits the pshet phase modulation on every tapping harmonic.
    Returned harmonics are amplitude and phase of modulation.
    """
    theta_ref = np.linspace(-np.pi, np.pi, tap_ft.shape[0], endpoint=False) + np.pi / tap_ft.shape[0]
    sigma_tracker = []

    sig = tap_ft[:, 0]
    sig_ft = np.fft.rfft(sig, norm='forward')
    tau = (np.abs(sig_ft[1]) + 1j * np.abs(sig_ft[2])) * np.exp(1j * np.pi / 2)

    params = Parameters()
    params.add('theta_0', min=-np.pi, max=0, value=-np.pi / 2)
    params.add('gamma', min=0, max=10, value=2.5)
    params.add('sigma_re', min=-1, max=1, value=np.real(tau))
    params.add('sigma_im', min=-1, max=1, value=np.imag(tau))
    params.add('offset', min=0, max=1, value=np.abs(sig_ft[0]))

    fit = minimize(pshet_obj_func, params, args=(theta_ref, sig), method='leastsqr')
    theta_0 = fit.params['theta_0'].value
    gamma = fit.params['gamma'].value
    sigma_re = fit.params['sigma_re'].value
    sigma_im = fit.params['sigma_im'].value
    sigma_tracker.append([sigma_re, sigma_im])

    for h in range(1, max_order + 1):
        sig = tap_ft[:, h]

        params = Parameters()
        params.add('theta_0', value=theta_0, vary=False)
        params.add('gamma', value=gamma, vary=False)
        params.add('offset', value=sig.mean(), vary=False)
        params.add('sigma_re', min=-1, max=1, value=sigma_re)
        params.add('sigma_im', min=-1, max=1, value=sigma_im)

        fit = minimize(pshet_obj_func, params, args=(theta_ref, sig), method='leastsqr')
        sigma_re = fit.params['sigma_re'].value
        sigma_im = fit.params['sigma_im'].value
        sigma_tracker.append([sigma_re, sigma_im])

    harmonics = []
    for s in sigma_tracker:
        sigma = s[0] + 1j * s[1]
        harmonics.append(sigma)
    harmonics = np.array(harmonics)
    return harmonics


def pshet_coefficients(ft: np.ndarray, gamma: float = 2.63, psi_R: float = 1.6, m: int = 1) -> np.ndarray:
    """ Computes coefficients for tapping demodulation.

    PARAMETERS
    ----------
    ft: np.ndarray
        array containing Fourier amplitudes. ref on axis=0, tap on axis=1
    gamma: float
        modulation depth
    psi_R: float
        offset in optical phase between sig and ref beam, determined by fitting reference modulation at phase of contact
    m: int
        sidebands to be evaluated are m and m+1

    RETURNS
    -------
    coefficients: np.ndarray
        complex coefficients for tapping demodulation
    """
    m = np.array([m, m + 1])
    scales = 1 / jv(m, gamma) * np.array([1, 1j])
    coefficients = (ft[m, :] * scales[:, np.newaxis]).sum(axis=0)

    phase = np.exp(1j * (psi_R + m[0] * np.pi / 2))
    coefficients *= phase
    return coefficients


def pshet_sidebands(tap_ft: np.ndarray, max_order=None):
    """ Demodulates pshet modulation by calculating harmonic content via fft, and summing adjacent sidebands
    """
    ft = corrected_fft(array=tap_ft, axis=0, correction=None)
    harmonics = pshet_coefficients(ft=ft)
    if max_order is not None:
        harmonics = harmonics[:max_order+1]
    return harmonics


def pshet(data: np.ndarray, signals: list, tap_res: int = 64, ref_res: int = 64, chopped='auto', tap_correction='fft',
          binning='binned_kernel', pshet_demod='fitting', normalize='auto', max_order=5) -> np.ndarray:
    """ Simple combination pshet demodulation functions. The sorting of chopped and pumped pulses is handled here.

    Parameters
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_res: float
        number of tapping bins
    ref_res: float
        number of reference bins
    tap_correction: str or None or float
        Type or value of phase correction along axis=-1, see phased_ft()
    chopped: str or bool
        determines whether chopping should be considered during binning, 'auto' -> True when Signals.chop in signals
    binning: 'binning', 'binned_kernel', 'kernel'
        determines routine to compute binned phase domain
    pshet_demod: 'sidebands', 'fitting'
        determines routine to demodulate pshet to return harmonics
    normalize: 'auto', True, False
        if True, sig_a is normalized to sig_a / sig_b. When 'auto', normalize is True when sig_b in signals
    max_order: int
        max harmonic order that should be returned.

    Returns
    -------
    coefficients: np.ndarray
        complex coefficients for tapping demodulation
    """
    if chopped == 'auto':
        chopped = Signals.chop in signals
    if normalize == 'auto':
        normalize = Signals.sig_b in signals

    if chopped:
        chopped_idx, pumped_idx = sort_chopped(data[:, signals.index(Signals.chop)])
        sig_and_phase = [pshet_phases(data=data[pumped_idx], signals=signals, normalize=normalize),
                         pshet_phases(data=data[chopped_idx], signals=signals, normalize=normalize)]
    else:
        sig_and_phase = [pshet_phases(data=data, signals=signals, normalize=normalize)]

    binning_func = [pshet_binning, pshet_binned_kernel,
                    pshet_kernel_average][['binning', 'binned_kernel', 'kernel'].index(binning)]
    binned = [binning_func(sig_and_phase=sig, tap_res=tap_res, ref_res=ref_res) for sig in sig_and_phase]
    if chopped:
        binned = (binned[0] - binned[1])  # / binned[1]
    else:
        binned = binned[0]

    tap_ft = corrected_fft(array=binned, axis=1, correction=tap_correction)

    pshet_func = [pshet_sidebands, pshet_fitting][['sidebands', 'fitting'].index(pshet_demod)]
    harmonics = pshet_func(tap_ft=tap_ft, max_order=max_order)
    return harmonics
