# functions to demodulate SNOM data acquired in pseudo-heterodyne (pshet) mode
import numpy as np
from scipy.special import jv
from scipy.stats import binned_statistic_2d
from lmfit import Parameters, minimize

from .demod_utils import kernel_interpolation_1d, kernel_interpolation_2d, pshet_obj_func,\
    corrected_fft, chop_pump_idx, chopped_data, pumped_data
from .demod_corrections import normalize_sig_a
from ..utility.signals import Signals
from ..utility.multiprocess import process_manager


def pshet_binning(data: np.ndarray, signals=list, tap_res: int = 64, ref_res: int = 64) -> np.ndarray:
    """ Performs binning of sig_a onto 2D tapping and pshet phase domain.

    PARAMETERS
    ----------
    data: np.ndarray
        data array with signals on axis=0 and samples on axis=1
    signals: list(Signals)
        signals acquired by DAQ. Must have same order as signals on axis=0 in data.
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
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    ref_p = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    returns = binned_statistic_2d(x=tap_p, y=ref_p, values=data[:, signals.index(Signals.sig_a)],
                                  statistic='mean', bins=[tap_res, ref_res],
                                  range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    binned = returns.statistic
    return binned.T


def pshet_kernel_average(data: np.ndarray, signals: list, tap_res: int = 64, ref_res: int = 64) -> np.ndarray:
    """ Averages and interpolates signal onto even grid using kernel smoothening

    PARAMETERS
    ----------
    data: np.ndarray
        data array with signals on axis=0 and samples on axis=1
    signals: list(Signals)
        signals acquired by DAQ. Must have same order as signals on axis=0 in data.
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
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    ref_p = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    tap_grid = np.linspace(-np.pi, np.pi, tap_res, endpoint=False) + np.pi / tap_res
    ref_grid = np.linspace(-np.pi, np.pi, ref_res, endpoint=False) + np.pi / ref_res
    binned = kernel_interpolation_2d(signal=data[:, signals.index(Signals.sig_a)],
                                     x_sig=tap_p, y_sig=ref_p, x_grid=tap_grid, y_grid=ref_grid)
    return binned


def pshet_binned_kernel(data: np.ndarray, signals: list, tap_res: int = 64, ref_res: int = 64) -> np.ndarray:
    """ Reference phase is divided into intervals, each on which 1D kernel interpolation is applied on the tap axis.

    PARAMETERS
    ----------
    data: np.ndarray
        data array with signals on axis=0 and samples on axis=1
    signals: list(Signals)
        signals acquired by DAQ. Must have same order as signals on axis=0 in data.
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
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    ref_p = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    ref_bounds = np.linspace(-np.pi, np.pi, ref_res + 1, endpoint=True)
    ref_bins = []
    for i in range(ref_res):
        idx = np.logical_and(ref_bounds[i] < ref_p, ref_p < ref_bounds[i + 1])
        b = np.vstack([data[idx, signals.index(Signals.sig_a)], tap_p[idx]])
        ref_bins.append(b)
        # todo Can we pack everything into this for loop?

    tap_grid = np.linspace(-np.pi, np.pi, tap_res, endpoint=False) + np.pi / tap_res
    collector = {}
    binned = []
    for i, b in enumerate(ref_bins):
        args = {'signal': b[0],
                'x_sig': b[1],
                'x_grid': tap_grid}
        process_manager(func=kernel_interpolation_1d, args=args, collector=collector, identifier=i)
        binned.append([])
        # binned.append(kernel_interpolation_1d(signal=b[0], x_sig=b[1], x_grid=tap_grid))
    for k, v in collector.items():
        binned[k] = v
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
    ft = corrected_fft(array=tap_ft, axis=0, phase_correction=None)
    harmonics = pshet_coefficients(ft=ft)
    if max_order is not None:
        harmonics = harmonics[:max_order+1]
    return harmonics


def pshet(data: np.ndarray, signals: list, tap_res: int = 64, ref_res: int = 64, chopped='auto', tap_correction='fft',
          binning='binned_kernel', pshet_demod='fitting', ratiometry='auto', max_order=5) -> np.ndarray:
    """ Simple combination pshet demodulation functions. Pump-probe demodulation,
    tapping phase correction, and ratiometric correction is also handled here.

    PARAMETERS
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
    ratiometry: 'auto', True, False
        if True, sig_a is normalized to sig_a / sig_b. When 'auto', normalize is True when sig_b in signals
    max_order: int
        max harmonic order that should be returned.

    RETURNS
    -------
    coefficients: np.ndarray
        complex coefficients for tapping demodulation
    """
    if ratiometry is True or (ratiometry == 'auto' and Signals.sig_b in signals):
        data = normalize_sig_a(data=data, signals=signals)

    if chopped is True or (chopped == 'auto' and Signals.chop in signals):
        chop_idx, pump_idx = chop_pump_idx(data[:, signals.index(Signals.chop)])
        data_chop = chopped_data(data=data, idx=chop_idx)
        data_pump = pumped_data(data=data, idx=pump_idx)
        data = [data_chop, data_pump]
    else:
        data = [data]

    binning_func = [pshet_binning, pshet_binned_kernel,
                    pshet_kernel_average][['binning', 'binned_kernel', 'kernel'].index(binning)]
    binned = [binning_func(data=d, signals=signals, tap_res=tap_res, ref_res=ref_res) for d in data]

    # first perform ft along tapping axis (axis=1)
    tap_ft = [corrected_fft(array=b, axis=1, phase_correction=tap_correction) for b in binned]
    # for pshet demodulation on reference axis (axis=0), sidebands can be determined via fft or modulation can be fitted
    pshet_func = [pshet_sidebands, pshet_fitting][['sidebands', 'fitting'].index(pshet_demod)]
    harmonics = [pshet_func(tap_ft=t, max_order=max_order) for t in tap_ft]

    if len(harmonics) == 2:
        harmonics = harmonics[1] / harmonics[0] - 1
    else:
        harmonics = harmonics[0]

    return harmonics
