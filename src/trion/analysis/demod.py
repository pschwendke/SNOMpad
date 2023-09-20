from warnings import warn
import numpy as np
from scipy.special import jv
from scipy.stats import binned_statistic, binned_statistic_2d
from lmfit.models import GaussianModel
from lmfit import Parameters, minimize

import pandas as pd

from .signals import Signals
from .modelling import pshet_obj_func
from .utils import kernel_interpolation_1d, kernel_interpolation_2d


# UTILITY: FFT # PHASING ###############################################################################################
def phase_offset(binned: np.ndarray, axis=-1) -> float:
    """Determine phase shift required to make FT real.

    Parameters
    ----------
    binned : np.ndarray, real
        Binned date. Real.
    axis : int
        Axis to perform FT. Use `tap_p` for `theta_C`, `ref_p` for `theta_0`.

    Returns
    -------
    phi : float
        Phase offset of 1st harmonic along given axis. Offset is averaged for 2D phase domains.

    Note
    ----
    Operate on the appropriate axis to determine the phase offsets from the
    binned data. For binned with shape (M, tap_nbins)
    """
    spec = np.fft.rfft(binned, axis=axis)
    phi = np.angle(spec.take(1, axis=axis))
    phi = phi - (phi > 0) * np.pi  # shift all results to negative quadrant.
    phi = phi.mean()  # Output should be a float. However, is this stable?
    return phi


def phased_ft(array: np.ndarray, axis: int = -1, correction=None) -> np.ndarray:
    """ Computes real fft of array along given axis. correction determines the type of phase correction.

    Parameters
    ----------
    array: np.ndarray
    axis: int
    correction: str or None or float
        If correction is None, the coefficients are rotated by 0°.
        If correction == 'fft', phase_offset() is used to determine the phase correction.
            A binned phase domain is assumed, binning offset of pi and half a bin are corrected
        If correction is float, the value is used for phase correction.

    Returns
    -------
    np.real(ft)

    """
    if np.any(np.isnan(array)):
        raise ValueError('The array has empty bins.')

    ft = np.fft.rfft(array, axis=axis, norm='forward')

    if correction == 'fft':
        phase = - phase_offset(array, axis=axis)
    elif type(correction) in [float, int]:
        phase = - correction  # You might want to flip the sign. Check this when using.
    elif correction is None:
        return np.real(ft)
    else:
        raise NotImplementedError

    if array.ndim == 1:  # shd
        phase *= np.arange(ft.shape[axis])
    elif array.ndim == 2 and axis == 1:  # pshet tap axis
        phase = np.arange(ft.shape[axis])[np.newaxis, :] * phase
    elif array.ndim == 2 and axis == 0:  # pshet ref axis
        phase = np.arange(ft.shape[axis])[:, np.newaxis] * phase

    ft *= np.exp(1j * phase)
    return np.real(ft)


def sort_chopped(chop: np.ndarray) -> tuple:
    """ Takes optical chop signal, returns boolean indices of chopped and pumped shots
    """
    hi_sig = chop[chop > np.median(chop)]
    lo_sig = chop[chop < np.median(chop)]
    model = GaussianModel()

    # fit gaussian to chop signal for pumped shots
    values, edges = np.histogram(hi_sig, bins=100)
    A = values.max()
    result = model.fit(data=values, x=edges[:-1], center=edges[:-1][values == A][0], amplitude=A, sigma=.001,
                       method='leastsqr')
    hilim = result.best_values['center'] - 3 * result.best_values['sigma']
    pumped = chop > hilim

    # fit gaussian to chop signal for chopped shots
    values, edges = np.histogram(lo_sig, bins=100)
    A = values.max()
    result = model.fit(data=values, x=edges[:-1], center=edges[:-1][values == A][0], amplitude=A, sigma=.001,
                       method='leastsqr')
    lolim = result.best_values['center'] + 3 * result.best_values['sigma']
    chopped = chop < lolim

    return chopped, pumped


# SHD DEMODULATION #####################################################################################################
def shd_phases(data: np.ndarray, signals: list, normalize=False) -> np.ndarray:
    """ Mainly calculates tap_p. If normalize == True, sig_b is used for normalization.
    """
    # ToDo: add balanced detection
    if normalize:
        signal = data[:, signals.index(Signals.sig_a)] / data[:, signals.index(Signals.sig_b)]
    else:
        signal = data[:, signals.index(Signals.sig_a)]
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    sig_and_phase = np.vstack([signal, tap_p]).T
    return sig_and_phase


def shd_binning(sig_and_phase: np.ndarray, tap_res: int = 64) -> np.ndarray:
    """ Bins sig_a into 1D tap_p phase domain. tap_y, tap_x must be included.

    PARAMETERS
    ----------
    sig_and_phase: np.ndarray
        data array with sig_a and tap_p on axis=1 and samples on axis=0
    tap_res: float
        number of tapping bins

    RETURNS
    -------
    binned: np.ndarray
        average signals for each bin between -pi, pi.
    """
    returns = binned_statistic(x=sig_and_phase[:, 1], values=sig_and_phase[:, 0],
                               statistic='mean', bins=tap_res, range=[-np.pi, np.pi])
    binned = returns.statistic
    return binned


def shd_kernel_average(sig_and_phase: np.ndarray, tap_res: int = 64) -> np.ndarray:
    """ Averages and interpolates signal onto even grid using kernel smoothening
    """
    tap_grid = np.linspace(-np.pi, np.pi, tap_res, endpoint=False) + np.pi / tap_res
    binned = kernel_interpolation_1d(signal=sig_and_phase[:, 0], x_sig=sig_and_phase[:, 1], x_grid=tap_grid)
    return binned


def shd(data: np.ndarray, signals: list, tap_res: int = 64, chopped='auto', tap_correction='fft',
        binning='binning', normalize='auto') -> np.ndarray:
    """ Simple combination shd demodulation functions. The sorting of chopped and pumped pulses is handled here.

    Parameters
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_res: float
        number of tapping bins
    tap_correction: str or None or float
        Type or value of phase correction along axis=-1, see phased_ft()
    chopped: str or bool
        determines whether chopping should be considered during binning, 'auto' -> True when Signals.chop in signals
    binning: 'binning', 'binned_kernel', 'kernel'
        determines routine to compute binned phase domain
    normalize: 'auto', True, False
        if True, sig_a is normalized to sig_a / sig_b. When 'auto', normalize is True when sig_b in signals

    Returns
    -------
    coefficients: np.ndarray
        real coefficients for tapping demodulation.
    """
    if chopped == 'auto':
        chopped = Signals.chop in signals
    if normalize == 'auto':
        normalize = Signals.sig_b in signals

    if chopped:
        chopped_idx, pumped_idx = sort_chopped(data[:, signals.index(Signals.chop)])
        sig_and_phase = [shd_phases(data=data[pumped_idx], signals=signals, normalize=normalize),
                         shd_phases(data=data[chopped_idx], signals=signals, normalize=normalize)]
    else:
        sig_and_phase = [shd_phases(data=data, signals=signals, normalize=normalize)]

    binning_func = [shd_binning, shd_kernel_average][['binning', 'kernel'].index(binning)]
    binned = [binning_func(sig_and_phase=sig, tap_res=tap_res) for sig in sig_and_phase]
    if chopped:
        binned = (binned[0] - binned[1])  # / binned[1]
    else:
        binned = binned[0]

    ft = phased_ft(array=binned, axis=0, correction=tap_correction)
    return ft


# PSHET DEMODULATION ###################################################################################################
def pshet_phases(data: np.ndarray, signals: list, normalize=False) -> np.ndarray:
    """ Mainly calculates tap_p and ref_p. If normalize == True, sig_b is used for normalization.
    """
    # ToDo: add balanced detection
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
    ft = phased_ft(array=tap_ft, axis=0, correction=None)
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

    tap_ft = phased_ft(array=binned, axis=1, correction=tap_correction)

    pshet_func = [pshet_sidebands, pshet_fitting][['sidebands', 'fitting'].index(pshet_demod)]
    harmonics = pshet_func(tap_ft=tap_ft, max_order=max_order)
    return harmonics


# NAIVE DISCRETE FT DEMODULATION #######################################################################################
def dft_lstsq(phi, sig, max_order: int):
    assert phi.ndim == 1
    assert max_order > 1
    orders = np.arange(1, max_order + 1)
    coeff = np.hstack([
        np.ones_like(phi)[:, np.newaxis],  # DC
        np.cos(orders[np.newaxis, :] * phi[:, np.newaxis]),  # cos
        np.sin(orders[np.newaxis, :] * phi[:, np.newaxis])  # sin
    ])
    assert coeff.shape == (phi.size, 2 * max_order + 1)
    soln, _, _, _ = np.linalg.lstsq(coeff, sig, rcond=None)
    ret = np.asarray(soln[:max_order + 1], dtype=complex)  # what happens if sig.ndim > 1?
    ret[1:] += 1j * soln[max_order + 1:]
    return ret


def dft_naive(phi, y, orders):
    """
    Perform a naive fourier transform using trapezoidal integration.

    The input arrays are reorganized such that `phi` is sorted. The first
    element is "looped back" to integrate across the entire circle.

    Parameters
    ----------
    phi : (N,) np.ndarray
        Values of phi.
    y : (N,) or (N,K) np.ndarray
        Values of y. If 2d, the DFT is performed for every column.
    orders : (M,) np.ndarray
        Demodulation orders

    Returns
    -------
    amp : (M,) or (M, K) np.ndarray
        Complex amplitude for the given orders.
    """
    assert phi.ndim == 1
    y_org_ndim = y.ndim
    if y.ndim == 1:
        y = y[:, np.newaxis]
    else:
        assert y.ndim == 2
    # sort by phi
    idx = np.argsort(phi).tolist()
    # we need to repeat the first elements at the end, so we integrate over the
    # whole circle, without missing the seam where it rolls around
    idx = idx + [idx[0]]  # append last element again
    y_s = y.take(idx, axis=0).T
    phi_s = phi.take(idx)
    phi_s[-1] += 2 * np.pi
    intgr = np.trapz(
        np.exp(1j * phi_s[np.newaxis, :] * orders[:, np.newaxis])[np.newaxis, :, :] * y_s[:, np.newaxis, :],
        phi_s,
        axis=-1
    ) / np.pi
    intgr[:, orders == 0] *= 0.5
    if y_org_ndim == 1:
        intgr = np.squeeze(intgr)
    else:
        intgr = intgr.T
    return intgr


def shd_naive(df: pd.DataFrame, max_order: int) -> pd.DataFrame:
    """
    Perform shd demodulation using naive discrete FT (DFT).

    Compute the tapping phase from `tap_y` and `tap_x`, then extracts Fourier
    coefficients of the signals using naive DFT.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of sample points. The columns indicate the signal type.
    max_order: int
        Largest order.

    Returns
    -------
    amps : pd.DataFrame
        Dataframe containing the Fourier coefficients. Row indicate order,
        columns indicate signal type.

    See also
    -------
    dft_naive: DFT used by this function.
    shd: Standard shd demodulation by binning and FT.

    """
    phi = np.arctan2(df["tap_y"], df["tap_x"])
    data = df.drop(columns=["tap_x", "tap_y"])
    cols = data.columns.copy()
    assert data.ndim == 2
    assert phi.ndim == 1
    assert data.shape[0] == phi.size
    assert data.shape[1] == len(cols)
    amps = dft_naive(phi.to_numpy(), data.to_numpy(), np.arange(max_order))
    return pd.DataFrame(amps, columns=cols)


# OLDER STUFF # FOR COMPATIBILITY ######################################################################################
_deprecation_warning = FutureWarning("This function is deprecated. Please use the `shd` and `pshet` set of functions.")


def shd_ft(binned: np.ndarray) -> np.ndarray:
    """ Performs fft on array on binned data. Empty bins (NANs) raise ValueError.

    PARAMETERS
    ----------
    binned: np.ndarray
        array of values for fft. Values should be on axis=-1

    RETURNS
    -------
    ft: np.ndarray
        complex amplitudes of Fourier components. Orientation is amplitudes on axis=0 and signals on axis=1.
    """
    warn(_deprecation_warning)
    if np.any(np.isnan(binned)):
        raise ValueError("The binned array has empty bins.")
    ft = np.fft.rfft(binned) / binned.shape[-1]
    return ft.T


def pshet_ft(binned: np.ndarray) -> np.ndarray:
    """ Performs fft on array on binned data. Empty bins (NANs) raise ValueError.

    PARAMETERS
    ----------
    binned: np.ndarray
        array of values for fft. tapping bins on axis=-1, reference bins on axis=-2

    RETURNS
    -------
    ft: np.ndarray
        complex amplitudes of Fourier components.
    """
    warn(_deprecation_warning)
    if np.any(np.isnan(binned)):
        raise ValueError("The binned array has empty bins.")
    ft = np.fft.rfft2(binned) / binned.shape[-1] / binned.shape[-2]
    return ft


def calc_phase(df, in_place=False):
    warn(_deprecation_warning)
    "Computes tap_p"
    if not in_place:
        df = df.copy()
    df["tap_p"] = np.arctan2(df["tap_y"], df["tap_x"])
    return df


def bin_no(df, n_bins, in_place=False):
    warn(_deprecation_warning)
    "Computes bin_no"
    if not in_place:
        df = df.copy()
    lo = -np.pi
    step = 2 * np.pi / n_bins
    df["bin_no"] = (df["tap_p"] - lo) // step
    return df


def binned_average(df, n_bins, compute_counts=True):
    warn(_deprecation_warning)
    df = df.copy()
    if "tap_p" not in df.columns:
        calc_phase(df, in_place=True)
    if "bin_no" not in df.columns:
        bin_no(df, n_bins, in_place=True)
    df = df.drop(columns=[c for c in df.columns
                          if c in ("tap_x", "tap_y", "tap_p")])
    grpd = df.groupby("bin_no")
    avg = grpd.mean()
    if compute_counts:
        avg["count"] = grpd.count().iloc[:, 0]
    step = 2 * np.pi / n_bins
    lo = -np.pi
    avg["phi"] = avg.index * step + lo + step / 2
    # avg = avg.drop(columns="bin_no")
    return avg


def binned_ft(avg):  # these names suck
    warn(_deprecation_warning)
    step = np.diff(avg["phi"].iloc[:2])[0]  # argh...
    sigs = avg.drop(columns="phi")
    return pd.DataFrame(np.fft.rfft(sigs, axis=0) * step / 2 / np.pi, columns=sigs.columns)


def dft_binned(phi, sig, n_bins=256):
    """
    Perform demodulation using binning.

    The values are first collected in `nbins` depending on the value of `phi`.
    The average of `sig` is computed for each of these bins. The averaged
    values are used to compute the fourier components.
    """
    # Somehow very fast?!
    warn(_deprecation_warning)
    df = pd.DataFrame(np.vstack([phi, sig]).T, columns=["tap_p", "sig"])
    return np.squeeze(binned_ft(binned_average(df, n_bins, compute_counts=False)).to_numpy())
