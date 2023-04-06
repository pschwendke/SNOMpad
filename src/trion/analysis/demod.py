# ToDo: standardize names of variables

from warnings import warn
import numpy as np
from scipy.special import jv
from scipy.stats import binned_statistic, binned_statistic_2d
from lmfit.models import GaussianModel
import pandas as pd

from .signals import Signals, all_detector_signals
from .modelling import pshet_fitmodulation


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
    binned data. For binned with shape (M, tap_nbins
    """
    spec = np.fft.rfft(binned, axis=axis)
    phi = np.angle(spec.take(1, axis=axis))
    # phi = phi - (phi > 0) * np.pi  # shift all results to negative quadrant.
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
        raise ValueError("The array has empty bins.")

    ft = np.fft.rfft(array, axis=axis, norm='forward')

    if correction == 'fft':
        phase = - phase_offset(array, axis=axis)
    elif type(correction) in [float, int]:
        phase = - correction  # You might want to flip the sign. Check this when using.
    elif correction is None:
        return np.real(ft)
    else:
        raise NotImplementedError

    if array.ndim == 2:  # shd
        phase *= np.arange(ft.shape[axis])
    elif array.ndim == 3 and axis == -1:  # pshet tap axis
        phase = np.arange(ft.shape[axis])[np.newaxis, np.newaxis, :] * phase
    elif array.ndim == 3 and axis == -2:  # pshet ref axis
        phase = np.arange(ft.shape[axis])[np.newaxis, :, np.newaxis] * phase

    ft *= np.exp(1j * phase)
    return np.real(ft)


def sort_chopped(chop: np.ndarray) -> tuple:
    """ Takes optical chop signal, returns boolean indices of chopped and pumped shots
    """
    upper = chop[chop > np.median(chop)]
    lower = chop[chop < np.median(chop)]
    model = GaussianModel()

    # fit gaussian to chop signal for pumped shots
    values, edges = np.histogram(upper, bins=100)
    A = values.max()
    result = model.fit(data=values, x=edges[:-1], center=edges[:-1][values == A][0], amplitude=A, sigma=.001,
                       method='leastsqr')
    hilim = result.best_values['center'] - 3 * result.best_values['sigma']
    pumped = chop > hilim

    # fit gaussian to chop signal for chopped shots
    values, edges = np.histogram(lower, bins=100)
    A = values.max()
    result = model.fit(data=values, x=edges[:-1], center=edges[:-1][values == A][0], amplitude=A, sigma=.001,
                       method='leastsqr')
    lolim = result.best_values['center'] + 3 * result.best_values['sigma']
    chopped = chop < lolim

    return chopped, pumped


# SHD DEMODULATION #####################################################################################################
def shd_binning(data: np.ndarray, signals: list, tap_nbins: int = 64) -> np.ndarray:
    """ Bins signals into 1D tap_p phase domain. tap_y, tap_x must be included. When Signals.chop is included in signals
        chopped and pumped are binned sepparately and the difference normalized by the chopped signal (probe only) is
        returned.

    PARAMETERS
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_nbins: int
        number of tapping bins

    RETURNS
    -------
    binned: np.ndarray
        average signals for each bin between -pi, pi. Signals on axis=0, values on axis=1.
    """
    detector_signals = [data[:, signals.index(det_sig)] for det_sig in all_detector_signals if det_sig in signals]
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])

    if Signals.chop in signals:
        chopped_idx, pumped_idx = sort_chopped(data[:, signals.index(Signals.chop)])
        pumped = binned_statistic(x=tap_p[pumped_idx], values=[s[pumped_idx] for s in detector_signals],
                                  statistic='mean', bins=tap_nbins, range=[-np.pi, np.pi])
        chopped = binned_statistic(x=tap_p[chopped_idx], values=[s[chopped_idx] for s in detector_signals],
                                   statistic='mean', bins=tap_nbins, range=[-np.pi, np.pi])

        binned = (pumped.statistic - chopped.statistic) / chopped.statistic
        return binned

    else:
        returns = binned_statistic(x=tap_p, values=detector_signals,
                                   statistic='mean', bins=tap_nbins, range=[-np.pi, np.pi])
        binned = returns.statistic
        return binned


def shd(data: np.ndarray, signals: list, tap_nbins: int = 64, tap_correction='fft') -> np.ndarray:
    """ Simple combination of shd_binning and shd_ft

    PARAMETERS
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_nbins: int
        number of tapping bins
    tap_correction: str or None or float
        Type or value of phase correction, see phased_ft()

    RETURNS
    -------
    ft: np.ndarray
        real amplitudes of Fourier components. Orientation is amplitudes on axis=0 and signals on axis=1.
    """
    binned = shd_binning(data=data, signals=signals, tap_nbins=tap_nbins)
    ft = phased_ft(array=binned, axis=-1, correction=tap_correction)
    return ft.T


# PSHET DEMODULATION ###################################################################################################
def pshet_binning(data: np.ndarray, signals: list, tap_nbins: int = 64, ref_nbins: int = 64) -> np.ndarray:
    """ Performs 2D binning on signals onto tap_p, ref_p domain. tap_y, tap_x, ref_x, ref_y must be included.
        When Signals.chop is included in signals chopped and pumped are binned sepparately and the difference
        normalized by the chopped signal (probe only) is returned.

    PARAMETERS
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_nbins: float
        number of tapping bins
    ref_nbins: float
        number of reference bins

    RETURNS
    -------
    binned: np.ndarray
        average signals for each bin between -pi, pi. Signals on axis=0,
        tapping bins on axis=2, reference bins on axis=1.
    """
    detector_signals = [data[:, signals.index(det_sig)] for det_sig in all_detector_signals if det_sig in signals]
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    ref_p = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])

    if Signals.chop in signals:
        chopped_idx, pumped_idx = sort_chopped(data[:, signals.index(Signals.chop)])
        pumped = binned_statistic_2d(x=tap_p[pumped_idx], y=ref_p[pumped_idx],
                                     values=[s[pumped_idx] for s in detector_signals], statistic='mean',
                                     bins=[tap_nbins, ref_nbins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        chopped = binned_statistic_2d(x=tap_p[chopped_idx], y=ref_p[chopped_idx],
                                      values=[s[chopped_idx] for s in detector_signals], statistic='mean',
                                      bins=[tap_nbins, ref_nbins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        binned = (pumped.statistic - chopped.statistic) / chopped.statistic
    else:
        returns = binned_statistic_2d(x=tap_p, y=ref_p, values=detector_signals, statistic='mean',
                                      bins=[tap_nbins, ref_nbins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        binned = returns.statistic

    return binned.transpose(0, 2, 1)


def pshet_coefficients(ft: np.ndarray, gamma: float = 2.63, psi_R: float = 1.6, m: int = 1) -> np.ndarray:
    """ Computes coefficients for tapping demodulation.

    PARAMETERS
    ----------
    ft: np.ndarray
        array containig Fourier amplitudes. signals on axis=0, ref on axis=1, tap on axis=2
    gamma: float
        modulation depth
    psi_R: float
        offset in optical phase between sig and ref beam, determined by fitting reference modulation at phase of contact
    m: int
        sidebands to be evaluated are m and m+1

    RETURNS
    -------
    coefficients: np.ndarray
        complex coefficients for tapping demodulation. tapping harmonics on axis=0, signals on axis=1
    """
    m = np.array([m, m+1])
    scales = 1 / jv(m, gamma) * np.array([1, 1j])
    coefficients = (ft[:, m, :] * scales[np.newaxis, :, np.newaxis]).sum(axis=1)

    phase = np.exp(1j * (psi_R + m[0] * np.pi / 2))
    coefficients *= phase

    return coefficients.T


def pshet(data: np.ndarray, signals: list, tap_nbins: int = 64, ref_nbins: int = 64, demod_params=None,
          tap_correction='fft', ref_correction='fft', gamma: float = 2.63, psi_R: float = 1.6, m: int = 1) -> np.ndarray:
    """ Simple combination of pshet_binning, pshet_ft, and pshet_coeff

    Parameters
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_nbins: float
        number of tapping bins
    ref_nbins: float
        number of reference bins
    tap_correction: str or None or float
        Type or value of phase correction along axis=-1, see phased_ft()
    ref_correction: str or None or float
        Type or value of phase correction along axis=-2, see phased_ft()
    gamma: float
        modulation depth
    psi_R: float
        offset in optical phase between sig and ref beam, determined by fitting reference modulation at phase of contact
    m: int
        sidebands to be evaluated are m and m+1
    demod_params
        Parameter for fitting function (until now). Pass dictionary to use pshet_fitmodulation()
        to determine psi_R and gamma

    Returns
    -------
    coefficients: np.ndarray
        complex coefficients for tapping demodulation. tapping harmonics on axis=0, signals on axis=1
    """
    binned = pshet_binning(data=data, signals=signals, tap_nbins=tap_nbins, ref_nbins=ref_nbins)
    ft = phased_ft(array=binned, axis=-1, correction=tap_correction)
    ft = phased_ft(array=ft, axis=-2, correction=ref_correction)

    if type(demod_params) is dict:  # if psi_R and gamma should be determined through fitting
        if not all([p in demod_params for p in ['rho', 'gamma', 'psi_R', 'offset', 'theta_0']]):
            # initial guesses
            demod_params['rho'] = .45
            demod_params['gamma'] = 2.63
            demod_params['psi_R'] = 0
            demod_params['offset'] = 0
            demod_params['theta_0'] = 1.5
        binned = pshet_binning(data=data, signals=signals, tap_nbins=tap_nbins, ref_nbins=ref_nbins)
        demod_params['tap_offset'] = phase_offset(binned=binned, axis=-1)  # fit modulation at phase of contact
        pshet_fitmodulation(binned=binned, fit_params=demod_params)
        psi_R, gamma = demod_params['psi_R'], demod_params['gamma']

    coefficients = pshet_coefficients(ft=ft, gamma=gamma, psi_R=psi_R, m=m)

    return coefficients


# NAIVE DISCRETE FT DEMODULATION #######################################################################################
def dft_lstsq(phi, sig, max_order: int):
    assert phi.ndim == 1
    assert max_order > 1
    orders = np.arange(1, max_order+1)
    coeff = np.hstack([
        np.ones_like(phi)[:, np.newaxis],  # DC
        np.cos(orders[np.newaxis, :] * phi[:, np.newaxis]),  # cos
        np.sin(orders[np.newaxis, :] * phi[:, np.newaxis])  # sin
    ])
    assert coeff.shape == (phi.size, 2*max_order+1)
    soln, _, _, _ = np.linalg.lstsq(coeff, sig, rcond=None)
    ret = np.asarray(soln[:max_order+1], dtype=complex)  # what happens if sig.ndim > 1?
    ret[1:] += 1j*soln[max_order+1:]
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
        np.exp(1j*phi_s[np.newaxis, :]*orders[:, np.newaxis])[np.newaxis, :, :] * y_s[:, np.newaxis, :],
        phi_s,
        axis=-1
    )/np.pi
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
    step = 2*np.pi/n_bins
    df["bin_no"] = (df["tap_p"] - lo)//step
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
    step = 2*np.pi/n_bins
    lo = -np.pi
    avg["phi"] = avg.index * step + lo + step/2
    # avg = avg.drop(columns="bin_no")
    return avg


def binned_ft(avg):  # these names suck
    warn(_deprecation_warning)
    step = np.diff(avg["phi"].iloc[:2])[0]  # argh...
    sigs = avg.drop(columns="phi")
    return pd.DataFrame(np.fft.rfft(sigs, axis=0)*step/2/np.pi, columns=sigs.columns)


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
