# ToDo: standardize names of variables
#  note standards in README (with equations, maybe pdf example)

# demod.py: functions for demodulations
from warnings import warn
import numpy as np
from scipy.special import jv
from scipy.stats import binned_statistic, binned_statistic_2d
import pandas as pd

from .signals import Signals, all_detector_signals


def shd_binning(data: np.ndarray, signals: list, tap_nbins: int = 64) -> np.ndarray:
    """ Bins signals into 1D tap_p phase domain. tap_y, tap_x must be included.

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
    returns = binned_statistic(x=tap_p, values=detector_signals,
                               statistic='mean', bins=tap_nbins, range=[-np.pi, np.pi])
    binned = returns.statistic
    return binned


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
    if np.any(np.isnan(binned)):
        raise ValueError("The binned array has empty bins.")
    ft = np.fft.rfft(binned) / binned.shape[-1]
    return ft.T


def shd(data: np.ndarray, signals: list, tap_nbins: int = 64) -> np.ndarray:
    """ Simple combination of shd_binning and shd_ft
    """
    binned = shd_binning(data, signals, tap_nbins)
    return shd_ft(binned)


def pshet_binning(data: np.ndarray, signals: list, tap_nbins: int = 64, ref_nbins: int = 64) -> np.ndarray:
    """ Performs 2D binning on signals onto tap_p, ref_p domain. tap_y, tap_x, ref_x, ref_y must be included.

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
    returns = binned_statistic_2d(x=tap_p, y=ref_p, values=detector_signals, statistic='mean',
                                  bins=[tap_nbins, ref_nbins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    binned = returns.statistic
    if binned.ndim == 2:
        return binned.T[np.newaxis, :, :]
    elif binned.ndim == 3:
        return binned.transpose(0, 2, 1)


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
    if np.any(np.isnan(binned)):
        raise ValueError("The binned array has empty bins.")
    ft = np.fft.rfft2(binned) / binned.shape[-1] / binned.shape[-2]
    return ft


def pshet_coeff(ft: np.ndarray, gamma: float = 2.63) -> np.ndarray:
    """ Computes coefficients for tapping demodulation. Positive and negative frequencies (for non phase corrected
    data) are not summed.

    PARAMETERS
    ----------
    ft: np.ndarray
        array containig Fourier amplitudes. signals on axis=0, ref on axis=1, tap on axis=2
    gamma: float
        modulation depth
    RETURNS
    -------
    coefficients: np.ndarray
        complex coefficients for tapping demodulation. amplitudes on axis=0, signals on axis=1
    """
    m = np.array([1, 2])
    scales = 1 / jv(m, gamma) * np.array([1, 1j])

    coefficients = (np.abs(ft[:, m, :]) * scales[np.newaxis, :, np.newaxis]).sum(axis=1)
    # neg_coefficients = (np.abs(ft[:, -m, :]) * scales[np.newaxis, :, np.newaxis]).sum(axis=1)
    return coefficients.T


def pshet(data: np.ndarray, signals: list, tap_nbins: int = 64, ref_nbins: int = 64, gamma: float = 2.63) -> np.ndarray:
    """ Simple combination of pshet_binning, pshet_ft, and pshet_coeff
    """
    ft = pshet_ft(pshet_binning(data, signals, tap_nbins, ref_nbins))
    return pshet_coeff(ft, gamma)


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


# older stuff, kept for compatibility ##################################################################################
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
    if not "tap_p" in df.columns:
        calc_phase(df, in_place=True)
    if not "bin_no" in df.columns:
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
