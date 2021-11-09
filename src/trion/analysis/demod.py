# demod.py: functions for demodulations
from itertools import product
from warnings import warn
import numpy as np
from scipy.special import jv
import pandas as pd

from .signals import is_tap_modulation


def empty_bins_in(df: pd.DataFrame) -> bool:
    """Checks binned 1D or 2D DataFrame for empty bins in index (1D) and column names (2D).
    Returns True if empty bins are detected."""

    # check for missing tap bins
    for n in range(df.shape[0]):
        if df.index[n] != n:
            return True

    # check for missing pshet bins
    if isinstance(df.columns, pd.MultiIndex):
        for channel in df.columns.get_level_values(0).drop_duplicates():
            for m in range(df[channel].shape[1]):
                if df[channel].columns[m] != m:
                    return True

    # check for empty (NAN) bins
    if np.isnan(df).any(axis=None):
        return True
    return False


def bin_index(phi, n_bins: int):
    """
    Compute the phase bin index.
    """
    lo = -np.pi
    step = 2*np.pi/n_bins
    return (phi - lo)//step


def bin_midpoints(n_bins, lo=-np.pi, hi=np.pi):
    """Compute the midpoints of phase bins"""
    span = hi - lo
    step = span/n_bins
    return (np.arange(n_bins)+0.5) * step + lo


_avg_drop_cols = set(  # tap_x, tap_y, tap_p...
    map("_".join,
        product(["tap", "ref"], ["x", "y", "p"])
        )
)


def shd_binning(df: pd.DataFrame, tap_nbins: int = 32) -> pd.DataFrame:
    """Perform sHD binned average on `df`.

    Compute the tapping phase `tap_p` from `tap_y` and `tap_x`, then partition
    the data points in an histogram based on `tap_p`, and compute the average
    per bin.

    The function performs some operations in-place, therefore affecting the
    contents of `df`. To avoid this, pass a copy, ex: `shd_binning(df.copy())`

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of sample points. The columns indicate the signal type.
    tap_nbins : int
        Number of bins to use. Default: 32

    Returns
    -------
    avg : pd.DataFrame
        Dataframe containing the average per bins. Row index contains the bin
        number for tapping. Columns indicate the signal type.
    """
    # compute phases
    df["tap_p"] = np.arctan2(df["tap_y"], df["tap_x"])
    # compute indices
    df["tap_n"] = bin_index(df["tap_p"], tap_nbins)
    # compute histogram
    avg = df.drop(columns=_avg_drop_cols & set(df.columns)
                  ).groupby(["tap_n"]).mean()
    # fill missing bins with nans
    if len(avg) != tap_nbins:
        for i in range(tap_nbins):
            if i not in avg.index:
                avg.loc[i] = np.full(avg.shape[1], np.nan)
        avg.sort_index(inplace=True)
    return avg


def shd_ft(avg: pd.DataFrame) -> pd.DataFrame:
    """Perform Fourier analysis for shd demodulation.

    Parameters
    ----------
    avg : pd.DataFrame
        Data points averaged per bins. Rows indicate bin, columns indicate
        signal type. Note that every bin should be present, this is not checked.

    Returns
    -------
    shd : pd.DataFrame
        Fourier components. Rows indicate tapping order `n`, columns indicate
        signal type.
    """
    # normalization factor: step/2/np.pi, with step = 2*np.pi/len(avg)
    return avg.apply(np.fft.rfft, axis=0)/len(avg)


def shd(df: pd.DataFrame, tap_nbins: int = 32) -> pd.DataFrame:
    """Perform shd demodulation by binning and FT.
    This is a function simply calls `shd_binnning` and `shd_ft` in sequence.
    """
    avg = shd_binning(df, tap_nbins)
    if empty_bins_in(avg):
        raise ValueError("The binned DataFrame has empty bins.")
    return shd_ft(avg)


def pshet_binning(df: pd.DataFrame, tap_nbins: int = 32, ref_nbins: int = 16) -> pd.DataFrame:
    """Perform pshet binned average on `df`.

    Bin the data points in a 2D phase space (ref and tap phase), and compute the average per
    bin. The function performs some operations in-place, therefore affecting the
    contents of `df`. To avoid this, pass a copy, ex: `pshet_binning(df.copy())`

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of sample points. The columns indicate the signal type.
    tap_nbins : int
        Number of bins for tapping modulation. Default: 32
    ref_nbins : int
        Number of bins for reference arm phase modulation. Default: 16

    Returns
    -------
    avg : pd.DataFrame
        Dataframe containing the average per bins. Row index contains the bin
        number for tapping. The column index is a `MultiIndex` where level 0 is
        the signal name as a string (ex: `"sig_a"`), and level 1 is the
        reference bin number. Therefore, the histogram for `sig_a` can be
        accessed via `avg["sig_a"]`.
    """
    # compute phases
    df["tap_p"] = np.arctan2(df["tap_y"], df["tap_x"])
    df["ref_p"] = np.arctan2(df["ref_y"], df["ref_x"])
    # compute indices
    df["tap_n"] = bin_index(df["tap_p"], tap_nbins)
    df["ref_n"] = bin_index(df["ref_p"], ref_nbins)
    # compute histogram
    avg = df.drop(columns=_avg_drop_cols & set(df.columns)
                  ).groupby(["tap_n", "ref_n"]).mean()
    return avg.unstack()


def pshet_ft(avg: pd.DataFrame) -> pd.DataFrame:
    """Fourier transform an averaged pshet dataframe.
    Returns pd.DataFrame with MultiIndex: sig and n. Rows are m.
    """
    # TODO: check if we can use a form of `pd.Dataframe.apply`
    # normalization is the same as 1D, but divided by both lengths
    keys = avg.columns.get_level_values(0).drop_duplicates()
    demod = [np.fft.rfft2(avg[k].to_numpy()) / avg[k].shape[0] / avg[k].shape[1]
             for k in keys
             ]
    r, c = demod[0].shape
    col_idx = pd.MultiIndex.from_product(
        (keys, range(c))
    )
    ret = pd.DataFrame(
        data=np.hstack(demod), columns=col_idx, index=range(r),
    )
    return ret


def pshet(df: pd.DataFrame, tap_nbins: int = 32, ref_nbins: int = 16) -> pd.DataFrame:
    """Perform pshet demodulation by binning and FT.
    This is a function simply calls `pshet_binnning` and `pshet_ft` in sequence.
    """
    avg = pshet_binning(df, tap_nbins, ref_nbins)
    if empty_bins_in(avg):
        raise ValueError("The binned DataFrame has empty bins.")
    return pshet_ft(avg)


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


def pshet_coefficients(df: pd.DataFrame, gamma: float = 2.63) -> pd.DataFrame:
    """ Qualitatively computes the coefficients of tapping order harmonics at the sidebands.
    Sidebands at m=1 and m=2 are evaluated.

    Parameters
    ----------
    df: pd.Dataframe
        demodulated pshet signal as returned by pshet_ft
    gamma: float
        modulation depth of the reference mirror.

    Returns
    -------
    coeffs : pd.Dataframe
        Near field amplitudes (complex) for tapping harmonics n.
        Index is signal, row is n.
    """
    m = np.array([1, 2])    # evaluating sidebands at m=1 and m=2
    scales = 1/jv(m, gamma) * np.array([1, 1j])

    nsigs = len(df.columns.get_level_values(0).drop_duplicates())
    coeffs = (df.loc[:, (slice(None), m)].abs() * np.tile(scales, nsigs)
              ).groupby(level=0, axis=1).sum()

    return coeffs


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
