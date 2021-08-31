# demod.py: functions for demodulations
from itertools import product
from warnings import warn
import numpy as np
import pandas as pd

from .signals import is_tap_modulation



def bin_index(phi, n_bins: int):
    """
    Compute the phase bin index.
    """
    lo = -np.pi
    step = 2*np.pi/n_bins
    return (phi - lo)//step


def bin_midpoints(n_bins, lo=-np.pi, hi = np.pi):
    """Compute the midpoints of phase bins"""
    span = hi - lo
    step = span/n_bins
    return (np.arange(n_bins)+0.5) * step + lo


_avg_drop_cols = set(  # tap_x, tap_y, tap_p...
    map("_".join,
        product(["tap", "ref"], ["x", "y", "p"])
        )
)


def shd_binning(df: pd.DataFrame, tap_nbins: int = 32):
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
    # TODO test
    #  smoke test passed
    # compute phases
    df["tap_p"] = np.arctan2(df["tap_y"], df["tap_x"])
    # compute indices
    df["tap_n"] = bin_index(df["tap_p"], tap_nbins)
    # compute histogram
    avg = df.drop(columns=_avg_drop_cols & set(df.columns)
                  ).groupby(["tap_n"]).mean()
    return avg


def shd_ft(avg: pd.DataFrame):
    """Perform Fourier analysis for sHD demodulation.

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
    # todo: add to test suite before modifying
    #  smoke test passed
    # normalization factor: step/2/np.pi, with step = 2*np.pi/len(avg)
    return avg.apply(np.fft.rfft, axis=0)/len(avg)


def shd(df: pd.DataFrame, tap_nbins: int = 32):
    """
    Perform sHD demodulation by binning and FT.

    This is a function simply calls `shd_binnning` and `shd_ft` in sequence.
    """
    # todo: add to test suite before modifying
    #  smoke test passed
    return shd_ft(shd_binning(df, tap_nbins))


def pshet_binning(df: pd.DataFrame, tap_nbins: int = 32, ref_nbins: int = 16):
    """Perform pshet binned average on `df`.

    Partition the data points in a 2d histogram, and compute the average per
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
        number for tapping. The column index is a `Multindex` where level 0 is
        the signal name as a string (ex: `"sig_a"`), and level 1 is the
        reference bin number. Therefore, the histogram for `sig_a` can be
        accessed via `avg["sig_a"]`.
    """
    # TODO test
    #  smoke test passed
    #  should be added to test suite before modification...
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


def pshet_ft(avg: pd.DataFrame):
    """Fourier transform an averaged pshet dataframe."""
    # TODO: check if we can use a form of `pd.Dataframe.apply`
    # TODO: test
    return {k: np.fft.rfft2(avg[k].to_numpy())  # scale is missing...
            for k in avg.columns.get_level_values(0).drop_duplicates()
            }


def pshet(df: pd.DataFrame, tap_nbins: int = 32, ref_nbins: int = 16):
    """
    Perform pshet demodulation by binning and FT.
    """
    # todo: test
    avg = pshet_binning(df, tap_nbins, ref_nbins)
    return pshet_ft(avg)


def dft_lstsq(phi, sig, max_order: int):
    assert phi.ndim == 1
    assert max_order > 1
    orders = np.arange(1, max_order+1)
    coeff = np.hstack([
        np.ones_like(phi)[:,np.newaxis], # DC
        np.cos(orders[np.newaxis,:] * phi[:,np.newaxis]), # cos
        np.sin(orders[np.newaxis,:] * phi[:,np.newaxis])  # sin
    ])
    assert coeff.shape == (phi.size, 2*max_order+1)
    soln, _, _, _ = np.linalg.lstsq(coeff, sig, rcond=None)
    ret = np.asarray(soln[:max_order+1], dtype=complex) # what happens if sig.ndim > 1?
    ret[1:] += 1j*soln[max_order+1:]
    return ret


def dft_naive(phi, y, orders):
    """
    Perform a naive fourier transform using trapezoidal integration.

    The input arrays (phi) are sorted before integration.

    Parameters
    ----------
    phi : (N,) np.ndarray
        Values of phi.
    y : (N,) np.ndarray
        Values of y.
    orders : (M,) np.ndarray
        Demodulation orders

    Returns
    -------
    amp : (M,) np.ndarray
        Complex amplitude for the given orders.
    """
    assert phi.ndim == 1
    if y.ndim == 1:
        y = y[np.newaxis,:]
    # sort by phi
    idx = np.argsort(phi).tolist()
    idx = idx + [idx[0]]  # append last element again
    # we need to repeat the first elements at the end, so we integrate over the
    # whole circle, without missing the seam.
    y_s = y.take(idx, axis=1)
    phi_s = phi.take(idx)
    phi_s[-1] += 2 * np.pi
    intgr = np.trapz(
        np.exp(1j*phi_s[np.newaxis, :]*orders[:, np.newaxis])*y_s,
        phi_s,
        axis=1
    )/np.pi
    intgr[orders==0] *= 0.5
    return intgr


#####  older stuff, kept for compatibility

_deprecation_warning = FutureWarning("This function is deprecated. Please use the `shd` and `pshet` set of functions.")

def calc_phase(df, in_place=False):
    warn(_deprecation_warning)
    "Computes tap_p"
    # todo: deprecate
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
        avg["count"] = grpd.count().iloc[:,0]
    step = 2*np.pi/n_bins
    lo = -np.pi
    avg["phi"] = avg.index * step + lo + step/2
    #avg = avg.drop(columns="bin_no")
    return avg


def binned_ft(avg):  # these names suck
    warn(_deprecation_warning)
    step = np.diff(avg["phi"].iloc[:2])[0] # argh...
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
    # TODO: expand to multiple signals
    df = pd.DataFrame(np.vstack([phi, sig]).T, columns=["tap_p", "sig"])
    return np.squeeze(binned_ft(binned_average(df, n_bins, compute_counts=False)).to_numpy())