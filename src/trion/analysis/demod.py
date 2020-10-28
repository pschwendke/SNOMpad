# demod.py: functions for demodulations

import numpy as np
import pandas as pd

from .signals import is_tap_modulation


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
    # TODO: check how to handle multiple y values...
    assert phi.ndim == 1
    idx = np.argsort(phi)
    y_s = y[idx]
    phi_s = phi[idx]
    return np.trapz(
        np.exp(1j*phi_s[np.newaxis, :]*orders[:, np.newaxis])*y_s[np.newaxis,:],
        phi_s,
        axis=1
    )/2/np.pi


def calc_phase(df, in_place=False):
    "Computes tap_p"
    if not in_place:
        df = df.copy()
    df["tap_p"] = np.arctan2(df["tap_y"], df["tap_x"])
    return df


def bin_no(df, n_bins, in_place=False):
    "Computes bin_no"
    if not in_place:
        df = df.copy()
    lo = -np.pi
    step = 2*np.pi/n_bins
    df["bin_no"] = (df["tap_p"] - lo)//step
    return df


def binned_average(df, n_bins, compute_counts=True):
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


def binned_ft(avg):
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
    # TODO: expand to multiple signals
    df = pd.DataFrame(np.vstack([phi, sig]).T, columns=["tap_p", "sig"])
    return np.squeeze(binned_ft(binned_average(df, n_bins, compute_counts=False)).to_numpy())


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


