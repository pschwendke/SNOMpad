# ToDo: rename normalize -> ratiometry, check that defaults are read from metadata (Measurement classes)
import numpy as np
from scipy.stats import binned_statistic

from ..utility.signals import Signals
from .demod_utils import kernel_interpolation_1d, corrected_fft, sort_chopped


def shd_phases(data: np.ndarray, signals: list, normalize=False) -> np.ndarray:
    """ Mainly calculates tap_p. If normalize == True, sig_b is used for normalization.
    """
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

    ft = corrected_fft(array=binned, axis=0, correction=tap_correction)
    return ft
