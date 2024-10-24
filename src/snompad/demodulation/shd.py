# functions to demodulate SNOM data acquired in self-homodyne (shd) mode
import numpy as np
from scipy.stats import binned_statistic

from snompad.utility.signals import Signals
from snompad.demodulation.utils import kernel_average_1d, corrected_fft, chop_pump_idx, chopped_data, pumped_data
from snompad.demodulation.corrections import normalize_sig_a


def shd_binning(data: np.ndarray, signals: list, tap_res: int = 64) -> np.ndarray:
    """ Bins sig_a into 1D tapping phase domain. tap_y, tap_x must be included.

    PARAMETERS
    ----------
    data: np.ndarray
        data array with signals on axis=0 and samples on axis=1
    signals: list(Signals)
        signals acquired by DAQ. Must have same order as signals on axis=0 in data.
    tap_res: float
        number of tapping bins

    RETURNS
    -------
    binned: np.ndarray
        average signals for each bin between -pi, pi.
    """
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    returns = binned_statistic(x=tap_p, values=data[:, signals.index(Signals.sig_a)],
                               statistic='mean', bins=tap_res, range=[-np.pi, np.pi])
    binned = returns.statistic
    return binned


def shd_kernel_average(data: np.ndarray, signals: list, tap_res: int = 128) -> np.ndarray:
    """ Averages and interpolates signal onto even grid using kernel smoothening. tap_y, tap_x must be included.

    PARAMETERS
    ----------
    data: np.ndarray
        data array with signals on axis=0 and samples on axis=1
    signals: list(Signals)
        signals acquired by DAQ. Must have same order as signals on axis=0 in data.
    tap_res: float
        number of tapping bins

    RETURNS
    -------
    binned: np.ndarray
        averaged and interpolated signals for each bin between -pi, pi.
    """
    tap_p = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    tap_grid = np.linspace(-np.pi, np.pi, tap_res, endpoint=False) + np.pi / tap_res
    binned = kernel_average_1d(signal=data[:, signals.index(Signals.sig_a)], x_sig=tap_p, x_grid=tap_grid)
    return binned


def shd(data: np.ndarray, signals: list, tap_res: int = 64, chopped='auto', tap_correction='fft',
        binning='binning', ratiometry='auto') -> np.ndarray:
    """ Simple combination shd demodulation functions (binning and spectral analysis). Pump-probe demodulation,
    tapping phase correction, and ratiometric correction is also handled here.

    PARAMETERS
    ----------
    data: np.ndarray
        array of data with signals on axis=1 and data points on axis=0. Signals must be in the same order as in signals.
    signals: list of Signals
        list of Signals (e.g. Signals.sig_a, Signals.tap_x). Must have same order as along axis=1 in data.
    tap_res: float
        number of tapping bins
    tap_correction: str or None or float
        Type or value of phase correction along axis=-1, see phased_ft()
    chopped: 'auto', True, False
        determines whether chopping should be considered during binning, 'auto' -> True when Signals.chop in signals
    binning: 'binning', 'binned_kernel', 'kernel'
        determines routine to compute binned phase domain
    ratiometry: 'auto', True, False
        if True, sig_a is normalized to sig_a / sig_b. When 'auto', normalize is True when sig_b in signals

    RETURNS
    -------
    coefficients: np.ndarray
        real coefficients for tapping demodulation.
    """
    # ToDo: It's annoying to have to pass Signals.xxx_x all the time. Find a way to also pass lists or arrays of strings
    if ratiometry is True or (ratiometry == 'auto' and Signals.sig_b in signals):
        data = normalize_sig_a(data=data, signals=signals)

    if chopped is True or (chopped == 'auto' and Signals.chop in signals):
        chop_idx, pump_idx = chop_pump_idx(data[:, signals.index(Signals.chop)])
        data_chop = chopped_data(data=data, idx=chop_idx)
        data_pump = pumped_data(data=data, idx=pump_idx)
        data = [data_chop, data_pump]
    else:
        data = [data]

    binning_func = [shd_binning, shd_kernel_average][['binning', 'kernel'].index(binning)]
    binned = [binning_func(data=d, signals=signals, tap_res=tap_res) for d in data]

    ft = [corrected_fft(array=b, axis=0, phase_correction=tap_correction) for b in binned]

    if len(ft) == 2:
        ft = ft[1] / ft[0] - 1
    else:
        ft = ft[0]

    return ft
