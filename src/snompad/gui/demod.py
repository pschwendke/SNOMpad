# demod functions to use in visualizer
# ToDo: do we need this? Just put it in that file.

import numpy as np

from ..utility import Signals
from ..demodulation.utils import chop_pump_idx
from ..demodulation import shd, pshet

from . import max_harm, signals


def demod_to_buffer(data, modulation: str, tap: int, ref: int, chop: bool, ratiometry: bool, abs_val: bool):
    rtn_value = 0
    try:
        if modulation == 'pshet':
            coefficients = np.abs(pshet(data=data, signals=signals, tap_res=tap, ref_res=ref, binning='binning',
                                        chopped=chop, pshet_demod='sidebands',
                                        ratiometry=ratiometry))
        elif modulation == 'shd':
            coefficients = shd(data=data, signals=signals, tap_res=tap, binning='binning',
                               chopped=chop, ratiometry=ratiometry)
        elif modulation == 'no mod':
            coefficients = np.zeros(max_harm+1)
            if chop:
                chopped_idx, pumped_idx = chop_pump_idx(data[:, signals.index(Signals.chop)])
                chopped = data[chopped_idx, signals.index(Signals.sig_a)].mean()
                pumped = data[pumped_idx, signals.index(Signals.sig_a)].mean()
                pump_probe = (pumped - chopped)  # / chopped
                coefficients[0] = chopped
                coefficients[1] = pumped
                coefficients[2] = pump_probe
                rtn_value = '(0) probe only -- (1) pump only -- (2) pump-probe'
            else:
                coefficients[0] = data[:, signals.index(Signals.sig_a)].mean()
                coefficients[1] = data[:, signals.index(Signals.sig_b)].mean()
                rtn_value = '(0) sig_a -- (1) sig_b'
        if abs_val:
            coefficients = np.abs(coefficients[: max_harm+1])
        else:
            coefficients = np.real(coefficients[: max_harm+1])

    except Exception as e:
        if 'empty bins' in str(e):
            rtn_value = 2
        else:
            rtn_value = str(e)
    return rtn_value, coefficients
