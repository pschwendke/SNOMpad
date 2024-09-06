# demod functions to use in visualizer

import numpy as np

# if we run the GUI froma subdirectory, the imports must be absolute
from snompad.utility import Signals
from snompad.demodulation.utils import chop_pump_idx
from snompad.demodulation import shd, pshet


def demod_to_buffer(data, signals, modulation: str, tap: int, ref: int, chop: bool, ratiometry: bool, abs_val: bool, max_harm):
    rtn = 0
    coeff = np.zeros(max_harm + 1)
    try:
        if modulation == 'pshet':
            coeff = np.abs(pshet(data=data, signals=signals, tap_res=tap, ref_res=ref, binning='binning',
                                        chopped=chop, pshet_demod='sidebands',
                                        ratiometry=ratiometry))
        elif modulation == 'shd':
            coeff = shd(data=data, signals=signals, tap_res=tap, binning='binning',
                               chopped=chop, ratiometry=ratiometry)
        elif modulation == 'no mod':
            # coeff = np.zeros(max_harm+1)
            if chop:
                chopped_idx, pumped_idx = chop_pump_idx(data[:, signals.index(Signals.chop)])
                chopped = data[chopped_idx, signals.index(Signals.sig_a)].mean()
                pumped = data[pumped_idx, signals.index(Signals.sig_a)].mean()
                pump_probe = (pumped - chopped)  # / chopped
                coeff[0] = chopped
                coeff[1] = pumped
                coeff[2] = pump_probe
                rtn = '(0) probe only -- (1) pump only -- (2) pump-probe difference'
            else:
                coeff[0] = data[:, signals.index(Signals.sig_a)].mean()
                coeff[1] = data[:, signals.index(Signals.sig_b)].mean()
                rtn = '(0) sig_a -- (1) sig_b'
        if abs_val:
            coeff = np.abs(coeff[: max_harm+1])
        else:
            coeff = np.real(coeff[: max_harm+1])

    except Exception as e:
        if 'empty bins' in str(e):
            rtn = 2
        else:
            rtn = str(e)
    return rtn, coeff
