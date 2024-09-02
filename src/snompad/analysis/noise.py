# scripts for noise analysis
import numpy as np
from lmfit import Parameters, minimize

from .utils import tip_frequency


def phase_fit_obj_func(params, t, data):
    model = params['amp'] * np.cos(params['f'] * 2 * np.pi * t + params['phi']) + params['offset']
    residuals = model - data
    return residuals


def phase_shifting(x, y):
    """ Evaluates the tuning of modulation phases tap_x,y or ref_x,y. A sample rate of 200_000 Hz is assumed.
    PARAMETERS
    ----------
    x: np.ndarray
        tap_x or ref_x
    y: np.ndarray
        tap_y or ref_y

    RETURNS
    -------
    tuple:
        df: difference in frequency of cos fits, should be very small
        dphi: phase difference, should be 90 °
        damp: difference in amplitude, should be small
        amp_avg: mean of modulation amplitude (radius of circle in x-y plot)
        amp_std: standard deviation of modulation amplitude (spread around circle in x-y plot)
        offsets: x, y offset potentials at DAQ
    """
    p = np.arctan2(y, x)
    t = np.arange(len(x)) * 5e-6
    f_tip = tip_frequency(tap_p=p)

    # phase_x
    params = Parameters()
    params.add('f', value=f_tip, min=200_000, max=400_000)
    params.add('phi', value=0, min=0, max=2*np.pi)
    params.add('amp', value=x.max(), min=0, max=1)
    params.add('offset', value=0, min=-1, max=1)
    fit = minimize(phase_fit_obj_func, params, args=(t, x), method='leastsqr')
    x_f = fit.params['f']
    x_phi = fit.params['phi']
    x_amp = fit.params['amp']
    x_offset = fit.params['offset']

    # phase_y
    params = Parameters()
    params.add('f', value=f_tip, min=200_000, max=400_000)
    params.add('phi', value=0, min=0, max=2 * np.pi)
    params.add('amp', value=y.max(), min=0, max=1)
    params.add('offset', value=0, min=-1, max=1)
    fit = minimize(phase_fit_obj_func, params, args=(t, y), method='leastsqr')
    y_f = fit.params['f']
    y_phi = fit.params['phi']
    y_amp = fit.params['amp']
    y_offset = fit.params['offset']

    # spreading
    r = np.sqrt(x**2 + y**2)
    std = r.std()
    avg = r.mean()

    # return variables
    df = x_f - y_f
    dphi = x_phi - y_phi
    damp = x_amp - y_amp
    amp_avg = avg
    amp_std = std
    offsets = (x_offset.value, y_offset.value)

    return df, dphi, damp, amp_avg, amp_std, offsets
