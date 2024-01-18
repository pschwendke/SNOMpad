# functions correct and polish datasets

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize


def linear_offset(values):
    """ Fits linear offset (linear function) to set of evenly spaced values, and returns offset values.
    """
    # Todo: This is probably more efficient with linear regression
    def obj_func(params, x, data):
        m = params['m']
        b = params['b']
        model = (m * x + b)
        residuals = data - model
        return residuals

    params = Parameters()
    params.add('m', value=1)
    params.add('b', value=1)

    x = np.linspace(0, 1, len(values))
    fit = minimize(obj_func, params, args=(x, values))
    offset = fit.params['m'] * x + fit.params['b']
    return offset


def planar_offset(data: xr.DataArray, x_min=None, x_max=None, y_min=None, y_max=None) -> xr.DataArray:
    """ Fits planar offset xr.DataArray with coordinates 'x' and 'y', and returns offset values with same shape.
    """
    # ToDo: can we also do this with linear regression?
    if x_min is None:
        x_min = data.x.values.min()
    if x_max is None:
        x_max = data.x.values.max()
    if y_min is None:
        y_min = data.y.values.min()
    if y_max is None:
        y_max = data.y.values.max()

    def obj_func(params, x, y, data):
        a = params['a']
        b = params['b']
        c = params['c']
        x, y = np.meshgrid(x, y)
        model = a * x + b * y + c
        residuals = data - model
        return residuals

    data_mask = data[np.logical_and(y_min < data.y, data.y < y_max), np.logical_and(x_min < data.x, data.x < x_max)]
    x_mask = data.x[np.logical_and(x_min < data.x, data.x < x_max)].values
    y_mask = data.y[np.logical_and(y_min < data.y, data.y < y_max)].values

    params = Parameters()
    params.add('a', value=1)
    params.add('b', value=1)
    params.add('c', value=1)
    fit = minimize(obj_func, params, args=(x_mask, y_mask, data_mask))
    a = fit.params['a']
    b = fit.params['b']
    c = fit.params['c']

    x, y = np.meshgrid(data.x.values, data.y.values)
    offset = a * x + b * y + c
    return offset


def subtract_linear(line: np.ndarray) -> np.ndarray:
    """ removes linear offset of values on 'line'.
    """
    correction = linear_offset(line)
    leveled = line - correction
    return leveled


def normalize_linear(line: np.ndarray) -> np.ndarray:
    """ fits linear offset and computes factor to set linear offset of values on line to 1.
    """
    offset = linear_offset(line)
    leveled = line / offset
    return leveled


def align_rows_subtract(data):
    """ removes linear offset from every row of array 'data'
    """
    output = data.copy()
    for n, l in enumerate(data):
        leveled = subtract_linear(line=l)
        output[n] = leveled
    return output


def align_rows_normalize(data):
    """ normalizes linear offset from every row of array 'data', i.e. sets linear offset to 1.
    """
    output = data.copy()
    for n, l in enumerate(data):
        leveled = normalize_linear(line=l)
        output[n] = leveled
    return output


def subtract_planar(data: xr.DataArray, x_min=None, x_max=None, y_min=None, y_max=None,
                    show_mask=False) -> xr.DataArray:
    """ returns DataArray with planar offset removed. Offset is evaluated in between x_min, x_max and y_min, y_max.
    """
    offset = planar_offset(data=data, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    output = data - offset

    if show_mask:
        np.abs(data).plot()
        plt.fill_between(np.linspace(x_min, x_max, 2), np.full(2, y_max), np.full(2, y_min), alpha=.4, color='red')
        plt.title('input data with selected mask in red')
        plt.show()

    return output


def normalize_planar(data: xr.DataArray, x_min=None, x_max=None, y_min=None, y_max=None,
                     show_mask=False) -> xr.DataArray:
    """ returns DataArray with planar offset removed. Offset is evaluated in between x_min, x_max and y_min, y_max.
    """
    offset = planar_offset(data=data, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    output = data / offset

    if show_mask:
        np.abs(data).plot()
        plt.fill_between(np.linspace(x_min, x_max, 2), np.full(2, y_max), np.full(2, y_min), alpha=.4, color='red')
        plt.title('input data with selected mask in red')
        plt.show()

    return output


def level_complex(data, x_min=None, x_max=None, y_min=None, y_max=None, show_mask=False):
    """ Combination of correction functions for complex valued (optical) image data. The procedure is:
    1. normalize amplitude row-wise to linear offset
    2. normalize amplitude to planar offset inside selected window (offset values become 1)
    3. subtract linear offset row wise in phase data
    4. subtract planar offset in phase data
    5. set median of phase data to 0
    6. construct and return complex valued data
    """
    amp = align_rows_normalize(np.abs(data))
    amp = normalize_planar(amp, x_min, x_max, y_min, y_max, show_mask)
    # amp -= amp.min()
    
    phase = data.copy()
    phase.values = np.angle(phase)
    phase = align_rows_subtract(phase)
    phase = subtract_planar(phase, x_min, x_max, y_min, y_max)
    phase -= phase.median()
    
    ret = amp * np.exp(1j * phase)
    return ret
