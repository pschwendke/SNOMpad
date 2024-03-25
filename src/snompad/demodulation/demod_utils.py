# tools and helper functions
import numpy as np
from .demod_corrections import phase_offset


# KERNEL INTERPOLATION #################################################################################################
def gaussian_kernel_1d(x, x0, sigma):
    """ Returns array with not normalized weights for x-coordinates,
    determined by kernel function centered at x0
    """
    dx = np.array([(x - x0) ** 2, (x - 2 * np.pi - x0) ** 2, (x + 2 * np.pi - x0) ** 2]).min(axis=0, initial=2*np.pi)
    gk = np.exp(-.5 * dx / sigma ** 2)
    return gk


def kernel_interpolation_1d(signal, x_sig, x_grid, sigma=None):
    """ Interpolates signal on coordinates x_sig onto passed coordinates x_grid,
    using gaussian kernel smoothing of width sigma.
    """
    if sigma is None:
        step = 2 * np.pi / len(x_grid)
        sigma = 1.1 * step

    out = np.zeros(x_grid.shape)
    for n, x_n in enumerate(x_grid):
        window = gaussian_kernel_1d(x_sig, x_n, sigma)
        out[n] = np.average(a=signal, weights=window)
    return out


def gaussian_kernel_2d(x, y, x0, y0, sigma):
    """ Returns array with not normalized weights for coordinates,
    determined by kernel function centered at x0, y0
    """
    dxdy = np.array([[(x - x0) ** 2, (y - y0) ** 2],
                     [(x - 2 * np.pi - x0) ** 2, (y - 2 * np.pi - y0) ** 2],
                     [(x + 2 * np.pi - x0) ** 2, (y + 2 * np.pi - y0) ** 2]]).min(axis=0, initial=2*np.pi)
    r = np.sqrt(dxdy.sum(axis=0))
    gk = np.exp(-.5 * r ** 2 / sigma ** 2)
    return gk


def kernel_interpolation_2d(signal, x_sig, y_sig, x_grid, y_grid, sigma=None):
    """ Interpolates signal on coordinates x_sig, y_sig onto passed grid x_grid, y_grid,
    using gaussian kernel smoothing of width sigma.
    """
    if sigma is None:
        step = 2 * np.pi / np.max([len(x_grid), len(y_grid)])
        sigma = .5 * step

    out = np.zeros((len(y_grid), len(x_grid)))
    for n, y_n in enumerate(y_grid):
        for m, x_m in enumerate(x_grid):
            gk = gaussian_kernel_2d(x_sig, y_sig, x_m, y_n, sigma)
            out[n, m] = np.average(a=signal, weights=gk)
    return out


# PSHET FITTING FUNCTIONS ##############################################################################################
def pshet_modulation(theta_ref, theta_0, gamma, sigma):
    """ Returns signal intensity after pshet modulation.
    """
    signal = sigma * np.exp(1j * gamma * np.sin(theta_ref - theta_0))
    return np.real(signal + np.conj(signal))


def pshet_obj_func(params, theta_ref, sig):
    """ Returns array of residuals, normalized to signal amplitude.
    """
    theta_0 = params['theta_0'].value
    gamma = params['gamma'].value
    sigma = params['sigma_re'].value + 1j * params['sigma_im'].value

    model = pshet_modulation(theta_ref=theta_ref, theta_0=theta_0, gamma=gamma, sigma=sigma)
    model += params['offset'].value

    amplitude = sig.max() - sig.min()
    residuals = (model - sig) / amplitude
    return residuals


# FOURIER TRANSFORMS ###################################################################################################
def corrected_fft(array: np.ndarray, axis: int = -1, correction=None) -> np.ndarray:
    """ Computes real fft of array along given axis. correction determines the type of phase correction.

    Parameters
    ----------
    array: np.ndarray
    axis: int
    correction: str or None or float
        If correction is None, the coefficients are rotated by 0°.
        If correction == 'fft', phase_offset() is used to determine the phase correction.
            A binned phase domain is assumed, binning offset of pi and half a bin are corrected
        If correction is float, the value is used for phase correction.

    Returns
    -------
    np.real(ft)

    """
    if np.any(np.isnan(array)):
        raise ValueError('The array has empty bins.')

    ft = np.fft.rfft(array, axis=axis, norm='forward')

    if correction == 'fft':
        phase = - phase_offset(array, axis=axis)
    elif type(correction) in [float, int]:
        phase = - correction  # You might want to flip the sign. Check this when using.
    elif correction is None:
        return np.real(ft)
    else:
        raise NotImplementedError

    if array.ndim == 1:  # shd
        phase *= np.arange(ft.shape[axis])
    elif array.ndim == 2 and axis == 1:  # pshet tap axis
        phase = np.arange(ft.shape[axis])[np.newaxis, :] * phase
    elif array.ndim == 2 and axis == 0:  # pshet ref axis
        phase = np.arange(ft.shape[axis])[:, np.newaxis] * phase

    ft *= np.exp(1j * phase)
    return np.real(ft)


# CHOPPING AND PUMP-PROBE ##############################################################################################
def sort_chopped(chop: np.ndarray) -> tuple:
    """ Takes optical chop signal, returns boolean indices of chopped and pumped shots
    """
    hi_sig = chop[chop > np.median(chop)]
    lo_sig = chop[chop < np.median(chop)]

    pumped = np.abs(chop - hi_sig.mean()) < (3 * hi_sig.std())
    chopped = np.abs(chop - lo_sig.mean()) < (3 * lo_sig.std())

    return chopped, pumped
