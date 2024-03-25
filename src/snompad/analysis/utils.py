# tools and helper functions
import numpy as np


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
