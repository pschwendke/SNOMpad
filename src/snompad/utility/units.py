from scipy.constants import physical_constants
import numpy as np

h = physical_constants['Planck constant in eV/Hz'][0]
c = physical_constants['speed of light in vacuum'][0]


def ev(wl):
    """ Converts values from wavelength (nm) into energy (eV)
    """
    e = h * c / (wl * 1e-9)
    return e  # eV


def nm(e):
    """ Converts values from energy (eV) into wavelength (nm)
    """
    wl = h * c / e
    return wl * 1e9  # nm


def epsilon(n, kappa=0):
    """ Converts refractive index n and extinction coefficient kappe into complex dielectric function
    """
    r = n ** 2 + kappa ** 2
    i = 2 * n * kappa
    return r + 1j * i


def refractive(epsilon):
    """ Calculates refractive index n from complex dielectric function epsilon
    """
    e1 = np.real(epsilon)
    e2 = np.imag(epsilon)
    return np.sqrt((e1 + np.sqrt(e1**2 + e2**2))/2)


def extinction(epsilon):
    """ Calculates extinction coefficient kappa from complex dielectric function epsilon
    """
    e1 = np.real(epsilon)
    e2 = np.imag(epsilon)
    return np.sqrt((-e1 + np.sqrt(e1**2 + e2**2))/2)
