# SNOM near-field models
from scipy.interpolate import interp1d
import numpy as np
from .units import nm


def eps_air(e):
    """ from Ciddor1996, https://doi.org/10.1364/AO.35.001566

    PARAMETERS
    ----------
    e
        energy of light in eV
    """
    wl = nm(e) * 1e-3  # um
    n = 1 + 0.05792105 / (238.0185 - wl**-2) + 0.00167917 / (57.362 - wl**-2)
    eps = n**2 + 0 * 1j
    return eps


def eps_pt(e):
    """ Dielectric function of Pt determined via REELS in Werner2009 https://doi.org/10.1063/1.3243762
    Determined between 0.5 eV and 4 eV via cubic spline interpolation (continuous 2nd derivative)

    PARAMETERS
    ----------
    e
        energy of light in eV
    """
    e_data = np.array([.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4.])
    eps_data = np.array([-635.547 + 1j * 216.023,
                            -294.704 + 1j * 68.276,
                            -165.622 + 1j * 29.962,
                            -103.964 + 1j * 16.097,
                            -69.907 + 1j * 10.023,
                            -49.132 + 1j * 7.046,
                            -35.496 + 1j * 5.511,
                            -26.009 + 1j * 4.755,
                            -19.067 + 1j * 4.506,
                            -13.748 + 1j * 4.718,
                            -9.570 + 1j * 5.574,
                            -6.797 + 1j * 7.492,
                            -7.017 + 1j * 9.412,
                            -7.938 + 1j * 8.097,
                            -6.730 + 1j * 6.530])
    eps_real = interp1d(x=e_data, y=np.real(eps_data), kind='cubic', bounds_error=True)
    eps_imag = interp1d(x=e_data, y=np.imag(eps_data), kind='cubic', bounds_error=True)
    eps = eps_real(e) + 1j * eps_imag(e)
    return eps


def beta(eps_s, eps_m=1):
    """ Reflection coefficient beta

    PARAMETERS
    ----------
    eps_s
        dielectric function of sample
    eps_m
        dielectric function of medium (between tip and sample). This is mostly taken to be 1.
    """
    return (eps_s - eps_m) / (eps_s + eps_m)


def alpha(a, epsilon, epsilon_m):
    """ polarizability of a conducting sphere, from Bohren1983, ISBN 9780471293408

    PARAMETERS
    ----------
    a
        radius of the sphere in m
    epsilon
        dielectric function of the sphere
    epsilon_m
        dielectric function of the medium surounding the sphere
    """
    ret = 4 * np.pi * a**3 * (epsilon - epsilon_m) / (epsilon + 2 * epsilon_m)
    return ret


def alpha_eff_point_dipole(alpha, beta, a, z):
    """ point dipole model as in Keilman2004, https://doi.org/10.1098/rsta.2003.1347

    PARAMETERS
    ----------
    alpha
        polarizability of tip, modeled as conducting sphere
    beta
        reflections coefficient of sample underneath tip
    a
        tip radius in m
    z
        height of tip above sample in m
    """
    ret = alpha * (1 + beta) / (1 - alpha * beta / 16 / np.pi / (a + z)**3)
    return ret
