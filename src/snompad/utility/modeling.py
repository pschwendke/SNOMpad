# SNOM near-field models
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
