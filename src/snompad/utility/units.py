from scipy.constants import physical_constants

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
