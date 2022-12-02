# Some functions to fit tapping and reference modulation, in order to do phase correction and determine gamma and psi_R

import numpy as np
from lmfit import Parameters, minimize

# ToDo: documentation


# SIGNAL MODELLING #####################################################################################################
def reference_modulation(theta, rho, gamma, theta_0, psi_R):
    """ essentially the reference beam phase
    """
    psi = gamma * np.sin(theta - theta_0)
    ret = rho * (1 + np.exp(1j * (psi - psi_R)))

    return ret


# FITTING FUNCTIONS ####################################################################################################
def pshet_fitmodulation(binned: np.ndarray, tap_offset: float, theta_0: float,
                        rho: float, gamma: float, psi_R: float, offset: float) -> tuple[float]:
    """ only performed on first signal, i.e. data[0, :, :]
    """

    def obj_func(parameters, x_data, y_data) -> float:
        model = reference_modulation(theta=x_data, rho=parameters['rho'], gamma=parameters['gamma'],
                                     theta_0=parameters['theta_0'], psi_R=parameters['psi_R'])
        chi_squared = (np.real(model) + parameters['offset'] - y_data) ** 2
        return chi_squared

    theta_tap = np.linspace(-np.pi, np.pi, binned.shape[-1])
    theta_ref = np.linspace(-np.pi, np.pi, binned.shape[-2])
    theta_C = np.pi - tap_offset - (tap_offset < 0) * np.pi  # make theta_C positive
    theta_C = np.median(theta_C)
    theta_C_idx = np.abs(theta_tap + theta_C).argmin()
    data = binned[0, :, theta_C_idx]

    params = Parameters()
    params.add('rho', value=rho, min=0, max=1)
    params.add('gamma', value=gamma, min=0, max=10)
    params.add('theta_0', value=theta_0, min=0, max=np.pi)
    params.add('psi_R', value=psi_R, min=-np.pi, max=np.pi)
    params.add('offset', value=offset, min=0, max=1)

    if offset == 0:
        fit = minimize(obj_func, params, args=(theta_ref, data), method='basinhopping')
    else:
        fit = minimize(obj_func, params, args=(theta_ref, data), method='leastsqr')
    rho = fit.params['rho'].value
    gamma = fit.params['gamma'].value
    theta_0 = fit.params['theta_0'].value
    psi_R = fit.params['psi_R'].value
    offset = fit.params['offset'].value

    return rho, gamma, theta_0, psi_R, offset
