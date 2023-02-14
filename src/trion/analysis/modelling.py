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
def pshet_fitmodulation(binned: np.ndarray, fit_params: dict) -> tuple[float]:
    """ only performed on first signal, i.e. data[0, :, :]
    fit_params should include rho, gamma, theta_0, psi_R, offset, tap_offset
    """
    def obj_func(parameters, x_data, y_data) -> float:
        model = reference_modulation(theta=x_data, rho=parameters['rho'], gamma=parameters['gamma'],
                                     theta_0=parameters['theta_0'], psi_R=parameters['psi_R'])
        chi_squared = (np.real(model) + parameters['offset'] - y_data) ** 2
        return chi_squared

    theta_tap = np.linspace(-np.pi, np.pi, binned.shape[-1])
    theta_ref = np.linspace(-np.pi, np.pi, binned.shape[-2])
    theta_C = np.pi - fit_params['tap_offset'] - (fit_params['tap_offset'] < 0) * np.pi  # make theta_C positive
    theta_C = np.median(theta_C)
    theta_C_idx = np.abs(theta_tap + theta_C).argmin()
    data = binned[0, :, theta_C_idx]

    params = Parameters()
    params.add('rho', value=fit_params['rho'], min=0, max=1)
    params.add('gamma', value=fit_params['gamma'], min=0, max=10)
    params.add('theta_0', value=fit_params['theta_0'], min=0, max=np.pi)
    params.add('psi_R', value=fit_params['psi_R'], min=-np.pi, max=np.pi)
    params.add('offset', value=fit_params['offset'], min=0, max=1)

    if fit_params['offset'] == 0:
        fit = minimize(obj_func, params, args=(theta_ref, data), method='basinhopping')
    else:
        fit = minimize(obj_func, params, args=(theta_ref, data), method='leastsqr')
    fit_params['rho'] = fit.params['rho'].value
    fit_params['gamma'] = fit.params['gamma'].value
    fit_params['theta_0'] = fit.params['theta_0'].value
    fit_params['psi_R'] = fit.params['psi_R'].value
    fit_params['offset'] = fit.params['offset'].value
