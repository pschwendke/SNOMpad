# Some functions to fit tapping and reference modulation, in order to do phase correction and determine gamma and psi_R

import numpy as np
from lmfit import Parameters, minimize

from trion.analysis.signals import Signals

# ToDo: documentation


# SIGNAL MODELLING #####################################################################################################
def reference_modulation(theta, rho, gamma, theta_0, psi_R):
    """ essentially the reference beam phase
    """
    psi = gamma * np.sin(theta - theta_0)
    ret = rho * (1 + np.exp(1j * (psi - psi_R)))

    return ret


def tapping_modulation(theta, theta_C):
    ret = np.zeros(len(theta))
    amps = [.05, -.015, -.0005, .0005]
    offset = .5
    for i, a in enumerate(amps):
        ret += a * (np.cos((i+1) * (theta - theta_C)))
    return ret + offset


def shd_signal(theta: np.ndarray, theta_C: float = 1.6) -> np.ndarray:
    """ Modeling of tapping modulation in shd mode. theta_C is the phase of contact.
    """
    sig = tapping_modulation(theta=theta, theta_C=theta_C)
    sig *= sig.conj()
    return np.real(sig)


def pshet_signal(theta_tap: np.ndarray, theta_ref: np.ndarray, theta_C: float = 1.6, theta_0: float = 1.8,
                 psi_R: float = 3, rho: float = .2, gamma: float = 2.63) -> np.ndarray:
    """ Modeling of tapping and pshet modulation in pshet mode. Returns 2D phase domain, a simple superposition of
    both modulations. The tap_p dependency of psi_R is neglected.
    """
    sig_tap = tapping_modulation(theta=theta_tap, theta_C=theta_C)
    sig_ref = reference_modulation(theta=theta_ref, rho=rho, gamma=gamma, theta_0=theta_0, psi_R=psi_R)
    sig = sig_tap + sig_ref
    sig *= sig.conj()
    return np.real(sig)


def shd_data(npts: int = 70_000, theta_C: float = 1.6, noise_level: float = 0):
    """ Returns npts # of simulated shd data points, similar as recorded by DAQ
    """
    tap_p = np.random.uniform(0, 2 * np.pi, npts)
    tap_x = np.cos(tap_p)
    tap_y = np.sin(tap_p)
    sigs = [Signals.sig_a, Signals.tap_x, Signals.tap_y]
    sig_a = shd_signal(theta=tap_p, theta_C=theta_C)
    sig_a += np.random.uniform(- sig_a.max() * noise_level, sig_a.max() * noise_level, npts)
    data = np.vstack([sig_a, tap_x, tap_y]).T

    return data, sigs


def pshet_data(npts: int = 200_000, theta_C: float = 1.6, theta_0: float = 1.8, psi_R: float = 3, rho: float = .2,
               gamma: float = 2.63, noise_level: float = 0):
    """ Returns npts # of simulated pshet data points, similar as recorded by DAQ
    """
    tap_p = np.random.uniform(0, 2 * np.pi, npts)
    ref_p = np.random.uniform(0, 2 * np.pi, npts)
    tap_x = np.cos(tap_p)
    tap_y = np.sin(tap_p)
    ref_x = np.cos(ref_p)
    ref_y = np.sin(ref_p)
    sigs = [Signals.sig_a, Signals.tap_x, Signals.tap_y, Signals.ref_x, Signals.ref_y]
    sig_a = pshet_signal(theta_tap=tap_p, theta_ref=ref_p, theta_C=theta_C, theta_0=theta_0, psi_R=psi_R, rho=rho,
                         gamma=gamma)
    sig_a += np.random.uniform(- sig_a.max() * noise_level, sig_a.max() * noise_level, npts)
    data = np.vstack([sig_a, tap_x, tap_y, ref_x, ref_y]).T

    return data, sigs


# FITTING FUNCTIONS ####################################################################################################
def pshet_fitmodulation(binned: np.ndarray, fit_params: dict):
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
