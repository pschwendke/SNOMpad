# Some functions to fit tapping and reference modulation, in order to do phase correction and determine gamma and psi_R

import numpy as np

from trion.utility.signals import Signals

# ToDo: documentation


# SIGNAL MODELLING # MAINLY FOR TESTING ################################################################################
def reference_modulation(theta, rho, gamma, theta_0, psi_R):
    """ essentially the reference beam phase
    """
    psi = gamma * np.sin(theta - theta_0)
    ret = rho * (1 + np.exp(1j * (psi - psi_R)))
    return ret


def tapping_modulation(theta, theta_C, amps=None):
    ret = np.zeros(len(theta))
    if amps is None:
        amps = [.5, .05, -.015, -.0005, .0005]
    for i, a in enumerate(amps):
        ret += a * (np.cos(i * (theta - theta_C)))
    return ret


def shd_signal(theta: np.ndarray, theta_C: float = 1.6, amps=None) -> np.ndarray:
    """ Modeling of tapping modulation in shd mode. theta_C is the phase of contact.
    """
    sig = tapping_modulation(theta=theta, theta_C=theta_C, amps=amps)
    return np.real(sig)


def pshet_signal(theta_tap: np.ndarray, theta_ref: np.ndarray, theta_C: float = 1.6, theta_0: float = 1.8,
                 psi_R: float = 3, rho: float = .2, gamma: float = 2.63, amps=None) -> np.ndarray:
    """ Modeling of tapping and pshet modulation in pshet mode. Returns 2D phase domain, a simple superposition of
    both modulations. The tap_p dependency of psi_R is neglected.
    """
    sig_tap = tapping_modulation(theta=theta_tap, theta_C=theta_C, amps=amps)
    sig_ref = reference_modulation(theta=theta_ref, rho=rho, gamma=gamma, theta_0=theta_0, psi_R=psi_R)
    sig = sig_tap + sig_ref
    sig *= sig.conj()
    return np.real(sig)


def shd_data(npts: int = 70_000, theta_C: float = 1.6, noise_level: float = 0, tap_nbins=None, amps=None):
    """ Returns npts # of simulated shd data points, similar as recorded by DAQ
    If tap_nbins is given, one data point per bin is returned
    """
    tap_p = np.random.uniform(0, 2 * np.pi, npts)
    if tap_nbins is not None:
        tap_p = np.linspace(0 + 2 * np.pi / tap_nbins, 2 * np.pi, tap_nbins, endpoint=False)
    tap_x = np.cos(tap_p)
    tap_y = np.sin(tap_p)
    sigs = [Signals.sig_a, Signals.tap_x, Signals.tap_y]
    sig_a = shd_signal(theta=tap_p, theta_C=theta_C, amps=amps)
    sig_a += np.random.uniform(- sig_a.max() * noise_level, sig_a.max() * noise_level, len(tap_p))
    data = np.vstack([sig_a, tap_x, tap_y]).T
    return data, sigs


def pshet_data(npts: int = 200_000, theta_C: float = 1.6, theta_0: float = 1.8, psi_R: float = 3, rho: float = .2,
               gamma: float = 2.63, noise_level: float = 0, amps=None):
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
                         gamma=gamma, amps=amps)
    sig_a += np.random.uniform(- sig_a.max() * noise_level, sig_a.max() * noise_level, npts)
    data = np.vstack([sig_a, tap_x, tap_y, ref_x, ref_y]).T
    return data, sigs


# FITTING FUNCTIONS # FOR DEMODULATION #################################################################################
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
