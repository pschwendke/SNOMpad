import numpy as np
from scipy.optimize import curve_fit


# MODULATION PHASE CORRECTION ##########################################################################################
# ToDo: documentation
def tap_fitfunc(theta, A, theta_C, offset):  # essentially the tip height
    ret = A * (1 + np.cos(theta - theta_C)) + offset  # 1+cos because signal increases with decreasing height
    return ret


def ref_fitfunc(theta, rho, gamma, theta_0, psi_rel, offset):  # essentially the reference beam phase
    psi = gamma * np.cos(theta - theta_0)
    ret = rho * np.cos(psi - psi_rel) + offset
    return ret


# ToDo: better names
def phase_of_contact(binned: np.ndarray) -> float:
    theta_tap = np.linspace(-np.pi, np.pi, binned.shape[-1])
    guess = [.05, 2, .5]  # A, theta_C, offset
    bounds = [[0, -10, 0],  # lower bounds
              [10, 10, 10]]  # upper bounds

    params, _ = curve_fit(tap_fitfunc, theta_tap, binned.squeeze(), p0=guess, bounds=bounds)
    A, theta_C, offset = params
    # ToDo: check variance for good convergence
    return theta_C


def pshet_mod_params(binned: np.array):
    theta_tap = np.linspace(-np.pi, np.pi, binned.shape[-1])
    theta_ref = np.linspace(-np.pi, np.pi, binned.shape[-2])
    guess = [.3, 2.63, 0, 3, 0.6]  # rho, gamma, theta_0, psi_rel, offset
    bounds = [[0, 0, -10, -10, 0],  # lower bounds
              [10, 10, 10, 10, 10]]  # upper bounds

    # fit reference modulation for every tip position
    params = []
    for i, _ in enumerate(theta_tap):
        row = binned.squeeze()[:, i]
        guess, _ = curve_fit(ref_fitfunc, theta_ref, row, p0=guess, bounds=bounds)
        params.append(guess)
        # ToDo: check variance for good convergence
    params = np.array(params)  # columns: rho, gamma, theta_0, psi_rel, offset

    # fit tapping modulation at theta_ref == theta_0
    theta_0 = (theta_ref - params[:, 2].mean()).argmin()  # theta_0 is fairly constant with tapping modulation
    row = binned.squeeze()[theta_0].squeeze()

    theta_C = phase_of_contact(row)
    # select reference modulation parameters at tapping phase of contact
    rho, gamma, theta_0, psi_rel, offset = [p[(theta_tap - theta_C).argmin()].squeeze()
                                            for p in np.split(params, 5, axis=1)]
    return theta_C, theta_0, gamma, psi_rel


# PHASE SHIFTING CORRECTION ############################################################################################
# ToDo: add functions to correct for badly calibrated phase shifting during acquisition
