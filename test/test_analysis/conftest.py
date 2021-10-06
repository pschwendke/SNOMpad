import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def shd_data_points(request) -> tuple[tuple, pd.DataFrame]:
    tap_nbins, tap_nharm, tap_harm_amps = request.param
    """ Generates one data point in the middle of each bin.

    Parameters
    ----------
    tap_nbins: int
        number of bins, also number of data points (one per bin)
    tap_nharm: int
        number of harmonics of the returned signal, also the lenght of tap_harm_amps
    tap_harm_amps: list
        list of complex amplitudes for creation of harmonics

    Returns
    -------
    request.param:
        all parameters that are passed to the function, in order to be accessible in test function.
    data: pd.DataFrame
        columns are ['sig_a', 'tap_x', 'tap_y']
        rows are data points (one per bin)
    """
    # tap phase
    tap_phase = np.arange(-np.pi, np.pi, 2 * np.pi / tap_nbins)
    tap_phase += np.pi / tap_nbins
    tap_x = np.cos(tap_phase)
    tap_y = np.sin(tap_phase)

    # generate data
    signal = np.zeros(tap_phase.shape)
    for n in range(tap_nharm):
        signal += np.abs(tap_harm_amps[n]) * np.cos(n * tap_phase + np.angle(tap_harm_amps[n]))
    data = pd.DataFrame(np.array([signal, tap_x, tap_y]).T, columns=['sig_a', 'tap_x', 'tap_y'])

    return request.param, data


@pytest.fixture(scope='session')
def pshet_data_points(request) -> tuple[tuple, pd.DataFrame]:
    tap_nbins, tap_nharm, tap_harm_amps, ref_nbins, ref_nharm, ref_harm_amps = request.param
    """ Generates one data point in the middle of each bin.

    Parameters
    ----------
    tap_nbins: int
        number of bins for tapping demodulation
    tap_nharm: int
        number of harmonics in tapping modulation
    tap_harm_amps: list
        list of complex amplitudes of harmonics of tapping modulation
    ref_nbins: int
        number of bins for pshet demodulation
    ref_nharm: int
        number of harmonics in pshet modulation
    ref_harm_amps: list
        list of complex amplitudes of harmonics of pshet modulation

    Returns
    -------
    request.param:
        all parameters that are passed to the function, in order to be accessible in test function.
    data: pd.DataFrame
        columns are ['sig_a', 'sig_b', 'tap_x', 'tap_y', 'ref_x', 'ref_y']
        rows are data points (one for each combination of tap and ref bin)
    """
    # generating phases
    tap_phase = np.arange(-np.pi, np.pi, 2 * np.pi / tap_nbins)
    tap_phase += np.pi / tap_nbins
    tap_x = np.cos(tap_phase)
    tap_y = np.sin(tap_phase)
    ref_phase = np.arange(-np.pi, np.pi, 2 * np.pi / ref_nbins)
    ref_phase += np.pi / ref_nbins
    ref_x = np.cos(ref_phase)
    ref_y = np.sin(ref_phase)

    # data: DataFrame with one row for every point in phase space
    signal = np.zeros((tap_nbins * ref_nbins, 6))
    for i in range(tap_nbins):
        for j in range(ref_nbins):
            row = i * ref_nbins + j
            tap_sig = sum(np.abs(tap_harm_amps[n]) * np.cos(n * tap_phase[i] + np.angle(tap_harm_amps[n]))
                          for n in range(tap_nharm))
            ref_sig = sum(np.abs(ref_harm_amps[n]) * np.cos(n * ref_phase[j] + np.angle(ref_harm_amps[n]))
                          for n in range(ref_nharm))
            data_point = tap_sig * ref_sig
            signal[row] = [data_point, -data_point, tap_x[i], tap_y[i], ref_x[j], ref_y[j]]
    data = pd.DataFrame(signal, columns=['sig_a', 'sig_b', 'tap_x', 'tap_y', 'ref_x', 'ref_y'])

    return request.param, data


@pytest.fixture(scope='session')
def noise_data(request) -> pd.DataFrame:
    npts = request.param
    """ Creates random (noise) data for sig_a and sig_b for n_points data points.
    tap_x,y and re_x,y are calculated for random phases.
    
    Parameters
    ----------
    npts: int
        number of data points in returned DataFrame

    Returns
    -------
    data_df: pd.DataFrame
        Channels as columns and data points as rows.
    """
    # creating some noise data
    tap_phase = np.random.uniform(-np.pi, np.pi, npts)
    ref_phase = np.random.uniform(-np.pi, np.pi, npts)
    tap_x = np.cos(tap_phase)
    tap_y = np.sin(tap_phase)
    ref_x = np.cos(ref_phase)
    ref_y = np.sin(ref_phase)
    sig_a = np.random.uniform(-np.pi, np.pi, npts)
    sig_b = np.random.uniform(-np.pi, np.pi, npts)

    data_array = np.array([sig_a, sig_b, tap_x, tap_y, ref_x, ref_y]).T
    data_df = pd.DataFrame(data_array, columns=['sig_a', 'sig_b', 'tap_x', 'tap_y', 'ref_x', 'ref_y'])

    return data_df
