import numpy as np
import pandas as pd
import pytest
from trion.analysis.demod import dft_naive, bin_midpoints, shd_naive
from string import ascii_lowercase


def randphi(npts):
    np.random.seed(2108312109)
    return np.random.uniform(-np.pi, np.pi, npts)


@pytest.fixture
def samplesingle(npts, amps, phi_func):
    # generate values of phi
    phi = phi_func(npts)
    amps = np.atleast_2d(amps)
    # generate signal as: y_j = sum_n a_n * exp(-i n phi_j)
    # axis 0 is pts (j), axis 1 is signal column, axis -1 is order (n)
    sig = amps * np.exp(-1j*np.arange(amps.shape[-1])[np.newaxis,np.newaxis,:]*phi[:,np.newaxis,np.newaxis])
    assert sig.shape[0] == npts
    assert sig.shape[1:] == (amps.shape)
    sig = np.squeeze(sig.real.sum(axis=-1))  # sum over orders
    return sig, phi


# some random amplitudes
randamp = (np.random.randn(8) + 1j * np.random.randn(8)) * np.exp(-np.arange(8) * 2)
randamp[0] = randamp[0].real  # DC can't phase shift... oopsie

# test set
cmplx_amps = [
    np.array([1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 1j, 0, 0, 0, 0]),
    np.array([0, 1, 1j, 0, 0, 0, 0]),
    np.array([1, 1, 1j, 0, 0, 0, 0]),
    randamp,
    randamp[:-2],
    # two dimensional
    np.array([[0, 1, 1j, 0, 0, 0, 0], [2, 0, 1, 1+1j, 0,0,0]]),
]


@pytest.mark.parametrize("npts", [10, 1000, 10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [bin_midpoints, randphi])
@pytest.mark.parametrize("nmax", [2, 8, 16])
def test_1d_shape(samplesingle, amps, nmax):
    # test the return shape
    sig, phi = samplesingle
    orders = np.arange(0, nmax)
    coeff = dft_naive(phi, sig, orders)
    if amps.ndim == 1:
        assert coeff.shape == (nmax,)
    else:
        assert coeff.shape == (len(orders), amps.shape[0])


@pytest.mark.parametrize("npts", [100, 1000, 10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [bin_midpoints, randphi])
def test_single_amps(samplesingle, amps, phi_func):
    # test that we get the same amplitudes back
    sig, phi = samplesingle
    orders = np.arange(0, amps.shape[-1])
    ret = dft_naive(phi, sig, orders)
    # use loose tolerance for random numbers, standard for midpoints
    tol = 10/phi.size if phi_func is randphi else 1E-8
    assert np.allclose(amps.T, ret, atol=tol)


@pytest.mark.parametrize("npts", [1000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [randphi])
def test_shd_naive(samplesingle, amps, phi_func):
    # shd_naive should get the same numbers out, but it a different format.
    sig, phi = samplesingle
    # prepare dataframe
    if sig.ndim == 1:
        cols = ["sig_a"]
    else:
        cols = [f"sig_{ascii_lowercase[i]}"
                for i in range(sig.shape[1])]
    cols.extend(["tap_x", "tap_y"])
    df = pd.DataFrame(
        data = np.vstack((
            sig.T, np.cos(phi), np.sin(phi))).T,
        columns=cols,
    )
    # get reference data
    ref = dft_naive(phi, sig, np.arange(amps.shape[-1]))
    # get data using shd_naive, compare with dft results.
    ret = np.squeeze(shd_naive(df, amps.shape[-1]).to_numpy())
    assert np.allclose(ret, ref)
