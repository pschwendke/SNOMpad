import numpy as np
import pytest
from trion.analysis.demod import dft_naive, bin_midpoints

def randphi(npts):
    return np.random.uniform(-np.pi, np.pi, npts)

@pytest.fixture
def samplesingle(npts, amps, phi_func):
    phi = phi_func(npts) # random numbers = bad
    sig = amps[np.newaxis,:] * np.exp(-1j*np.arange(amps.size)[np.newaxis,:]*phi[:,np.newaxis])
    assert sig.shape == (npts, amps.size)
    sig = sig.real.sum(axis=1)
    return sig, phi

randamp = (np.random.randn(8) + 1j * np.random.randn(8)) * np.exp(-np.arange(8) * 2)
randamp[0] = randamp[0].real  # DC can't be complex... oopsie
cmplx_amps = [
    np.array([1, 1, 1j, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0]),
    np.array([1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 1j, 0, 0, 0, 0]),
    np.array([0, 1, 1j, 0, 0, 0, 0]),
    randamp,
]

@pytest.mark.parametrize("npts", [10, 1000, 10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [bin_midpoints, randphi])
@pytest.mark.parametrize("nmax", [2, 8, 16])
def test_1d_shape(samplesingle, amps, nmax):
    sig, phi = samplesingle
    orders = np.arange(0, nmax)
    coeff = dft_naive(phi, sig, orders)
    assert coeff.shape == (nmax,)

@pytest.mark.parametrize("npts", [1000, 10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [bin_midpoints, randphi])
def test_single_amps(samplesingle, amps):
    sig, phi = samplesingle
    orders = np.arange(0, len(amps))
    ret = dft_naive(phi, sig, orders)
    assert np.allclose(ret, amps, atol=1/phi.size)
