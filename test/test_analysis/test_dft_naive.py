import numpy as np
import pytest
from trion.analysis.demod import dft_naive, bin_midpoints

def randphi(npts):
    return np.random.uniform(-np.pi, np.pi, npts)

@pytest.fixture
def samplesingle(npts, amps, phi_func):
    phi = phi_func(npts) # random numbers = bad
    amps = np.atleast_2d(amps)
    sig = amps * np.exp(-1j*np.arange(amps.shape[-1])[np.newaxis,np.newaxis,:]*phi[:,np.newaxis,np.newaxis])  # axis 0 is pts, axis 1? is signal axis -1 is order
    assert sig.shape[0] == npts
    assert sig.shape[1:] == (amps.shape)
    sig = np.squeeze(sig.real.sum(axis=-1))
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
    np.array([[0, 1, 1j, 0, 0, 0, 0], [2, 0, 1, 1+1j, 0,0,0]]),
]

@pytest.mark.parametrize("npts", [10, 1000, 10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [bin_midpoints, randphi])
@pytest.mark.parametrize("nmax", [2, 8, 16])
def test_1d_shape(samplesingle, amps, nmax):
    sig, phi = samplesingle
    orders = np.arange(0, nmax)
    coeff = dft_naive(phi, sig, orders)
    if amps.ndim == 1:
        assert coeff.shape == (nmax,)
    else:
        assert coeff.shape == (amps.shape[0], nmax)

@pytest.mark.parametrize("npts", [100, 1000, 10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("phi_func", [bin_midpoints, randphi])
def test_single_amps(samplesingle, amps, phi_func):
    sig, phi = samplesingle
    orders = np.arange(0, amps.shape[-1])
    ret = dft_naive(phi, sig, orders)
    tol = 10/phi.size if phi_func is randphi else 1E-8
    assert np.allclose(ret, amps, atol=tol)
