import numpy as np
import pytest
from trion.analysis.demod import dft_naive, bin_midpoints


@pytest.fixture
def samplesingle(npts, amps):
    phi = bin_midpoints(npts) # random numbers = bad
    sig = amps[np.newaxis,:] * np.exp(-1j*np.arange(amps.size)[np.newaxis,:]*phi[:,np.newaxis])
    assert sig.shape == (npts, amps.size)
    sig = sig.real.sum(axis=1)
    return sig, phi

cmplx_amps = [
    np.array([1, 1, 1j, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0]),
    np.array([1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 1j, 0, 0, 0, 0]),
    np.array([0, 1, 1j, 0, 0, 0, 0]),
    (np.random.randn(8) + 1j * np.random.randn(8)) * np.exp(-np.arange(8) * 2),
]

@pytest.mark.parametrize("npts", [10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
@pytest.mark.parametrize("nmax", [2, 8, 16])
def test_1d_shape(samplesingle, amps, nmax):
    sig, phi = samplesingle
    orders = np.arange(0, nmax)
    coeff = dft_naive(phi, sig, orders)
    assert coeff.shape == (nmax,)

@pytest.mark.parametrize("npts", [10_000])
@pytest.mark.parametrize("amps", cmplx_amps)
def test_single_amps(samplesingle, amps):
    sig, phi = samplesingle
    orders = np.arange(0, len(amps))
    ret = dft_naive(phi, sig, orders)
    assert np.allclose(ret, amps)