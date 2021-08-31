import numpy as np
import pytest
from trion.analysis.demod import dft_naive


@pytest.fixture
def sample1d(npts, amps):
    phi = np.random.uniform(-np.pi, np.pi, npts) # random numbers = bad
    sig = amps[np.newaxis,:] * np.exp(-1j*np.arange(amps.size)[np.newaxis,:]*phi[:,np.newaxis])
    assert sig.shape == (npts, amps.size)
    sig = sig.real.sum(axis=1)
    return np.vstack((sig, np.cos(phi), np.sin(phi))).T


@pytest.mark.parametrize("npts", [100000])
@pytest.mark.parametrize("amps", [
    np.array([0, 1, 0, 0, 0, 0, 0])
])
@pytest.mark.parametrize("nmax", [8,])
def test_1d_shape(sample1d, amps, nmax):
    phi = np.arctan2(sample1d[:,-1], sample1d[:,-2])
    orders = np.arange(0, nmax)
    coeff = dft_naive(phi, sample1d[:,0], orders)
    assert coeff.shape == (nmax,)
    assert False