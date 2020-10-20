# using pytest
import pytest
from trion.expt.buffer import CircularArrayBuffer, ExtendingArrayBuffer
import numpy as np
import pandas as pd
from itertools import product, chain
from common import exp_configs, n_samples


def gen_test_data(npts, nsigs):
    return np.arange(1, npts+1)[:,np.newaxis] * 10.0**np.arange(0, nsigs)[np.newaxis,:]

buf_cfs = (
    [23, 3, 10],
    [50, 5, 10],
    [1_000, 20, 100]
)
cases = [
    [e]+b for e, b in product(exp_configs, buf_cfs)
]
@pytest.mark.parametrize(
    "exp, npts, nchunks, bufsize",
    cases
)
def test_circ_buffer(exp, npts, nchunks, bufsize):
    sigs = exp.signals()
    buf = CircularArrayBuffer(max_size=bufsize, vars=sigs)
    data = gen_test_data(npts, len(sigs))
    #nchunks = data.shape[0] // nchunks
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    n = 5
    ret = buf.get(n)
    assert ret.shape == (n, len(sigs))
    assert np.allclose(ret, data[-5:,:])


# TODO: there's something wrong with the ExtendingArrayBuffer, likely an off-by-one
@pytest.mark.parametrize(
    "exp, npts, nchunks, bufsize",
    cases,
)
def test_extending_buffer(exp, npts, nchunks, bufsize):
    sigs = exp.signals()
    buf = ExtendingArrayBuffer(vars=sigs, chunk_size=bufsize)
    data = gen_test_data(npts, len(sigs))
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    assert buf.size >= npts  # extented appropriately
    assert np.allclose(data, buf.buf[:npts, :])  # data is correct
    buf.truncate()
    assert buf.size == npts # truncation works
    assert buf.i == buf.size
    # check tail is ok
    tail_len = 10
    tail = buf.get(tail_len)
    assert tail.shape[0] == tail_len
    assert np.allclose(data[-tail_len:,:], tail)
    assert np.allclose(data, buf.get(buf.size))


@pytest.mark.xfail
def test_buffers():
    # for all types. Use this to formalize the common behavior.
    # assert np.allclose(buf.get(len=buf.size), data)
    # assert buf.size == n_samples
    raise NotImplementedError

cases = list(product(
    [
        CircularArrayBuffer,
        #ExtendingArrayBuffer,
    ],
    exp_configs,
    n_samples,
))

@pytest.mark.parametrize(
    "buffer_type, exp, n_samples",
    cases
)
def test_export(tmp_path, buffer_type, exp, n_samples):
    sigs = exp.signals()
    buf = buffer_type(max_size=n_samples, vars=sigs)
    data = np.arange(0.0, n_samples*len(sigs)).reshape((n_samples, len(sigs)))
    buf.put(data)
    pth = tmp_path / "test.csv"
    buf.export(pth, len=buf.size)
    del buf
    loaded = pd.read_csv(pth)
    loaded_data = loaded.to_numpy()
    loaded_cols = loaded.columns
    assert np.allclose(loaded_data, data)
    assert all(loaded_cols == sigs)
