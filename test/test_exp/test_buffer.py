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
    [1_000, 20, 100],
    [1_000, 1, 100], # chunks will be bigger than size.
)


cases = list(product(
    [
        CircularArrayBuffer,
        ExtendingArrayBuffer,
    ],
    exp_configs,
    n_samples,
))

@pytest.mark.parametrize(
    "cls, exp, npts",
    cases,
)
def test_buffer_get(cls, exp, npts):
    sigs = exp.signals()
    buf = cls(vars=sigs, size=npts+10)
    data = gen_test_data(npts, len(sigs))
    buf.put(data)
    # check getting all gets the complete valid buffer
    assert np.any(np.isnan(buf.buf))
    ret = buf.get(npts)
    assert ret.shape[0] <= npts
    assert ret.shape[1] == len(sigs)
    assert np.all(~np.isnan(ret))
    assert np.allclose(data, ret)
    # get correct length
    n = npts//3
    ret = buf.get(n)
    assert ret.shape[0] == n
    assert np.allclose(data[:n], ret)
    ofst = n
    ret = buf.get(n, ofst)
    assert ret.shape[0] == n
    assert np.allclose(data[ofst:n+ofst], ret)


@pytest.mark.parametrize(
    "cls, exp, npts",
    cases,
)
def test_buffers(cls, exp, npts):
    # for all types. Use this to formalize the common behavior.
    sigs = exp.signals()
    buf = cls(vars=sigs, size=npts)
    data = gen_test_data(npts, len(sigs))
    buf.put(data)
    buf.finish()
    assert buf.size == npts
    assert np.allclose(buf.get(len=buf.size), data)
    n = buf.size//3
    assert np.allclose(buf.head(n), data[:n,:])
    assert np.allclose(buf.tail(n), data[-n:, :])


@pytest.mark.parametrize(
    "buffer_type, exp, n_samples",
    cases
)
def test_export(tmp_path, buffer_type, exp, n_samples):
    sigs = exp.signals()
    buf = buffer_type(size=n_samples, vars=sigs)
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


cases_chunked = [
    [e]+b for e, b in product(
        exp_configs,
        buf_cfs,
    )
]

@pytest.mark.parametrize(
    "exp, npts, nchunks, bufsize",
    cases_chunked,
)
def test_circ_buffer(exp, npts, nchunks, bufsize):
    sigs = exp.signals()
    buf = CircularArrayBuffer(size=bufsize, vars=sigs)
    data = gen_test_data(npts, len(sigs))
    #nchunks = data.shape[0] // nchunks
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    ret = buf.get(bufsize)
    assert ret.shape == (bufsize, len(sigs))
    assert np.allclose(ret, data[-bufsize:,:])

@pytest.mark.parametrize(
    "exp, npts, nchunks, bufsize",
    cases_chunked,
)
def test_extending_buffer(exp, npts, nchunks, bufsize):
    sigs = exp.signals()
    buf = ExtendingArrayBuffer(vars=sigs, size=bufsize)
    data = gen_test_data(npts, len(sigs))
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    assert buf.size >= npts  # extented appropriately
    assert np.allclose(data, buf.buf[:npts, :])  # data is correct
    buf.truncate()
    assert buf.size == npts  # truncation works
    assert buf.i == buf.size
    # check tail is ok
    tail_len = 10
    tail = buf.tail(tail_len)
    assert tail.shape[0] == tail_len
    assert np.allclose(data[-tail_len:, :], tail)
    assert np.allclose(data, buf.get(buf.size))

