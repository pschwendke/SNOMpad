# using pytest
import pytest

from trion.utility.signals import Signals
from trion.expt.buffer import (
    CircularArrayBuffer, ExtendingArrayBuffer, H5Buffer, Overfill,
)
import numpy as np
import pandas as pd
from common import exp_configs, n_samples


def gen_test_data(npts, nsigs):
    return np.arange(1, npts+1)[:,np.newaxis] * 10.0**np.arange(0, nsigs)[np.newaxis,:]


@pytest.fixture(scope="function", params=exp_configs)
def exp(request):
    return request.param

@pytest.fixture(scope="function", params=n_samples)
def npts(request):
    return request.param

@pytest.fixture(scope="function", params=[
    CircularArrayBuffer,
    ExtendingArrayBuffer,
    H5Buffer,
])
def buffer(request, exp, npts, tmp_path):
    sigs = exp.signals()
    kwargs = dict(
        vars=sigs,
        size=npts+10,
    )
    cls = request.param
    assert exp is not None
    if cls == H5Buffer:
        kwargs["fname"] = tmp_path / "test_buffer.h5"
        kwargs["experiment"] = exp
    buf = cls(**kwargs)
    return buf

@pytest.fixture
def data(npts, exp):
    nsigs = len(exp.signals())
    return np.arange(1, npts+1)[:,np.newaxis] * 10.0**np.arange(0, nsigs)[np.newaxis,:]


def test_buffer_get(buffer, exp, npts, data):
    sigs = exp.signals()
    buffer.put(data)
    # check getting all gets the complete valid buffer
    ret = buffer.get(npts)
    assert ret.shape[0] <= npts
    assert ret.shape[1] == len(sigs)
    assert np.all(~np.isnan(ret))
    assert np.allclose(data, ret)
    # get correct length
    n = npts//3
    ret = buffer.get(n)
    assert ret.shape[0] == n
    assert np.allclose(data[:n], ret)
    ofst = n
    ret = buffer.get(n, ofst)
    assert ret.shape[0] == n
    assert ret.shape[0] == n
    assert np.allclose(data[ofst:n+ofst], ret)


def test_buffers(buffer, npts, data):
    # for all types. Use this to formalize the common behavior.
    buffer.put(data)
    buffer.finish()
    assert buffer.size == npts
    assert np.allclose(buffer.get(n=buffer.size), data)
    n = buffer.size//3
    assert np.allclose(buffer.head(n), data[:n,:])
    assert np.allclose(buffer.tail(n), data[-n:, :])

@pytest.mark.parametrize("nget", [2,7,100])
@pytest.mark.parametrize("nchunks", [3,5,20,1])
def test_tail(buffer, data, nget, nchunks):
    m = 0  # manually keep an index to our position in data.
    for chunk in np.array_split(data, nchunks, axis=0):
        buffer.put(chunk)
        m += chunk.shape[0]
        ret = buffer.tail(nget)
        assert ret.shape[0] >= 1
        assert ret.shape[1] == data.shape[1]
        assert ret.shape[0] <= nget
        assert buffer.size < nget or ret.shape[0] == nget
        assert np.count_nonzero(np.isnan(ret)) == 0
        # check we actually have the tail...
        assert np.all(ret == data[m-ret.shape[0]:m])


def test_rotating_tail():
    sigs = [Signals.sig_a, Signals.tap_x]
    npts = 1000
    nget = 10
    buf = CircularArrayBuffer(vars=sigs, size=npts)
    data = np.arange(len(sigs)*npts).reshape((-1, len(sigs)))
    buf.put(data)
    ret = buf.tail(nget)
    assert np.all(ret == data[-nget:])


def test_export(tmp_path, buffer, exp, data):
    sigs = exp.signals()
    buffer.put(data)
    pth = tmp_path / "test.csv"
    buffer.export(pth, len=buffer.size)
    del buffer
    loaded = pd.read_csv(pth)
    loaded_data = loaded.to_numpy()
    loaded_cols = [Signals[c] for c in loaded.columns]
    assert np.allclose(loaded_data, data)
    assert loaded_cols == sigs


@pytest.mark.parametrize("bufsize", [10, 100])
@pytest.mark.parametrize("nchunks", [3,5,20,1])
def test_circ_buffer(exp, npts, nchunks, bufsize, data):
    # Test getting entire buffer
    sigs = exp.signals()
    buf = CircularArrayBuffer(size=bufsize, vars=sigs)
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    buf.finish()
    ret = buf.get(bufsize)
    assert ret.shape == (min(bufsize, npts), len(sigs))
    assert np.allclose(ret, data[-bufsize:,:])


@pytest.mark.parametrize("bufsize", [10, 100])
@pytest.mark.parametrize("nchunks", [3,5,20,1])
def test_extending_buffer(exp, npts, nchunks, bufsize, data):
    sigs = exp.signals()
    buf = ExtendingArrayBuffer(vars=sigs, size=bufsize)
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    assert buf.size >= npts  # extanded appropriately
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

@pytest.mark.parametrize("bufsize", [10, 100])
@pytest.mark.parametrize("cls", [ExtendingArrayBuffer, H5Buffer])
@pytest.mark.parametrize("overfill", [Overfill.clip, Overfill.raise_])
def test_overfill(cls, exp, npts, bufsize, data, tmp_path, overfill):
    sigs = exp.signals()
    dn = 3
    initsize = min(bufsize, npts-dn)
    kwargs = dict(
        vars=sigs,
        size=initsize,
        max_size=npts-dn,
        overfill=overfill,
    )
    if cls == H5Buffer:
        kwargs["fname"] = tmp_path / "test_buffer.h5"
        kwargs["experiment"] = exp
    buf = cls(**kwargs)
    assert buf.i == 0
    assert buf.buf_size == initsize
    assert buf.buf.shape[0] == initsize
    assert buf.max_size == npts-dn
    if overfill is Overfill.raise_:
        with pytest.raises(ValueError):
            buf.put(data)
    else:
        buf.put(data)
        assert buf.size == buf.max_size # test expanded to maximum
        assert buf.get(buf.size).shape[0] == buf.max_size
