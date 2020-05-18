# using pytest
import pytest
from trion.analysis.signals import Experiment, Scan, Acquisition, Detector
from trion.expt.buffer import CircularArrayBuffer
import numpy as np
from itertools import product, chain


exp_configs = (
    Experiment(Scan.point, Acquisition.shd, Detector.single),
    Experiment(Scan.point, Acquisition.shd, Detector.dual),
)

buf_cfs = (
    [23, 3, 10],
    [50, 5, 10],
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
    data = np.arange(1, npts+1)[:,np.newaxis] * 10.0**np.arange(0, len(sigs))[np.newaxis,:]
    #nchunks = data.shape[0] // nchunks
    for chunk in np.array_split(data, nchunks, axis=0):
        buf.put(chunk)
    n = 5
    ret = buf.get(n)
    assert ret.shape == (n, len(sigs))
    assert np.allclose(ret, data[-5:,:])
