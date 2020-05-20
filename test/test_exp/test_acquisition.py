from itertools import product
import os.path as pth
import numpy as np
import pytest
from trion.expt.acquisition import single_point
from trion.analysis.signals import Signals
from common import exp_configs, n_samples
from gen_ref_single_point import make_fname


@pytest.mark.parametrize("n", n_samples)
def test_truncate(n):
    sigs = [Signals.sig_A, Signals.sig_B]
    ret = single_point("DevT", sigs, n, truncate=True)
    assert ret.shape == (n, len(sigs))

@pytest.mark.parametrize(
    "exp, n",
    product(exp_configs, n_samples)
)
def test_single_point(exp, n):
    sigs = exp.signals()
    data = single_point("DevT", sigs, n, truncate=True)
    assert data.shape[1] == len(sigs)
    assert data.shape[0] >= n
    ref = np.load(pth.join(pth.dirname(__file__), make_fname(exp, n)))
    assert np.allclose(data, ref) # checking data consistency.
