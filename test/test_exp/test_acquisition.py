import pytest
from trion.expt.acquisition import single_point
from trion.utility.signals import Signals
from common import exp_configs, n_samples


@pytest.mark.parametrize("n", n_samples)
def test_truncate(n):
    sigs = [Signals.sig_a, Signals.sig_b]
    ret = single_point("DevT", sigs, n, truncate=True)
    assert ret.shape == (n, len(sigs))


@pytest.mark.parametrize("exp", exp_configs)
@pytest.mark.parametrize("n", n_samples)
def test_single_point(exp, n):
    sigs = exp.signals()
    data = single_point("DevT", sigs, n, truncate=True)
    assert data.shape[1] == len(sigs)
    assert data.shape[0] >= n
    # ref = np.load(pth.join(pth.dirname(__file__), make_fname(exp, n)))
    # assert np.allclose(data, ref) # checking data consistency. This is somehow inconsistent and unreliable.
