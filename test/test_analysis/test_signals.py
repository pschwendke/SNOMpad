from trion.analysis.signals import (
    Signals, Experiment, Scan, Acquisition, Detector
)
import pytest


def test_sortable():
    assert len(list(Signals)) == len(Signals.__members__) # make sure these are actually the members
    assert sorted(Signals) == list(Signals)
    values = [s.value for s in Signals]
    assert all(isinstance(v, str) for v in values)


exp_configs = {
    "shd single": (
        Experiment(Scan.point, Acquisition.shd, Detector.single),
        {Signals.sig_A, Signals.tap_x, Signals.tap_y}
    ),
    "shd dual": (
        Experiment(Scan.point, Acquisition.shd, Detector.dual),
        {Signals.sig_A, Signals.sig_B, Signals.tap_x, Signals.tap_y}
    )
}


@pytest.mark.parametrize(
    "exp, signals",
    exp_configs.values(),
    ids=exp_configs.keys(),
)
def test_expt_configs(exp, signals):
    # test the behavior of currently supported experimental configurations.
    assert exp.is_valid()
    res = exp.signals()
    assert sorted(res) == res
    assert set(res) == signals
