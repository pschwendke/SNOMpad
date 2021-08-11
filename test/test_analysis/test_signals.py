from trion.analysis.signals import (
    Signals, Scan, Demodulation, Detector, signal_colormap
)
from trion.analysis.experiment import Experiment
import pytest


def test_sortable():
    assert len(list(Signals)) == len(Signals.__members__) # make sure these are actually the members
    assert sorted(Signals) == list(Signals)
    values = [s.value for s in Signals]
    assert all(isinstance(v, str) for v in values)


exp_configs = {
    "shd single": (
        Experiment(Scan.point, Demodulation.shd, Detector.single),
        {Signals.sig_a, Signals.tap_x, Signals.tap_y}
    ),
    "shd dual": (
        Experiment(Scan.point, Demodulation.shd, Detector.dual),
        {Signals.sig_a, Signals.sig_b, Signals.tap_x, Signals.tap_y}
    ),
    "pshet single": (
        Experiment(Scan.point, Demodulation.pshet, Detector.single),
        {Signals.sig_a, Signals.tap_x, Signals.tap_y, Signals.ref_x, Signals.ref_y}
    ),
    "pshet dual": (
        Experiment(Scan.point, Demodulation.pshet, Detector.dual),
        {Signals.sig_a, Signals.sig_b, Signals.tap_x, Signals.tap_y, Signals.ref_x, Signals.ref_y}
    ),

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

@pytest.mark.parametrize(
    "scale", [255, 1], ids=["qt", "mpl"]
)
def test_cmap(scale):
    cmap = signal_colormap(scale=scale)
    assert all(s in cmap for s in Signals)
    assert all(s.name in cmap for s in Signals)
    assert all(all(cmap[s] == cmap[s.name]) for s in Signals)
    assert all(all(v <= scale) for v in cmap.values())
