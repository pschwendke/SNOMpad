import pytest
import nidaqmx.system
from trion.expt.cfuncs import self_cal


names = [d.name for d in nidaqmx.system.System.local().devices]
@pytest.mark.parametrize("dev_name", names)
def test_self_cal(dev_name):
    assert self_cal(dev_name)
