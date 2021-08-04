import pytest
import nidaqmx.system
from trion.expt.cfuncs import self_cal

# TODO: add in a common file, and use as a fixture. A lot of other function use this.

#names = [d.name for d in nidaqmx.system.System.local().devices]

# @pytest.mark.parametrize("dev_name", names)
# def test_self_cal(dev_name):
#     assert self_cal(dev_name)
