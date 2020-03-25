# describe: setup a measurement, and describe it.
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits, AcquisitionType

SEP = " ".join(["-"]*30)
# setup system

# setup task
with ni.Task("signals") as task:
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:2", 
        min_val=-10, max_val=10,
        units=VoltageUnits.VOLTS,
    ) 
    task.timing.cfg_samp_clk_timing(
        rate=200000,
        # source=channel If '', use onboard clock
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=100000
    )
    print("task.name:", task.name)
    print("task.channels:", task.channels)
    print("task.channel_names:", task.channel_names)
    print(SEP)
    # check channels
    chan = task.ai_channels[0]
    print("chan:", chan)
    print(SEP)
    #timing
    tmg = task.timing
    print("tmg:", tmg)
    #print("tmg.master_timebase_src:", tmg.master_timebase_src) > nidaqmx.errors.DaqError: Specified property is not supported by the device or is not applicable to the task.
    print(SEP)
