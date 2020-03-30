import numpy as np
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits, AcquisitionType, READ_ALL_AVAILABLE
from nidaqmx.stream_readers import AnalogMultiChannelReader
from time import sleep

##### SETUP
n_tot = 100000
sample_rate = 200000

with ni.Task("signals") as task:
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:1", 
        min_val=-10, max_val=10,
    ) 
    n_channels = task.number_of_channels
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=n_tot,
    )
    reader = AnalogMultiChannelReader(task.in_stream)
    read_buffer = np.ones((n_channels, n_tot))*-1000 # impossible output
    i = 0
    ##### START
    task.start()
    while not task.is_task_done() and i < n_tot:
        sleep(0.01) # pretend to be busy with other tasks
        n = reader._in_stream.avail_samp_per_chan
        if n == 0: continue
        n = min(n, n_tot-i) # prevent reading too many samples
        ##### READ
        tmp = np.ones((n_channels, n)) * -1001
        r = reader.read_many_sample(
            tmp, 
            number_of_samples_per_channel=n
        )
        read_buffer[:,i:i+n] = tmp
        i += r
    ##### STOP AND CHECK RESULTS
    task.stop()
    assert np.all(read_buffer > -1000)
