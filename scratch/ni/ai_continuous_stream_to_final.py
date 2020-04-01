# try acquiring and streaming directly into an external buffer.
# ie: see if we can avoid an intermediary buffer.

import numpy as np
from tqdm import tqdm
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits, AcquisitionType, READ_ALL_AVAILABLE
from nidaqmx.stream_readers import (AnalogSingleChannelReader, 
    AnalogMultiChannelReader, AnalogUnscaledReader)
import logging
from time import time, sleep
logging.basicConfig(level=logging.DEBUG)

n_samp = 100000
sample_rate = 200000

with ni.Task("signals") as task:
    # NIDAQmx indexes by 0 but includes endpoint, ie: ai0:2 -> ai0, ai1, ai2
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:1", 
        min_val=-10, max_val=10,
    ) 
    n_channels = task.number_of_channels
    task.timing.cfg_samp_clk_timing(
        rate=200000,
        # source= If '', use onboard clock
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=n_samp,
    )
    reader = AnalogMultiChannelReader(task.in_stream)
    output_buffer = np.ones((n_channels, n_samp))*-1000 # impossible output
    i = 0
    pbar = tqdm(total=n_samp)
    def n_samp_acq_cb(task, event_type, n, data):
        pbar.update(n)
        return 0

    task.register_every_n_samples_acquired_into_buffer_event(
        sample_rate//50, n_samp_acq_cb)
    
    task.start()
    t0 = time()
    while not task.is_task_done() or i >= n_samp:
        sleep(0.01)
        n = reader._in_stream.avail_samp_per_chan
        pbar.set_description(f"{i} + {n}")
        if n == 0: continue
        tmp = output_buffer[:,i:i+n] # this one is neither C nor F contiguous
        
        i += reader.read_many_sample(
            tmp, 
            number_of_samples_per_channel=n
        )
        
    task.stop()
    t1 = time()
    pbar.close()
    assert np.all(output_buffer >= -10.0)


    
    