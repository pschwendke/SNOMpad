# try a finite acquisition
import numpy as np
#import tqdm
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader, AnalogSingleChannelReader
import logging
from time import time
logging.basicConfig(level=logging.DEBUG)

def task_done_cb(task, status, data):
    logging.info(f"task: {type(task)}, {task}")
    logging.info(f"status: {type(status)}, {status}")
    logging.info(f"data: {type(data)}, {data}")
    return 0

def n_samp_acq_cb(task, event_type, n, data):
    print(f"progress: {n}")
    return 0

# let's acquire 3 channels, 200 khz, 100 000 times
with ni.Task("signals") as task:
    # NIDAQmx indexes by 0 but includes endpoint, ie: ai0:2 -> ai0, ai1, ai2
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:2", 
        min_val=-10, max_val=10,
        units=VoltageUnits.VOLTS,
    ) 
    task.register_done_event(task_done_cb) # this now requires explicit call to start
    task.register_every_n_samples_acquired_into_buffer_event(10000, n_samp_acq_cb)
    task.timing.cfg_samp_clk_timing(
        rate=200000,
        # source= If '', use onboard clock
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=100000
    )
    logging.info("Setting up reader")
    #tmg = task.timing
    #logging.info(f"Timing: {tmg}")
    #trg = task.triggers
    #logging.info(f"Trigger: {trg}")
    logging.info("Starting task")
    task.start()
    t0 = time()
    task.wait_until_done()
    dt = time()-t0
    logging.info(f"took: {dt} s")
    # implicit: stop()
    v = task.read()#number_of_samples_per_channel=10)
    
    logging.info(f"Value len: {len(v)}")
    logging.info(f"Typeof: {type(v)}")
    logging.info(f"Done? {task.is_task_done()}")
