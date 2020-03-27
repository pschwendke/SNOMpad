# try a finite acquisition
import numpy as np
from tqdm import tqdm
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits, AcquisitionType, READ_ALL_AVAILABLE
import logging
from time import time, sleep
logging.basicConfig(level=logging.DEBUG)

def task_done_cb(task, status, data):
    logging.info(f"task: {type(task)}, {task}")
    logging.info(f"status: {type(status)}, {status}")
    logging.info(f"data: {type(data)}, {data}")
    return 0

n_samp = 100000
# let's acquire 3 channels, 200 khz, 100 000 times
with ni.Task("signals") as task:
    # NIDAQmx indexes by 0 but includes endpoint, ie: ai0:2 -> ai0, ai1, ai2
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:2", 
        min_val=-10, max_val=10,
        units=VoltageUnits.VOLTS,
    ) 
    n_channels = task.number_of_channels
    #task.register_done_event(task_done_cb) # this now requires explicit call to start
    task.timing.cfg_samp_clk_timing(
        rate=200000,
        # source= If '', use onboard clock
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=n_samp,
    )
    with tqdm(total=n_samp) as pbar:
        def n_samp_acq_cb(task, event_type, n, data):
            # there is a bug with tqdm: it is sometimes shown when pbar context
            # manager exits. It is not due to nidaqmx firing the event at an
            # unexpected moment.
            pbar.update(n)
            return 0
        
        task.register_every_n_samples_acquired_into_buffer_event(10000, n_samp_acq_cb)

        #tmg = task.timing
        #logging.info(f"Timing: {tmg}")
        #trg = task.triggers
        #logging.info(f"Trigger: {trg}")

        #logging.info("Starting task")
        task.start()
        t0 = time()
        task.wait_until_done()
        t1 = time()
        v = task.read(number_of_samples_per_channel=READ_ALL_AVAILABLE) #number_of_samples_per_channel=10)
        task.stop()
        t2 = time()
    logging.info(f"wait took: {t1-t0:.04f}")
    logging.info(f"total took: {t2-t0:.04f}")
    logging.info(f"type(v): {type(v)}")
    logging.info(f"len(v): {len(v)}") # len is 3
    #logging.info(f"v: {v}") 
    logging.info(f"type(v[0]): {type(v[0])}")
    logging.info(f"len(v[0]): {len(v[0])}")
    logging.info(f"Done? {task.is_task_done()}")
    logging.info("Finishing task...")

print("Script finished")

