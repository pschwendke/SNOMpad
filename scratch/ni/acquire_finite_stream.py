# try a finite acquisition
import numpy as np
from tqdm import tqdm
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits, AcquisitionType, READ_ALL_AVAILABLE
from nidaqmx.stream_readers import (AnalogSingleChannelReader, 
    AnalogMultiChannelReader, AnalogUnscaledReader)
import logging
from time import time, sleep
logging.basicConfig(level=logging.DEBUG)

# for reading raw samples: task.in_stream.read
# this object (task.in_stream) enables low-level control of reading
# for unscaled or scaled arrays, use the objects in nidaqmx.stream_readers

n_samp = 100000
with ni.Task("signals") as task:
    # NIDAQmx indexes by 0 but includes endpoint, ie: ai0:2 -> ai0, ai1, ai2
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:2", 
        min_val=-10, max_val=10,
        units=VoltageUnits.VOLTS,
    ) 
    n_channels = task.number_of_channels
    task.timing.cfg_samp_clk_timing(
        rate=200000,
        # source= If '', use onboard clock
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=n_samp,
    )
    reader = AnalogMultiChannelReader(task.in_stream)
    # array must have flags ['C_CONTIGUOUS', 'WRITEABLE']
    buf = np.zeros((n_channels, n_samp), order="C")
    pbar = tqdm(total=n_samp)
    def n_samp_acq_cb(task, event_type, n, data):
        # there is a bug with tqdm: it is sometimes shown when pbar context
        # manager exits. It is not due to nidaqmx firing the event at an
        # unexpected moment.
        pbar.update(n)
        return 0

    task.register_every_n_samples_acquired_into_buffer_event(
        10000, n_samp_acq_cb)
    task.start()
    t0 = time()
    task.wait_until_done()
    t1 = time()
    reader.read_many_sample(
        buf, number_of_samples_per_channel=READ_ALL_AVAILABLE
    )
    task.stop()
    t2 = time()

    pbar.close()
    logging.info(f"wait took: {t1-t0:.04f}")
    logging.info(f"total took: {t2-t0:.04f}")
    logging.info(f"type(v): {type(buf)}")
    logging.info(f"len(v): {len(buf)}") # len is 3
    #logging.info(f"v: {v}") 
    if n_samp <= 10:
        print(buf.T)
    logging.info(f"type(v[0]): {type(buf[0])}")
    #logging.info(f"len(v[0]): {len(v[0])}")
    logging.info(f"Done? {task.is_task_done()}")
    logging.info("Finishing task...")
