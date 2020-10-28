import collections
import re
import logging

import numpy
import numpy as np
import pytest
import random
import time

import nidaqmx
from nidaqmx.constants import (
    Edge, TriggerType, AcquisitionType, LineGrouping, Level, TaskMode, ProductCategory)
from nidaqmx.utils import flatten_channel_string
from nidaqmx.tests.helpers import generate_random_seed
from tqdm import tqdm

logging.basicConfig(format="%(levelname)-7s | %(relativeCreated)6d - %(message)s", level=logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

dev = "Dev1"
outname = "trfunc_wide_sine.npz"

number_of_samples = int(1E3)
sample_rate = 1E6
freqs = np.arange(0, 5E5+1, 1E3)
amp = 0.5
offset = 0
t = np.arange(0, number_of_samples)/sample_rate

tail_len = 500

data_out = []

with nidaqmx.Task("write") as write_task, nidaqmx.Task("read") as read_task, \
        nidaqmx.Task("clock") as sample_clk_task:
    # prepare sample clock
    logging.debug("Setting up clock task.")
    sample_clk_task.co_channels.add_co_pulse_chan_freq(
        f'{dev}/ctr0', freq=sample_rate)
    sample_clk_task.timing.cfg_implicit_timing(
        samps_per_chan=number_of_samples+tail_len)
    sample_clk_task.control(TaskMode.TASK_COMMIT)
    sample_clk_terminal = f"/{dev}/Ctr0InternalOutput"

    # prepare write task
    logging.debug("Setting up output task.")
    write_task.ao_channels.add_ao_voltage_chan(
        f"{dev}/ao0:1", max_val=5, min_val=-5,
    )
    write_task.timing.cfg_samp_clk_timing(
        sample_rate, source=sample_clk_terminal,
        active_edge=Edge.RISING, samps_per_chan=number_of_samples+tail_len)
    
    # prepare read task
    logging.debug("Setting up input task.")
    read_task.ai_channels.add_ai_voltage_chan(
        f"{dev}/ai0:1", max_val=5, min_val=-5,
    )
    read_task.timing.cfg_samp_clk_timing(
        sample_rate, source=sample_clk_terminal,
        active_edge=Edge.RISING, samps_per_chan=number_of_samples)

    tail = np.ones((2, tail_len))*offset
    
    for target_freq in tqdm(freqs, desc="Measuring..."):
        
        y = amp*np.sin(t*2*np.pi*target_freq)+offset
        payload = np.repeat(y.reshape((1, -1)), 2, axis=0)
        payload = np.hstack((payload, tail))
        assert numpy.asanyarray(payload).shape == (2, number_of_samples+tail_len)

        write_task.write(payload)


        logging.info(f"Measuring f={target_freq:5.02e}")
        read_task.start()
        write_task.start()
        sample_clk_task.start()

        values_read = np.array(read_task.read(
            number_of_samples_per_channel=number_of_samples, timeout=2))

        read_task.stop()
        write_task.stop()
        sample_clk_task.stop()

        data_out.append(values_read)
        
        # numpy.testing.assert_allclose(
        #     values_read, payload, rtol=0.05, atol=0.005)

    logging.info("Done...")

data_out = np.array(data_out)
assert data_out.shape == (freqs.size, 2, number_of_samples)

logging.info(f"Saving to: {outname}")
np.savez_compressed(outname, freqs = freqs, t=t, data=data_out)