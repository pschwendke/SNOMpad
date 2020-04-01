# try chunked acquisition: a long-running acquisition where we read a finite
# number of samples into a buffer, that then sends them to a final destination
# device FIFO -> nidaq buffer -> acquisition buffer -> final location
import numpy as np
from tqdm import tqdm
import nidaqmx as ni
from nidaqmx.constants import VoltageUnits, AcquisitionType, READ_ALL_AVAILABLE
from nidaqmx.stream_readers import (AnalogSingleChannelReader, 
    AnalogMultiChannelReader, AnalogUnscaledReader)
import logging
from time import time, sleep
logging.basicConfig(level=logging.DEBUG)

class AcqBuffer():
    """Chunked acquisition buffer"""
    def __init__(n_chan, samp_per_chunk):
        #self._n = samp_per_chunk
        self.buf = np.zeros((n_chan, samp_per_chunk))
        self.i = 0

# UNFINISHED

samp_per_chunk = 1000
sample_rate = 10000
n_tot = 5000



with ni.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:1")
    # this is an uncached property, don't use in a tight loop.
    n_chan = task.number_of_channels  

    # setup buffers
    acq_buf = np.zeros((n_chan, samp_per_chunk))
    final_buf = np.zeros(())

    task.timing.cfg_samp_clk_timing(
        rate=sample_rate, sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=n_tot # used to determine nidaqmx buffer size.
    )