#!python3
# try finite scripted acquisition using our API. Save to memory, to file at
# the end.

import logging
import numpy as np
from tqdm import tqdm
from trion.expt.daq import DaqController
from trion.expt.buffer import CircularArrayBuffer
logging.basicConfig(level=logging.DEBUG)

n_tot = 100_000
variables = ["sig_A", "tap_x", "tap_y"]

ctrl = DaqController("DevT", clock_channel="")

buffer = CircularArrayBuffer(vars=variables, max_size=n_tot)
ctrl.setup(buffer)

