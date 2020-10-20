#!python3
# try finite scripted acquisition using our API. Save to memory, to file at
# the end.

import logging
from time import sleep
import numpy as np
from tqdm import tqdm
from trion.expt.daq import DaqController
from trion.expt.buffer import CircularArrayBuffer
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-5s - %(name)-s - %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

n_tot = 150_000
variables = ["sig_A", "tap_x", "tap_y"]

ctrl = DaqController("DevT", clock_channel="")

buffer = CircularArrayBuffer(vars=variables, size=n_tot)
ctrl.setup(buffer=buffer)

ctrl.start()
#t0 = time()
n_read = 0
pbar = tqdm(desc="n_sample", total= n_tot)
while not ctrl.is_done() and n_read < n_tot:
    sleep(0.01)
    n = ctrl.reader.read()
    pbar.update(n)
    n_read += n
    #logging.info(f"{n_read}")
ctrl.stop()
pbar.close()
ctrl.close()
logging.info(f"Number of samples read: {n_read}")
logging.info(f"Final array position: {buffer.i}")
assert not np.any(np.isnan(buffer.buf))

import matplotlib.pyplot as plt
data = buffer.get(len=n_tot)
x = np.arange(n_tot)
for i, curve in enumerate(data.T):
    plt.plot(curve, label=variables[i])

plt.legend()

plt.show()
