from time import sleep
import numpy as np
from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.daq import DaqController

signals = ["sig_A", "tap_x", "tap_y"]
n_samples = 100_000

ctrl = DaqController("DevT", clock_channel="")
buffer = ExtendingArrayBuffer(vars=signals, max_size=n_samples)
n_read = 0
ctrl.setup(buffer=buffer).start()
try:
    while True:
        try:
            if ctrl.is_done() or n_read >= n_samples:
                break
            sleep(0.01)
            n = ctrl.reader.read()
            n_read += n
        except KeyboardInterrupt: # manual termination
            break
finally:
    ctrl.stop().close()

np.save("output.npy", buffer.buf, allow_pickle=False)