# acquisition scripts
import logging
from time import sleep
from typing import Iterable
import numpy as np

from trion.analysis.signals import Signals
from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.daq import DaqController


def single_point(device: str, signals: Iterable[Signals], clock_channel: str,
                 n_samples: int, truncate: bool=False, pbar=None, ):
    """
    Perform single point acquisition.

    Parameters
    ----------
    device : str
        Name of NI device.
    signals: Iterable[Signals]
        Signals to acquire.
    clock_channel : str
        Channel to use as sample clock
    n_samples : int
        Number of samples to acquire. If < 0, acquires continuously.
    truncate : bool
        Truncate output to exact number of points.
    pbar : progressbar or None
        Progressbar to use. Calls pbar.update(n_read).
    """
    if n_samples < 0:
        n_samples = np.inf
    ctrl = DaqController(device, clock_channel=clock_channel)
    buffer = ExtendingArrayBuffer(vars=signals, max_size=n_samples)
    n_read = 0

    ctrl.setup(buffer=buffer)
    ctrl.start()
    try:
        while True:
            try:
                if ctrl.is_done() or n_read >= n_samples:
                    break
                sleep(0.01)
                n = ctrl.reader.read()
                if pbar is not None:
                    pbar.update(n)
                n_read += n
            except KeyboardInterrupt:
                logging.warning("Acquisition interrupted by user.")
                break
    finally:
        ctrl.stop()
        ctrl.close()
        logging.info("Acquisition finished.")
    data = buffer.buf
    if truncate and np.isfinite(n_samples):
        data = data[:n_samples,:]
    return data
