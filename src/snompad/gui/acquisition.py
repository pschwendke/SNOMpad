# classes to acquire data from DAQ

import numpy as np
from time import sleep

from ..acquisition.buffer import CircularArrayBuffer
from ..drivers import DaqController
from ..utility import Signals

from main import go
from user_messages import err_code, err_msg

acquisition_buffer = None
buffer_size = 200_000

signals = [
    Signals.sig_a,
    Signals.sig_b,
    Signals.tap_x,
    Signals.tap_y,
    Signals.ref_x,
    Signals.ref_y,
    Signals.chop
]

class Acquisitor:
    def __init__(self) -> None:
        self.idle_loop()  # to make the thread 'start listening'

    def idle_loop(self):
        """ when 'GO' button is not active
        """
        while not go:
            sleep(.01)
        self.acquisition_loop()

    def acquisition_loop(self):
        """ when 'GO' button is active
        """
        global acquisition_buffer, err_code, err_msg
        # harm_scaling = np.zeros(max_harm + 1)
        acquisition_buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
        daq = DaqController(dev='Dev1', clock_channel='pfi0')
        daq.setup(buffer=acquisition_buffer)
        try:
            daq.start()
            while go:
                sleep(.01)
                daq.reader.read()
        except Exception as e:
            # ToDo check if there are 'usual suspects' and catch them specifically
            if err_code == 0:
                err_code = 10
                err_msg = str(e)
            raise
        finally:
            daq.close()
            self.idle_loop()
