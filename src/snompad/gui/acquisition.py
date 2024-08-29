# classes to acquire data from DAQ

import numpy as np
from time import sleep

from ..acquisition.buffer import CircularArrayBuffer
from ..drivers import DaqController

from . import buffer_size, signals

acquisition_buffer = None


class Acquisitor:
    def __init__(self) -> None:
        self.go = False
        self.buffer = None
        self.idle_loop()  # to make the thread 'start listening'

    def idle_loop(self):
        """ when 'GO' button is not active
        """
        while not self.go:
            sleep(.01)
        self.acquisition_loop()

    def acquisition_loop(self):
        """ when 'GO' button is active
        """
        # ToDo: figure out how to read messages from here
        # global acquisition_buffer, err_code, err_msg
        # harm_scaling = np.zeros(max_harm + 1)
        self.buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
        daq = DaqController(dev='Dev1', clock_channel='pfi0')
        daq.setup(buffer=self.buffer)
        try:
            daq.start()
            while self.go:
                sleep(.01)
                daq.reader.read()
        except Exception as e:
            # ToDo check if there are 'usual suspects' and catch them specifically
            # if err_code == 0:
            #     err_code = 10
            #     err_msg = str(e)
            raise
        finally:
            daq.close()
            self.idle_loop()
