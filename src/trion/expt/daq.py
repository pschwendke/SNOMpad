# daq.py: controller for the data acquisition card
"""
Controller for the data acquisition card. 

This is a simplified interface to the DAQ specific to our needs. This defines
the following classes:

DaqController: Handles the nidaqmx machinery specific to our needs.
    This is mostly a Facade. Needs to know the signals, the trigger channel,
    eventually the PLL channel.
Readers: Adapter between DaqController and Buffers. 
    Generally handles the constraints of both APIs.
    Acquires analog and digital tasks in lock-step if necessary.
    Converts the data format between nidaqmx and our destination.
    Selects the read mode (streaming vs by copy).
    Maybe selects callback vs manual.. let's try callback first
    May break this up into multiple classes...

Ideally, the DaqController should not know the Buffer in any way, and just pass
it to the reader.
"""


import logging
import typing
import numpy as np
import attr
import nidaqmx as ni
from nidaqmx import DaqError
from nidaqmx.error_codes import DAQmxErrors
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx._task_modules.read_functions import _read_analog_f_64
from nidaqmx.constants import READ_ALL_AVAILABLE, FillMode, AcquisitionType
from ..analysis.utils import is_optical_signal
import warnings
import re
logger = logging.getLogger(__name__)

# please don't change this...
default_channel_map = {
    "sig_A": "ai0",
    "sig_B": "ai1",
    "sig_d": "ai0",
    "sig_s": "ai1",
    "tap_x": "ai2",
    "tap_y": "ai3",
    "ref_x": "ai4",
    "ref_y": "ai5",
    # chop is to be determined
}

class TAMR(AnalogMultiChannelReader): # TAMR est une sous-classe
    """
    Transposed Analog Multichannel Reader

    This class is similar to AnalogMultiChannelReader, except it uses arrays
    with shape (n_samp_per_chan, n_chan) instead of (n_chan, n_samp_per_chan).
    This is consitent with the C-ordering of the data buffer, required by
    nidaqmx. For C-ordered arrays, it is much less costly to extend along axis
    0, therefore it should be used for the number of channels.
    """
    def _verify_array(self, data, number_of_samples_per_channel,
                      is_many_chan, is_many_samp):
        if not self._verify_array_shape:
            return
        channels_to_read = self._in_stream.channels_to_read
        number_of_channels = len(channels_to_read.channel_names)
        array_shape = (number_of_samples_per_channel, number_of_channels)
        if array_shape is not None and data.shape != array_shape:
            raise DaqError(
                'Read cannot be performed because the NumPy array passed into '
                'this function is not shaped correctly. You must pass in a '
                'NumPy array of the correct shape based on the number of '
                'channels in task and the number of samples per channel '
                'requested.\n\n'
                'Shape of NumPy Array provided: {0}\n'
                'Shape of NumPy Array required: {1}'
                .format(data.shape, array_shape),
                DAQmxErrors.UNKNOWN.value, task_name=self._task.name)

    def read_many_sample(self, data, 
            number_of_samples_per_channel=READ_ALL_AVAILABLE, timeout=10.0):
        number_of_samples_per_channel = (
            self._task._calculate_num_samps_per_chan(
                number_of_samples_per_channel))

        self._verify_array(data, number_of_samples_per_channel, True, True)
        
        return _read_analog_f_64(self._handle, data,
            number_of_samples_per_channel, timeout,
            fill_mode=FillMode.GROUP_BY_SCAN_NUMBER)


@attr.s(order=False)
class TrionsAnalogReader(): # for not pump-probe
    _analog_stream = attr.ib(default=None, init=False)
    buffer = attr.ib(default=None)
    channel_map: typing.Mapping[str, str] = attr.ib(
        default=default_channel_map, kw_only=True
    )
    _ni_reader = attr.ib(init=False, default=None)

    @property
    def analog_stream(self):
        return self._analog_stream

    @property
    def vars(self):
        return self.buffer.vars

    @analog_stream.setter
    def analog_stream(self, in_stream):
        logger.debug("Setting analog stream")
        self._analog_stream = in_stream
        self._ni_reader = TAMR(self.analog_stream)

    # def register_callbacks(self): ?
    #     # enables: thing.register_callbacks().start()

    #     raise NotImplementedError()
    #     return self
    
    @property
    def avail_samp(self):
        return self.analog_stream.avail_samp_per_chan

    def read(self, n=np.infty):
        # return number of samples read?
        n = min(n, self.avail_samp)
        if n == 0:
            return 0
        tmp = np.full((n, len(self.analog_stream.channels_to_read.channel_names)),
                       np.nan)
        r = self._ni_reader.read_many_sample(
                tmp,
                number_of_samples_per_channel=n
        )
        self.buffer.put(tmp)
        return r
    

@attr.s(order=False) # this thing auto-generates __init__ for us
class DaqController():
    """
    Controller for DAQ.

    Parameters
    ----------
    dev : str
        Device name (ex: Dev0)
    clock_channel : str
        Physical channel to use as a clock. Triggers occur on rising edges.
    sample_rate : float
        Expected maximum sample rate.
    sig_range : float, default=2
        Range of optical signal channels, in V. (I think this must be symmetric)
    phase_range : float, default=10
        Range of modulation channels, in V. (I think this must be symmetric...)
    channel_map : dict[str, str], default : default_channel_map
        Mapping from signal type to 

    Attributes
    ----------
    tasks: list of ni.Task
        Wrapped tasks
    reader: Reader object (or list thereof?)
        Wrapped reader. Cannot be set on init
    """
    dev: str = attr.ib()
    clock_channel: str = attr.ib(kw_only=True)
    sample_rate: float = attr.ib(kw_only=True, default=200_000)
    sig_range: float = attr.ib(default=2, kw_only=True)
    phase_range: float = attr.ib(default=10, kw_only=True)
    tasks: typing.Sequence[ni.Task] = attr.ib(factory=list, kw_only=True)
    reader = attr.ib(init=False, default=None)

    def __attrs_post_init__(self): # well... sorry for this..
        # Just log some information.
        logger.info(f"Daq Controller using {self.dev}.")

    # - how to handle read mode? we can directly stream to destination, or use a 
    #   buffer.. strategy...
    # - Should we maintain the acquistion loop ourselves or use callbacks?
    #   We can delegate this to the reader and buffer objects...
    #   I'm tempted to say callbacks will be better. Careful with GUI threads
    #   though... Let's try, fix it later if there is stutter...
    #
    
    def setup(self,
              buffer,
              ) -> 'DaqController':
        """
        Prepare for acquisition.

        Generate tasks, setup timing, reader objects and acquisition buffer.
        This is a clear method where the Controller object acts as a facade, so
        it handles a lot..

        Parameters
        ----------
        buffer : trion.expt.AbstractBuffer
            Buffer for acquisition.

        Notes
        -----
        The mapping from signal name to physical channel is determined by the 
        `channel_map` attribute, which you probably shouldn't change...
        """
        # TODO: will need to be redone when adding chopping

        # generate tasks
        analog = ni.Task("analog")
        # Get variable names from buffer if they are not supplied.
        
        self.reader = TrionsAnalogReader(buffer=buffer)
        # add the analog channels
        for var in self.reader.vars:
            c = self.reader.channel_map[var]
            lim = self.sig_range if is_optical_signal(var) else self.phase_range
            logger.debug(f"Adding variable: {var}, channel {c}, lim: {lim}")
            analog.ai_channels.add_ai_voltage_chan(
                f"{self.dev}/{c}", min_val=-lim, max_val=lim)
        self.tasks.append(analog)
        self.reader.analog_stream = analog.in_stream
        
        self.setup_timing()
        return self

    def setup_timing(self):
        for t in self.tasks:
            t.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                source=self.clock_channel,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=int(self.sample_rate/10), # 100 ms worth of buffer?
            )
        return self

    def start(self) -> 'DaqController':
        # Finalise setting up the reader and buffer, ie: connect the callbacks?
        # start the actual acquisiition
        for t in self.tasks:
            t.start()
        # fire a START TRIGGER event
        return self

    def is_done(self) -> bool:
        return all(t.is_task_done() for t in self.tasks) # hmm...

    # def read(self, dest):
    #     # or delegate this to the reader object? Delegate
    #     raise NotImplementedError()

    def stop(self) -> 'DaqController':
        for t in self.tasks:
            t.stop()
        # undo the actions of `self.start`
        # signal end of read to reader (he can then delegate to buffer
        # if necessary)""
        return self

    def close(self):
        with warnings.catch_warnings():
            # silence a warning from nidaqmx when trying to close the same task multiple times
            warnings.filterwarnings(
                "ignore",
                message=r"Attempted to close NI-DAQmx task of name .* but task was already closed"
            )
            for t in self.tasks:
                t.close()

    def self_calibrate(self) -> 'DaqController':
        raise NotImplementedError()
        return self

    def __del__(self):
        #logger.debug("DaqController.__del__")
        try:
            self.close()
        except Exception:
            pass
        