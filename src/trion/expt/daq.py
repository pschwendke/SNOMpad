# daq.py: controller for the data acquisition card
"""
Controller for the data acquisition card. 

This is a simplified interface to the DAQ specific to our needs. This defines
the following classes:

DaqController: Handles the nidaqmx machinery specific to our needs.
    This is mostly a Facade.
Readers: Adapter between DaqController and Buffers. 
    Generally handles the constraints of both APIs.
    Maintains synchronization when both analog and digital tasks are needed.
    Selects the read method (callback vs manual).
    Converts the data format between nidaqmx and our destination.
    May factorize this into subclassses...
Buffers: Manage the output buffer.
    Handle opening, sync (flushing), closing, expansion, truncation. Must track
    its current index to enable reading by other classes without involving the
    reader. Must handle two read modes: manually put data and 
    provide view into buffer.
    The destination buffer may also be the entrance into an analysis pipeline...

Ideally, the DaqController should not know the Buffer in any way...
"""
# try to keep this vanilla as much as possible.

# what does the controller need to know?
# - acquisition channels
#   do the controller care which channel is which? Probably we should just let
#   the user describe the mapping using names: map ai0 to optical+...
# - triggering channels
# - PLL eventually...

# we also need:
# - a subclass of reader object for our needs, including one that reads
#   both analog and digital tasks in lock-step. This essentially converts the
#   data into the output of nidaqmx to our standard data specification. It also
#   handles whether acquisiton is streamed or uses an intermediary buffer.
# - a buffer object that manages to push the acquired data into the destination,
#   handles array expansion, truncation, file flushing, yada. It should
#   also track it's current index in order to be read live without intervention
#   by the reader. It must handle two cases: directly put data in there, or 
#   provide a (direct) view into buffer.

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
logger = logging.getLogger(__name__)

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



@attr.s(order=False) # this thing auto-generates __init__ for us
class DaqController(object):
    """
    Controller for DAQ.

    Parameters
    ----------
    dev : str
        Device name (ex: Dev0)
    channel_map : dict (str: str)
        Mapping between signal type and physical channel.
        Which way? (ai0: sig_diff?)
        What to use for the trion channels? (default mapping...)
            - ai0: sig_diff
            - ai1: sig_sum (optional)
            - ai2: tap_x
            - ai3: tap_y
            - ai4: ref_x (optional)
            - ai5: ref_y (optional)
            - chopper? ...
    clock_channel : str
        Physical channel to use as a clock. Triggers occur on rising edges.
    sample_rate : float
        Expected maximum sample rate.
    sig_range : float, default=2
        Range of optical signal channels, in V. (I think this must be symmetric)
    phase_range : float, default=10
        Range of modulation channels, in V. (I think this must be symmetric...)

    Attributes
    ----------
    tasks: list of ni.Task
        Wrapped tasks
    reader: Reader object (or list thereof?)
        Wrapped reader. Cannot be set on init
    """
    dev: str = attr.ib()
    channel_map: typing.Mapping[str, str] = attr.ib()
    clock_channel: str = attr.ib(kw_only=True)
    sample_rate: float = attr.ib(kw_only=True)
    sig_range: float = attr.ib(default=2, kw_only=True)
    phase_range: float = attr.ib(default=10, kw_only=True)
    tasks: typing.Sequence[ni.Task] = attr.ib(factory=list, kw_only=True)
    reader = attr.ib(init=False, default=None)

    # - how to handle read mode? we can directly stream to destination, or use a 
    #   buffer.. strategy...
    # - Should we maintain the acquistion loop ourselves or use callbacks?
    #   We can delegate this to the reader and buffer objects...
    #   I'm tempted to say callbacks will be better. Careful with GUI threads
    #   though... Let's try, fix it later if there is stutter...
    #
    # - How should we handle mapping between physical channel and detector?
    #   - We can use the channel names provided by ni, or build our own.
    #   - Should we use the order of the physical channels, or a consistent
    #     order of detectors? We should probably use the physical order for the
    #     acquistion, and make the mapping available as metadata. Then, the
    #     processing code can keep a consistent order.
    #   

    def setup(self, n_samp = int(1E5)):
        """
        Prepare for acquisition.

        Generate tasks, setup timing, reader objects and acquisition buffer.
        """
        # TODO: will need to be redone when adding chopping
        # generate tasks
        analog = ni.Task("analog")
        
        for c, role in sorted(self.channel_map.keys()):
            v_range = self.sig_range if is_optical_signal(role) else self.phase_range
            analog.ai_channels.add_ai_voltage_chan(
                f"{self.dev}/{c}", min_val=-v_range, max_val=v_range)
        self.tasks.append(analog)
        # configure timing
        for t in self.tasks:
            t.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=int(self.sample_rate/10) # 100 ms worth of buffer?
            )
        # create reader and acquisition buffer
        self.reader = TAMR(analog.in_stream)
        # setup buffer

    def start(self):
        raise NotImplementedError()

    def is_done(self):
        return all(t.is_task_done() for t in self.tasks) # hmm...

    def read(self, dest):
        # or delegate this to the reader object?
        raise NotImplementedError()

    def stop(self):
        for t in self.tasks:
            t.stop()

    def close(self):
        for t in self.tasks:
            t.close()
        # TODO: handle buffer...
    
    def self_calibrate(self):
        raise NotImplementedError()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        
