import numpy as np
import nidaqmx as ni
from nidaqmx import DaqError
from nidaqmx.error_codes import DAQmxErrors
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx._task_modules.read_functions import _read_analog_f_64
from nidaqmx.constants import READ_ALL_AVAILABLE, FillMode, AcquisitionType
from time import sleep

# try to see if we can make the direct read to buffer work.
# Two possibilities: 
# 1. use shape (n_chan, n_samp_per_chan) and F order.
# 2. use shape (n_samp_per_chan, n_chan) and C order.
# Once this is done, try with a mmap array...

# Changing to F order (NO!)
# This requires changes to the functions such as 
# nidaqmx._task_modules.read_functions._read_analog_f_64.
# However, this function sets the argtypes of the ctypes function. A change here
# will be global. The argument type is thus a global state that depends on the
# first call. This is an unacceptable side-effect.
# Furthermore: 
# - pyHDF5 uses C-order.
# - C-order is the logical order, as we deal with a list of records for every
#   trigger.
# 
# Transposing to (n_samp_per_chan, n_chan)
# - need to change the _verify_array method to use tranposed shape.
# - While you're at it, maybe it should always use a 2D array. If required,
#   you can then squeeze. Note that (1,n) and (n,1) arrays are both C and F
#   contiguous.
# - change the call to _read_analog_f_64 to use the fill_mode argument.
# Both changes can be done by subclassing AnalogMultiChannelReader object.
# Can an call this a TAMR (Transposed AnalogMultichannelReader)

# Eventually, we will need to read from both a analog and a digital task.
# This can (maybe?) be taken care of by (another) subclass. (MADMR)
# We can handle synchronization by wrapping both input channels, reading the
# lowest number of available channels...

class TAMR(AnalogMultiChannelReader): # TAMR est une sous-classe
    """
    Transposed Analog Multichannel Reader
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


##### SETUP
n_tot = 100000
sample_rate = 200000

with ni.Task("signals") as task:
    task.ai_channels.add_ai_voltage_chan(
        "DevT/ai0:1", 
        min_val=-10, max_val=10,
    ) 
    n_channels = task.number_of_channels
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=n_tot,
    )
    reader = TAMR(task.in_stream)
    read_buffer = np.ones((n_tot, n_channels))*-1000 # impossible output
    i = 0
    ##### START
    task.start()
    while not task.is_task_done() and i < n_tot:
        sleep(0.01) # pretend to be busy with other tasks
        n = reader._in_stream.avail_samp_per_chan
        if n == 0: continue
        n = min(n, n_tot-i) # prevent reading too many samples
        ##### READ
        i += reader.read_many_sample(
            read_buffer[i:i+n, :], # read directly into array using a view
            number_of_samples_per_channel=n
        )
    ##### STOP AND CHECK RESULTS
    task.stop()
    assert np.all(read_buffer > -1000)
print("Complete")
print(read_buffer[:10,:])
# Yes, this works
import matplotlib.pyplot as plt
plt.figure()
x = np.arange(n_tot)
plt.plot(x, read_buffer[:,0])
plt.plot(x, read_buffer[:,1])
plt.show()
