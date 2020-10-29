import collections
import re

import numpy
import numpy as np
import pytest
import random
import time

import nidaqmx
from nidaqmx.constants import (
    Edge, TriggerType, AcquisitionType, LineGrouping, Level, TaskMode, ProductCategory)
from nidaqmx.utils import flatten_channel_string
from nidaqmx.tests.helpers import generate_random_seed

def x_series_device():
    system = nidaqmx.system.System.local()

    for device in system.devices:
        if (not device.dev_is_simulated and
                device.product_category == ProductCategory.X_SERIES_DAQ and
                len(device.ao_physical_chans) >= 2 and
                len(device.ai_physical_chans) >= 4 and
                len(device.do_lines) >= 8 and
                (len(device.di_lines) == len(device.do_lines)) and
                len(device.ci_physical_chans) >= 4):
            return device


x_series_device = x_series_device()
seed = generate_random_seed()
random.seed(seed)

number_of_samples = int(1E6)
sample_rate = 1E6

ChannelPair = collections.namedtuple(
        'ChannelPair', ['output_channel', 'input_channel'])


# def _get_analog_loopback_channels(device):
#     loopback_channel_pairs = []



#     for ao_physical_chan in device.ao_physical_chans:
#         device_name, ao_channel_name = ao_physical_chan.name.split('/')

#         loopback_channel_pairs.append(
#             ChannelPair(
#                 ao_physical_chan.name,
#                 '{0}/_{1}_vs_aognd'.format(device_name, ao_channel_name)
#             ))

#     return loopback_channel_pairs

# # Select a random loopback channel pair on the device.
# loopback_channel_pairs = _get_analog_loopback_channels(
#     x_series_device)

# number_of_channels = random.randint(2, len(loopback_channel_pairs))
# channels_to_test = random.sample(
#     loopback_channel_pairs, number_of_channels)
channels_to_test = [ 
    ChannelPair("Dev1/ao0", "Dev1/ai0"),
    ChannelPair("Dev1/ao1", "Dev1/ai1")
]
number_of_channels = len(channels_to_test)
print(channels_to_test)

target_freq = 1/1E-4
amp = 0.5
t = np.arange(0, number_of_samples)/sample_rate
y = amp*np.sin(t*2*np.pi*target_freq)
print("max {}", np.max(y))
print("min {}", np.min(y))

with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task, \
        nidaqmx.Task() as sample_clk_task:
    # Use a counter output pulse train task as the sample clock source
    # for both the AI and AO tasks.
    sample_clk_task.co_channels.add_co_pulse_chan_freq(
        '{0}/ctr0'.format(x_series_device.name), freq=sample_rate)
    sample_clk_task.timing.cfg_implicit_timing(
        samps_per_chan=number_of_samples)
    sample_clk_task.control(TaskMode.TASK_COMMIT)

    samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(
        x_series_device.name)

    write_task.ao_channels.add_ao_voltage_chan(
        flatten_channel_string(
            [c.output_channel for c in channels_to_test]),
        max_val=5, min_val=-5)
    write_task.timing.cfg_samp_clk_timing(
        sample_rate, source=samp_clk_terminal,
        active_edge=Edge.RISING, samps_per_chan=number_of_samples)

    read_task.ai_channels.add_ai_voltage_chan(
        flatten_channel_string(
            [c.input_channel for c in channels_to_test]),
        max_val=10, min_val=-10)
    read_task.timing.cfg_samp_clk_timing(
        sample_rate, source=samp_clk_terminal,
        active_edge=Edge.FALLING, samps_per_chan=number_of_samples)

    # Generate values to test.
    values_to_test = np.repeat(y.reshape((1, -1)), number_of_channels, axis=0)
    assert numpy.asanyarray(values_to_test).shape == (number_of_channels, number_of_samples)
    write_task.write(values_to_test)

    # Start the read and write tasks before starting the sample clock
    # source task.
    read_task.start()
    write_task.start()
    sample_clk_task.start()

    values_read = read_task.read(
        number_of_samples_per_channel=number_of_samples, timeout=2)

    numpy.testing.assert_allclose(
        values_read, values_to_test, rtol=0.05, atol=0.005)