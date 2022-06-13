# acquisition scripts
import numpy as np
import xarray as xr
import pandas as pd
import logging
from time import sleep
from typing import Iterable
from tqdm import tqdm
from itertools import product

from trion.analysis.signals import Signals, Demodulation, Scan
from trion.analysis.io import export_data
from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.buffer.base import Overfill
from trion.expt.daq import DaqController
from trion.expt.scans import ContinuousScan, SteppedScan

import nidaqmx
from nidaqmx.constants import (Edge, TaskMode)

logger = logging.getLogger(__name__)


def single_point(device: str, signals: Iterable[Signals], n_samples: int,
                 clock_channel: str = '', truncate: bool = False, pbar=None, ):
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
    buffer = ExtendingArrayBuffer(vars=signals, max_size=n_samples, overfill=Overfill.clip)
    n_read = 0

    ctrl.setup(buffer=buffer)
    ctrl.start()
    try:
        while True:
            try:
                if ctrl.is_done() or n_read >= n_samples:
                    break
                sleep(0.001)
                n = ctrl.reader.read()
                if pbar is not None:
                    pbar.update(n)
                n_read += n
            except KeyboardInterrupt:
                logger.warning("Acquisition interrupted by user.")
                break
    finally:
        ctrl.stop()
        ctrl.close()
        logger.info("Acquisition finished.")
    data = buffer.buf
    if truncate and np.isfinite(n_samples):
        data = data[:n_samples, :]
    return data


def transfer_func_acq(
        device: str,
        read_channels: Iterable[str], write_channels: Iterable[str],
        freqs, amp: float, offset: float = 0,
        n_samples: int = int(1E3), sample_rate: float = 1E6,
        pbar=None, logger=None,
):
    """
    Perform transfer function measurement using a slow passage method.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    t = np.arange(0, n_samples)/sample_rate
    tail_len = int(5E-4*sample_rate)  # 500 us of tail
    measured = []

    with nidaqmx.Task("write") as write_task, nidaqmx.Task("read") as read_task, \
            nidaqmx.Task("clock") as sample_clk_task:
        # prepare sample clock
        logger.debug("Setting up clock task.")
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
            f'{device}/ctr0', freq=sample_rate)
        sample_clk_task.timing.cfg_implicit_timing(
            samps_per_chan=n_samples + tail_len)
        sample_clk_task.control(TaskMode.TASK_COMMIT)
        sample_clk_terminal = f"/{device}/Ctr0InternalOutput"

        # prepare write task
        logger.debug("Setting up output task.")
        ao_channel_names = [device+"/"+c for c in write_channels]
        logger.debug("Write channel names: "+repr(ao_channel_names))
        for c in ao_channel_names:
            write_task.ao_channels.add_ao_voltage_chan(
                c, max_val=5, min_val=-5,
            )
        write_task.timing.cfg_samp_clk_timing(
            sample_rate, source=sample_clk_terminal,
            active_edge=Edge.RISING,
            samps_per_chan=n_samples + tail_len)

        # prepare read task
        logger.debug("Setting up input task.")
        ai_channel_names = [device+"/"+c for c in read_channels]
        logger.debug("Read channel names: "+repr(ao_channel_names))
        for c in ai_channel_names:
            read_task.ai_channels.add_ai_voltage_chan(
                c, max_val=5, min_val=-5,
            )
        read_task.timing.cfg_samp_clk_timing(
            sample_rate, source=sample_clk_terminal,
            active_edge=Edge.RISING, samps_per_chan=n_samples)

        tail = np.ones((write_task.number_of_channels, tail_len)) * offset
        if pbar is None:
            pbar = tqdm(freqs, disable=True)
        for target_freq in pbar:
            if target_freq == 0:
                y = np.ones_like(t)*amp  # +offset
            else:
                y = amp * np.sin(t * 2 * np.pi * target_freq) + offset
            payload = np.repeat(y.reshape((1, -1)),
                                write_task.number_of_channels,
                                axis=0)
            payload = np.hstack((payload, tail))
            assert payload.shape == (write_task.number_of_channels, n_samples + tail_len)

            write_task.write(payload)
            logger.debug(f"Measuring f={target_freq:5.02e}")

            read_task.start()
            write_task.start()
            sample_clk_task.start()

            values_read = np.array(read_task.read(
                number_of_samples_per_channel=n_samples, timeout=2))

            read_task.stop()
            write_task.stop()
            sample_clk_task.stop()

            measured.append(values_read)

    logger.info("Done...")
    measured = np.array(measured)
    assert measured.shape == (freqs.size, len(read_channels), n_samples)
    return measured


class SteppedRetraction(SteppedScan):
    def __init__(self, signals: Iterable[Signals], mod: Demodulation, z_size: int = 0.2, z_res: int = 200,
                 x_target=None, y_target=None, npts: int = 75_000, setpoint: int = 0.8):
        """
        Parameters
        ----------
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        mod: Demodulation
            type of modulation of optical signals, e.g. pshet
        z_size: int
            height or distance dz of the retraction curve in microns
        z_res: int
            number of steps (pixels) acquired during retraction curve
        x_target: float
            x coordinate of retraction curve. Must be passed together with y_target
        y_target: float
            y coordinate of retraction curve. Must be passed together with x_target
        npts: int
            number of samples per pixel acquired by the DAQ
        setpoint: int
            AFM setpoint when engaged
        """
        super().__init__(signals, mod)
        self.afm_sampling_time = 50  # /ms
        self.acquisition_mode = Scan.stepped_retraction
        self.z_size = z_size
        self.z_res = z_res
        self.x_center = None
        self.y_center = None
        self.x_target = x_target
        self.y_target = y_target
        self.npts = npts
        self.setpoint = setpoint

    def start(self):
        self.prepare()
        if self.x_target is not None and self.y_target is not None:
            logger.info(f'Moving sample to target position x={self.x_target:.2f}, y={self.y_target:.2f}')
            self.afm.goto_xy(self.x_target, self.y_target)
        targets = np.linspace(0, self.z_size, self.z_res, endpoint=True)
        self.afm.prepare_retraction(self.mod, self.z_size, self.z_res, self.afm_sampling_time)
        self.afm.engage(self.setpoint)

        try:
            logger.info('Starting scan')
            self.afm.start()
            self.x_center, self.y_center, init_z, _, _ = self.afm.get_current()
            afm_tracking = []
            pbar = tqdm(targets)
            for i, t in enumerate(pbar):
                _, _, z, _, _ = self.afm.get_current()
                dist = abs(z - init_z) * 1E6  # micrometers!
                while dist < t:
                    _, _, z, _, _ = self.afm.get_current()
                    dist = abs(z - init_z) * 1E6
                    pbar.set_postfix(target=t, dist=dist, z=z)
                    sleep(0.01)
                    if self.afm.scan.IsCompleted:
                        logger.warning('Scan is completed while there are targets. (%d of %d)', i, len(targets))
                        self.z_res = i
                        break
                if self.afm.scan.IsCompleted:
                    break
                self.afm.scan.Suspend()
                pre_values = list(self.afm.get_current())[2:]
                data = single_point(self.device, self.signals, self.npts, self.clock_channel)
                post_values = list(self.afm.get_current())[2:]
                export_data(f'{self.data_folder}/pixel_{i:05d}.npz', data, self.signals)
                afm_tracking.append([t] + pre_values + post_values)
                self.afm.scan.Resume()

            while not self.afm.scan.IsCompleted:
                sleep(1)
        finally:
            self.disconnect()

        afm_tracking = np.array(afm_tracking)
        names = ['z_target', 'Z_pre', 'M1A_pre', 'M1P_pre', 'Z_post', 'M1A_post', 'M1P_post']
        self.afm_data = xr.DataArray(data=afm_tracking, dims=('px', 'ch'),
                                     coords={'px': np.arange(self.z_res), 'ch': names})
        self.export()

        logging.info('Done')
        return self.afm_data


class SteppedImage(SteppedScan):
    def __init__(self, signals: Iterable[Signals], mod: Demodulation, x_center: float, y_center: float,
                 x_res: int, y_res: int, x_size: float, y_size: float, npts: int = 75_000, setpoint: int = 0.8):
        """
        Parameters
        ----------
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        mod: Demodulation
            type of modulation of optical signals, e.g. pshet
        x_center: float
            x value in the center of the acquired image
        y_center: float
            y value in the center of the acquired image
        x_res: int
            number of pixels along x-axis (horizontal)
        y_res: int
            number of pixels along y-axis (vertical)
        x_size: float
            size of image in x direction (in micrometres)
        y_size: float
            size of image in y direction (in micrometres)
        npts: int
            number of samples per pixel acquired by the DAQ
        setpoint: int
            AFM setpoint when engaged
        """
        super().__init__(signals, mod)
        self.acquisition_mode = Scan.stepped_image
        self.x_size = x_size
        self.y_size = y_size
        self.x_res = x_res
        self.y_res = y_res
        self.x_center = x_center
        self.y_center = y_center
        self.npts = npts
        self.setpoint = setpoint

    def start(self):
        self.prepare()
        x_pos, y_pos = [
            np.linspace(-1, 1, n, endpoint=True) * s / 2 + p
            for n, s, p in zip((self.x_res, self.y_res), (self.x_size, self.y_size), (self.x_center, self.y_center))
        ]
        targets = list(product(y_pos, x_pos))
        self.afm.set_pshet(self.mod)

        try:
            self.afm.engage(self.setpoint)
            logger.info('Starting scan')
            tracked_values = []
            for i, (y, x) in enumerate(tqdm(targets)):
                self.afm.goto_xy(x, y)
                tracked_values.append([x, y] + list(self.afm.get_current())[2:])
                data = single_point(self.device, self.signals, self.npts, self.clock_channel)
                export_data(f'{self.data_folder}/pixel_{i:05d}.npz', data, self.signals)
        finally:
            self.disconnect()

        df = pd.DataFrame(tracked_values, columns=['x', 'y', 'Z', 'M1A', 'M1P'])
        df['i'] = df.index
        afm_data = {}
        for c in ['Z', 'M1A', 'M1P'] + ['i']:
            ch = df[['x', 'y', c]].groupby(['y', 'x']).sum().unstack()
            da = xr.DataArray(ch, dims=('y', 'x'), coords={'x': x_pos * 1E-6, 'y': y_pos * 1E-6})
            if c in ['Z', 'M1A']:
                da.attrs['z_unit'] = 'm'
            else:
                da.attrs['z_unit'] = ''
            afm_data[c] = da

        self.afm_data = xr.Dataset(afm_data)
        self.export()

        logging.info('Done')
        return self.afm_data


class ContinuousRetraction(ContinuousScan):
    def __init__(self, signals: Iterable[Signals], mod: Demodulation, z_size: float = 0.2, npts: int = 5_000,
                 x_target=None, y_target=None, setpoint: int = 0.8, z_res: int = 200, afm_sampling_time: int = 300):
        """
        Parameters
        ----------
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        mod: Demodulation
            type of modulation of optical signals, e.g. pshet
        z_size: int
            height or distance dz of the retraction curve
        npts: int
            number of samples from the DAQ that are saved in one pixel
        x_target: float
            x coordinate of retraction curve. Must be passed together with y_target
        y_target: float
            y coordinate of retraction curve. Must be passed together with x_target
        setpoint: int
            AFM setpoint when engaged
        z_res: int
            number of pixels of utilized NeaScan approach curve routine
        afm_sampling_time: int
            time that NeaScan samples for every pixel (in ms). Measure for acquisition speed
        """
        super().__init__(signals, mod)
        self.acquisition_mode = Scan.continuous_retraction
        self.z_size = z_size
        self.z_res = z_res
        self.x_target = x_target
        self.y_target = y_target
        self.setpoint = setpoint
        self.npts = npts
        self.afm_sampling_time = afm_sampling_time

    def start(self):
        self.prepare()
        try:
            if self.x_target is not None and self.y_target is not None:
                logger.info(f'Moving sample to target position x={self.x_target:.2f}, y={self.y_target:.2f}')
                self.afm.goto_xy(self.x_target, self.y_target)
            self.afm.prepare_retraction(self.mod, self.z_size, self.z_res, self.afm_sampling_time)
            self.afm.engage(self.setpoint)
            tracked_data, _ = self.acquire()
        finally:
            self.disconnect()

        self.afm_data = tracked_data
        self.export()

        logger.info('Done')
        return self.afm_data


class ContinuousImage(ContinuousScan):
    def __init__(self, signals: Iterable[Signals], mod: Demodulation, x_center: float, y_center: float, x_res: int,
                 y_res: int, x_size: float, y_size: float, afm_sampling_time: float, afm_angle: float = 0,
                 npts: int = 5_000, setpoint: int = 0.8):
        """
        Parameters
        ----------
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        mod: Demodulation
            type of modulation of optical signals, e.g. pshet
        x_center: float
            x value in the center of the acquired image
        y_center: float
            y value in the center of the acquired image
        x_res: int
            number of pixels along x-axis (horizontal)
        y_res: int
            number of pixels along y-axis (vertical)
        x_size: float
            size of image in x direction (in micrometres)
        y_size: float
            size of image in y direction (in micrometres)
        afm_sampling_time: int
            time that NeaScan samples for every pixel (in ms). Measure for acquisition speed
        afm_angle: float
            rotation of the scan frame (in degrees)
        npts: int
            number of samples from the DAQ that are saved in one pixel
        setpoint: int
            AFM setpoint when engaged
        """
        super().__init__(signals, mod)
        self.acquisition_mode = Scan.continuous_image
        self.x_size = x_size
        self.y_size = y_size
        self.x_res = x_res
        self.y_res = y_res
        self.x_center = x_center
        self.y_center = y_center
        self.afm_angle = afm_angle
        self.afm_sampling_time = afm_sampling_time
        self.setpoint = setpoint
        self.npts = npts

    def start(self):
        self.prepare()
        try:
            self.afm.prepare_image(self.mod, self.x_center, self.y_center, self.x_size, self.y_size,
                                   self.x_res, self.y_res, self.afm_angle, self.afm_sampling_time)
            self.afm.engage(self.setpoint)
            tracked_data, nea_data = self.acquire()
        finally:
            self.disconnect()

        nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in nea_data.items()}
        self.afm_data = xr.Dataset(nea_data)
        for k, v in self.afm_data.items():
            if k in ['Z', 'R-Z', 'M1A', 'R-M1A']:
                v.attrs['z_unit'] = 'm'
            else:
                v.attrs['z_unit'] = ''

        self.afm_data['tracked_data'] = tracked_data
        self.export()

        logger.info('Done')
        return self.afm_data
