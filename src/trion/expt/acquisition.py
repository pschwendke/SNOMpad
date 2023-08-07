# ToDo: standardize units
#  standardize names of parameters
#  note standards in README

# acquisition scripts
import numpy as np
import xarray as xr
import pandas as pd
import logging
from time import sleep
from typing import Iterable
from tqdm import tqdm
from itertools import product

from trion.analysis.signals import Signals, Scan
from trion.expt.buffer import ExtendingArrayBuffer
from trion.expt.buffer.base import Overfill
from trion.expt.daq import DaqController
from trion.expt.scans import ContinuousScan, BaseScan

import nidaqmx
from nidaqmx.constants import (Edge, TaskMode)

logger = logging.getLogger(__name__)


def single_point(device: str, signals: Iterable[Signals], npts: int,
                 clock_channel: str = 'pfi0', truncate: bool = True, pbar=None):
    """
    Perform single point acquisition of DAQ data.

    Parameters
    ----------
    device : str
        Name of NI device.
    signals: Iterable[Signals]
        Signals to acquire.
    clock_channel : str
        Channel to use as sample clock
    npts : int
        Number of samples to acquire. If < 0, acquires continuously.
    truncate : bool
        Truncate output to exact number of points.
    pbar : progressbar or None
        Progressbar to use. Calls pbar.update(n_read).
    """
    if npts < 0:
        npts = np.inf
    ctrl = DaqController(device, clock_channel=clock_channel)
    buffer = ExtendingArrayBuffer(vars=signals, size=npts, overfill=Overfill.clip)
    n_read = 0

    ctrl.setup(buffer=buffer)
    ctrl.start()
    try:
        while True:
            try:
                if ctrl.is_done() or n_read >= npts:
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
        ctrl.close()
        logger.info("Acquisition finished.")
    data = buffer.buf
    if truncate and np.isfinite(npts):
        data = data[:npts, :]
    return data


class ContinuousPoint(ContinuousScan):
    def __init__(self, modulation: str, n: int, npts: int = 5_000, t=None, t_unit=None, t0_mm=None, delay_tracker=None,
                 setpoint: float = 0.8, chopped=False, signals=None, x_target=None, y_target=None, in_contact=True,
                 stop=False):
        """ Collects n samples of optical and AFM data without scan. Data is saved in chunks of npts length.

        Parameters
        ----------
        modulation: str
            either 'shd' or 'pshet'
        n: int
            number of samples to acquire. When n == 0, acquisition is continuous until stop is True
        npts: int
            number of samples from the DAQ that are saved in one chunk
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        delay_tracker: None or position property of DLStage()
            if not None, delay values (in mm) will be tracked and saved with every chunk
        setpoint: float
            AFM setpoint when engaged
        chopped: bool
            set True if acquiring pump-probe with chopper
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        x_target: float
            x coordinate of retraction curve. Must be passed together with y_target
        y_target: float
            y coordinate of retraction curve. Must be passed together with x_target
        in_contact: bool
            If True, the samples are acquired while AFM is in contact
        stop: bool
            will be evaluated after every chunk. stop is True stops acquisition.
        """
        super().__init__(modulation=modulation, npts=npts, setpoint=setpoint, signals=signals,
                         chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm)
        self.acquisition_mode = Scan.point
        self.xy_unit = 'um'
        self.x_target = x_target
        self.y_target = y_target
        self.in_contact = in_contact
        self.delay_tracker = delay_tracker
        self.target = n
        self.stop = stop

    def prepare(self):
        super().prepare()
        if self.target == 0:
            self.target = np.inf
        if self.x_target is not None and self.y_target is not None:
            logger.info(f'Moving sample to target position x={self.x_target:.2f} um, y={self.y_target:.2f} um')
            self.afm.goto_xy(self.x_target, self.y_target)

    def acquire(self):
        excess = 0
        chunk_idx = 0
        afm_tracking = []
        while chunk_idx * self.npts < self.target and self.stop is False:
            current = [chunk_idx] + list(self.afm.get_current())
            if self.delay_tracker is not None:
                current.append(self.delay_tracker())
            afm_tracking.append(current)

            n_read = excess
            while n_read < self.npts:
                sleep(0.001)
                n = self.ctrl.reader.read()
                n_read += n
            excess = n_read - self.npts
            data = self.buffer.get(n=self.npts, offset=excess)
            self.file['daq_data'].create_dataset(str(chunk_idx), data=data, dtype='float32')
            chunk_idx += 1

        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']
        if self.delay_tracker is not None:
            tracked_channels.append('t')
        for i, c in enumerate(tracked_channels[1:]):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0]})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array(self.t)})
            self.afm_data[c] = da

    def start(self):
        try:
            self.prepare()
            if self.in_contact:
                self.afm.engage(self.setpoint)
            self.ctrl.start()
            logger.info('starting single point acquisition')
            self.acquire()
        finally:
            self.disconnect()
        self.export()
        logger.info('point acquisition complete')


class SteppedRetraction(BaseScan):
    def __init__(self, modulation: str, z_size: int = 0.2, z_res: int = 200, signals=None, chopped=False, t=None,
                 t_unit=None, t0_mm=None, x_target=None, y_target=None, npts: int = 75_000, setpoint: float = 0.8):
        """
        Parameters
        ----------
        modulation: str
            type of modulation of optical signals, e.g. 'pshet'
        z_size: int
            height or distance dz of the retraction curve in micrometers
        z_res: int
            number of steps (pixels) acquired during retraction curve
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        chopped: bool
            set True if acquiring pump-probe with chopper
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        x_target: float
            x coordinate of retraction curve in um. Must be passed together with y_target
        y_target: float
            y coordinate of retraction curve in um. Must be passed together with x_target
        npts: int
            number of samples per chunk acquired by the DAQ
        setpoint: float
            AFM setpoint when engaged
        """
        super().__init__(modulation=modulation, signals=signals, chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm)
        self.afm_sampling_milliseconds = 50
        self.acquisition_mode = Scan.stepped_retraction
        self.xy_unit = 'um'
        self.z_size = z_size
        self.z_res = z_res
        self.x_center = None
        self.y_center = None
        self.x_target = x_target
        self.y_target = y_target
        self.npts = npts
        self.setpoint = setpoint

    def start(self):
        try:
            self.prepare()
            if self.x_target is not None and self.y_target is not None:
                logger.info(f'Moving sample to target position x={self.x_target:.2f} um, y={self.y_target:.2f} um')
                self.afm.goto_xy(self.x_target, self.y_target)
            targets = np.linspace(0, self.z_size, self.z_res, endpoint=True)
            tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']
            self.afm.prepare_retraction(self.modulation, self.z_size, self.z_res, self.afm_sampling_milliseconds)
            self.afm.engage(self.setpoint)

            logger.info('Starting scan')
            self.afm.start()
            self.x_center, self.y_center, init_z, _, _ = self.afm.get_current()
            afm_tracking = []
            pbar = tqdm(targets)
            for i, t in enumerate(pbar):
                # wait for target z-position during retraction scan
                _, _, z, _, _ = self.afm.get_current()
                dist = abs(z - init_z)
                while dist < t:
                    _, _, z, _, _ = self.afm.get_current()
                    dist = abs(z - init_z)
                    pbar.set_postfix(target=t, dist=dist, z=z)
                    sleep(0.01)
                    if self.afm.scan.IsCompleted:
                        logger.warning(f'Scan is completed while there are targets. ({i} of {len(targets)})')
                        self.z_res = i
                        break
                if self.afm.scan.IsCompleted:
                    break
                self.afm.scan.Suspend()
                # acquire npts # of samples
                data = single_point(self.device, self.signals, self.npts, self.clock_channel)
                current = list(self.afm.get_current())
                self.file['daq_data'].create_dataset(str(i), data=data, dtype='float32')
                afm_tracking.append([t, i] + current)
                self.afm.scan.Resume()

            while not self.afm.scan.IsCompleted:
                sleep(1)
        finally:
            self.disconnect()

        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        for i, c in enumerate(tracked_channels):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='z_target', coords={'z_target': afm_tracking[:, 0]})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array(self.t)})
            self.afm_data[c] = da

        self.export()
        logging.info('Scan complete')


class SteppedImage(BaseScan):
    def __init__(self, modulation: str, x_center: float, y_center: float, x_res: int, y_res: int,
                 x_size: float, y_size: float, npts: int = 75_000, setpoint: float = 0.8,
                 signals=None, chopped=False, t=None, t_unit=None, t0_mm=None):
        """
        Parameters
        ----------
        modulation: str
            type of modulation of optical signals, e.g. pshet
        x_center: float
            x value in the center of the acquired image (in micrometres)
        y_center: float
            y value in the center of the acquired image (in micrometres)
        x_res: int
            number of pixels along x-axis (horizontal)
        y_res: int
            number of pixels along y-axis (vertical)
        x_size: float
            size of image in x direction (in micrometres)
        y_size: float
            size of image in y direction (in micrometres)
        npts: int
            number of samples per chunk acquired by the DAQ
        setpoint: float
            AFM setpoint when engaged
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        chopped: bool
            set True if acquiring pump-probe with chopper
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        """
        super().__init__(modulation=modulation, signals=signals, chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm)
        self.acquisition_mode = Scan.stepped_image
        self.xy_unit = 'um'
        self.x_size = x_size
        self.y_size = y_size
        self.x_res = x_res
        self.y_res = y_res
        self.x_center = x_center
        self.y_center = y_center
        self.npts = npts
        self.setpoint = setpoint

    def start(self):
        x_pos, y_pos = [
            np.linspace(-1, 1, n, endpoint=True) * s / 2 + p
            for n, s, p in zip((self.x_res, self.y_res), (self.x_size, self.y_size), (self.x_center, self.y_center))
        ]
        targets = list(product(y_pos, x_pos))
        tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']

        try:
            self.prepare()
            self.afm.set_pshet(self.modulation)
            self.afm.engage(self.setpoint)
            logger.info('Starting scan')
            afm_tracking = []
            for i, (y, x) in enumerate(tqdm(targets)):
                self.afm.goto_xy(x, y)
                data = single_point(self.device, self.signals, self.npts, self.clock_channel)
                self.file['daq_data'].create_dataset(str(i), data=data, dtype='float32')
                current = list(self.afm.get_current())
                afm_tracking.append([x, y, i] + current)
        finally:
            self.disconnect()

        df = pd.DataFrame(afm_tracking, columns=tracked_channels)
        self.afm_data = xr.Dataset()
        for c in tracked_channels:
            ch = df[['x', 'y', c]].groupby(['y', 'x']).sum().unstack()
            da = xr.DataArray(ch, dims=('y', 'x'), coords={'x': x_pos, 'y': y_pos})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array(self.t)})
            self.afm_data[c] = da
        self.export()
        logging.info('Scan complete')


class SteppedLineScan(BaseScan):
    def __init__(self, modulation: str, x_start: float, y_start: float, x_stop: float, y_stop: float, res: int,
                 npts: int, chopped=False, signals=None, t=None, t_unit=None, t0_mm=None, setpoint: float = 0.8):
        """
        Parameters
        ----------
        modulation: str
            type of modulation of optical signals, e.g. 'pshet'
        x_start: float
            x value of starting point of line scan, in um
        y_start: float
            y value of starting point of line scan, in um
        x_stop: float
            x value of final point of line scan, in um
        y_stop: float
            y value of final point of line scan, in um
        res:
            resolution of stepped line scan
        npts: int
            number of samples per chunk acquired by the DAQ
        chopped: bool
            set True if acquiring pump-probe with chopper
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        setpoint: float
            AFM setpoint when engaged
        """
        super().__init__(modulation=modulation, signals=signals, chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm)
        self.acquisition_mode = Scan.stepped_line
        self.xy_unit = 'um'
        self.x_start = x_start
        self.y_start = y_start
        self.x_stop = x_stop
        self.y_stop = y_stop
        self.linescan_res = res
        self.npts = npts
        self.setpoint = setpoint

    def start(self):
        x_pos = np.linspace(self.x_start, self.x_stop, self.linescan_res)
        y_pos = np.linspace(self.y_start, self.y_stop, self.linescan_res)
        targets = list(zip(y_pos, x_pos))
        tracked_channels = ['x_target', 'y_target', 'x', 'y', 'z', 'amp', 'phase']

        try:
            self.prepare()
            self.afm.set_pshet(self.modulation)
            self.afm.engage(self.setpoint)
            logger.info('Starting scan')
            afm_tracking = []
            for i, (y, x) in enumerate(tqdm(targets)):
                self.afm.goto_xy(x, y)
                data = single_point(self.device, self.signals, self.npts, self.clock_channel)
                self.file['daq_data'].create_dataset(str(i), data=data, dtype='float32')
                current = list(self.afm.get_current())
                afm_tracking.append([i, x, y] + current)
        finally:
            self.disconnect()

        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        for i, c in enumerate(tracked_channels):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0]})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array(self.t)})
            self.afm_data[c] = da
        self.export()
        logging.info('Scan complete')


class ContinuousRetraction(ContinuousScan):
    def __init__(self, modulation: str, z_size: float = 0.2, npts: int = 5_000, setpoint: float = 0.8, signals=None,
                 chopped=False, t=None, t_unit=None, t0_mm=None, x_target=None, y_target=None, z_res: int = 200,
                 afm_sampling_ms: int = 300):
        """
        Parameters
        ----------
        modulation: str
            either 'shd' or 'pshet'
        z_size: int
            height or distance dz of the retraction curve
        npts: int
            number of samples from the DAQ that are saved in one chunk
        setpoint: float
            AFM setpoint when engaged
        chopped: bool
            set True if acquiring pump-probe with chopper
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        x_target: float
            x coordinate of retraction curve. Must be passed together with y_target
        y_target: float
            y coordinate of retraction curve. Must be passed together with x_target
        z_res: int
            number of pixels of utilized NeaScan approach curve routine
        afm_sampling_ms: int
            time that NeaScan samples for every pixel (in ms). Measure for acquisition speed
        """
        super().__init__(modulation=modulation, npts=npts, setpoint=setpoint, signals=signals,
                         chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm)
        self.acquisition_mode = Scan.continuous_retraction
        self.xy_unit = 'um'
        self.z_size = z_size
        self.z_res = z_res
        self.x_target = x_target
        self.y_target = y_target
        self.afm_sampling_ms = afm_sampling_ms

    def prepare(self):
        super().prepare()
        if self.x_target is not None and self.y_target is not None:
            logger.info(f'Moving sample to target position x={self.x_target:.2f} um, y={self.y_target:.2f} um')
            self.afm.goto_xy(self.x_target, self.y_target)
        self.afm.prepare_retraction(self.modulation, self.z_size, self.z_res, self.afm_sampling_ms)


class ContinuousImage(ContinuousScan):
    def __init__(self, modulation: str, x_center: float, y_center: float, x_res: int, y_res: int,
                 x_size: float, y_size: float, afm_sampling_ms: float, afm_angle_deg: float = 0, signals=None,
                 chopped=False, t=None, t_unit=None, t0_mm=None, npts: int = 5_000, setpoint: float = 0.8):
        """
        Parameters
        ----------
        modulation: str
            type of modulation of optical signals, e.g. 'pshet'
        x_center: float
            x value in the center of the acquired image (um)
        y_center: float
            y value in the center of the acquired image (um)
        x_res: int
            number of pixels along x-axis (horizontal)
        y_res: int
            number of pixels along y-axis (vertical)
        x_size: float
            size of image in x direction (in micrometres)
        y_size: float
            size of image in y direction (in micrometres)
        afm_sampling_ms: int
            time that NeaScan samples for every pixel (in ms). Measure for acquisition speed
        afm_angle_deg: float
            rotation of the scan frame (in degrees)
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        chopped: bool
            set True if acquiring pump-probe with chopper
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        npts: int
            number of samples from the DAQ that are saved in one chunk
        setpoint: float
            AFM setpoint when engaged
        """
        super().__init__(modulation=modulation, npts=npts, setpoint=setpoint, signals=signals,
                         chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm)
        self.acquisition_mode = Scan.continuous_image
        self.xy_unit = 'um'
        self.x_size = x_size
        self.y_size = y_size
        self.x_res = x_res
        self.y_res = y_res
        self.x_center = x_center
        self.y_center = y_center
        self.afm_angle_deg = afm_angle_deg
        self.afm_sampling_ms = afm_sampling_ms

    def prepare(self):
        super().prepare()
        self.afm.prepare_image(self.modulation, self.x_center, self.y_center, self.x_size, self.y_size,
                               self.x_res, self.y_res, self.afm_angle_deg, self.afm_sampling_ms)


class ContinuousLineScan(ContinuousScan):
    def __init__(self, modulation: str, x_start: float, y_start: float, x_stop: float, y_stop: float,
                 res: int = 200, n_lines: int = 10, afm_sampling_ms: float = 50, npts: int = 5_000,
                 chopped=False, signals=None, t=None, t_unit=None, t0_mm=None, setpoint: float = 0.8):
        """
        Parameters
        ----------
        modulation: str
            type of modulation of optical signals, e.g. 'pshet'
        x_start: float
            x value of starting point of line scan, in um
        y_start: float
            y value of starting point of line scan, in um
        x_stop: float
            x value of final point of line scan, in um
        y_stop: float
            y value of final point of line scan, in um
        res: int
            resolution along fast axis (along line scan) for underlying AFM scan
        n_lines: int
            resolution along slow axis (number of passes, trace and retrace) for underlying AFM scan
        afm_sampling_ms: int
            time that NeaScan samples for every pixel (in ms). Measure for acquisition speed
        npts: int
            number of samples from the DAQ that are saved in one chunk
        chopped: bool
            set True if acquiring pump-probe with chopper
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        setpoint: float
            AFM setpoint when engaged
        """
        super().__init__(modulation=modulation, signals=signals, chopped=chopped, t=t, t_unit=t_unit, t0_mm=t0_mm,
                         npts=npts, setpoint=setpoint)
        self.acquisition_mode = Scan.continuous_line
        self.xy_unit = 'um'
        self.x_start = x_start
        self.y_start = y_start
        self.x_stop = x_stop
        self.y_stop = y_stop
        self.x_res = res
        self.y_res = n_lines
        self.afm_sampling_ms = afm_sampling_ms

    def prepare(self):
        super().prepare()
        x_center = .5 * (self.x_start + self.x_stop)
        y_center = .5 * (self.y_start + self.y_stop)
        dx = np.abs(self.x_stop - self.x_start)
        dy = np.abs(self.y_stop - self.x_start)
        angle = np.atan2(dy, dx)
        length = np.sqrt(dx**2 + dy**2)
        self.afm.prepare_image(mod=self.modulation, x_center=x_center, y_center=y_center, x_size=length, y_size=0,
                               x_res=self.x_res, y_res=self.y_res, angle=angle, sampling_time_ms=self.afm_sampling_ms)


class DelayScan(BaseScan):
    # Wanted behaviour:
    # other scan class can be passed and are collected at every delay position
    # at the end, one file is saved
    # continuous only works with continuous point class
    # t can be ps, fs, s or mm
    # continuous for n_step == 0
    def __init__(self, modulation: str, scan: str, t_start: float, t_stop: float, t_unit: str, t0_mm: float,
                 n_step: int, continuous: bool = True, scale: str = 'lin', **scan_kwargs):
        """
        Parameters
        ----------
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        """
        super().__init__(modulation=modulation, t=150)  # some value because we don't know the unit yet
        self.acquisition_mode = Scan.delay_collection
        self.scan_kwargs = scan_kwargs
        self.t_unit = t_unit
        self.t0_mm = t0_mm

        if scale == 'lin':
            self.t_targets = np.linspace(t_start, t_stop, n_step)
        elif scale == 'log':
            self.t_targets = np.logspace(np.log10(t_start), np.log10(t_stop), n_step)
        else:
            raise NotImplementedError(f'scale has to be lin or log. "{scale}" was passed.')
        
        try:
            i = ['point', 'retraction', 'line', 'image'].index(scan)
        except ValueError:
            raise ValueError(f'Implemented scans are "point", "retraction", "line", and "image". "{scan}" was passed.')
        if continuous:
            self.scan_class = [ContinuousPoint, ContinuousRetraction, ContinuousLineScan, ContinuousImage][i]
        else:
            self.scan_class = [ContinuousPoint, SteppedRetraction, SteppedLineScan, SteppedImage][i]

    def single_scan(self, t: float):
        scan = self.scan_class(modulation=self.modulation, **self.scan_kwargs)
        scan.start()
        # ToDo Monkey patch the preparation ...
        #  or pass argument stand_alone = False

        # TODo: move delay stage
        # ToDo: when t_0 is given, make it reference and make it work with time arguments

    def start(self):
        try:
            self.prepare()
            for i, t in enumerate(self.t_targets):
                logger.info(f'Delay position {i} of {len(self.t_targets)}: t = {t} {self.t_unit}')
                self.single_scan(t=t)

        finally:
            self.disconnect()

        # ToDo format data ???

        self.export()
        logger.info('Delay scan complete')


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
