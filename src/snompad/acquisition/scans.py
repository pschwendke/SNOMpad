# ToDo: standardize units
#  standardize names of parameters
#  note standards in README

# acquisition scripts
import numpy as np
import xarray as xr
import logging
from time import sleep
from typing import Iterable
from tqdm import tqdm
from itertools import product
from datetime import datetime, timedelta

from ..utility.signals import Signals, Scan
from ..acquisition.buffer import ExtendingArrayBuffer
from ..acquisition.buffer.base import Overfill
from ..drivers import DaqController
from ..file_handlers.neascan import to_numpy
from ..acquisition.base import ContinuousScan, BaseScan

# import nidaqmx
# from nidaqmx.constants import (Edge, TaskMode)

logger = logging.getLogger(__name__)


def single_point(device: str, signals: Iterable[Signals], npts: int,
                 clock_channel: str = 'pfi0', truncate: bool = True):
    """
    Perform single point acquisition of DAQ data.

    PARAMETERS
    ----------
    device : str
        Name of NI device.
    signals: Iterable[Signals]
        Signals to acquire.
    clock_channel : str
        Channel to use as sample clock
    npts : int
        Number of pulses to acquire. If < 0, acquires continuously.
    truncate : bool
        Truncate output to exact number of points.
    """
    if npts < 0:
        npts = np.inf
    logger.debug(f'single_point: Acquiring {npts} samples ({", ".join([s.name for s in signals])})')
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
                n_read += n
            except KeyboardInterrupt:
                logger.warning('Acquisition interrupted by user.')
                break
    finally:
        ctrl.close()
        logger.debug('single_point: Acquisition finished.')
    data = buffer.buf
    if truncate and np.isfinite(npts):
        data = data[:npts, :]
    return data


# ToDo delete this after testing
# class ContinuousPoint(ContinuousScan):
#     def __init__(self, modulation: str, n: int, npts: int = 5_000, t=None, t_unit=None, t0_mm=None, metadata=None,
#                  setpoint: float = 0.8, pump_probe=False, signals=None, x_target=None, y_target=None, in_contact=True,
#                  parent_scan=None, delay_idx=None, ratiometry=False):
#         """ Collects n samples of optical and AFM data without scan. Data is saved in chunks of npts length.
#
#         Parameters
#         ----------
#         modulation: str
#             either 'shd' or 'pshet'
#         n: int
#             number of samples to acquire.
#         npts: int
#             number of samples from the DAQ that are saved in one chunk
#         t: float
#             if not None, delay stage will be moved to this position before scan
#         t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
#             unit of t value. Needs to ge vien when t is given
#         t0_mm: float
#             position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
#         metadata: dict
#             dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
#             Specified variables, e.g. self.name have higher priority than passed metadata.
#         setpoint: float
#             AFM setpoint when engaged
#         pump_probe: bool
#             set True if acquiring pump-probe with chopper
#         signals: Iterable[Signals]
#             signals that are acquired from the DAQ
#         x_target: float
#             x coordinate of acquisition. Must be passed together with y_target
#         y_target: float
#             y coordinate of acquisition. Must be passed together with x_target
#         in_contact: bool
#             If True, the samples are acquired while AFM is in contact
#         parent_scan: BaseScan
#             when a BaseScan object is passed, no file is created, but data saved into parent's file
#         delay_idx: str or int
#             identifier for this scan in the scope of the parent_scan, e.g. 'delay_pos_001'
#         ratiometry: bool
#             if True, sig_b is acquired as well
#         """
#         super().__init__(modulation=modulation, npts=npts, setpoint=setpoint, signals=signals, metadata=metadata,
#                          pump_probe=pump_probe, t=t, t_unit=t_unit, t0_mm=t0_mm, ratiometry=ratiometry,
#                          parent_scan=parent_scan, delay_idx=delay_idx)
#         self.acquisition_mode = Scan.point
#         self.xy_unit = 'um'
#         self.x_target = x_target
#         self.y_target = y_target
#         self.in_contact = in_contact
#         self.n = n
#
#     def prepare(self):
#         super().prepare()
#         logger.info(f'ContinuousPoint: acquiring {self.n} samples')
#         if self.x_target is not None and self.y_target is not None:
#             logger.info(f'ContinuousPoint: Moving sample to target position x={self.x_target:.2f} um, '
#                         f'y={self.y_target:.2f} um')
#             self.afm.goto_xy(self.x_target, self.y_target)
#
#     def acquire(self):
#         logger.info('ContinuousPoint: Starting acquisition')
#         # excess = 0
#         chunk_idx = 0
#         afm_tracking = []
#         daq_tracking = {}
#         while chunk_idx * self.npts < self.n:
#             logger.debug(f'ContinuousPoint: chunk # {chunk_idx}')
#             current = [chunk_idx] + list(self.afm.get_current())
#             afm_tracking.append(current)
#             logger.debug('ContinuousPoint: Got AFM data')
#             # n_read = excess
#             n_read = 0
#             nloop = 0  # for debugging, we are getting stuck somewhere around here.
#             # ToDo This was a quick patch. Revisit this. Is there a better solution, or should this go everywhere?
#             try:
#                 while n_read < self.npts:
#                     n = self.ctrl.reader.read()
#                     n_read += n
#                     nloop += 1
#                     sleep(0.001)
#                     if nloop > self.npts / 20:
#                         logging.error(f'ConinuousPoint: Acquisition loop exceeded allowed repetitions: nloop={nloop}.'
#                                       f' {n_read} samples were read from DAQ, instead of {self.npts}')
#                         break
#             except KeyboardInterrupt:
#                 logger.error('ContinuousPoint: Aborting current chunk')
#                 logger.debug(f"ContinuousPoint: {n_read} + {n} <? {self.npts}")
#             # excess = (n_read - self.npts) * int(n_read > self.npts)
#             # now all read samples are dumped into the buffer. Chunk size is not defined anymore
#             logger.debug(f'ContinuousPoint: After read data, nloop {nloop}')
#             data = self.buffer.tail(n=n_read)
#             daq_tracking[chunk_idx] = data
#             logger.debug('ContinuousPoint: after DAQ get.')
#             chunk_idx += 1
#
#         self.file.write_daq_data(data=daq_tracking)
#         afm_tracking = np.array(afm_tracking)
#         self.afm_data = xr.Dataset()
#         tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']
#         for i, c in enumerate(tracked_channels[1:]):
#             da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0].astype('int')})
#             if self.t is not None:
#                 da = da.expand_dims(dim={'t': np.array([self.t])})
#                 da = da.expand_dims(dim={'delay_pos_mm': np.array([self.delay_stage.position])})
#             self.afm_data[c] = da
#         self.stop_time = datetime.now()
#
#     def routine(self):
#         self.prepare()
#         if self.in_contact and self.parent_scan is None:
#             self.afm.engage(self.setpoint)
#         self.ctrl.start()
#         self.acquire()


class SteppedRetraction(BaseScan):
    def __init__(self, modulation: str, z_size: float = 0.2, z_res: int = 200, signals=None, pump_probe=False, t=None,
                 t_unit=None, t0_mm=None, x_target=None, y_target=None, npts: int = 50_000, setpoint: float = 0.8,
                 metadata=None, ratiometry=False):
        """
        PARAMETERS
        ----------
        modulation: str
            type of modulation of optical signals, e.g. 'pshet'
        z_size: float
            height or distance dz of the retraction curve in micrometers
        z_res: int
            number of steps (pixels) acquired during retraction curve
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        pump_probe: bool
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
            number of samples acquired by the DAQ at every position
        setpoint: float
            AFM setpoint when engaged
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        ratiometry: bool
            if True, sig_b is acquired as well
        """
        super().__init__(modulation=modulation, signals=signals, pump_probe=pump_probe, metadata=metadata,
                         t=t, t_unit=t_unit, t0_mm=t0_mm, setpoint=setpoint, npts=npts, ratiometry=ratiometry)
        self.afm_sampling_ms = 50
        self.acquisition_mode = Scan.stepped_retraction
        self.xy_unit = 'um'
        self.z_size = z_size
        self.z_res = z_res
        self.x_center = None
        self.y_center = None
        self.x_target = x_target
        self.y_target = y_target

    def routine(self):
        self.prepare()
        logger.info(f'SteppedRetraction: preparing stepped retraction: size={self.z_size:.2f}, resolution={self.z_res}')
        if self.x_target is not None and self.y_target is not None:
            logger.info(f'SteppedRetraction: Moving sample to target position x={self.x_target:.2f} um,'
                        f'y={self.y_target:.2f} um')
            self.afm.goto_xy(self.x_target, self.y_target)
        targets = np.linspace(0, self.z_size, self.z_res, endpoint=True)
        tracked_channels = ['z_target', 'x', 'y', 'z', 'amp', 'phase']
        self.afm.prepare_retraction(self.modulation, self.z_size, self.z_res, self.afm_sampling_ms)
        self.afm.engage(self.setpoint)

        logger.info('SteppedRetraction: Starting acquisition')
        self.afm.start()
        self.x_center, self.y_center, init_z, _, _ = self.afm.get_current()
        afm_tracking = []
        daq_tracking = {}
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
                    logger.warning(f'SteppedRetraction: Scan is completed while there are targets. '
                                   f'({i} of {len(targets)})')
                    self.z_res = i
                    break
            if self.afm.scan.IsCompleted:
                break
            self.afm.scan.Suspend()
            # acquire npts # of samples
            data = single_point(self.device, self.signals, self.npts, self.clock_channel)
            current = list(self.afm.get_current())
            daq_tracking[i] = data
            afm_tracking.append([i, t] + current)
            self.afm.scan.Resume()

        while not self.afm.scan.IsCompleted:
            sleep(1)
        logger.info('SteppedRetraction: Acquisition complete')
        self.stop_time = datetime.now()

        self.file.write_daq_data(data=daq_tracking)
        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        for i, c in enumerate(tracked_channels):
            da = xr.DataArray(data=afm_tracking[:, i + 1], dims='idx', coords={'idx': afm_tracking[:, 0].astype('int')})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array([self.t])})
            self.afm_data[c] = da


class SteppedImage(BaseScan):
    def __init__(self, modulation: str, x_center: float, y_center: float, x_res: int, y_res: int, x_size: float, y_size:
                 float, npts: int = 75_000, setpoint: float = 0.8, metadata=None, signals=None, pump_probe=False, 
                 ratiometry=False, t=None, t_unit=None, t0_mm=None):
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
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        pump_probe: bool
            set True if acquiring pump-probe with chopper
        ratiometry: bool
            if True, sig_b is acquired as well
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        """
        super().__init__(modulation=modulation, signals=signals, pump_probe=pump_probe, metadata=metadata,
                         t=t, t_unit=t_unit, t0_mm=t0_mm, setpoint=setpoint, npts=npts, ratiometry=ratiometry)
        self.acquisition_mode = Scan.stepped_image
        self.xy_unit = 'um'
        self.x_size = x_size
        self.y_size = y_size
        self.x_res = x_res
        self.y_res = y_res
        self.x_center = x_center
        self.y_center = y_center

    def routine(self):
        x_pos, y_pos = [
            np.linspace(-1, 1, n, endpoint=True) * s / 2 + p
            for n, s, p in zip((self.x_res, self.y_res), (self.x_size, self.y_size), (self.x_center, self.y_center))
        ]
        targets = list(product(y_pos, x_pos))
        tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']

        self.prepare()
        logger.info(f'SteppedImage: Preparing stepped image scan: center {self.x_center:.2f},{self.y_center:.2f}, '
                    f'size={self.x_size:.2f},{self.y_size:.2f}')
        self.afm.engage(self.setpoint)
        logger.info('SteppedImage: Starting acquisition')
        afm_tracking = []
        daq_tracking = {}
        for idx, (y, x) in enumerate(tqdm(targets)):
            self.afm.goto_xy(x, y)
            data = single_point(self.device, self.signals, self.npts, self.clock_channel)
            daq_tracking[idx] = data
            current = list(self.afm.get_current())
            afm_tracking.append([idx] + current)

        logger.info('SteppedImage: Acquisition complete')
        self.stop_time = datetime.now()
        self.file.write_daq_data(data=daq_tracking)
        self.afm_data = xr.Dataset()
        for c, ch_name in enumerate(tracked_channels):
            ch = np.array([[afm_tracking[targets.index((y, x))][c] for x in x_pos] for y in y_pos])
            da = xr.DataArray(ch, dims=('y_target', 'x_target'), coords={'x_target': x_pos, 'y_target': y_pos})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array([self.t])})
            self.afm_data[ch_name] = da


class SteppedLineScan(BaseScan):
    def __init__(self, modulation: str, x_start: float, y_start: float, x_stop: float, y_stop: float, res: int,
                 npts: int, pump_probe=False, signals=None, ratiometry=False, t=None, t_unit=None, t0_mm=None,
                 setpoint: float = 0.8, metadata=None):
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
        pump_probe: bool
            set True if acquiring pump-probe with chopper
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        ratiometry: bool
            if True, sig_b is acquired as well
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        setpoint: float
            AFM setpoint when engaged
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        """
        super().__init__(modulation=modulation, signals=signals, pump_probe=pump_probe, metadata=metadata,
                         t=t, t_unit=t_unit, t0_mm=t0_mm, setpoint=setpoint, npts=npts, ratiometry=ratiometry)
        self.acquisition_mode = Scan.stepped_line
        self.xy_unit = 'um'
        self.x_start = x_start
        self.y_start = y_start
        self.x_stop = x_stop
        self.y_stop = y_stop
        self.x_res = res
        self.y_res = 1  # only one pass during stepped acquisition -> 1 line
        dx = self.x_stop - self.x_start
        dy = self.y_stop - self.y_start
        if self.y_start == self.y_stop:
            self.afm_angle_deg = 0
        else:
            self.afm_angle_deg = np.arctan2(dy, dx) / 2 / np.pi * 360
        self.x_size = np.sqrt(dx ** 2 + dy ** 2)

    def routine(self):
        x_pos = np.linspace(self.x_start, self.x_stop, self.x_res)
        y_pos = np.linspace(self.y_start, self.y_stop, self.x_res)
        targets = list(zip(y_pos, x_pos))
        tracked_channels = ['x_target', 'y_target', 'x', 'y', 'z', 'amp', 'phase']

        self.prepare()
        logger.info(f'SteppedLineScan: Preparing stepped line scan: {x_pos[0]:.2f},{y_pos[0]:.2f} to '
                    f'{x_pos[-1]:.2f},{y_pos[-1]:.2f}')
        self.afm.engage(self.setpoint)
        logger.info('SteppedLineScan: Starting acquisition')
        afm_tracking = []
        daq_tracking = {}
        for i, (y, x) in enumerate(tqdm(targets)):
            self.afm.goto_xy(x, y)
            data = single_point(self.device, self.signals, self.npts, self.clock_channel)
            daq_tracking[i] = data
            current = list(self.afm.get_current())
            afm_tracking.append([i, x, y] + current)

        logger.info('SteppedLineScan: Acquisition complete')
        self.stop_time = datetime.now()
        self.file.write_daq_data(data=daq_tracking)
        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        for i, c in enumerate(tracked_channels):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0].astype('int')})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array([self.t])})
            self.afm_data[c] = da


class ContinuousRetraction(ContinuousScan):
    def __init__(self, modulation: str, z_size: float = 0.2, npts: int = 5_000, setpoint: float = 0.8, signals=None,
                 ratiometry=False, pump_probe=False, t=None, t_unit=None, t0_mm=None, x_target=None, y_target=None,
                 z_res: int = 200, afm_sampling_ms: int = 300, metadata=None):
        """
        Parameters
        ----------
        modulation: str
            either 'shd' or 'pshet'
        z_size: float
            height or distance dz of the retraction curve
        npts: int
            number of samples from the DAQ that are saved in one chunk
        setpoint: float
            AFM setpoint when engaged
        pump_probe: bool
            set True if acquiring pump-probe with chopper
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        signals: Iterable[Signals]
            signals that are acquired from the DAQ
        ratiometry: bool
            if True, sig_b is acquired as well
        x_target: float
            x coordinate of retraction curve. Must be passed together with y_target
        y_target: float
            y coordinate of retraction curve. Must be passed together with x_target
        z_res: int
            number of pixels of utilized NeaScan approach curve routine
        afm_sampling_ms: int
            time that NeaScan samples for every pixel (in ms). Measure for acquisition speed
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        """
        super().__init__(modulation=modulation, npts=npts, setpoint=setpoint, signals=signals, metadata=metadata,
                         pump_probe=pump_probe, t=t, t_unit=t_unit, t0_mm=t0_mm, ratiometry=ratiometry)
        self.acquisition_mode = Scan.continuous_retraction
        self.xy_unit = 'um'
        self.z_size = z_size
        self.z_res = z_res
        self.x_target = x_target
        self.y_target = y_target
        self.afm_sampling_ms = afm_sampling_ms

    def prepare(self):
        super().prepare()
        logger.info(f'ContinuousRetraction: preparing continuous retraction: size={self.z_size:.2f}, '
                    f'resolution={self.z_res}')
        if self.x_target is not None and self.y_target is not None:
            logger.info(f'ContinuousRetraction: Moving sample to target position x={self.x_target:.2f} um, '
                        f'y={self.y_target:.2f} um')
            self.afm.goto_xy(self.x_target, self.y_target)
        self.afm.prepare_retraction(self.modulation, self.z_size, self.z_res, self.afm_sampling_ms)


class ContinuousImage(ContinuousScan):
    def __init__(self, modulation: str, x_center: float, y_center: float, x_res: int, y_res: int,
                 x_size: float, y_size: float, afm_sampling_ms: float, afm_angle_deg: float = 0, signals=None,
                 pump_probe=False, t=None, t_unit=None, t0_mm=None, npts: int = 5_000, setpoint: float = 0.8,
                 metadata=None, ratiometry=False):
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
        pump_probe: bool
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
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        ratiometry: bool
            if True, sig_b is acquired as well
        """
        super().__init__(modulation=modulation, npts=npts, setpoint=setpoint, signals=signals, metadata=metadata,
                         pump_probe=pump_probe, t=t, t_unit=t_unit, t0_mm=t0_mm, ratiometry=ratiometry)
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
        logger.info(f'ContinuousImage: preparing continuous image scan: center {self.x_center:.2f},{self.y_center:.2f},'
                    f' size={self.x_size:.2f},{self.y_size:.2f}')
        self.afm.prepare_image(self.modulation, self.x_center, self.y_center, self.x_size, self.y_size,
                               self.x_res, self.y_res, self.afm_angle_deg, self.afm_sampling_ms)


class ContinuousLineScan(ContinuousScan):
    def __init__(self, modulation: str, x_start: float, y_start: float, x_stop: float, y_stop: float,
                 res: int = 200, n_lines: int = 10, afm_sampling_ms: float = 50, npts: int = 5_000,
                 pump_probe=False, signals=None, t=None, t_unit=None, t0_mm=None, setpoint: float = 0.8,
                 metadata=None, ratiometry=False):
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
        pump_probe: bool
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
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        ratiometry: bool
            if True, sig_b is acquired as well
        """
        super().__init__(modulation=modulation, signals=signals, pump_probe=pump_probe, t=t, t_unit=t_unit, t0_mm=t0_mm,
                         npts=npts, setpoint=setpoint, metadata=metadata, ratiometry=ratiometry)
        self.acquisition_mode = Scan.continuous_line
        self.xy_unit = 'um'
        self.x_start = x_start
        self.y_start = y_start
        self.x_stop = x_stop
        self.y_stop = y_stop
        self.x_res = res
        self.y_res = n_lines
        self.afm_sampling_ms = afm_sampling_ms
        self.x_center = .5 * (self.x_start + self.x_stop)
        self.y_center = .5 * (self.y_start + self.y_stop)
        dx = self.x_stop - self.x_start
        dy = self.y_stop - self.y_start
        if self.y_start == self.y_stop:
            self.afm_angle_deg = 0
        else:
            self.afm_angle_deg = np.arctan2(dy, dx) / 2 / np.pi * 360
        self.x_size = np.sqrt(dx ** 2 + dy ** 2)

    def prepare(self):
        super().prepare()

        logger.info(f'ContinuousLineScan: preparing continuous line scan: length={self.x_size:.3f} um, '
                    f'angle={self.afm_angle_deg:.2f} deg')
        self.afm.prepare_image(mod=self.modulation, x_center=self.x_center, y_center=self.y_center,
                               x_size=self.x_size, y_size=0, x_res=self.x_res, y_res=self.y_res,
                               angle=self.afm_angle_deg, sampling_time_ms=self.afm_sampling_ms)


class DelayScan(BaseScan):
    def __init__(self, modulation: str, npts: int,
                 t_unit: str, t0_mm=None, t_start: float = None, t_stop: float = None, t_res: int = None,
                 t_targets: np.ndarray = None, x_target: float = None, y_target: float = None,
                 setpoint: float = 0.8, ratiometry: bool = False, in_contact: bool = None, metadata=None
                 ):
        """
        PARAMETERS
        ----------
        modulation: str
            either 'shd', 'pshet', or 'none'. 'none' only works for point scans.
        npts: int
            number of samples to acquire at each delay position
        t_unit: str
            Unit of t values. Has to be one of 'm', 'mm', 's', 'ps', 'fs'
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        t_start: float
            first position of delay stage
        t_stop: float
            last position of delay stage
        t_res: int
            Number of steps between t_start and t_stop
        t_targets: np.ndarray
            time delay targets.
        x_target: float
            x coordinate of delay scan in um. Must be passed together with y_target
        y_target: float
            y coordinate of delay scan in um. Must be passed together with x_target
        setpoint: float
            AFM setpoint when engaged
        ratiometry: bool
            if True, sig_b is acquired as well for ratiometric signal correction
        in_contact: bool
            when False, AFM probe is not engaged during acquisition
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        """
        super().__init__(modulation=modulation, pump_probe=True, t_unit=t_unit, t0_mm=t0_mm, metadata=metadata,
                         setpoint=setpoint, npts=npts, ratiometry=ratiometry)
        self.acquisition_mode = Scan.delay_scan
        self.t_start = t_start
        self.t_stop = t_stop
        self.t_res = t_res
        self.t_targets = t_targets
        if t_targets is None:
            self.t_targets = np.linspace(self.t_start, self.t_stop,
                                         self.t_res)  # ToDo: the sign seems to be flipped. Think about this.
        self.x_target = x_target
        self.y_target = y_target
        self.in_contact = in_contact

    def routine(self):
        self.prepare()
        logger.info('DelayScan: Preparing')
        logger.debug(f'Scan parameters:\n    t_start: {self.t_start}\n    t_stop: {self.t_stop}'
                     f'\n    t_res: {self.t_res}\n    t_unit: {self.t_unit}\n    t_0(mm): {self.t0_mm}')
        if self.x_target is not None and self.y_target is not None:
            logger.info(f'DelayScan: Moving to target position x={self.x_target:.2f} um, y={self.y_target:.2f} um')
            self.afm.goto_xy(self.x_target, self.y_target)
        if self.in_contact is not False:
            self.afm.engage(self.setpoint)
        logger.info('DelayScan: Starting acquisition')
        daq_tracking = {}
        afm_tracking = []
        elapsed = timedelta(seconds=0)
        for i, t in enumerate(self.t_targets):
            self.delay_stage.move_abs(val=t, unit=self.t_unit)
            self.delay_stage.wait_for_stage()
            self.delay_stage.log_status()
            stage_position = self.delay_stage.position
            step_time = datetime.now() - self.date - elapsed  # s/it
            elapsed = step_time + elapsed  # timedelta
            logger.info(f'Delay position {i+1} of {len(self.t_targets)}: '
                        f't = {t:.2f} {self.t_unit}  ({stage_position:.2f} mm) '
                        f'[{str(elapsed).split(".")[0]}, {step_time.total_seconds():.1} s/it]')
            data = single_point(self.device, self.signals, self.npts, self.clock_channel, truncate=False)
            daq_tracking[i] = data
            current = list(self.afm.get_current())
            afm_tracking.append([i, t, stage_position] + current)

        logger.info('DelayScan: Acquisition complete')
        self.stop_time = datetime.now()
        self.delay_stage.move_abs(val=self.t_targets[0], unit=self.t_unit)

        self.file.write_daq_data(data=daq_tracking)
        self.afm_data = xr.Dataset()
        tracked_channels = ['idx', 't_target', 't_pos', 'x', 'y', 'z', 'amp', 'phase']
        afm_tracking = np.array(afm_tracking)
        for i, c in enumerate(tracked_channels[1:]):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0].astype('int')})
            self.afm_data[c] = da
        

class NoiseScan(ContinuousScan):
    def __init__(self, sampling_seconds: float, signals=None, x_target=None, y_target=None,
                 npts: int = 5_000, setpoint: float = 0.8):
        """
        Parameters
        ----------
        signals: iterable
            list of Signals. If None, the signals will be determined by modulation and pump_probe
        sampling_seconds: float
            number of seconds to acquire
        x_target: float
            x coordinate of acquisition in um. Must be passed together with y_target
        y_target: float
            y coordinate of acquisition in um. Must be passed together with x_target
        npts: int
            number of samples per chunk acquired by the DAQ
        setpoint: float
            AFM setpoint when engaged
        """
        if signals is None:
            signals = [Signals['sig_a'], Signals['tap_x'], Signals['tap_y']]
        super().__init__(npts=npts, setpoint=setpoint, modulation='shd', signals=signals)
        self.acquisition_mode = Scan.noise_sampling
        self.x_target, self.y_target = x_target, y_target
        self.afm_sampling_ms = 0.4
        self.x_res, self.y_res = int(sampling_seconds * 1e3 / self.afm_sampling_ms // 1), 1
        self.x_size, self.y_size = 0, 0
        self.afm_angle_deg = 0

    def prepare(self):
        super().prepare()
        if self.x_target is None or self.y_target is None:
            self.x_target, self.y_target = self.afm.nea_mic.TipPositionX, self.afm.nea_mic.TipPositionY
        self.afm.prepare_image(self.modulation, self.x_target, self.y_target, self.x_size, self.y_size,
                               self.x_res, self.y_res, self.afm_angle_deg, self.afm_sampling_ms, serpent=True)

    def acquire(self):
        # ToDo this could be merged at some time with parent class
        logger.info('NoiseScan: Starting scan')
        # excess = 0
        daq_data = []
        while not self.afm.scan.IsCompleted:
            n_read = 0
            while n_read < self.npts:  # this slicing seems redundant. Someone should rewrite this.
                sleep(.001)
                n = self.ctrl.reader.read()
                n_read += n
            # excess = n_read - self.npts
            daq_data.append(self.buffer.tail(n=n_read))
            # now all read samples are dumped into the buffer. Chunk size is not defined anymore
            # ToDo: clean this up

        daq_data = {0: np.vstack(daq_data)}  # just one chunk / pixel
        self.file.write_daq_data(data=daq_data)

        logger.info('NoiseScan: Acquisition complete')
        self.nea_data = to_numpy(self.nea_data)
        self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x'), coords={'x': np.linspace(0, 1, self.x_res), 'y': [0]}
                                         ) for k, v in self.nea_data.items()}
        self.nea_data = xr.Dataset(self.nea_data)
        logger.info('NoiseScan: Scan complete')
        self.stop_time = datetime.now()
