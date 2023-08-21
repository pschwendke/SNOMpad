import logging
import h5py
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep

from trion.analysis.signals import Scan, Demodulation, Signals
from trion.expt.daq import DaqController
from trion.expt.buffer import CircularArrayBuffer
from trion.expt.nea_ctrl import NeaSNOM, to_numpy
from trion.expt.dl_ctrl import DLStage
from trion.__init__ import __version__

logger = logging.getLogger(__name__)


def xr_to_h5_datasets(ds: xr.Dataset, group: h5py.Group):
    for dim, coord in ds.coords.items():  # create dimension scales
        group[dim] = coord.values
        group[dim].make_scale(dim)
    for ch in ds:
        da = ds[ch]
        dset = group.create_dataset(name=ch, data=da.values)  # data
        for k, v in da.attrs.items():  # copy all metadata / attributes
            dset.attrs[k] = v
        for dim in da.coords.keys():  # attach dimension scales
            n = da.get_axis_num(dim)
            dset.dims[n].attach_scale(group[dim])


class BaseScan(ABC):
    def __init__(self, modulation: str, signals=None, device='Dev1', clock_channel='pfi0',
                 t=None, t_unit=None, t0_mm=None, chopped=False, parent_scan=None, identifier=None):
        """
        Parameters
        ----------
        modulation: str
            either 'shd' or 'pshet'
        signals: iterable
            list of Signals. If None, the signals will be determined by modulation and chopped
        device: str
            should be 'Dev1' for now.
        clock_channel: str
            should be 'pfi0', as labelled on the DAQ
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        chopped: bool
            if True, chop signal will be acquired
        parent_scan: BaseScan
            when a BaseScan object is passed, no file is created, but data saved into parent's file
        identifier: str
            identifier for this scan in the scope of the parent_scan, e.g. 'delay_pos_001'
        """
        try:
            self.modulation = Demodulation[modulation]
        except KeyError:
            logger.error(f'BaseScan only takes "shd" and "pshet" as modulation values. "{modulation}" was passed.')
        if signals is None:
            sig_list = {'shd': ['sig_a', 'tap_x', 'tap_y'], 'pshet': ['sig_a', 'tap_x', 'tap_y', 'ref_x', 'ref_y']}
            signals = [Signals[s] for s in sig_list[modulation]]
            if chopped:
                signals.append(Signals['chop'])
        self.signals = signals
        self.device = device
        self.clock_channel = clock_channel
        self.trion_version = __version__
        self.t = t
        self.t_unit = t_unit
        self.t0_mm = t0_mm
        self.chopped = chopped
        self.parent_scan = parent_scan
        self.identifier = identifier
        self.acquisition_mode = None  # should be defined in subclass __init__
        self.afm_data = None
        self.nea_data = None
        self.date = None
        self.name = None
        self.file = None
        self.stop_time = None
        self.neaclient_version = None
        self.neaserver_version = None
        self.afm = None
        self.delay_stage = None
        self.metadata_list = ['name', 'date', 'user', 'sample', 'tip', 'acquisition_mode', 'acquisition_time_s',
                              'modulation', 'signals', 'chopped',
                              'light_source', 'probe_color_nm', 'pump_color_nm', 'probe_FWHM_nm', 'pump_FWHM_nm',
                              'probe_power_mW', 'pump_power_mW',
                              'x_size', 'y_size', 'z_size', 'x_center', 'y_center', 'x_res', 'y_res', 'z_res',
                              'x_start', 'x_stop', 'y_start', 'y_stop', 'linescan_res', 'xy_unit',
                              't', 't_start', 't_stop', 't_unit', 't0_mm',
                              'device', 'clock_channel', 'npts', 'setpoint', 'tip_velocity_um/s', 'afm_sampling_ms',
                              'afm_angle_deg', 'tapping_frequency_Hz', 'ref_mirror_frequency_Hz',
                              'trion_version', 'neaclient_version', 'neaserver_version']

    def prepare(self):
        logger.info('Preparing Scan')
        self.date = datetime.now()
        self.name = self.date.strftime('%y%m%d-%H%M%S_') + self.acquisition_mode.value
        self.connect()

        if self.parent_scan is None:
            filename = self.name + '.h5'
            logger.info(f'Creating scan file: {filename}')
            self.file = h5py.File(filename, 'w')
        else:  # acts as a separate file, for all that we care about:
            self.file = self.parent_scan.file.create_group(self.identifier)
        self.file.create_group('afm_data')  # afm data (xyz, amp, phase) tracked during acquisition
        self.file.create_group('daq_data')  # chunks of daq data, saved with current afm data as attributes
        self.file.create_group('nea_data')  # data returned by NeaScan API

        if self.t_unit is not None:
            if self.t0_mm is not None:
                self.delay_stage.reference = self.t0_mm
            logger.info(f'Moving to delay position: {self.t} {self.t_unit}')
            self.delay_stage.move_abs(val=self.t, unit=self.t_unit)
            self.delay_stage.wait_for_stage()
            self.delay_stage.log_status()

    def connect(self):
        """
        connect to all relevant devices, or inherit from outer scope
        """
        if self.parent_scan is None:
            logger.info('Connecting')
            self.afm = NeaSNOM()
            if self.t_unit is not None:
                logger.info('Initializing delay stage')
                self.delay_stage = DLStage()
                ret = self.delay_stage.prepare()
                if ret is not True:
                    raise RuntimeError('Error preparing delay stage')
        else:
            logger.info('Inheriting connections from parent scan')
            self.afm = self.parent_scan.afm
            self.delay_stage = self.parent_scan.delay_stage
        self.neaclient_version = self.afm.nea_mic.ClientVersion
        self.neaserver_version = self.afm.nea_mic.ServerVersion

    def disconnect(self):
        """
        disconnect from all relevant devices if not inherited
        """
        if self.parent_scan is None:
            logger.info('Disconnecting')
            self.afm.disconnect()
            self.file.close()
            if self.delay_stage is not None:
                self.delay_stage.disconnect()

    @abstractmethod
    def routine(self):
        """
        acquisition routine, specific for each scan class
        """

    def start(self):
        """
        starts the acquisition
        """
        try:
            self.routine()
        finally:
            self.disconnect()
        
    def export(self):
        """
        Export xr.Datasets in self.afm_data and self.nea_data to hdf5 file. Collect and export metadata.
        """
        if self.afm_data is not None:
            logger.info('Saving tracked AFM data')
            xr_to_h5_datasets(ds=self.afm_data, group=self.file['afm_data'])
        if self.nea_data is not None:
            logger.info('Saving NeaScan data')
            xr_to_h5_datasets(ds=self.nea_data, group=self.file['nea_data'])

        logger.info('Collecting metadata')
        metadata = {}
        # ToDo collect metadata from lab book
        for m in self.metadata_list:
            if m in self.__dict__.keys():
                metadata[m] = self.__dict__[m]
            else:
                metadata[m] = None
        metadata['date'] = self.date.strftime('%Y-%m-%d_%H:%M:%S')
        metadata['acquisition_time_s'] = str(self.stop_time - self.date)
        metadata['modulation'] = self.modulation.value
        metadata['signals'] = [s.value for s in self.signals]
        metadata['acquisition_mode'] = self.acquisition_mode.value
        for k, v in metadata.items():
            if v is not None:
                logger.debug(f'writing metadata ({k}: {v})')
                self.file.attrs[k] = v


class ContinuousScan(BaseScan):
    def __init__(self, npts: int, setpoint: float, **kwargs):
        """
        Parameters
        ----------
        npts: int
            number of samples from the DAQ that are saved in one chunk
        setpoint: float
            AFM setpoint when engaged
        **kwargs:
            keyword arguments that are passed to BaseScan.__init__()
        """
        super().__init__(**kwargs)
        self.ctrl = None
        self.buffer = None
        if npts > 100_000:
            logger.error(f'npts was reduced to max chunk size of 100_000. npts={npts} was passed.')
            npts = 100_000
        self.npts = npts
        self.setpoint = setpoint

    def connect(self):
        super().connect()
        self.ctrl = DaqController(self.device, clock_channel=self.clock_channel)
        self.buffer = CircularArrayBuffer(vars=self.signals, size=100_000)
        self.ctrl.setup(buffer=self.buffer)

    def disconnect(self):
        super().disconnect()
        self.ctrl.close()

    def acquire(self):
        logger.info('Starting acquisition')
        excess = 0
        chunk_idx = 0
        afm_tracking = []
        while not self.afm.scan.IsCompleted:
            print(f'Chunk no {chunk_idx}. Scan progress: {self.afm.scan.Progress * 100:.2f} %', end='\r')
            current = self.afm.get_current()
            afm_tracking.append([chunk_idx] + list(current))

            n_read = excess
            while n_read < self.npts:
                sleep(0.001)
                n = self.ctrl.reader.read()
                n_read += n
            excess = n_read - self.npts
            data = self.buffer.get(n=self.npts, offset=excess)
            self.file['daq_data'].create_dataset(str(chunk_idx), data=data, dtype='float32')
            chunk_idx += 1
        print('\nAcquisition complete')
        logger.info('Acquisition complete')  # ToDo this is still a bit off. Clean up at some point.
        self.stop_time = datetime.now()

        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']
        for i, c in enumerate(tracked_channels[1:]):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0]})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array(self.t)})
            self.afm_data[c] = da
        # ToDo neadata should maybe be transformed in nea_ctr.py
        if self.acquisition_mode == Scan.continuous_image:
            self.nea_data = to_numpy(self.nea_data)
            self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in self.nea_data.items()}
            # ToDo get coordinates from NeaScan
            self.nea_data = xr.Dataset(self.nea_data)

    def routine(self):
        self.prepare()
        self.afm.engage(self.setpoint)
        self.ctrl.start()
        self.nea_data = self.afm.start()
        self.acquire()
        self.export()


class NoiseScan(ContinuousScan):
    def __init__(self, sampling_seconds: float, signals: None, x_target=None, y_target=None,
                 npts: int = 5_000, setpoint: float = 0.8):
        """
        Parameters
        ----------
        signals: iterable
            list of Signals. If None, the signals will be determined by modulation and chopped
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
        self.afm_sampling_milliseconds = 0.4
        self.x_res, self.y_res = int(sampling_seconds * 1e3 / self.afm_sampling_milliseconds // 1), 1
        self.x_size, self.y_size = 0, 0
        self.afm_angle = 0

    def prepare(self):
        super().prepare()
        if self.x_target is None or self.y_target is None:
            self.x_target, self.y_target = self.afm.nea_mic.TipPositionX, self.afm.nea_mic.TipPositionY
        self.afm.prepare_image(self.modulation, self.x_target, self.y_target, self.x_size, self.y_size,
                               self.x_res, self.y_res, self.afm_angle, self.afm_sampling_milliseconds, serpent=True)

    def acquire(self):
        # ToDo this could be merged at some time with parent class
        logger.info('Starting scan')
        excess = 0
        daq_data = []
        while not self.afm.scan.IsCompleted:
            print(f'Scan progress: {self.afm.scan.Progress * 100:.2f} %', end='\r')
            n_read = excess
            while n_read < self.npts:  # this slicing seems redundant. Someone should rewrite this.
                sleep(.001)
                n = self.ctrl.reader.read()
                n_read += n
            excess = n_read - self.npts
            daq_data.append(self.buffer.get(n=self.npts, offset=excess))

        daq_data = np.vstack(daq_data)
        self.file['daq_data'].create_dataset('1', data=daq_data, dtype='float32')  # just one chunk / pixel

        logger.info('Acquisition complete')
        self.nea_data = to_numpy(self.nea_data)
        self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in self.nea_data.items()}
        self.nea_data = xr.Dataset(self.nea_data)
        logger.info('Scan complete')
        self.stop_time = datetime.now()
