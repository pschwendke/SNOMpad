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
    def __init__(self, modulation: str, signals=None, pump_probe=False, setpoint: float = 0.8, npts: int = 5_000,
                 device='Dev1', clock_channel='pfi0',
                 metadata: dict = None, t: float = None, t_unit: str = None, t0_mm: float = None,
                 parent_scan=None, delay_idx=None):
        """
        Parameters
        ----------
        modulation: str
            either 'shd' or 'pshet'
        signals: iterable
            list of Signals. If None, the signals will be determined by modulation and pump_probe
        pump_probe: bool
            if True, chop signal will be acquired and delay stage will be connected and moved to t position.
            t and t_unit must be passed as well.
        device: str
            should be 'Dev1' for now.
        clock_channel: str
            should be 'pfi0', as labelled on the DAQ
        setpoint: float
            AFM setpoint when engaged
        npts: int
            number of samples from the DAQ that are saved in one chunk
        metadata: dict
            dictionary of metadata that will be written to acquisition file, if key is in self.metadata_keys.
            Specified variables, e.g. self.name have higher priority than passed metadata.
        t: float
            if not None, delay stage will be moved to this position before scan
        t_unit: str in ['m', 'mm', 's', 'ps', 'fs']
            unit of t value. Needs to ge vien when t is given
        t0_mm: float
            position for t_0 on the delay stage. Needs to be given when t_unit is 's', 'ps', or 'fs'
        parent_scan: BaseScan
            when a BaseScan object is passed, no file is created, but data saved into parent's file
        delay_idx: str or int
            identifier for this scan in the scope of the parent_scan, e.g. 'delay_pos_001'
        """
        try:
            self.modulation = Demodulation[modulation]
        except KeyError:
            logger.error(f'BaseScan only takes "shd" and "pshet" as modulation values. "{modulation}" was passed.')
        if signals is None:
            sig_list = {'shd': ['sig_a', 'tap_x', 'tap_y'], 'pshet': ['sig_a', 'tap_x', 'tap_y', 'ref_x', 'ref_y']}
            signals = [Signals[s] for s in sig_list[modulation]]
            if pump_probe:
                signals.append(Signals['chop'])
        if npts > 200_000:
            logger.error(f'npts was reduced to max chunk size of 200_000. npts={npts} was passed.')
            npts = 200_000
        self.npts = npts
        self.signals = signals
        self.device = device
        self.clock_channel = clock_channel
        self.trion_version = __version__
        self.t = t
        self.t_unit = t_unit
        self.t0_mm = t0_mm
        self.setpoint = setpoint
        self.pump_probe = pump_probe
        self.parent_scan = parent_scan
        self.delay_idx = delay_idx
        self.connected = False
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
        self.log_handler = None
        self.metadata = metadata
        self.metadata_keys = ['name', 'date', 'user', 'sample', 'tip', 'acquisition_mode', 'acquisition_time',
                              'modulation', 'signals', 'pump_probe',
                              'light_source', 'probe_color_nm', 'pump_color_nm', 'probe_FWHM_nm', 'pump_FWHM_nm',
                              'probe_power_mW', 'pump_power_mW',
                              'x_size', 'y_size', 'z_size', 'x_center', 'y_center', 'x_res', 'y_res', 'z_res',
                              'x_start', 'x_stop', 'y_start', 'y_stop', 'linescan_res', 'xy_unit',
                              't', 't_start', 't_stop', 't_unit', 't0_mm', 'delay_idx', 'scan_type',
                              'device', 'clock_channel', 'npts', 'setpoint', 'tip_velocity_um/s', 'afm_sampling_ms',
                              'afm_angle_deg', 'tapping_frequency_Hz', 'tapping_amp_nm', 'ref_mirror_frequency_Hz',
                              'trion_version', 'neaclient_version', 'neaserver_version']

    def prepare(self):
        self.date = datetime.now()
        self.name = self.date.strftime('%y%m%d-%H%M%S_') + self.acquisition_mode.value
        if self.parent_scan is None:
            self.log_handler = logging.FileHandler(filename=self.name + '.log', encoding='utf-8')
            self.log_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
            logging.getLogger().addHandler(self.log_handler)

            logger.info('Preparing Scan')
            filename = self.name + '.h5'
            logger.info(f'Creating scan file: {filename}')
            self.file = h5py.File(filename, 'w')
        else:  # acts as a separate file, for all that we care about:
            logger.info(f'Creating sub file for delay position: {self.delay_idx}')
            self.file = self.parent_scan.file.create_group(f'delay_idx_{self.delay_idx}')
        self.connected = self.connect()
        self.file.create_group('afm_data')  # afm data (xyz, amp, phase) tracked during acquisition
        self.file.create_group('daq_data')  # chunks of daq data, saved with current afm data as attributes
        self.file.create_group('nea_data')  # data returned by NeaScan API

        if self.pump_probe:
            if self.t0_mm is not None:
                self.delay_stage.reference = self.t0_mm
            if self.t is not None:
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
            self.afm.set_pshet(self.modulation)
            if self.pump_probe:
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
        return True

    def disconnect(self):
        """
        disconnect from all relevant devices if not inherited
        """
        if self.parent_scan is None:
            logger.info('Disconnecting')
            # if self.afm is not None:
            self.afm.disconnect()
            self.afm = None
            # if self.file is not None:
            self.file.close()
            if self.delay_stage is not None:
                self.delay_stage.disconnect()
                self.delay_stage = None
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler.close()
        return False

    def __del__(self):
        logger.debug('BaseScan.__del__()')
        if self.connected:
            self.disconnect()

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
            self.export()
        except KeyboardInterrupt:
            logger.error('Acquisition was interrupted by user')
        finally:
            self.connected = self.disconnect()
        
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
        metadata_collector = {}
        for m in self.metadata_keys:
            if m in self.__dict__.keys():
                metadata_collector[m] = self.__dict__[m]
            elif self.metadata is not None and m in self.metadata.keys():
                metadata_collector[m] = self.metadata[m]
            else:
                metadata_collector[m] = None
        metadata_collector['date'] = self.date.strftime('%Y-%m-%d_%H:%M:%S')
        metadata_collector['acquisition_time'] = str(self.stop_time - self.date)
        metadata_collector['modulation'] = self.modulation.value
        metadata_collector['signals'] = [s.value for s in self.signals]
        metadata_collector['acquisition_mode'] = self.acquisition_mode.value
        for k, v in metadata_collector.items():
            if v is not None:
                logger.debug(f'writing metadata ({k}: {v})')
                self.file.attrs[k] = v


class ContinuousScan(BaseScan):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs:
            keyword arguments that are passed to BaseScan.__init__()
        """
        super().__init__(**kwargs)
        self.ctrl = None
        self.buffer = None

    def connect(self):
        super().connect()
        self.ctrl = DaqController(self.device, clock_channel=self.clock_channel)
        self.buffer = CircularArrayBuffer(vars=self.signals, size=100_000 + self.npts)
        self.ctrl.setup(buffer=self.buffer)
        return True

    def disconnect(self):
        super().disconnect()
        self.ctrl.close()
        self.ctrl = None
        return False

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
        logger.info('Acquisition complete')
        self.stop_time = datetime.now()

        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']
        for i, c in enumerate(tracked_channels[1:]):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0].astype('int')})
            if self.t is not None:
                da = da.expand_dims(dim={'t': np.array([self.t])})
            self.afm_data[c] = da
        # ToDo neadata should maybe be transformed in nea_ctr.py
        if self.acquisition_mode == Scan.continuous_image:
            self.nea_data = to_numpy(self.nea_data)
            self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in self.nea_data.items()}
            # ToDo get coordinates from NeaScan
            self.nea_data = xr.Dataset(self.nea_data)
        if self.acquisition_mode in [Scan.continuous_retraction, Scan.continuous_line]:
            # ToDo neadata must be turned into xr.dataset
            self.nea_data = None

    def routine(self):
        self.prepare()
        self.afm.engage(self.setpoint)
        self.ctrl.start()
        self.nea_data = self.afm.start()
        self.acquire()
