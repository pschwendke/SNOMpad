import logging
import h5py
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep

from trion.analysis.experiment import Measurement, load
from trion.analysis.signals import Scan, Demodulation, Signals
from trion.expt.daq import DaqController
from trion.expt.buffer import CircularArrayBuffer
from trion.expt.nea_ctrl import NeaSNOM, to_numpy
from trion.expt.dl_ctrl import DLStage
from trion.__init__ import __version__

logger = logging.getLogger(__name__)


class BaseScan(ABC):
    def __init__(self, modulation: str, signals=None, device='Dev1', clock_channel='pfi0',
                 delay_position_mm=None, chopped=False):
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
        delay_position_mm: float
            if not None, delay stage will be moved to this position before scan
        chopped: bool
            if True, chop signal will be acquired
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
        self.delay_position_mm = delay_position_mm
        self.chopped = chopped
        self.acquisition_mode = None  # should be defined in subclass __init__
        self.afm_data = None
        self.nea_data = None
        self.date = None
        self.name = None
        self.file = None
        self.filename = None
        self.neaclient_version = None
        self.neaserver_version = None
        self.stop_time = None
        self.afm = None
        self.delay_stage = None
        self.metadata_list = ['name', 'date', 'user', 'sample', 'tip', 'acquisition_mode', 'acquisition_time_s',
                              'modulation', 'signals', 'chopped',
                              'light_source', 'probe_color_nm', 'pump_color_nm', 'probe_FWHM_nm', 'pump_FWHM_nm',
                              'probe_power_mW', 'pump_power_mW',
                              'delay_position_mm', 't0_mm',
                              'x_size', 'y_size', 'z_size', 'x_center', 'y_center', 'x_res', 'y_res', 'z_res',
                              'x_start', 'x_stop', 'y_start', 'y_stop', 'linescan_res', 'xy_unit',
                              'device', 'clock_channel', 'npts', 'setpoint', 'tip_velocity_um/s', 'afm_sampling_ms',
                              'afm_angle_deg', 'tapping_frequency_Hz', 'ref_mirror_frequency_Hz',
                              'trion_version', 'neaclient_version', 'neaserver_version']

    def prepare(self):
        logger.info('Preparing Scan')
        self.connect()
        self.date = datetime.now()
        self.name = self.date.strftime('%y%m%d-%H%M%S_') + self.acquisition_mode.value

        if self.delay_position_mm is not None:
            logger.info('Initializing delay stage')
            ret = self.delay_stage.prepare()
            if ret is True:
                logger.info(f'Moving to delay position: {self.delay_position_mm} mm')
                self.delay_stage.move_abs(val=self.delay_position_mm, unit='mm')
                self.delay_stage.wait_for_stage()
            self.delay_stage.log_status()

        logger.info(f'Creating scan file: {self.filename}')
        self.filename = self.name + '.h5'
        self.file = h5py.File(self.filename, 'w')
        self.file.create_group('afm_data')  # afm data (xyz, amp, phase) tracked during acquisition
        self.file.create_group('daq_data')  # chunks of daq data, saved with current afm data as attributes
        self.file.create_group('nea_data')  # data returned by NeaScan API

    def connect(self):
        """
        connect to all relevant devices
        """
        logger.info('Connecting')
        self.afm = NeaSNOM()
        self.neaclient_version = self.afm.nea_mic.ClientVersion
        self.neaserver_version = self.afm.nea_mic.ServerVersion
        if self.delay_position_mm is not None:
            self.delay_stage = DLStage()

    def disconnect(self):
        """
        disconnect from all relevant devices
        """
        logger.info('Disconnecting')
        self.stop_time = datetime.now()
        self.afm.disconnect()
        if self.delay_stage is not None:
            self.delay_stage.disconnect()
        self.file.close()

    @abstractmethod
    def start(self) -> Measurement:
        """
        starts the acquisition, collects data in hdf5 file, returns Measurement object
        """

    def export(self):
        """
        Export xr.Datasets in self.afm_data and self.nea_data to hdf5 file. Collect and export metadata.
        """
        def xr_to_h5_dataset(ds: xr.Dataset, group: h5py.Group):
            for dim, coord in ds.coords.items():  # create dimension scales
                group[dim] = coord.values
                group[dim].make_scale(dim)
            for ch in ds:
                da = ds[ch]
                dset = group.create_dataset(name=ch, data=da.values)  # data
                dset.attrs = da.attrs  # copy all metadata / attributes
                for dim in da.coords.keys():  # attach dimension scales
                    n = da.get_axis_num(dim)
                    dset.dims[n].attach_scale(group[dim])

        if self.afm_data is not None:
            logger.info('Saving tracked AFM data')
            xr_to_h5_dataset(ds=self.afm_data, group=self.file['afm_data'])
        if self.nea_data is not None:
            logger.info('Saving NeaScan data')
            xr_to_h5_dataset(ds=self.nea_data, group=self.file['nea_data'])

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
        self.file.attrs = metadata


class ContinuousScan(BaseScan):
    def __init__(self, npts: int, setpoint: float, **kwargs):
        super().__init__(**kwargs)
        self.ctrl = None
        self.buffer = None
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

        afm_tracking = np.array(afm_tracking)
        self.afm_data = xr.Dataset()
        tracked_channels = ['idx', 'x', 'y', 'z', 'amp', 'phase']
        for i, c in enumerate(tracked_channels[1:]):
            da = xr.DataArray(data=afm_tracking[:, i+1], dims='idx', coords={'idx': afm_tracking[:, 0]})
            if self.delay_position_mm is not None:
                da = da.expand_dims(dim={'t': np.array(self.delay_position_mm)})
            self.afm_data[c] = da
        if self.acquisition_mode == Scan.continuous_image:
            self.nea_data = to_numpy(self.nea_data)
            self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in self.nea_data.items()}
            # ToDo get coordinates from NeaScan
            self.nea_data = xr.Dataset(self.nea_data)

    def start(self) -> Measurement:
        try:
            self.prepare()
            self.afm.engage(self.setpoint)
            self.ctrl.start()
            logger.info('Starting scan')
            self.nea_data = self.afm.start()
            self.acquire()
        finally:
            self.disconnect()

        self.export()
        logger.info('Scan complete')
        return load(self.filename)


#  ToDo: rewrite this
# class NoiseScan(BaseScan):
#     def __init__(self, signals: Iterable[Signals], sampling_seconds: float,
#                  x_target=None, y_target=None, npts: int = 5_000, setpoint: float = 0.8):
#         super().__init__(signals, modulation=Demodulation.shd)
#         self.x_target, self.y_target = x_target, y_target
#         self.npts = npts
#         self.setpoint = setpoint
#
#         self.acquisition_mode = Scan.noise_sampling
#         self.afm_sampling_milliseconds = 0.4
#         self.x_res, self.y_res = int(sampling_seconds * 1e3 / self.afm_sampling_milliseconds // 1), 1
#         self.x_size, self.y_size = 0, 0
#         self.afm_angle = 0
#         self.ctrl = None
#         self.buffer = None
#         self.acquire_daq = False
#         if Signals.tap_x in self.signals:
#             self.acquire_daq = True
#             self.daq_data_filename = None
#
#     def prepare(self):
#         super().prepare()
#         if self.x_target is None or self.y_target is None:
#             self.x_target, self.y_target = self.afm.nea_mic.TipPositionX, self.afm.nea_mic.TipPositionY
#         self.afm.prepare_image(self.modulation, self.x_target, self.y_target, self.x_size, self.y_size,
#                                self.x_res, self.y_res, self.afm_angle, self.afm_sampling_milliseconds, serpent=True)
#
#     def connect(self):
#         super().connect()
#         if self.acquire_daq:
#             self.ctrl = DaqController(self.device, clock_channel=self.clock_channel)
#             self.buffer = CircularArrayBuffer(vars=self.signals, size=100_000)
#             self.ctrl.setup(buffer=self.buffer)
#
#     def disconnect(self):
#         super().disconnect()
#         if self.acquire_daq:
#             self.ctrl.close()
#
#     def acquire(self):
#         if self.acquire_daq:
#             self.ctrl.start()
#             excess = 0
#             daq_data = []
#             while not self.afm.scan.IsCompleted:
#                 print(f'Scan progress: {self.afm.scan.Progress * 100:.2f} %', end='\r')
#
#                 # ToDo: maybe an ExtendingArrayBuffer of sufficient size would be better. skip the loop.
#                 n_read = excess
#                 while n_read < self.npts:
#                     sleep(.001)
#                     n = self.ctrl.reader.read()
#                     n_read += n
#                 excess = n_read - self.npts
#                 daq_data.append(self.buffer.get(n=self.npts, offset=excess))
#
#             daq_data = np.vstack(daq_data)
#             self.file['daq_data'].create_dataset('1', data=daq_data, dtype='float32')  # just one chunk / pixel
#         else:
#             while not self.afm.scan.IsCompleted:
#                 sleep(.1)
#
#         logger.info('Acquisition complete')
#         self.nea_data = to_numpy(self.nea_data)
#
#     def start(self) -> Measurement:
#         try:
#             self.prepare()
#             self.afm.engage(self.setpoint)
#             logger.info('Starting scan')
#             self.nea_data = self.afm.start()
#             self.acquire()
#         finally:
#             self.disconnect()
#
#         self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in self.nea_data.items()}
#         self.nea_data = xr.Dataset(self.nea_data)
#
#         self.export()
#         logger.info('Scan complete')
#         return load(self.filename)
