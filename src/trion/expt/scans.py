import os
import logging
import h5py
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep
from typing import Iterable

from trion.analysis.io import export_data
from trion.analysis.experiment import Measurement, Noise
from trion.expt.daq import DaqController
from trion.expt.buffer import CircularArrayBuffer
from trion.analysis.signals import Scan, Demodulation, Signals
from trion.expt.nea_ctrl import NeaSNOM, to_numpy
from trion.__init__ import __version__

logger = logging.getLogger(__name__)


class BaseScan(ABC):
    def __init__(self, signals, modulation, device='Dev1', clock_channel='pfi0'):
        self.signals = signals
        self.modulation = modulation
        self.device = device
        self.clock_channel = clock_channel
        self.trion_version = __version__
        self.chopped = None
        self.afm_data = None
        self.nea_data = None
        self.date = None
        self.file = None
        self.name = None
        self.neaclient_version = None
        self.neaserver_version = None
        self.stop_time = None
        self.afm = None
        self.acquisition_mode = None
        self.metadata_list = ['name', 'date', 'user', 'sample', 'tip', 'acquisition_mode', 'acquisition_time_s',
                              'modulation', 'signals', 'chopped',
                              'light_source', 'probe_color_nm', 'pump_color_nm', 'probe_FWHM_nm', 'pump_FWHM_nm',
                              'probe_power_mW', 'pump_power_mW',
                              'delay_position_mm', 't0_mm',
                              'x_size', 'y_size', 'z_size', 'x_center', 'y_center', 'x_res', 'y_res', 'z_res', 'xy_unit',
                              'device', 'clock_channel', 'npts', 'setpoint', 'tip_velocity_um/s', 'afm_sampling_ms',
                              'afm_angle_deg', 'tapping_frequency_Hz', 'ref_mirror_frequency_Hz',
                              'trion_version', 'neaclient_version', 'neaserver_version']

    def prepare(self):
        logger.info('Preparing Scan')
        self.connect()
        self.date = datetime.now()
        self.name = self.date.strftime('%y%m%d-%H%M%S_') + self.acquisition_mode.value

        filename = self.name + '.h5'
        logger.info(f'Creating scan file: {filename}')
        self.file = h5py.File(filename, 'w')
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

    def disconnect(self):
        """
        disconnect from all relevant devices
        """
        logger.info('Disconnecting')
        self.stop_time = datetime.now()
        self.afm.disconnect()
        self.file.close()

    @abstractmethod
    def start(self) -> Measurement:
        """
        starts the acquisition, collects data in self.afm_data, returns Measurement object
        """
        pass

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
                for k, v in da.attrs.items():  # copy all metadata / attributes
                    dset.attrs[k] = v
                for dim in da.coords.keys():  # attach dimension scales
                    n = da.get_axis_num(dim)
                    dset.dims[n].attach_scale(group[dim])

        if self.afm_data is not None:
            logger.info('Saving tracked AFM data')
            xr_to_h5_dataset(ds=self.afm_data, group=self.file['afm_data'])
        if self.nea_data is not None:
            logger.info('Saving NeaScan data')
            xr_to_h5_dataset(ds=self.nea_data, group=self.file['nea_data'])
        # ToDo export log

        logger.info('Collecting metadata')
        metadata = {}
        # ToDo collect metadata from lab book
        for m in self.metadata_list:
            if m in self.__dict__.keys():
                metadata[m] = self.__dict__[m]
            else:
                metadata[m] = None
        metadata['date'] = self.date.strftime('%Y-%m-%dT%H:%M:%S')
        metadata['acquisition_time_s'] = str(self.stop_time - self.date)
        metadata['modulation'] = self.modulation.value
        metadata['signals'] = [s.value for s in self.signals]
        metadata['acquisition_mode'] = self.acquisition_mode.value
        self.file.attrs = metadata


class ContinuousScan(BaseScan):
    def __init__(self, signals, modulation):
        super().__init__(signals, modulation)
        self.ctrl = None
        self.buffer = None
        self.npts = None

    def connect(self):
        super().connect()
        self.ctrl = DaqController(self.device, clock_channel=self.clock_channel)
        self.buffer = CircularArrayBuffer(vars=self.signals, size=100_000)
        self.ctrl.setup(buffer=self.buffer)

    def disconnect(self):
        super().disconnect()
        self.ctrl.close()

    def start(self) -> Measurement:
        pass

    def acquire(self):
        # TODO fix progress bars
        logger.info('Starting scan')
        self.ctrl.start()
        self.nea_data = self.afm.start()

        tracked_channels = ['x', 'y', 'z', 'M1A', 'M1P']
        excess = 0
        pix_count = 0
        afm_tracking = []
        while not self.afm.scan.IsCompleted:
            print(f'Pixel no {pix_count}. Scan progress: {self.afm.scan.Progress * 100:.2f} %', end='\r')
            current = self.afm.get_current()
            afm_tracking.append(list(current))

            n_read = excess
            while n_read < self.npts:
                sleep(0.001)
                n = self.ctrl.reader.read()
                n_read += n
            excess = n_read - self.npts
            data = self.buffer.get(n=self.npts, offset=excess)

            dset = self.file['daq_data'].create_dataset(f'{pix_count:05}', data=data)
            tracked = {ch: current[i] for i, ch in enumerate(tracked_channels)}
            dset.attrs = tracked

            pix_count += 1
        print('\nAcquisition complete')

        afm_tracking = np.array(afm_tracking)
        channels = ['x', 'y', 'z', 'M1A', 'M1P']
        index = np.arange(pix_count)
        self.afm_data = xr.Dataset()
        for i, c in enumerate(channels):
            da = xr.DataArray(data=afm_tracking[:, i], dims='idx', coords={'idx': index})
            self.afm_data[c] = da
        if self.acquisition_mode == Scan.continuous_image:
            self.nea_data = to_numpy(self.nea_data)
            self.nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in self.nea_data.items()}
            # ToDo get coordinates from NeaScan
            self.nea_data = xr.Dataset(self.nea_data)
            for k, v in self.nea_data.items():
                if k in ['Z', 'R-Z', 'M1A', 'R-M1A']:
                    v.attrs['z_unit'] = 'm'
                else:
                    v.attrs['z_unit'] = ''


class NoiseScan(BaseScan):
    def __init__(self, signals: Iterable[Signals], sampling_seconds: float,
                 x_target=None, y_target=None, npts: int = 5_000, setpoint: int = 0.8):
        super().__init__(signals, modulation=Demodulation.shd)
        self.x_target, self.y_target = x_target, y_target
        self.npts = npts
        self.setpoint = setpoint

        self.acquisition_mode = Scan.noise_sampling
        self.afm_sampling_milliseconds = 0.4
        self.x_res, self.y_res = int(sampling_seconds * 1e3 / self.afm_sampling_milliseconds // 1), 1
        self.x_size, self.y_size = 0, 0
        self.afm_angle = 0
        self.ctrl = None
        self.buffer = None
        self.acquire_daq = False
        if Signals.tap_x in self.signals:
            self.acquire_daq = True
            self.daq_data_filename = None

    def prepare(self):
        super().prepare()
        if self.x_target is None or self.y_target is None:
            self.x_target, self.y_target = self.afm.nea_mic.TipPositionX, self.afm.nea_mic.TipPositionY
        self.afm.prepare_image(self.modulation, self.x_target, self.y_target, self.x_size, self.y_size,
                               self.x_res, self.y_res, self.afm_angle, self.afm_sampling_milliseconds, serpent=True)

    def connect(self):
        super().connect()
        if self.acquire_daq:
            self.ctrl = DaqController(self.device, clock_channel=self.clock_channel)
            self.buffer = CircularArrayBuffer(vars=self.signals, size=100_000)
            self.ctrl.setup(buffer=self.buffer)

    def disconnect(self):
        super().disconnect()
        if self.acquire_daq:
            self.ctrl.close()

    def acquire(self) -> dict:
        # TODO fix progress bars
        nea_data = self.afm.start()

        if self.acquire_daq:
            self.ctrl.start()
            excess = 0
            daq_data = []
            while not self.afm.scan.IsCompleted:
                print(f'Scan progress: {self.afm.scan.Progress * 100:.2f} %', end='\r')

                # ToDo: maybe an ExtendingArrayBuffer of sufficient size would be better. skip the loop.
                n_read = excess
                while n_read < self.npts:
                    sleep(.001)
                    n = self.ctrl.reader.read()
                    n_read += n
                excess = n_read - self.npts
                daq_data.append(self.buffer.get(n=self.npts, offset=excess))

            daq_data = np.vstack(daq_data)
            self.file['daq_data'].create_dataset('00001', data=daq_data)  # just one chunk / pixel
        else:
            while not self.afm.scan.IsCompleted:
                sleep(.1)

        logger.info('Acquisition complete')
        nea_data = to_numpy(nea_data)
        return nea_data

    def start(self) -> Noise:
        try:
            self.prepare()
            self.afm.engage(self.setpoint)
            logger.info('Starting scan')
            nea_data = self.acquire()
        finally:
            self.disconnect()

        nea_data = {k: xr.DataArray(data=v, dims=('y', 'x')) for k, v in nea_data.items()}
        self.afm_data = xr.Dataset(nea_data)
        for k, v in self.afm_data.items():
            if k == ['z', 'r-z', 'M1A', 'r-M1A']:
                v.attrs['z_unit'] = 'm'
            else:
                v.attrs['z_unit'] = ''
        self.export()
        logger.info('Scan complete')

        return Noise(self.afm_data)