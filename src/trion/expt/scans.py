import os
import logging
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep

from trion.analysis.io import export_data
from trion.expt.daq import DaqController
from trion.expt.buffer import CircularArrayBuffer
from trion.analysis.signals import Scan
from trion.expt.nea_ctrl import NeaSNOM, to_numpy
from trion.__init__ import __version__

logger = logging.getLogger(__name__)


class BaseScan(ABC):
    def __init__(self, signals, mod, device='Dev1', clock_channel='pfi0'):
        self.signals = signals
        self.mod = mod
        self.device = device
        self.clock_channel = clock_channel
        self.trion_version = __version__
        self.afm_data = None
        self.start_time = None
        self.data_folder = None
        self.neaclient_version = None
        self.neaserver_version = None
        self.stop_time = None
        self.afm = None
        self.afm_data = None
        self.acquisition_mode = None
        self.metalist = ['x_size', 'y_size', 'z_size', 'x_res', 'y_res', 'z_res', 'x_center', 'y_center', 'setpoint',
                         'tip_velocity', 'afm_sampling_time', 'afm_angle', 'device', 'clock_channel', 'npts',
                         'data_folder', 'trion_version', 'neaclient_version', 'neaserver_version']

    def prepare(self):
        logger.info('Preparing Scan')
        self.connect()
        self.start_time = datetime.now()

        logger.info('Creating data folder')
        self.data_folder = self.start_time.strftime('%y%m%d-%H%M%S_pixel_files')
        try:
            os.mkdir(self.data_folder)
        except FileExistsError:
            logger.warning('Data directory already exists. Previously recorded files will be overwritten.')

    @abstractmethod
    def connect(self):
        """
        connect to all relevant devices
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        disconnect from all relevant devices
        """
        pass

    def export(self):
        logger.info('Saving AFM data')
        self.afm_data.attrs['date'] = self.start_time.strftime('%Y-%m-%dT%H:%M:%S')
        self.afm_data.attrs['acquisition_time'] = str(self.stop_time - self.start_time)
        self.afm_data.attrs['modulation'] = self.mod.value
        self.afm_data.attrs['signals'] = [s.value for s in self.signals]
        self.afm_data.attrs['acquisition_mode'] = self.acquisition_mode.value
        for k, v in self.__dict__.items():
            if k in self.metalist and v is not None:
                self.afm_data.attrs[k] = v

        if self.acquisition_mode in [Scan.stepped_image, Scan.continuous_image]:
            # ToDo: check if units are still in accordance to NeaSNOM object
            self.afm_data.attrs['x_offset'] = (self.x_center - self.x_size / 2) * 1E-6
            self.afm_data.attrs['y_offset'] = (self.y_center - self.y_size / 2) * 1E-6

        filename = self.start_time.strftime('%y%m%d-%H%M%S_') + f'{self.acquisition_mode.value}.nc'
        self.afm_data.to_netcdf(filename)


class SteppedScan(BaseScan):
    def connect(self):
        logger.info('Connecting')
        self.afm = NeaSNOM()
        self.neaclient_version = self.afm.nea_mic.ClientVersion
        self.neaserver_version = self.afm.nea_mic.ServerVersion

    def disconnect(self):
        logger.info('Disconnecting')
        self.afm.disconnect()
        self.stop_time = datetime.now()


class ContinuousScan(BaseScan):
    def __init__(self, signals, mod):
        super().__init__(signals, mod)
        self.ctrl = None
        self.buffer = None
        self.npts = None

    def connect(self):
        logger.info('Connecting')
        self.afm = NeaSNOM()
        self.neaclient_version = self.afm.nea_mic.ClientVersion
        self.neaserver_version = self.afm.nea_mic.ServerVersion
        self.ctrl = DaqController(self.device, clock_channel=self.clock_channel)
        self.buffer = CircularArrayBuffer(vars=self.signals, size=100_000)
        self.ctrl.setup(buffer=self.buffer)

    def disconnect(self):
        logger.info('Disconnecting')
        self.afm.disconnect()
        self.stop_time = datetime.now()
        self.ctrl.close()

    def acquire(self):
        # TODO fix progress bars
        logger.info('Starting scan')
        self.ctrl.start()
        nea_data = self.afm.start()

        excess = 0
        pix_count = 0
        afm_tracking = []
        while not self.afm.scan.IsCompleted:
            print(f'Pixel no {pix_count}. Scan progress: {self.afm.scan.Progress * 100:.2f} %', end='\r')
            afm_tracking.append(list(self.afm.get_current()))

            n_read = excess
            while n_read < self.npts:
                sleep(0.001)
                n = self.ctrl.reader.read()
                n_read += n
            excess = n_read - self.npts
            data = self.buffer.get(n=self.npts, offset=excess)

            export_data(f'{self.data_folder}/pixel_{pix_count:05d}.npz', data, self.signals)
            pix_count += 1
        print('\nScan complete')

        afm_tracking = np.array(afm_tracking)
        tracked_data = xr.DataArray(data=afm_tracking, dims=('px', 'ch'),
                                    coords={'px': np.arange(pix_count), 'ch': ['x', 'y', 'Z', 'M1A', 'M1P']})
        if self.acquisition_mode == Scan.continuous_image:
            nea_data = to_numpy(nea_data)

        return tracked_data, nea_data
