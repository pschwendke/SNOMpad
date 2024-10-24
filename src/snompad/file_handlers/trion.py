# code to read TRION data from .nc files
import numpy as np
import xarray as xr
import logging
from glob import glob

from snompad.file_handlers.base import AcquisitionReader

logger = logging.getLogger(__name__)


class ReadTrionAcquisition(AcquisitionReader):
    """ class to read scans that were saved to xarray .nc files
    """
    def __repr__(self):
        return f'TRION acuisition file reader: {self.filename}'

    def open_file(self):
        logger.info(f'Opening file: {self.filename}')
        self.afm_data = xr.load_dataset(self.filename)
        try:
            self.afm_data = self.afm_data['__xarray_dataarray_variable__']
        except KeyError:
            pass
        if 'name' not in self.afm_data.attrs.keys():
            name = self.afm_data.attrs['date'].replace('T', '_').replace(':', '').replace('-', '')
            name += '_' + self.afm_data.attrs['acquisition_mode']
            self.afm_data.attrs['name'] = name
        for k, v in self.afm_data.attrs.items():
            self.metadata[k] = v

    def close_file(self):
        logger.info(f'Closing file: {self.filename}')
        self.afm_data.close()

    def read_data(self):
        """ reads afm and nea data from file and stores xr.Datasets. A wrapper is built around DAQ data to save memory.
        Use .load_daq_data() to return dict of all daq data chunks.
        """
        logger.info('AFM data already read')

        class DaqDataDict(dict):
            def __getitem__(self, item):
                filename = super().__getitem__(item)
                npz = np.load(filename)
                data = np.vstack([i for i in npz.values()]).T
                return data

        logger.info('Constructing wrapper around DAQ data')
        directory = '/'.join(self.filename.split('/')[:-1]) + '/'.join(self.filename.split('\\')[:-1]) + '/'
        data_folder = directory + self.afm_data.attrs['data_folder']
        filenames = glob(data_folder + '/*.npz')
        self.daq_data = DaqDataDict()
        for f in filenames:
            fname = f.split('/')[-1].split('\\')[-1]
            idx = fname.split('_')[1].split('.')[0]
            self.daq_data[int(idx)] = f

    def load_daq_data(self):
        """ load all acquired DAQ data chunks into memory

        RETURNS
        -------
        daq_data: dict
            keys are integer chunk index and values are np.ndarray
        """
        logger.info('Reading DAQ data from file')
        directory = '/'.join(self.filename.split('/')[:-1]) + '/'.join(self.filename.split('\\')[:-1]) + '/'
        data_folder = directory + self.afm_data.attrs['data_folder']
        filenames = glob(data_folder + '/*.npz')
        daq_data = {}
        for f in filenames:
            fname = f.split('/')[-1].split('\\')[-1]
            idx = fname.split('_')[1].split('.')[0]

            npz = np.load(f)
            data = np.vstack([i for i in npz.values()]).T
            daq_data[int(idx)] = data
        return daq_data
