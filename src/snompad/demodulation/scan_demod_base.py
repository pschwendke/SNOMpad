# base class to load scann data acquired with TRION or SNOMpad
import numpy as np
import xarray as xr
import h5py
import inspect
from abc import ABC, abstractmethod

from . import shd, pshet
from ..utility.signals import Scan, Demodulation, Signals
from ..file_handlers.hdf5 import ReadH5Acquisition, WriteH5Demodulation, ReadH5Demodulation, h5_to_xr_dataset, xr_to_h5_datasets
from ..file_handlers.trion import ReadTrionAcquisition


class BaseScanDemod(ABC):
    """ Base class to load, demodulate, plot, and export acquired data.
    """
    def __init__(self, filename: str):
        if filename[-3:] == '.h5':
            self.scan = ReadH5Acquisition(filename)
        elif filename[-3:] == '.nc':
            self.scan = ReadTrionAcquisition(filename)
        else:
            raise NotImplementedError(f'filetype not supported: {filename}')
        self.afm_data = self.scan.afm_data
        self.nea_data = self.scan.nea_data
        self.daq_data = self.scan.daq_data
        self.metadata = self.scan.metadata
        self.name = self.metadata['name']
        self.mode = Scan[self.metadata['acquisition_mode']]
        self.signals = [Signals[s] for s in self.metadata['signals']]
        self.modulation = Demodulation[self.metadata['modulation']]
        self.demod_data = None
        self.demod_cache = {}
        self.demod_file = None

    def __str__(self) -> str:
        ret = 'Metadata:\n'
        for k, v in self.metadata.items():
            ret += f'{k}: {v}\n'
        if self.afm_data is not None:
            ret += '\nAFM data:\n'
            ret += self.afm_data.__str__()
        if self.nea_data is not None:
            ret += '\nNeaScan data:\n'
            ret += self.nea_data.__str__()
        return ret

    def __repr__(self) -> str:
        return f'<SNOMpad measurement: {self.name}>'

    def __del__(self):
        self.scan.close_file()
        self.close_demod_file()

    def close_demod_file(self):
        if self.demod_file is not None:
            self.demod_file.close()

    def demod_params(self, old_params: dict, kwargs: dict):
        """ collects demod parameters from arguments and function default values
        """
        param_list = ['tap_res', 'ref_res', 'chopped', 'normalize', 'tap_correction', 'binning', 'pshet_demod',
                      'max_order', 'r_res', 'z_res', 'direction', 'line_no', 'trim_ratio', 'demod_npts']
        # ToDo: max_order seems not to be tracked. Check this.
        demod_func = [shd, pshet][[Demodulation.shd, Demodulation.pshet].index(self.modulation)]
        signature = inspect.signature(demod_func)
        parameters = old_params.copy()  # parameters stored in demod_data dataset before demodulation
        parameters.update({k: v.default for k, v in signature.parameters.items() if k in param_list})  # func defaults
        parameters.update({k: v for k, v in kwargs.items() if k in param_list})  # arguments passed to demod functions
        hash_key = ''.join(sorted([str(parameters[p]) for p in param_list if p in parameters.keys()]))
        parameters['hash_key'] = hash(hash_key)
        return parameters

    def demod_filename(self, filename: str = 'auto'):
        """ creates or opens a file to read and write demodulated data

        Parameters
        ----------
        filename: str
            When file exists, it will be loaded into self.demod_cache. If it does not exist it will be created. Auto
            generates file name from scan name.
        """
        if filename == 'auto':
            filename = self.name + '_demod.h5'
        try:
            self.demod_file = h5py.File(filename, 'r+')
            if self.demod_file.attrs['name'] != self.name:
                raise RuntimeError(f'The demod file was created for the scan {self.demod_file.attrs["name"]}'
                                   f'You have loaded the scan file {self.name}')
            for key, group in self.demod_file.items():
                if key not in self.demod_cache.keys():
                    self.demod_cache[key] = h5_to_xr_dataset(group=group)
        except FileNotFoundError:
            self.demod_file = h5py.File(filename, 'w-')
            self.demod_file.attrs['name'] = self.name

    def cache_to_file(self):
        """ Write demod_cache to file
        """
        if self.demod_file is not None:
            for key, dset in self.demod_cache.items():
                if key not in self.demod_file.keys():
                    self.demod_file.create_group(key)
                    xr_to_h5_datasets(ds=dset, group=self.demod_file[key])

    @abstractmethod
    def demod(self):
        """ Collect and demodulate raw data, to produce NF images, retraction curves, spectra etc.
        """

    @abstractmethod
    def plot(self):
        """ Plot (and save) demodulated data, e.g. images or curves
        """
