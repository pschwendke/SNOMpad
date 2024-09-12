import inspect
from abc import ABC, abstractmethod

from ..demodulation import shd, pshet
from ..utility.signals import Scan, Demodulation, Signals
from ..file_handlers.hdf5 import ReadH5Acquisition, H5DemodulationFile
from ..file_handlers.trion import ReadTrionAcquisition


class BaseScanDemod(ABC):
    """ Base class to load, demodulate, plot, and export acquired data.
    """
    def __init__(self, filename: str):
        # define these first because they __del__ will try to close them
        self.scan_file = None
        self.demod_file = None
        # load scan file
        if filename[-3:] == '.h5':
            self.scan_file = ReadH5Acquisition(filename)
        elif filename[-3:] == '.nc':
            self.scan_file = ReadTrionAcquisition(filename)
        else:
            raise NotImplementedError(f'Filetype not supported: {filename}')
        self.afm_data = self.scan_file.afm_data
        self.nea_data = self.scan_file.nea_data
        self.daq_data = self.scan_file.daq_data
        self.metadata = self.scan_file.metadata
        self.name = self.metadata['name']
        self.mode = Scan[self.metadata['acquisition_mode']]
        self.signals = [Signals[s] for s in self.metadata['signals']]
        self.modulation = Demodulation[self.metadata['modulation']]
        # attrs to store demod data
        self.demod_data = None
        self.demod_filetype = 'hdf5'
        self.demod_cache = {}

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
        if self.scan_file is not None:
            self.scan_file.close_file()
        if self.demod_file is not None:
            self.demod_file.close_file()

    def demod_params(self, params: dict, kwargs: dict) -> dict:
        """ collects demod parameters from arguments and function defaults to describe cached demod data
        """
        param_list = ['tap_res', 'ref_res', 'chopped', 'ratiometry', 'tap_correction', 'binning', 'pshet_demod',
                      'max_order', 'r_res', 'z_res', 'direction', 'line_no', 'trim_ratio', 'demod_npts']
        demod_func = [shd, pshet][[Demodulation.shd, Demodulation.pshet].index(self.modulation)]
        signature = inspect.signature(demod_func)
        parameters = params.copy()  # parameters stored in demod_data dataset before demodulation
        parameters.update({k: v.default for k, v in signature.parameters.items() if k in param_list})  # func defaults
        parameters.update({k: v for k, v in kwargs.items() if k in param_list})  # arguments passed to demod function
        return parameters

    def open_demod_file(self):
        """ Creates or opens a file to read and write demodulated data
        """
        if self.demod_filetype == 'hdf5':
            self.demod_file = H5DemodulationFile(name=self.name)
        else:
            raise NotImplementedError(f'Demod file filetype: {self.demod_filetype} is not implemented')
        self.demod_file.load_demod_data(cache=self.demod_cache)
        self.demod_file.write_metadata(metadata=self.metadata)

    def cache_to_file(self):
        """ Write demod_cache to file
        """
        if self.demod_file is None:
            self.open_demod_file()
        self.demod_file.write_demod_data(cache=self.demod_cache)

    def add_metadata(self, key: str, value):
        self.metadata[key] = value
        self.demod_file.write_metadata(metadata=self.metadata)

    @abstractmethod
    def demod(self):
        """ Collect and demodulate raw data, to produce NF images, retraction curves, spectra etc.
        """

    @abstractmethod
    def plot(self):
        """ Plot (and save) demodulated data, e.g. images or curves
        """


class BaseDemodReader(ABC):
    """ Base class do open, view, and modify demod files.
    """
    def __init__(self, filename: str):
        if filename[-9:] == '_demod.h5':
            self.file = H5DemodulationFile(name=filename)
        else:
            raise NotImplementedError(f'Filetype not supported: {filename}')
        self.metadata = self.file.load_metadata()
        self.demod_data = None
        self.demod_cache = {}
        self.file.load_demod_data(cache=self.demod_cache)
        self.index_cache()

    def __repr__(self) -> str:
        return f'SNOMpad demodulation file reader: {self.file.filename}'

    def __str__(self) -> str:
        ret = 'Metadata:\n'
        for k, v in self.metadata.items():
            ret += f'{k}: {v}\n'
        return ret

    def __del__(self):
        self.file.close_file()

    def add_metadata(self, key: str, value):
        self.metadata[key] = value
        self.file.write_metadata(metadata=self.metadata)

    def write_demod_data(self):
        """ Write demod_cache to file
        """
        self.file.write_demod_data(cache=self.demod_cache)

    def index_cache(self):
        indexed = {}
        for i, (key, dset) in enumerate(self.demod_cache.items()):
            indexed[i] = dset
        self.demod_cache = indexed

    def show_cache(self):
        out = 'Cached demod data:\n'
        for i, dset in self.demod_cache.items():
            out += f'{i}\n'
            params = dset.attrs['demod_params']
            for k, v in params.items():
                out += f'\u0009{k}: {v}\n'
        print(out)

    def select(self, index: int):
        self.demod_data = self.demod_cache[index]

    @abstractmethod
    def plot(self):
        """ Plot (and save) demodulated data, e.g. images or curves
        """
