# base classes for reading and writing of Acquisition data to ensure compatibility
from abc import ABC, abstractmethod


class AcquisitionReader(ABC):
    """ Base class to read scan data and return xarrays for afm and nea data, and a dict for daq data.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.daq_data = None
        self.afm_data = None
        self.nea_data = None
        self.file = None
        self.metadata = {}
        self.open_file()
        self.read_data()

    def __del__(self):
        self.close_file()

    @abstractmethod
    def open_file(self):
        """ open file and read metadata
        """
        pass

    @abstractmethod
    def close_file(self):
        """ opened scan file should be closed
        """
        pass

    @abstractmethod
    def read_data(self):
        """ data contained in possible external files or file structures should be collected and presented
        """
        pass

    @abstractmethod
    def load_daq_data(self):
        """ all daq data should be loaded into a dictionary and stored in memory
        """
        pass


# class AcquisitionWriter(ABC):
#     """ we will need this if we have more than one filetype to write to
#     """


class DemodulationFile(ABC):
    """ Base class to read and write demodulated data from and to files
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        self.open_file()

    def __del__(self):
        self.close_file()

    def __repr__(self) -> str:
        return f'<SNOMpad demodulation file reader: {self.filename}>'

    @abstractmethod
    def open_file(self):
        pass

    @abstractmethod
    def close_file(self):
        pass

    @abstractmethod
    def load_demod_data(self, cache: dict):
        """ Demod data is loaded from file and placed in cache.
        """
        pass

    @abstractmethod
    def write_demod_data(self, cache: dict):
        """ Demod cache is written do demod file.
        """
        pass

    @abstractmethod
    def load_metadata(self) -> dict:
        """ Read metadata from file and return dict
        """
        pass

    @abstractmethod
    def write_metadata(self, metadata: dict):
        """ Write metadata dict to file
        """
        pass
