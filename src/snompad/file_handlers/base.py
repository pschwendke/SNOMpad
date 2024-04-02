# base classes for reading and writing of Acquisition data to ensure compatibility
from abc import ABC, abstractmethod


class AcquisitionReader(ABC):
    """ base class to read scan data and return xarrays for afm and nea data, and a dict for daq data.
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
