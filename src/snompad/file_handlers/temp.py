import tempfile
import h5py
import logging

logger = logging.getLogger(__name__)


class H5Buffer:
    """ This is just a wrapper to put a dictionary of 32-bit data arrays in a temporary hdf5 file.
    """
    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile()
        logger.info(f'H5Buffer: Created temporary file: {self.temp_file.name}')
        self.h5_file = h5py.File(self.temp_file, 'w')

    def __getitem__(self, key):
        return self.h5_file[key]

    def __setitem__(self, key, value):
        self.h5_file.create_dataset(str(key), data=value, dtype='float32')  # ToDo: maybe 16 bit is even enough???
        # self.h5_file.flush()  # this is probably not necessary

    def __del__(self):
        logger.debug('H5Buffer: __del__()')
        self.cleanup()

    def keys(self):
        return self.h5_file.keys()

    def items(self):
        return self.h5_file.items()

    def cleanup(self):
        logger.info('H5Buffer: closing buffer')
        self.h5_file.close()
        self.temp_file.close()
