import tempfile
import h5py


class H5Buffer:
    """ This is just a wrapper to put a dictionary of 32-bit data arrays in a temporary hdf5 file.
    """
    def __init__(self):
        tf = tempfile.TemporaryFile()
        self.file = h5py.File(tf, 'w')

    def __getitem__(self, key):
        return self.file[key]

    def __setitem__(self, key, value):
        self.file.create_dataset(key, data=value, dtype='float32')  # maybe 16 bit is even enough???

    def __del__(self):
        self.cleanup()

    def keys(self):
        return self.file.keys()

    def items(self):
        return self.file.items()

    def cleanup(self):
        self.file.close()  # garbage collector closes tf
