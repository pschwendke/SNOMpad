#buffer.py: Acquisition buffers
"""
Buffers: Manage the output buffer.
    Handle opening, sync (flushing), closing, expansion, truncation. Must track
    its current index to enable reading by other classes without involving the
    reader. Must handle two read modes: manually put data and 
    provide view into buffer.
    The destination buffer could also be the entrance into an analysis pipeline...

NOT THREADSAFE.
"""
from abc import ABC, abstractmethod
import logging
import numpy as np
import os.path as pth
from ..analysis.io import export_data
logger = logging.getLogger(__name__)


class AbstractBuffer(ABC):
    """Defines the API for Buffer objects.
    """
    def __init__(self, *, vars):
        self.i = 0
        self._vrs = vars

    @property
    @abstractmethod
    def size(self):
        """Current buffer size"""
        pass


    # those are only for file-based...
    # @abstractmethod
    # def open(self, name, mode: str): # not sure about the arguments
    #     """Open a buffer."""
    #     pass

    @abstractmethod
    def close(self) -> 'AbstractBuffer':
        pass

    @abstractmethod
    def finish(self) -> 'AbstractBuffer':
        """
        Cleanup after acquisition finished.
        """
        pass

    # TODO: implement a "save" method.

    # may not be an abstract method: potentially useless for h5py...
    # @abstractmethod 
    # def writing_view(self, len): # I don't like the names of these...
    #     """
    #     Provide a view into underlying buffer, for direct writing.
    #     """
    #     pass

    # @abstractmethod 
    # def reading_view(self, len): # I don't like the names of these...
    #     """
    #     Provide a view into underlying buffer for direct reading.
    #     """
    #     pass

    @abstractmethod 
    def put(self, data) -> 'AbstractBuffer':
        """
        Put data into buffer. Handles expansion if necessary
        """
        pass

    @abstractmethod
    def get(self, len: int):
        """
        Get last values from buffer.
        """
        pass

    @property
    def vars(self):
        """Get the vars names present in buffer (ie: columns)"""
        return self._vrs

    def export(self, filename, len: int=None):
        """
        Export buffer contents to file.

        Parameters
        ----------
        filename: str or path-like
            Output file name
        len: int or None (default)
            Number of points to export, counting from end. If len is None
            (default), exports the entire buffer.
        """
        self.finish()
        len = self.size if (len is None) else len
        export_data(filename, self.get(len), self.vars)


class ArrayBuffer(AbstractBuffer):
    def __init__(self, *, init_size, dtype=np.float, **kw):
        super().__init__(**kw)
        self.buf = np.full((init_size, len(self.vars)), np.nan, dtype=dtype)

    @property
    def size(self):
        return self.buf.shape[0]


class CircularArrayBuffer(ArrayBuffer):
    """
    A simple circular buffer using a numpy array.
    """
    def __init__(self, *, max_size, **kw):
        super().__init__(**{"init_size" : max_size, **kw})

    def put(self, data):
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j <= self.size:
            self.buf[self.i:j,:] = data[:,:]
        else: # rotate
            # lengths of first and second segments (`pre` and `post`)
            pre_len = self.size - self.i
            post_len = j - self.size
            self.buf[self.i:, :] = data[:pre_len, :]
            self.buf[:post_len, :] = data[pre_len:, :]

        self.i = (self.i + n) % self.size
        return self

    def get(self, len):
        r0 = self.i
        # rotate to bring oldest value to the start of array.
        return np.roll(self.buf, -r0, axis=0)[-len:,:]

    def close(self):
        return self

    def finish(self):
        return self


class ExtendingArrayBuffer(ArrayBuffer):
    """
    An array that extends as required.
    """
    def __init__(self, *, chunk_size=20_000, max_size=2_000_000, **kw):
        super().__init__(**{"init_size" : chunk_size, **kw})
        self.chunk_size = chunk_size
        self.max_size = max_size
    
    def put(self, data):
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j > self.size:
            self.expand().put(data)
        else:
            self.buf[self.i:j,:] = data[:,:]
            self.i = j
        return n

    def fill(self, data):
        """Fill with as much as possible"""
        logger.debug("Filling array.")
        avail = self.size-self.i
        self.buf[self.i:self.size,:] = data[:avail,:]
        self.i += avail
        return avail

    def get(self, len):
        # returns by view
        r0 = self.i-1
        return self.buf[r0-len:r0,:]

    def finish(self):
        self.truncate()
        return self    
    
    def close(self):
        self.finish()
        return self
    
    def truncate(self):
        logger.debug(f"Truncating buffer to: {self.i}")
        self.buf = self.buf[:self.i,:]
        return self

    def expand(self, by=None):
        if by is None:
            by=self.chunk_size
        if self.size + by > self.max_size:
            raise ValueError(f"Cannot expand buffer beyond max_size: {self.size+by} > {self.max_size}.")
        logger.debug(f"Expanding buffer to: {self.size+by}")
        self.buf = np.vstack(
            (self.buf,
             np.full((by, len(self.vars)), np.nan),
            ),
        )
        return self
        

# Factory function?
