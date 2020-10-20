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
from math import ceil

import numpy as np
import os.path as pth
from ..analysis.io import export_data
logger = logging.getLogger(__name__)


class AbstractBuffer(ABC):
    def __init__(self, *, vars):
        """
        Defines the API for Buffer objects.

        Parameters
        ----------
        vars : iterable of Signals


        Attributes
        ----------
        i : int
            Index of next value to be written
        size : int, read-only
            Current size of buffer.
        """
        self.i = 0
        self._vrs = vars

    @property
    @abstractmethod
    def size(self):
        """
        Current buffer size.
        """
        pass

    @abstractmethod
    def close(self) -> 'AbstractBuffer':
        """
        Close the buffer.
        """
        pass

    @abstractmethod
    def finish(self) -> 'AbstractBuffer':
        """
        Cleanup after acquisition finished.
        """
        pass


    @abstractmethod 
    def put(self, data) -> 'AbstractBuffer':
        """
        Put data into buffer. Handles expansion if necessary.

        Parameters
        ----------
        data : array_like of shape (N,M)
            Data to be added. N points with M variables.
        """
        pass

    @abstractmethod
    def get(self, len: int, offset: int=0):
        """
        Get valid values from buffer.

        This method does not return values that may be present in the buffer
        but were not written (ie: allocated space).

        Parameters
        ----------
        len : int
            Number of points to return.
        offset : int
            Start position

        Returns
        -------
        data : np.ndarray of shape (N,M)
            Data points of up to `len` points and `M` columns.
        """
        pass

    def tail(self, len):
        """
        Get `len` last values from buffer.
        """
        offset = self.i-len
        return self.get(len, offset)

    def head(self, len):
        """
        Get `len` values from start of buffer.
        """
        return self.get(len)

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
        export_data(filename, self.tail(len), self.vars)


class ArrayBuffer(AbstractBuffer):
    def __init__(self, *, size, dtype=np.float, **kw):
        """
        Base class for buffers based on numpy arrays.

        Parameters
        ----------
        size : int
            Initial size of the underlying buffer
        dtype : type
            dtype for the underlying buffer

        Attributes
        ----------
        buf : np.ndarray
            Underlying data buffer. Depending on the exact concrete class, this
            buffer may contain invalid data.
        buf_size : int, read-only
            Size of the underlying buffer (number of points).
        """
        super().__init__(**kw)
        self.buf = np.full((size, len(self.vars)), np.nan, dtype=dtype)

    @property
    def buf_size(self):
        return self.buf.shape[0]


class CircularArrayBuffer(ArrayBuffer):
    def __init__(self, *a, **kw):
        """
        A simple circular buffer using a numpy array.

        Parameters
        ----------
        vars : iterable of Signals
            Signals contained in the buffer (ie: columns)
        size : int
            Size of the circular buffer.
        dtype : type
            dtype of the buffer. Defaults to np.float.

        Attributes
        ----------
        i : int
            Index of next value to be written
        size : int, read-only
            Current size of buffer. This is the number of valid data points
        buf : (N, M) np.ndarray
            Underlying buffer with a total capacity of N points for M columns.
        buf_size : int, read-only
            Size of the underlying buffer (number of points).
        n_written : int, read-only
            Total number of points written.
        """
        super().__init__(**kw)
        self._nwritten = 0

    @property
    def n_written(self):
        return self._nwritten

    @property
    def size(self):
        return min(self.n_written, self.buf_size)

    def put(self, data):
        if data.shape[0] <= self.buf_size:
            self._put_single(data)
        else:
            nchunks = ceil(data.shape[0]/self.buf_size)
            for chunk in np.array_split(data, nchunks, axis=0):
                self._put_single(chunk)
        return self

    def _put_single(self, data):
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j <= self.buf_size:
            self.buf[self.i:j,:] = data[:,:]
        else: # rotate
            # lengths of first and second segments (`pre` and `post`)
            pre_len = self.buf_size - self.i
            post_len = j - self.buf_size
            self.buf[self.i:, :] = data[:pre_len, :]
            self.buf[:post_len, :] = data[pre_len:, :]

        self._nwritten += n
        self.i = j % self.buf_size

    def get(self, len: int, offset=0):
        # truncate to show only valid values
        len = min(len, self.size)
        # rotate to bring oldest value to the start of array.
        return np.roll(self.buf, self.size-self.i-offset, axis=0)[:len,:]

    def close(self):
        return self

    def finish(self):
        return self


class ExtendingArrayBuffer(ArrayBuffer):
    def __init__(self, *, max_size=2_000_000, **kw):
        """
        An array that extends as required.
        """
        defaults = dict(size=20_000)
        kw = {**defaults, **kw}
        super().__init__(**kw)
        self.chunk_size = kw["size"]
        self.max_size = max_size

    @property
    def size(self):
        return self.i
    
    def put(self, data):
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j > self.buf_size:
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

    def get(self, len, offset=0):
        return self.buf[offset:offset+len,:]
    #
    # def tail(self, len):
    #     # returns by view
    #     r0 = self.i # there I fixed it...
    #     return self.buf[r0-len:r0,:]

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
