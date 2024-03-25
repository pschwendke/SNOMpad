import logging
import numpy as np
from . import AbstractBuffer, Overfill

logger = logging.getLogger(__name__)


class ArrayBuffer(AbstractBuffer):
    def __init__(self, *, size, dtype=float, **kw):
        """ Base class for buffers based on numpy arrays.

        PARAMETERS
        ----------
        vars : iterable of Signals
            Signals contained in the buffer (ie: columns)
        size : int
            Initial size of the underlying buffer
        dtype : type
            dtype for the underlying buffer
        overfill: Overfill enum
            Behavior on buffer overfill. Defaults to Overfill.raise_

        ATTRIBUTES
        ----------
        i : int, property
            Index of next value to be written
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
        """ A simple circular buffer using a numpy array. The rotating buffer cannot overfill. It's "overfill" parameter
        is ignored. It is nevertheless always set to `Overfill.ignore`

        PARAMETERS
        ----------
        vars : iterable of Signals
            Signals contained in the buffer (ie: columns)
        size : int
            Initial size of the underlying buffer
        dtype : type
            dtype for the underlying buffer

        ATTRIBUTES
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
        kw["overfill"] = Overfill.ignore
        super().__init__(*a, **kw)
        self._nwritten = 0

    @property
    def n_written(self):
        return self._nwritten

    @property
    def size(self):
        return min(self.n_written, self.buf_size)

    def _flat_idx(self, row_idx):
        """ Generate a wrapping flat index from the row indices
        """
        assert np.asanyarray(row_idx).ndim == 1
        arr_idx = np.array(np.meshgrid(row_idx, np.arange(len(self.vars)), indexing="ij")).reshape((2, -1))
        return np.ravel_multi_index(arr_idx, self.buf.shape, mode="wrap")

    def put(self, data, overfill=Overfill.ignore) -> int:
        n = data.shape[0]
        row_idx = self.i + np.arange(n)
        idx = self._flat_idx(row_idx)
        # data is larger than the entire buffer, we drop the unnecessary part
        m = min(n, self.buf_size)
        self.buf.flat[idx[-m*data.shape[1]:]] = data[-m:, :]
        self.i += n
        self._nwritten += n
        return n

    def get(self, n: int, offset=0):
        # truncate to show only valid values
        start = self.i - self.size + offset
        n = min(n, self.size-offset)  # if we don't include -offset, we can roll beyond the oldest value
        stop = start + n
        row_idx = np.arange(start, stop)
        return self.buf.flat[self._flat_idx(row_idx)].reshape((n, len(self.vars)))

    def tail(self, n):
        """ Get `len` last values from buffer.
        """
        n = min(n, self.size)
        row_idx = np.arange(self.i-n, self.i)
        return self.buf.flat[self._flat_idx(row_idx)].reshape((n, len(self.vars)))

    def close(self):
        pass

    def finish(self):
        return self


class ExtendingArrayBuffer(ArrayBuffer):
    def __init__(self, *, max_size=2_000_000, **kw):
        """ An array that extends as required.

        Parameters
        ----------
        vars : iterable of Signals
            Signals contained in the buffer (ie: columns)
        size : int
            Size of expansion chunks. Defaults to 20_000
        dtype : type
            dtype of the buffer. Defaults to np.float.
        overfill : Overfill enum
            Behavior on buffer overfill. Defaults to Overfill.raise_
        max_size: int
            Maximum size of buffer. Defaults to 2_000_000.

        Attributes
        ----------
        i : int, property
            Index of next value to be written
        size : int, read-only, property
            Number of valid data points
        buf : np.ndarray
            Underlying data buffer. Depending on the exact concrete class, this
            buffer may contain invalid data.
        buf_size : int, read-only
            Size of the underlying buffer (number of points).
        chunk_size : int
            Default size of expansion. D
        """
        defaults = dict(size=min(20_000, max_size))
        kw = {**defaults, **kw}
        super().__init__(**kw)
        self.chunk_size = kw["size"]
        self.max_size = max_size

    @property
    def size(self):
        return self.i

    def put(self, data) -> int:
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j > self.buf_size:
            try:
                self.expand()
            except ValueError as e:
                if not str(e).startswith("Cannot expand buffer beyond max_size"):
                    raise
                # we are overfilling
                if self.overfill is Overfill.clip:
                    r = self.expand_max().fill(data)
                elif self.overfill is Overfill.ignore:
                    return 0
                else:
                    raise
            else:
                r = self.put(data)
        else:
            self.buf[self.i:j, :] = data[:, :]
            self.i = j
            r = n
        return r

    def fill(self, data):
        """ Fill with as much as possible
        """
        logger.debug("Filling array.")
        avail = min(self.buf_size-self.i, data.shape[0])
        self.buf[self.i:self.i+avail, :] = data[:avail, :]
        self.i += avail
        return avail

    def get(self, n, offset=0):
        end = min(self.i, offset+n)
        return self.buf[offset:end, :]

    def finish(self):
        self.truncate()
        return self

    def close(self):
        self.finish()

    def truncate(self):
        logger.debug(f"Truncating buffer to: {self.i}")
        self.buf = self.buf[:self.i, :]
        return self

    def expand(self, by=None):
        """ Expand buffer by the given number of frames. By default, expand by `self.chunk_size`
        """
        if by is None:
            by = self.chunk_size
        if by < 0:
            raise ValueError("Cannot expand buffer by negative value.")
        if self.buf_size + by > self.max_size:
            raise ValueError(f"Cannot expand buffer beyond max_size: {self.size+by} > {self.max_size}.")
        logger.debug(f"Expanding buffer to: {self.size+by}")
        self.buf = np.vstack((self.buf, np.full((by, len(self.vars)), np.nan),),)
        return self

    def expand_max(self):
        return self.expand(by=max(self.max_size - self.buf_size, 0))
