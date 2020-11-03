from abc import ABC, abstractmethod
from enum import Enum, auto

from trion.analysis.io import export_data


class Overfill(Enum):
    """Control behavior on overfilling."""
    raise_ = auto()
    clip = auto()
    ignore = auto()


class AbstractBuffer(ABC):
    def __init__(self, *a, vars, overfill=Overfill.raise_, **kw):
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
        super().__init__(*a, **kw)
        self.i = 0
        self._vrs = vars
        self.overfill = overfill

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Current buffer size.
        """
        pass

    @property
    @abstractmethod
    def buf_size(self) -> int:
        pass

    @abstractmethod
    def close(self):
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
    def put(self, data) -> int:
        """
        Put data into buffer. Handles expansion if necessary.

        Parameters
        ----------
        data : array_like of shape (N,M)
            Data to be added. N points with M variables.
        overfill : Overfill
            Behavior on overfill.

        Returns
        -------
        n : int
            number of points written
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
        offset = max(0, self.i-len)
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