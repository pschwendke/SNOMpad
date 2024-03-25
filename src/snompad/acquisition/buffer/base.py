from abc import ABC, abstractmethod
from enum import Enum, auto


class Overfill(Enum):
    """ Control behavior on overfilling
    """
    raise_ = auto()
    clip = auto()
    ignore = auto()


class AbstractBuffer(ABC):
    def __init__(self, *a, vars, overfill=Overfill.raise_, **kw):
        """ Defines the API for Buffer objects.

        PARAMETERS
        ----------
        vars : iterable of Signals
            Variables contained in databuffer
        overfill : Overfill enum
            Behavior on overfill.

        ATTRIBUTES
        ----------
        i : int, property
            Index of next value to be written
        size : int, read-only, property
            Number of valid data points
        buf_size : int, read-only property
            Size of current buffer.
        """
        super().__init__(*a, **kw)
        self.i = 0
        self._vrs = vars
        self.overfill = overfill

    @property
    @abstractmethod
    def size(self) -> int:
        """ Current buffer size.
        """
        pass

    @property
    @abstractmethod
    def buf_size(self) -> int:
        pass

    @abstractmethod
    def close(self):
        """ Close the buffer.
        """
        pass

    @abstractmethod
    def finish(self) -> 'AbstractBuffer':
        """ Cleanup after acquisition finished.
        """
        pass

    @abstractmethod
    def put(self, data) -> int:
        """ Put data into buffer. Handles expansion if necessary.

        PARAMETERS
        ----------
        data : array_like of shape (N,M)
            Data to be added. N points with M variables.

        RETURNS
        -------
        n : int
            number of points written
        """
        pass

    @abstractmethod
    def get(self, n: int, offset: int = 0):
        """ Get valid values from buffer.

        This method does not return values that may be present in the buffer
        but were not written (ie: allocated space).

        PARAMETERS
        ----------
        n : int
            Number of points to return.
        offset : int
            Start position

        RETURNS
        -------
        data : np.ndarray of shape (N,M)
            Data points of up to `len` points and `M` columns.
        """
        pass

    def tail(self, n):
        """ Get `len` last values from buffer.
        """
        offset = max(0, self.i - n)
        return self.get(n, offset)

    def head(self, n):
        """ Get `n` values from start of buffer.
        """
        return self.get(n)

    @property
    def vars(self):
        """ Get the vars names present in buffer (ie: columns)
        """
        return self._vrs
