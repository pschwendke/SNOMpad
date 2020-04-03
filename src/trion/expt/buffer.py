#buffer.py: Acquisition buffers
"""
Buffers: Manage the output buffer.
    Handle opening, sync (flushing), closing, expansion, truncation. Must track
    its current index to enable reading by other classes without involving the
    reader. Must handle two read modes: manually put data and 
    provide view into buffer.
    The destination buffer may also be the entrance into an analysis pipeline...
"""
from abc import ABCMeta, abstractmethod # There go my dreams of keeping this vanilla...

class AbstractBuffer(meta=ABCMeta):
    """Defines the API for Buffer objects."""
    def __init__(self):
        self.current = 0

    @abstractmethod
    def open(self, name, mode: str): # not sure about the arguments
        """Open a buffer."""
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def sync(self):
        pass

    @abstractmethod
    def expand(self, val: int):
        """
        Expand dataset.
        
        Not sure about the value: should we expand 'to' or expand 'by', or...
        """
        pass

    @abstractmethod
    def truncate(self):
        """
        Remove unused space.
        """
        pass

    # may not be an abstract method: potentially useless for h5py...
    @abstractmethod 
    def writing_view(self, len): # I don't like the names of these...
        """
        Provide a view into underlying buffer, for direct writing.
        """
        pass

    @abstractmethod 
    def reading_view(self, len): # I don't like the names of these...
        """
        Provide a view into underlying buffer for direct reading.
        """
        pass

    @abstractmethod 
    def put(self, data): # maybe we should overload __setitem__?
        """
        Put data into buffer. Handles expansion if necessary
        """
        pass

    @abstractmethod
    def get(self, len, vars=None): # maybe we should overload __getitem__?
        """
        Get last values from buffer. Optionally select a subset of columns.
        """
        pass

# First subclass: a rotating numpy buffer