#buffer.py: Acquisition buffers
"""
Buffers: Manage the output buffer.
    Handle opening, sync (flushing), closing, expansion, truncation. Must track
    its current index to enable reading by other classes without involving the
    reader. Must handle two read modes: manually put data and 
    provide view into buffer.
    The destination buffer may also be the entrance into an analysis pipeline...
"""
from abc import ABCMeta, ABC, abstractmethod # There go my dreams of keeping this vanilla...
import numpy as np

class AbstractBuffer(ABC):
    """Defines the API for Buffer objects."""
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

    # @abstractmethod
    # def close(self):
    #     pass

    # @abstractmethod
    # def sync(self):
    #     pass

    # @abstractmethod
    # def expand(self, val: int):
    #     """
    #     Expand dataset.
        
    #     Not sure about the value: should we expand 'to' or expand 'by', or...
    #     """
    #     pass

    # @abstractmethod
    # def truncate(self):
    #     """
    #     Remove unused space.
    #     """
    #     pass

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
    def put(self, data): # maybe we should overload __setitem__?
        """
        Put data into buffer. Handles expansion if necessary
        """
        pass

    @abstractmethod
    def get(self, len, offset=0): # maybe we should overload __getitem__?
        """
        Get last values from buffer.
        """
        pass

    @property
    def vars(self):
        """Get the vars names present in buffer (ie: columns)"""
        return self._vrs
    

class CircularArrayBuffer(AbstractBuffer):
    """
    A simple circular buffer using a numpy array.
    """
    def __init__(self, *, max_size, **kw):
        super().__init__(**kw)
        self.buf = np.full((max_size, len(self.vars)), np.nan)

    @property
    def size(self):
        return self.buf.shape[0]

    def expand(self, *a, **kw):
        raise TypeError("Rotating buffer cannot be expanded.")

    def truncate(self, *a, **kw):
        raise TypeError("Rotating buffer cannot be expanded.")

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

    def get(self, len, offset=0):
        r0 = self.i+offset
        return np.roll(self.buf, -r0, axis=0)[len::-1,:]
        
    def writing_view(self, len):
        raise NotImplementedError

    def reading_view(self, len):
        raise NotImplementedError



        

# Factory function?
