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
from .base import AbstractBuffer, Overfill
from .array import CircularArrayBuffer, ExtendingArrayBuffer
from .h5 import H5Buffer
