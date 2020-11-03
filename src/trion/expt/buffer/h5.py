"""
Acquisition buffer based on H5Py.

Try to keep this flat and simple, I don't think links etc.

Organization:
- root group must contain metadata:
    - buffer specification version
    - scan type (single point, approach curve, image, 3D, custom)
    - Scan info (dimensions etc)
    - acquistion parameters:
        - DAQ
            -device name
            - channels in use
            - limits
            - trigger channel
        - SNOM
            - none!
- root group contains only metadata, perhaps a few global results
- raw group: the complete data table
    - metadata:
        - column names
    - dataset  names: p{idx:06d}, each corresponds to a pixel
        - metadata:
            - pixel coordinates (x, y, z) (if SNOM is present)
            - pixel index (i, j, k...)

Start with a flat thing...
"""
import logging
import os.path as pth

import numpy as np
import h5py

from .base import AbstractBuffer, Overfill

logger = logging.getLogger(__name__)

h5py.get_config().track_order = True


# How do we handle the different frames?
# we could have them just be proxys for different datasets...
# we need a way to track the frame indices...
#
# we need to abstract out the scan!

class H5Buffer(AbstractBuffer):
    raw_fmt = "raw/frame{self.frame_idx:06d}"

    def __init__(self, *a,
                 fname=None,
                 mode="x",
                 frame_idx=0,
                 size=20_000,
                 dtype=np.float,
                 axes=None,
                 #chunk_size=5000,
                 h5_kw=None,
                 max_size = 2_000_000,
                 **kw):
        """
        Buffer backed by an H5Py file.
        """
        if pth.exists(fname):
            raise RuntimeError("Opening exisiting files is not supported.")
        super().__init__(*a, **kw)
        h5_kw = {} if h5_kw is None else h5_kw
        self.h5f = h5py.File(fname, mode, **h5_kw)
        self.h5f.attrs["version"] = "0.1"
        self.h5f.attrs["scan_type"] = "single_point"
        self.h5f.attrs["frame_idx"] = frame_idx

        self.chunk_size = size
        self.raw = self.h5f.create_group("/raw", track_order=True)
        self.raw.attrs["vars"] = [v.value for v in self._vrs]
        self.buf = self.h5f.create_dataset(
            self.raw_fmt.format(**locals()),
            shape=(size, len(self.vars)),
            dtype=dtype,
            fillvalue=np.nan,
            chunks=(self.chunk_size, len(self.vars)),
            maxshape=(max_size, len(self.vars)),
        )

    @property
    def size(self):
        return self.i

    @property
    def buf_size(self) -> int:
        return self.buf.shape[0]

    @property
    def max_size(self) -> int:
        return self.buf.maxshape[0]

    def put(self, data) -> int:
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j > self.buf_size:
            try:
                self.expand()
            except ValueError as e:
                if "dimension cannot exceed the existing maximal size" in  str(e):
                    raise
                # we are overfilling
                if self.overfill is Overfill.clip:
                    r = self.expand_max().fillup(data)
                elif self.overfill is Overfill.ignore:
                    return 0
                else:
                    raise
            else:
                r = self.put(data)
        else:
            self.buf[self.i:j,:] = data[:,:]
            self.i = j
            r = n
        return r

    def expand(self, by=None):
        by = by or self.chunk_size
        self.buf.resize(self.buf_size+by, axis=0)
        return self

    def expand_max(self):
        self.buf.resize(self.max_size-self.buf_size, axis=0)
        return self

    def fillup(self, data):
        avail = self.max_size - self.i
        return self.put(data[:avail])

    def get(self, len, offset=0):
        return self.buf[offset:offset+len,:]

    def finish(self) -> AbstractBuffer:
        self.truncate()
        # prepare next dataset?
        return self

    def close(self):
        self.finish()
        self.h5f.close()

    def truncate(self) -> AbstractBuffer:
        self.buf = self.buf[:self.i,:]
        return self

    @property  # good candidate for an ABC MultiFrameBuffer
    def frame_idx(self) -> int:
        return self.h5f.attrs["frame_idx"]

    @frame_idx.setter
    def frame_idx(self, value):
        self.h5f.attrs["frame_idx"] = value

    def set_frame(self, idx) -> "H5Buffer":
        # good candidate for an ABC MultiFrameBuffer
        raise NotImplementedError()

    def prepare_frame(self, idx) -> "H5Buffer":
        raise NotImplementedError()

    def next_frame(self):
        #
        raise NotImplementedError()


