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

from .base import AbstractBuffer

logger = logging.getLogger(__name__)

h5py.get_config().track_order = True


# How do we handle the different frames?
# we could have them just be proxys for different datasets...

class H5Buffer(AbstractBuffer):
    raw_fmt = "raw/pix{self.pix_idx:06d}"

    def __init__(self, *a,
                 fname=None,
                 mode="x",
                 pix_idx=0,
                 size=20_000,
                 dtype=np.float,
                 #chunk_size=5000,
                 h5_kw=None,
                 **kw):
        """
        Buffer backed by an H5Py file.
        """
        super().__init__(*a, **kw)
        if pth.exists(fname):
            raise RuntimeError("Opening exisiting files is not supported.")
        h5_kw = {} if h5_kw is None else h5_kw
        self.h5f = h5py.File(fname, mode, **h5_kw)
        self.h5f.attrs["version"] = "0.1"
        self.h5f.attrs["scan_type"] = "single_point"
        self.pix_idx = pix_idx
        self.chunk_size = size
        self.raw = self.h5f.create_group("/raw", track_order=True)
        self.raw.attrs["vars"] = [v.value for v in self._vrs]
        self.buf = self.h5f.create_dataset(
            self.raw_fmt.format(**locals()),
            shape=(size, len(self.vars)),
            dtype=dtype,
            chunks=(self.chunk_size, len(self.vars)),
            maxshape=(2_000_000, len(self.vars)),
        )

    @property
    def size(self):
        return self.i

    @property
    def buf_size(self) -> int:
        return self.buf.shape[0]

    def put(self, data) -> int:
        n = np.asarray(data).shape[0]
        j = self.i+n
        if j > self.buf_size:
            self.expand().put(data)
        else:
            self.buf[self.i:j,:] = data[:,:]
            self.i = j
        return n

    def expand(self, by=None):
        by = by or self.chunk_size
        self.buf.resize(self.buf.shape[0]+by, axis=0)
        return self

    def get(self, len, offset=0):
        return self.buf[offset:offset+len,:]

    def finish(self):
        self.truncate()
        # prepare next dataset?
        return self

    def close(self):
        self.finish()
        self.h5f.close()

    def truncate(self):
        self.buf = self.buf[:self.i,:]
        return self

