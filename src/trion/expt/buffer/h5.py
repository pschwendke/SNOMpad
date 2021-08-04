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
from enum import Enum

import numpy as np
import h5py

import attr

from .base import AbstractBuffer, Overfill
from ...analysis.experiment import Experiment

logger = logging.getLogger(__name__)

h5py.get_config().track_order = False


# How do we handle the different frames?
# we could have them just be proxys for different datasets...
# we need a way to track the frame indices...
#
# we need to abstract out the scan!

class H5Buffer(AbstractBuffer):
    raw_fmt = "raw/frame{idx:06d}"

    def __init__(self, *,
                 fname=None,
                 mode: str = "x",
                 frame_idx: int = 0,
                 size: int = 20_000,
                 dtype: type = float,
                 max_size: int = 2_000_000,
                 h5_kw: dict = None,
                 experiment: Experiment = None,
                 **kw):
        """
        Buffer backed by an H5Py file.

        Parameters
        ----------
        vars : iterable of Signals
            Signals contained in the buffer (ie: columns)
        size : int
            Size of h5py chunks. Defaults to 20_000
        dtype : type
            dtype of the buffer. Defaults to np.float.
        overfill: Overfill enum
            Behavior on buffer overfill. Defaults to Overfill.raise_
        fname : str or file-like
            H5py file name
        mode : str, one of "w", "r", "r+", "a", "w-" "x"
            Mode in which to open file. Refer to `h5py.File` documentation.
            Defaults to "x".
        frame_idx : positive int
            Index of next frame. Defaults to 0.
        max_size : positive int
            Maximum size of an frame. Defaults to 2_000_000
        h5_kw : dict
            Extra arguments file `h5py.File`.
        """
        if pth.exists(fname):
            raise RuntimeError("Opening exisiting files is not supported.")
        super().__init__(**kw)
        h5_kw = {} if h5_kw is None else h5_kw
        self.h5f = h5py.File(fname, mode, **h5_kw)
        self.h5f.attrs["version"] = "0.1"
        self.frame_idx = frame_idx
        self.buf = None
        self.dtype = dtype
        self.max_size = max_size

        self.experiment = experiment

        self.chunk_size = size
        self.raw = self.h5f.create_group("/raw", track_order=True)
        self.raw.attrs["vars"] = [v.value for v in self._vrs]
        self.prepare_frame(self.frame_idx)


    @property
    def experiment(self):  # TODO: add to test suite
        cfg = {n: self.h5f.attrs.get(n, None)
               for n in attr.fields(Experiment)}
        return Experiment.from_dict(**cfg)

    @experiment.setter
    def experiment(self, exp):
        for k, v in exp.to_dict().items():
            if isinstance(v, Enum):
                v = v.name
            self.h5f.attrs[k] = v

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
            try:
                self.expand()
            except ValueError as e:
                if not str(e).startswith("Unable to set extend dataset (dimension cannot exceed the existing maximal size"):
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
        by = self.chunk_size if by is None else by
        if by < 0:
            raise ValueError("Cannot expand buffer by negative value.")
        self.buf.resize(self.buf_size+by, axis=0)
        return self

    def expand_max(self):
        self.buf.resize(self.max_size-self.buf_size, axis=0)
        return self

    def fillup(self, data):
        avail = self.max_size - self.i
        return self.put(data[:avail])

    def get(self, len, offset=0):
        end = min(self.i, offset + len)
        return self.buf[offset:end,:]

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

    def prepare_frame(self, idx: int) -> "H5Buffer":
        # should we determine the metadata from self.experiment?
        # I don't think so, I think we should supply the metadata.
        logger.debug(f"Creating frame: {idx}")
        self.buf = self.h5f.create_dataset(
            self.raw_fmt.format(idx=idx),
            shape=(self.chunk_size, len(self.vars)),
            dtype=self.dtype,
            fillvalue=np.nan,
            chunks=(self.chunk_size, len(self.vars)),
            maxshape=(self.max_size, len(self.vars)),
        )
        self.i = 0
        self.frame_idx = idx
        return self


    def next_frame(self) -> "H5Buffer":
        next_frame_idx = self.frame_idx + 1
        self.prepare_frame(next_frame_idx)
        return self


