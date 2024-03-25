# code to read and write SNOMpad data from and to hdf5 files

import numpy as np
import xarray as xr
import logging
import h5py

from snompad.utility.signals import Signals

logger = logging.getLogger(__name__)


def h5_to_xr_dataset(group: h5py.Group):
    """ takes an hdf5 group that has previously been converted from an xr.Dataset, and reproduces this xr.Dataset.
    Datasets labeled with _real and _imag are combined into one complex valued DataArray.
    """
    ds = xr.Dataset()
    ds.attrs = group.attrs
    for ch, dset in group.items():
        if ch[:11] == 'attr_group_':
            ds.attrs[ch[11:]] = {k: v for k, v in dset.attrs.items()}
        elif dset.dims[0].keys():
            values = np.array(dset)
            dims = [d.keys()[0] for d in dset.dims]
            da = xr.DataArray(data=values, dims=dims, coords={d: np.array(group[d]) for d in dims})
            da.attrs = dset.attrs
            ds[ch] = da
    return ds


def xr_to_h5_datasets(ds: xr.Dataset, group: h5py.Group):
    """ Takes an xr.Dataset and formats and writes its contents to an hdf5 group. Complex valued DataArrays are split
    into two datasets (_real, and _imag).
    """
    for dim, coord in ds.coords.items():  # create dimension scales
        group[dim] = coord.values
        group[dim].make_scale(dim)
    for ch in ds:
        da = ds[ch]
        if da.dtype == 'O':  # variable 'idx' is array of arrays, i.e. a np.ndarray of type 'object'
            dset = group.create_dataset(name=ch, shape=da.shape, dtype=h5py.vlen_dtype(int))
            for i in np.ndindex(da.shape):
                dset[i] = da.values[i]
        else:
            dset = group.create_dataset(name=ch, data=da.values)  # data
        for k, v in da.attrs.items():  # copy all metadata / attributes
            dset.attrs[k] = v
        for dim in da.coords.keys():  # attach dimension scales
            n = da.get_axis_num(dim)
            dset.dims[n].attach_scale(group[dim])
    for k, v in ds.attrs.items():
        if type(v) == dict:  # for nested attributes, i.e. dicts in dicts
            attr_group = group.create_group(name=f'attr_group_{k}')
            for key, val in v.items():
                attr_group.attrs[key] = val
        else:
            group.attrs[k] = v


class H5Acquisition:
    """ class to create and write acquired data into an hdf5 file
    """
    def __init__(self):
        self.file = None
        self.filename = None

    def __repr__(self):
        return f'<SNOMpad acquisition file: {self.filename}>'

    def __del__(self):
        self.close_file()

    def create_file(self, filename: str):
        """ create hdf5 file with scan name
        """
        logger.info(f'Creating scan file: {filename}')
        self.filename = filename
        self.file = h5py.File(filename, 'w')

    def close_file(self):
        """ close hdf5 file
        """
        logger.debug(f'Closing acquisition file: {self.filename}')
        self.file.close()

    def write_daq_data(self, data: dict):
        """ write dictionary of acquired daq chunks into group. Chunks are individual hdf5 datasets
        """
        logger.info('Saving acquired DAQ data')
        n_digits = int(np.log10(len(data.keys()))) + 1
        self.file.create_group('daq_data')
        for k, v in data.items():
            self.file['daq_data'].create_dataset(str(k).zfill(n_digits), data=v, dtype='float32')

    def write_afm_data(self, data: xr.Dataset):
        """ write xr.Dataset of afm data (xyz, amp, phase) tracked during acquisition into group
        """
        logger.info('Saving tracked AFM data')
        self.file.create_group('afm_data')
        xr_to_h5_datasets(ds=data, group=self.file['afm_data'])

    def write_nea_data(self, data: xr.Dataset):
        """ write xr.Dataset of NeaScan data returned by API after acquisition into group
        """
        logger.info('Saving NeaScan data')
        self.file.create_group('nea_data')
        xr_to_h5_datasets(ds=data, group=self.file['nea_data'])

    def write_metadata(self, key, val):
        """ write single instance (key and value) of metadata into root group attrs
        """
        logger.info(f'Saving metadata: {key}: {val}')
        self.file.attrs[key] = val
