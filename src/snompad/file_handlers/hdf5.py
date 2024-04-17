# code to read and write SNOMpad data from and to hdf5 files
import numpy as np
import xarray as xr
import logging
import h5py

from .base import AcquisitionReader

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


class WriteH5Acquisition:
    """ class to create and write acquired data into an hdf5 file
    """
    def __init__(self):
        self.file = None
        self.filename = None

    def __repr__(self):
        return f'<SNOMpad acquisition file writer: {self.filename}>'

    def __del__(self):
        self.close_file()

    def create_file(self, name: str):
        """ create hdf5 file with scan name
        """
        logger.info(f'WriteH5Acquisition: Creating scan file: {name}.h5')
        self.filename = name + '.h5'
        self.file = h5py.File(self.filename, 'w')

    def close_file(self):
        """ close hdf5 file
        """
        logger.debug(f'WriteH5Acquisition: Closing acquisition file: {self.filename}')
        self.file.close()

    def write_daq_data(self, data: dict):
        """ write dictionary of acquired daq chunks into group. Chunks are individual hdf5 datasets
        """
        logger.info('WriteH5Acquisition: Saving acquired DAQ data')
        n_digits = int(np.log10(len(data.keys()))) + 1
        self.file.create_group('daq_data')
        for k, v in data.items():
            self.file['daq_data'].create_dataset(str(k).zfill(n_digits), data=v, dtype='float32')

    def write_afm_data(self, data: xr.Dataset):
        """ write xr.Dataset of afm data (xyz, amp, phase) tracked during acquisition into group
        """
        logger.info('WriteH5Acquisition: Saving tracked AFM data')
        self.file.create_group('afm_data')
        xr_to_h5_datasets(ds=data, group=self.file['afm_data'])

    def write_nea_data(self, data: xr.Dataset):
        """ write xr.Dataset of NeaScan data returned by API after acquisition into group
        """
        logger.info('WriteH5Acquisition: Saving NeaScan data')
        self.file.create_group('nea_data')
        xr_to_h5_datasets(ds=data, group=self.file['nea_data'])

    def write_metadata(self, key, val):
        """ write single instance (key and value) of metadata into root group attrs
        """
        logger.info(f'WriteH5Acquisition: Saving metadata: {key}: {val}')
        self.file.attrs[key] = val


class ReadH5Acquisition(AcquisitionReader):
    """ class to read DAQ, AFM and NeaSpec data and metadata from hdf5 file
    """
    def __repr__(self):
        return f'<SNOMpad acquisition file reader: {self.filename}>'

    def open_file(self):
        logger.info(f'Opening file: {self.filename}')
        self.file = h5py.File(self.filename, 'r')
        self.metadata = {}
        for k, v in self.file.attrs.items():
            self.metadata[k] = v

    def close_file(self):
        logger.info(f'Closing file: {self.filename}')
        self.file.close()

    def read_data(self):
        """ reads afm and nea data from file and stores xr.Datasets. A wrapper is built around DAQ data to save memory.
        Use .load_daq_data() to return dict of all daq data chunks.
        """
        if self.file['afm_data'].keys():
            logger.info('Reading AFM data from file')
            self.afm_data = h5_to_xr_dataset(group=self.file['afm_data'])
        if self.file['nea_data'].keys():
            logger.info('Reading NeaScan data from file')
            self.nea_data = h5_to_xr_dataset(group=self.file['nea_data'])

        class DaqDataDict(dict):
            def __getitem__(self, item):
                dataset = super().__getitem__(item)
                return np.array(dataset)
        logger.info('Constructing wrapper around DAQ data')
        self.daq_data = DaqDataDict()
        for k, v in self.file['daq_data'].items():
            self.daq_data[int(k)] = self.file['daq_data'][k]

    def load_daq_data(self):
        """ load all acquired DAQ data chunks into memory

        RETURNS
        -------
        daq_data: dict
            keys are integer chunk index and values are np.ndarray
        """
        logger.info('Reading DAQ data from file')
        daq_data = {}
        for k, v in self.file['daq_data'].items():
            daq_data[int(k)] = v
        return daq_data


class WriteH5Demodulation:
    def __init__(self):
        pass


class ReadH5Demodulation:
    def __init__(self):
        pass
