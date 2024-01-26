# io.py: read and write utility function
import numpy as np
import pandas as pd
import xarray as xr
import logging
import h5py

from datetime import datetime
from itertools import takewhile
from os.path import splitext
from copy import copy
from glob import glob
from gwyfile.objects import GwyContainer, GwyDataField, GwySIUnit

from .signals import Signals

logger = logging.getLogger(__name__)


# EXPORTING DATA TO NPZ, NPY, CSV ######################################################################################
def export_npz(filename, data, header, compress=True):
    """
    Export a single point data table as an npz archive.

    Parameters
    ----------
    filename: str
        Output file name
    data: np.ndarray of shape (N, M)
        Experimental data, containing N data points in M columns.
    header: list of str, length `M`
        name of columns in data
    """
    payload = dict(zip(header, data.T))
    if compress:
        np.savez_compressed(filename, **payload)
    else:
        np.savez(filename, **payload)


def export_npy(filename, data, header=None):
    """
    Export a single point data table as an npy file.

    Parameters
    ----------
    filename: str
        Output file name
    data: np.ndarray of shape (N, M)
        Experimental data, containing N data points in M columns.
    """
    if header is not None:
        logger.warning("The '.npy' format cannot save header information.")
    np.save(filename, data, allow_pickle=False)


def export_csv(filename, data, header):
    """
    Export a single point data table as a csv file.

    Uses pandas.

    Parameters
    ----------
    filename: str
        Output file name. Also accepts a buffer string.
    data: np.ndarray of shape (N, M)
        Experimental data, containing N data points in M columns.
    header: list of str, length `M`
        name of columns in data
    """

    df = pd.DataFrame(data, columns=header)
    df.to_csv(filename, index=False)


writers = {
    ".npz": export_npz,
    ".csv": export_csv,
    ".npy": export_npy,
}


def export_data(filename, data, header):
    """
    Export data to file. Exact export behavior depends on data and file extension.

    """
    ext = splitext(filename)[1]
    if ext not in writers:
        raise ValueError(f"Unsupported extension: {ext}. Should be one of: "+", ".join(writers))
    if not data.ndim == 2:
        raise ValueError(f"Can only save 2d arrays as .npz, but shape is {data.shape}")
    if len(header) != data.shape[1]:
        raise ValueError(f"Mismatched header and data. header: {len(header)}, data: {data.shape[1]}")
    header = copy(header)
    for i, v in enumerate(header):
        if isinstance(v, Signals):
            header[i] = v.value
    writer = writers[ext]
    writer(filename, data.astype('float32'), header)


# READING AND WRITING GWYDDION FORMAT ##################################################################################
def export_gwy(filename: str, data: xr.Dataset):
    """ Exports xr.Dataset as gwyddion file. xr.DataArrays become separate channels. DataArray attributes are saved
    as metadata. Attributes must include x,y_size and x,y_center. x,y_offset is defined as the top left corner
    of the image, with coordinate values increasing downwards and to the right.

    Parameters
    ----------
    filename: str
        filename of gwyddion file. Should end on '.gwy'
    data: xr.Dataset
        xr.Dataset of xr.DataArrays of x/y data
    """
    container = GwyContainer()
    metadata = data.attrs.copy()
    metastrings = {k: str(v) for k, v in metadata.items()}
    metacontainer = GwyContainer(metastrings)

    x_offset = (metadata['x_center'] - metadata['x_size'] / 2)
    y_offset = (metadata['y_center'] - metadata['y_size'] / 2)
    try:
        if metadata['xy_unit'] == 'um':
            x_offset *= 1e-6
            y_offset *= 1e-6
            metadata['x_size'] *= 1e-6
            metadata['y_size'] *= 1e-6
        if metadata['xy_unit'] == 'nm':
            x_offset *= 1e-9
            y_offset *= 1e-9
            metadata['x_size'] *= 1e-9
            metadata['y_size'] *= 1e-9
    except KeyError:
        pass
    xy_unit = 'm'

    for i, (t, d) in enumerate(data.data_vars.items()):
        image_data = d.values.astype('float64')  # only double precision floats in gwy files
        if image_data.ndim != 2:
            raise RuntimeError(f'Expected 2-dimensional data, got dimension {image_data.ndim} instead')
        try:
            z_unit = d.attrs['z_unit']
            if z_unit == 'um':
                image_data *= 1e-6
                z_unit = 'm'
            if z_unit == 'nm':
                image_data *= 1e-9
                z_unit = 'm'
        except KeyError:
            z_unit = ''

        container['/' + str(i) + '/data/title'] = t  # ToDo this somehow does not work for the first channel
        container['/' + str(i) + '/data'] = GwyDataField(image_data,
                                                         xreal=metadata['x_size'],
                                                         yreal=metadata['y_size'],
                                                         xoff=x_offset,
                                                         yoff=y_offset,
                                                         si_unit_xy=GwySIUnit(unitstr=xy_unit),
                                                         si_unit_z=GwySIUnit(unitstr=z_unit),
                                                         )
        if 'optical' in t and 'amp' in t:
            container['/' + str(i) + '/base/palette'] = 'Warm'
        container['/' + str(i) + '/meta'] = metacontainer

    container['/filename'] = filename
    container.tofile(filename)


def load_gsf(filename):
    """Reads gwyddion gsf files Gwyddion Simple Field 1.0

    The script looks for XRes and YRes, calculates the length of binary data and cuts that from the end.
    Metadata is read separately until zero padding is reached (raises ValueError).

    Parameters
    ----------
    filename: string

    Returns
    -------
    data: numpy array
        array of shape (XRes, YRes) containing image data
    metadata: dictionary
        dictionary values are strings
    """
    metadata = {}
    data = None
    XRes = None
    YRes = None
    with open(filename, 'rb') as file:
        first_line = file.readline().decode('utf8')
        if first_line != 'Gwyddion Simple Field 1.0\n':
            logger.error(f'Expected "Gwyddion Simple Field 1.0", got "{first_line}" instead')

        # first determine the size of the binary (data) section
        while XRes is None or YRes is None:
            try:
                name, value = file.readline().decode('utf8').split('=')
                logging.debug(f'reading header: {name}: {value}')
                if name == 'XRes':
                    XRes = int(value)
                if name == 'YRes':
                    YRes = int(value)
            except ValueError as e:
                logging.error('While looking for XRes, YRex the following exception occurred:\n' + str(e))
                break
            except UnicodeDecodeError as e:
                logging.error('While looking for XRes, YRex the following exception occurred:\n' + str(e))
                break
        binary_size = XRes * YRes * 4  # 4: binary is somehow indexed in bytes
        # and read the binary data
        bindata = file.read()[-binary_size:]

    # open the file again to parse the metadata
    with open(filename, 'rb') as file:  # ToDo: Can't this be done in the previous while loop?
        file.readline()
        for line in file.read()[:-binary_size].split(b'\n'):
            logging.debug(f'metadata: {line}')
            try:
                name, value = line.decode('utf8').split('=')
                metadata[name] = value
            except ValueError:
                logging.debug('ValueError while reading metadata (expected)')
                break
            except UnicodeDecodeError as e:
                logging.error('While parsing metadata the following exception occurred:\n' + str(e))
                break

    if len(bindata) == XRes * YRes * 4:
        logging.debug('binary data found ... decoding to np.array')
        data = np.frombuffer(bindata, dtype=np.float32)
        data = data.reshape(YRes, XRes)
    else:
        logging.error('binary data not found or of the wrong shape')

    return data, metadata


def combine_gsf(filenames: list, names: list = None) -> xr.Dataset:
    """ Takes list of .gsf file filenames and combines them into one xr.Dataset. Metadata is written to attributes,
    so that Dataset can be exported to .gwy file directly.
    """
    ds = xr.Dataset()
    if names and len(names) != len(filenames):
        logger.error('Names and filenames must have same length')
        names = None

    for i, f in enumerate(filenames):
        if names:
            name = names[i]
        else:
            name = f.split('/')[-1]
            name = name.split('.')[0]
        image, imdata = load_gsf(f)

        # all x/y image data are saved to the Dataset
        old_attrs = ds.attrs.copy()
        ds.attrs['x_offset'] = float(imdata['XOffset'])
        ds.attrs['y_offset'] = float(imdata['YOffset'])
        ds.attrs['x_size'] = float(imdata['XReal'])
        ds.attrs['y_size'] = float(imdata['YReal'])
        ds.attrs['x_res'] = int(imdata['XRes'])
        ds.attrs['y_res'] = int(imdata['YRes'])
        ds.attrs['xy_unit'] = imdata['XYUnits']
        if i > 0 and old_attrs != ds.attrs:
            logging.error('Metadata of .gsf files do not match.')

        x = np.linspace(ds.attrs['x_offset'], ds.attrs['x_offset'] + ds.attrs['x_size'], ds.attrs['x_res'])
        y = np.linspace(ds.attrs['y_offset'], ds.attrs['y_offset'] + ds.attrs['y_size'], ds.attrs['y_res'])
        da = xr.DataArray(data=image, dims=('y', 'x'), coords={'x': x, 'y': y})
        da.attrs['z_unit'] = imdata['ZUnits']
        ds[name] = da

    return ds


# LOAD DATA ACQUIRED WITH NEASCAN ######################################################################################
def load_approach(fname: str) -> pd.DataFrame:
    """Loads an approach curve as a pandas.DataFrame.

    Parameters
    ----------
    fname : str
        File to load.

    Returns
    -------
    frame : pd.DataFrame
        Approach curve.

    The header of the file, containing the acquisition parameters are stored as
    `dataframe.attrs["header"]`.
    """
    with open(fname, encoding="utf-8") as f:
        meta = list(takewhile(lambda ln: ln.startswith("#"), f))
    meta = "".join(meta)
    frame = pd.read_table(fname, comment="#").dropna(axis="columns")
    frame.attrs["header"] = meta
    return frame


def load_nea_metadata(filename) -> dict:
    """Formats neaspec metadata text file (from image scan) in dictionary

    Parameters
    ----------
    filename: string
        name of file containing metadata (in current directory)

    Returns
    -------
    metadata: dictionary
        dictionary containing all metadata with the keys

    name
    project
    date    # datetime object
    mode    # e.g. AFM/PShet
    neascanversion

    x_center    # m
    y_center    # m
    x_size    # m
    y_size    # m
    x_offset  # m
    y_offset  # m
    x_res    # pixel
    y_res    # pixel
    rotation    # degrees

    tipfreq    # Hz
    tipamp    # mV
    tapamp    # nm
    setpoint    # %
    integration    # ms
    tipspeed    # m/s
    pgain
    igain
    dgain
    M1Ascaling    # nm/V

    demodulation    # e.g. Fourier
    modfreq    # Hz
    modamp    # mV
    modoffset    # mV
    """
    with open(filename, 'r') as file:
        original_data = {}
        for line in file.read().split('\n')[1:]:
            line = line[2:]
            segments = line.split('\u0009')
            try:
                name = segments.pop(0)
                unit = segments.pop(0)
                values = segments
                original_data[name] = values
            except IndexError:
                pass

    metadata = {'name': original_data['Description:'][0],
                'project': original_data['Project:'][0],
                'date': datetime.strptime(original_data['Date:'][0], '%m/%d/%Y %H:%M:%S'),
                'mode': original_data['Scan:'][0],
                'neascanversion': original_data['Version:'][0],

                'x_center': float(original_data['Scanner Center Position (X, Y):'][0]) * 1e-6,
                'y_center': float(original_data['Scanner Center Position (X, Y):'][1]) * 1e-6,
                'rotation': float(original_data['Rotation:'][0]),

                'tipfreq_Hz': float(original_data['Tip Frequency:'][0].replace(',', '')),
                'tipamp_mV': float(original_data['Tip Amplitude:'][0].replace(',', '')),
                'tapamp_nm': float(original_data['Tapping Amplitude:'][0].replace(',', '')),
                'setpoint': float(original_data['Setpoint:'][0]),
                'integration_ms': float(original_data['Integration time:'][0]),
                'p_gain': float(original_data['Regulator (P, I, D):'][0]),
                'i_gain': float(original_data['Regulator (P, I, D):'][1]),
                'd_gain': float(original_data['Regulator (P, I, D):'][2]),
                'M1Ascaling_nm/V': float(original_data['M1A Scaling:'][0]),

                'demodulation': original_data['Demodulation Mode:'][0],
                'modfreq_Hz': float(original_data['Modulation Frequency:'][0].replace(',', '')),
                'modamp_mV': float(original_data['Modulation Amplitude:'][0].replace(',', '')),
                'modoffset_mV': float(original_data['Modulation Offset:'][0].replace(',', ''))}

    return metadata


def load_nea_image(folder: str) -> xr.Dataset:
    """ Takes directory of NeaSCAN image (the one where all .gsf files are located), combines the .gsf files,
    and parses the metadata .txt file. A Dataset with image data and metadata as attributes is returned
    """
    if folder[-1] == '/':  # redundant if called by nea_to_gwy.py
        folder = folder[:-1]
    channels = ['Z', 'R-Z', 'M1A', 'R-M1A', 'M1P', 'R-M1P', 'O0A', 'R-O0A', 'O1A', 'O1P', 'R-O1A', 'R-O1P',
                'O2A', 'O2P', 'R-O2A', 'R-O2P', 'O3A', 'O3P', 'R-O3A', 'R-O3P', 'O4A', 'O4P', 'R-O4A', 'R-O4P']

    scan_name = folder.split('/')[-1].split('\\')[-1]  # works on Win and Mac. There is probably a better way though...
    metadata = load_nea_metadata(f'{folder}/{scan_name}.txt')
    filenames = []
    names = []
    for c in channels:
        g = glob(folder + f'/* {c} raw.gsf')
        if g:  # not all channels are necessarily saved
            filenames.append(g[0])  # takes first one, but there should be only one
            names.append(c)
    ds = combine_gsf(filenames=filenames, names=names)
    ds.attrs = {**ds.attrs, **metadata}

    return ds


# READING AND WRITING HDF5 FILES #######################################################################################
def h5_to_xr_dataset(group: h5py.Group):
    """ takes an hdf5 group that has previously been converted from an xr.Dataset, and reproduces this xr.Dataset.
    Datasets labeled with _real and _imag are combined into one complex valued DataArray.
    """
    ds = xr.Dataset()
    for ch, dset in group.items():
        if dset.dims[0].keys() and ch[-5:] != '_imag':
            if ch[-5:] == '_real':
                values = np.array(dset) + 1J * np.array(group[f'{ch[:-5]}_imag'])
            else:
                values = np.array(dset)
            dims = [d.keys()[0] for d in dset.dims]
            da = xr.DataArray(data=values, dims=dims, coords={d: np.array(group[d]) for d in dims})
            da.attrs = dset.attrs
            ds[ch] = da
    ds.attrs = group.attrs
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
        if da.dtpye == 'complex':  # split into two DataArrays if complex
            da = [da.real, da.imag]
            ch = [f'{ch}_real', f'{ch}_imag']
        else:
            da = [da]
            ch = [str(ch)]
        for name, data in zip(ch, da):
            dset = group.create_dataset(name=name, data=data.values)  # data
            for k, v in data.attrs.items():  # copy all metadata / attributes
                dset.attrs[k] = v
            for dim in data.coords.keys():  # attach dimension scales
                n = data.get_axis_num(dim)
                dset.dims[n].attach_scale(group[dim])
