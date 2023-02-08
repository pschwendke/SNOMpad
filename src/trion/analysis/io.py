# io.py: read and write utility function
import numpy as np
import pandas as pd
import xarray as xr
import logging

from itertools import takewhile
from os.path import splitext
from copy import copy
from gwyfile.objects import GwyContainer, GwyDataField

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


# READING AND WRITING GWYDDION FORMA ###################################################################################
def export_gwy(filename: str, data: xr.Dataset):
    """ Exports xr.Dataset as gwyddion file. xr.DataArrays become separate channels. DataArray attributes are saved
    as metadata. Attributes must include x,y_size and x,y_offset. x,y_offset is defined as the top left corner
    of the image, with coordinate values increasing downwards and to the right.

    Parameters
    ----------
    filename: str
        filename of gwyddion file. Should end on '.gwy'
    data: xr.Dataset
        xr.Dataset of xr.DataArrays of x/y data
    """
    container = GwyContainer()
    metadata = data.attrs
    metastrings = {k: str(v) for k, v in metadata.items()}
    metacontainer = GwyContainer(metastrings)

    for i, (t, d) in enumerate(data.data_vars.items()):
        image_data = d.values.astype('float64')  # only double precision floats in gwy files
        if image_data.ndim != 2:
            raise RuntimeError(f'Expected 2-dimensional data, got dimension {image_data.ndim} instead')
        try:
            z_unit = d.attrs['z_unit']
        except KeyError:
            z_unit = ''
        try:
            xy_unit = metadata['xy_unit']
        except KeyError:
            xy_unit = 'm'

        container['/' + str(i) + '/data/title'] = t  # ToDo this somehow does not work for the first channel
        container['/' + str(i) + '/data'] = GwyDataField(image_data,
                                                         xreal=metadata['x_size'],
                                                         yreal=metadata['y_size'],
                                                         xoff=metadata['x_offset'],
                                                         yoff=metadata['y_offset'],
                                                         si_unit_xy=xy_unit,
                                                         si_unit_z=z_unit,
                                                         )
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
