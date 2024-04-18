import numpy as np
import xarray as xr
import logging
from gwyfile.objects import GwyContainer, GwyDataField, GwySIUnit

logger = logging.getLogger(__name__)


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

        container[f'/{i}/data/title'] = t  # This does not work for the first channel. Maybe a bug in gwyfile..
        container[f'/{i}/data'] = GwyDataField(image_data,
                                               xreal=metadata['x_size'],
                                               yreal=metadata['y_size'],
                                               xoff=x_offset,
                                               yoff=y_offset,
                                               si_unit_xy=GwySIUnit(unitstr=xy_unit),
                                               si_unit_z=GwySIUnit(unitstr=z_unit),
                                               )
        if 'optical' in t and 'amp' in t:
            container[f'/{i}/base/palette'] = 'Warm'
        container[f'/{i}/meta'] = metacontainer
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
        array of shape (x_res, y_res) containing image data
    metadata: dictionary
        dictionary values are strings
    """
    metadata = {}
    data = None
    x_res = None
    y_res = None
    with open(filename, 'rb') as file:
        first_line = file.readline().decode('utf8')
        if first_line != 'Gwyddion Simple Field 1.0\n':
            logger.error(f'Expected "Gwyddion Simple Field 1.0", got "{first_line}" instead')

        # first determine the size of the binary (data) section
        while x_res is None or y_res is None:
            try:
                name, value = file.readline().decode('utf8').split('=')
                logging.debug(f'reading header: {name}: {value}')
                if name == 'XRes':
                    x_res = int(value)
                if name == 'YRes':
                    y_res = int(value)
            except ValueError as e:
                logging.error('While looking for x_res, YRex the following exception occurred:\n' + str(e))
                break
            except UnicodeDecodeError as e:
                logging.error('While looking for x_res, YRex the following exception occurred:\n' + str(e))
                break
        binary_size = x_res * y_res * 4  # 4: binary is somehow indexed in bytes
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

    if len(bindata) == x_res * y_res * 4:
        logging.debug('binary data found ... decoding to np.array')
        data = np.frombuffer(bindata, dtype=np.float32)
        data = data.reshape(y_res, x_res)
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
