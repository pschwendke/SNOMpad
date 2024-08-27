import numpy as np
import xarray as xr
import logging
from datetime import datetime
from glob import glob
from .gwyddion import combine_gsf


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
    if folder[-1] == '/':
        folder = folder[:-1]
    channels = ['Z', 'R-Z', 'M1A', 'R-M1A', 'M1P', 'R-M1P', 'O0A', 'R-O0A', 'O1A', 'O1P', 'R-O1A', 'R-O1P',
                'O2A', 'O2P', 'R-O2A', 'R-O2P', 'O3A', 'O3P', 'R-O3A', 'R-O3P',
                'O4A', 'O4P', 'R-O4A', 'R-O4P', 'O5A', 'O5P', 'R-O5A', 'R-O5P']

    scan_name = folder.split('/')[-1].split('\\')[-1]  # works on Win and Mac. There is probably a better way though...
    metadata = load_nea_metadata(f'{folder}/{scan_name}.txt')
    filenames = []
    names = []
    for c in channels:
        g = glob(folder + f'/* {c} raw.gsf')
        if g:  # not all channels are necessarily saved
            filenames.append(g[0])  # takes first one, but there should be only one
            names.append(c)
    if not filenames:
        raise RuntimeError('load_nea_image: no .gsf files found in directory')
    ds = combine_gsf(filenames=filenames, names=names)
    ds.attrs = {**ds.attrs, **metadata}

    return ds


def to_numpy(data: dict) -> dict:
    """ Converts data, as returned from NeaSNOM.scan(), to numpy arrays.
    (stolen from neaSpec SDK sample script)
    """
    for key in data.keys():
        d2 = data[key].GetData()
        limit1 = d2.GetUpperBound(0)+1
        limit2 = d2.GetUpperBound(1)+1
        array = np.zeros([limit1, limit2])
        for a in range(limit1):
            for b in range(limit2):
                array[a, b] = d2[a, b]
        data[key] = array
    return data
# ToDo make a 1D version of this or retraction curves
