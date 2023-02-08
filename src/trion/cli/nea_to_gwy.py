# script to collect gsf files saved by NeaSCAN and metadata and write to single gwy file

import sys
import os
import argparse
import logging

import numpy as np
import xarray as xr

from datetime import datetime
from glob import glob
from tqdm import tqdm

from trion.analysis.io import load_gsf, export_gwy


def load_metadata(filename) -> dict:
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


def load_images(filenames: list, names: list = None) -> xr.Dataset:
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


def load_neascan_image(folder: str) -> xr.Dataset:
    """ Takes directory of NeaSCAN image (the one where all .gsf files are located), combines the .gsf files,
    and parses the metadata .txt file. A Dataset with image data and metadata as attributes is returned
    """
    if folder[-1] == '/':
        folder = folder[:-1]
    channels = ['Z', 'R-Z', 'M1A', 'R-M1A', 'M1P', 'R-M1P', 'O0A', 'R-O0A', 'O1A', 'O1P', 'R-O1A', 'R-O1P',
                'O2A', 'O2P', 'R-O2A', 'R-O2P', 'O3A', 'O3P', 'R-O3A', 'R-O3P', 'O4A', 'O4P', 'R-O4A', 'R-O4P']

    filenames = [glob(folder + f'/* {c} raw.gsf')[0] for c in channels]  # takes first one, but there should be only one
    ds = load_images(filenames, channels)
    metadata = load_metadata(glob(folder + '/*.txt')[0])  # takes first text file, but again there should be no other
    ds.attrs = {**ds.attrs, **metadata}

    return ds


if __name__ == '__main__':
    logging.basicConfig(format="%(levelname)-7s - %(name)s - %(message)s")
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='load all image files saved by NeaSCAN during one image acquisition,'
                    'and save them to one .gwy file.',
        usage='nea_to_gwy.py [-h] input [-output] [-v]'
    )
    parser.add_argument('input', help='directory containing all .gsf files saved by NeaSCAN')
    parser.add_argument('-output', default='', help='output file name of .gwy file')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')

    args = parser.parse_args()
    loglev = max(logger.level - 10 * args.verbose, 10)
    logger.setLevel(loglev)
    logger.debug('arguments: ' + str(args))

    input_folder = args.input
    output_filename = args.output

    if output_filename == '':
        output_filename = input_folder.split('/')[-1]
        output_filename = '_'.join(output_filename.split(' ')[:2]).replace('-', '')
    if output_filename[-4:] != '.gwy':
        output_filename += '.gwy'

    if not os.path.exists(input_folder):
        logger.error(f'Input path does not exist. Given input path\n{input_folder}')
        sys.exit(1)

    image_data = load_neascan_image(input_folder)
    export_gwy(filename=output_filename, data=image_data)
