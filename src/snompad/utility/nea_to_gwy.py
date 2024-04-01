# script to collect gsf files saved by NeaSCAN and metadata and write to single gwy file
import sys
import os
import argparse
import logging

from ..file_handlers.gwyddion import export_gwy
from ..file_handlers.neascan import load_nea_image


logging.basicConfig(format="%(levelname)-7s - %(name)s - %(message)s")
logger = logging.getLogger()

parser = argparse.ArgumentParser(
    description='load all image files saved by NeaSCAN during one image acquisition,'
                'and save them to one .gwy file.',
    usage='nea_to_gwy.py [-h] input [-o] [-v]'
)
parser.add_argument('input', help='directory containing all .gsf files saved by NeaSCAN')
parser.add_argument('-o', '--output', default='', help='output file name of .gwy file')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')

args = parser.parse_args()
loglev = max(logger.level - 10 * args.verbose, 10)
logger.setLevel(loglev)
logger.debug('arguments: ' + str(args))

input_folder = args.input
if input_folder[-1] == '/':
    input_folder = input_folder[:-1]
output_filename = args.output

if output_filename == '':
    output_filename = input_folder.split('/')[-1]
    output_filename = '_'.join(output_filename.split(' ')[:2]).replace('-', '')
if output_filename[-4:] != '.gwy':
    output_filename += '.gwy'

if not os.path.exists(input_folder):
    logger.error(f'Input path does not exist. Given input path\n{input_folder}')
    sys.exit(1)

image_data = load_nea_image(input_folder)
export_gwy(filename=output_filename, data=image_data)
