# script to collect gsf files saved by NeaSCAN and metadata and write to single gwy file
import os

from ..file_handlers.gwyddion import export_gwy
from ..file_handlers.neascan import load_nea_image


def nea_to_gwy(input_folder: str, output_filename: str):
    """ collects all .gsf files, as saved by neaSCAN, and combines them to a single .gwy file.
    """
    if not os.path.exists(input_folder):
        raise RuntimeError(f'Input path does not exist. Given input path\n{input_folder}')

    if output_filename == '':
        output_filename = input_folder.split('/')[-1]
        output_filename = '_'.join(output_filename.split(' ')[:2]).replace('-', '')
    if output_filename[-4:] != '.gwy':
        output_filename += '.gwy'

    image_data = load_nea_image(input_folder)
    export_gwy(filename=output_filename, data=image_data)
    return True


if __name__ == '__main__':
    directory = input('directory containing all .gsf files saved by NeaSCAN:\n')
    filename = input('path and filename where .gwy file should be created (leave blank to auto generate):\n')

    done = False
    while not done:
        try:
            done = nea_to_gwy(input_folder=directory, output_filename=filename)
        except RuntimeError as e:
            if 'path does not exist' in str(e):
                directory = input(f'directory does not exist:\n{directory}\nTry again (type "exit" to cancel):\n')
            elif 'no .gsf files found in directory' in str(e):
                directory = input(f'No gsf files found in:\n{directory}\nTry again (type "exit" to cancel):\n')
            else:
                raise
        if directory == 'exit':
            break
