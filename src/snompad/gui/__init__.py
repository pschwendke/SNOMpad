import os

path = __path__[0]
path += '/..'


def launch_gui():
    """ launches the SNOMpad GUI.
    """
    os.system(f'bokeh serve --show {path}')
