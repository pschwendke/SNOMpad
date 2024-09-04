import os
from snompad import __path__

# GUI LAUNCHER #########################################################################################################
path = __path__[0] + '/gui'


def launch_gui():
    """ launches the SNOMpad GUI.
    """
    os.system(f'bokeh serve --show {path}')


if __name__ == '__main__':
    launch_gui()
