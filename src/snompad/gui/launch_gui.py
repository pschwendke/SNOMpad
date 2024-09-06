import os
from snompad.gui import __path__
# print(__path__)

def launch_gui():
    """ launches the SNOMpad GUI.
    """
    os.system(f'bokeh serve --show {__path__[0]}')


if __name__ == '__main__':
    launch_gui()
