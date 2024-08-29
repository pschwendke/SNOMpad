import os

from ..utility import Signals

# SHARED GUI PARAMETERS (CONSTANT) #####################################################################################
buffer_size = 200_000
max_harm = 7  # highest harmonics that is plotted, should be lower than 8 (because of display of signal/noise)
harm_plot_size = 40  # of values on x-axis when plotting harmonics

signals = [
    Signals.sig_a,
    Signals.sig_b,
    Signals.tap_x,
    Signals.tap_y,
    Signals.ref_x,
    Signals.ref_y,
    Signals.chop
]


# GUI LAUNCHER #########################################################################################################
path = __path__[0]
path += '/..'


def launch_gui():
    """ launches the SNOMpad GUI.
    """
    os.system(f'bokeh serve --show {path}')
