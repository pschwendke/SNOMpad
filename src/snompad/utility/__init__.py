from snompad.utility.signals import Signals, Demodulation, Scan
from snompad.utility.acquisition_logger import notebook_logger

import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap


# FOR CONVERSION OF CARRIAGE POSITION TO DELAY TIME ####################################################################
c = 299_792_458   # m/s
n_air = 1.00028520   # 400 - 1_000 nm wavelength [https://doi.org/10.1364/AO.47.004856]
c_air = c/n_air


# STANDARD COLORS FOR PLOTTING #########################################################################################
# ToDo: replace definitions elsewhere (scan_demod and GUI)
plot_colors = {i: c for i, c in enumerate(cc.glasbey_category10)}
plot_colors['z'] = cc.glasbey_category10[95]
plot_colors['amp'] = cc.glasbey_category10[92]
plot_colors['phase'] = cc.glasbey_category10[94]

color_maps = {
    'afm_z': cc.m_dimgray,
    'afm_amp': cc.m_gray,
    'afm_phase': cc.m_gray,
    'optical_amp': cc.m_fire
}
cyclic_colors = cc.CET_C3[128:] + cc.CET_C3[:128]
color_maps['optical_phase'] = LinearSegmentedColormap.from_list('cyclic', cyclic_colors)
