from .shd import shd
from .pshet import pshet

import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap


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
