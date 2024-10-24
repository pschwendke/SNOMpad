import numpy as np
import colorcet as cc
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import LinearColorMapper

from snompad.gui.utils import rgb_to_hex
from snompad.gui.buffer import PlottingBuffer
from snompad.gui.main import max_harm, phase_plot_sample_size, harm_plot_size
from snompad.utility import plot_colors


# SIGNAL TAB ###########################################################################################################
def make_sig_fig():
    default_visible = [3, 4, 5]
    buffer = PlottingBuffer(n=harm_plot_size, names=[str(h) for h in range(max_harm + 1)])
    fig = figure(height=600, width=1000, tools='pan,ywheel_zoom,box_zoom,reset,save', active_scroll='ywheel_zoom',
                 active_drag='pan', sizing_mode='scale_both')
    for h in range(max_harm+1):
        fig.line(x='t', y=str(h), source=buffer.buffer, line_color=rgb_to_hex(plot_colors[h]),
                 line_width=2, syncable=False, legend_label=f'{h}', visible=h in default_visible)
    fig.legend.location = 'center_left'
    fig.legend.click_policy = 'hide'
    return fig, buffer


def make_phase_fig():
    channels = ['sig_a', 'sig_b', 'chop']
    init_data = {c: np.zeros(phase_plot_sample_size // 20) for c in channels}
    init_data.update({'theta': np.linspace(-np.pi, np.pi, phase_plot_sample_size // 20)})
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=400, width=400, tools='pan,ywheel_zoom,box_zoom,reset,save',
                 active_scroll='ywheel_zoom', active_drag='pan', sizing_mode='fixed')
    for n, c in enumerate(channels):
        fig.scatter(x='theta', y=c, source=plot_data, marker='dot', line_color=rgb_to_hex(plot_colors[n]),
                    size=10, syncable=False, legend_label=c)
    fig.legend.location = 'top_right'
    fig.legend.click_policy = 'hide'
    return fig, plot_data


# TUNING TAB ###########################################################################################################
def make_binning_fig():
    init_data = {'binned': [np.random.uniform(size=(64, 64))]}
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=600, width=600, toolbar_location=None, sizing_mode='fixed')
    cmap = LinearColorMapper(palette=cc.bgy, nan_color='#FF0000')
    plot = fig.image(image='binned', source=plot_data, color_mapper=cmap,
                     dh=2*np.pi, dw=2*np.pi, x=-np.pi, y=-np.pi, syncable=False)
    cbar = plot.construct_color_bar(padding=1)
    fig.add_layout(cbar, 'right')
    return fig, plot_data


def make_tap_shift_fig():
    init_data = {'tap_x': [np.zeros(1)], 'tap_y': [np.zeros(1)]}
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=300, width=300, toolbar_location=None, sizing_mode='fixed')
    fig.scatter(x='tap_x', y='tap_y', source=plot_data,
                line_color='blue', marker='dot', size=10, syncable=False)
    return fig, plot_data


def make_ref_shift_fig():
    init_data = {'ref_x': [np.zeros(1)], 'ref_y': [np.zeros(1)]}
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=300, width=300, toolbar_location=None, sizing_mode='fixed')
    fig.scatter(x='ref_x', y='ref_y', source=plot_data,
                line_color='blue', marker='dot', size=10, syncable=False)
    return fig, plot_data


# RETRACTION TAB #######################################################################################################
def setup_retr_amp_fig():
    init_data = {
        'z': np.linspace(0, 1, 100),
        'amp': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM tapping amplitude', aspect_ratio=3)
    fig.xaxis.axis_label = 'dz (µm)'
    fig.line(x='z', y='amp', source=plot_data)
    return fig, plot_data


def setup_retr_phase_fig():
    init_data = {
        'z': np.linspace(0, 1, 100),
        'phase': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM tapping phase', aspect_ratio=3)
    fig.xaxis.axis_label = 'dz (µm)'
    fig.line(x='z', y='phase', source=plot_data)
    return fig, plot_data


def setup_retr_optical_fig():
    init_data = {'z': np.linspace(0, 1, 100)}
    init_data.update({str(o): np.ones(100) for o in range(max_harm + 1)})
    plot_data = ColumnDataSource(init_data)

    fig = figure(title='optical data', aspect_ratio=3)
    fig.xaxis.axis_label = 'dz (µm)'
    for o in range(max_harm + 1):
        line = fig.line(x='z', y=str(o), source=plot_data, legend_label=f'abs({o})', line_color=plot_colors[o])
        if o > 4:
            line.visible = False
    fig.legend.click_policy = 'hide'
    return fig, plot_data
