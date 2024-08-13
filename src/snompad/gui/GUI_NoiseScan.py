import numpy as np
import sys
import os
import colorcet as cc
from datetime import datetime
from h5py import File
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Button, NumericInput, Div
from bokeh.layouts import column, row, gridplot

from ..utility.signals import Signals
from ..acquisition.scans import NoiseScan
from snompad.analysis.scans import Noise

import logging

if __name__ == '__main__':
    os.system('bokeh serve --show GUI_NoiseScan.py')

logger = logging.getLogger()
logger.setLevel('INFO')

signals = [
    Signals.sig_a,
    Signals.sig_b,
    Signals.tap_x,
    Signals.tap_y
]

date = datetime.now()
directory = f'Z:/data/_DATA/SNOM/{date.strftime("%Y")}/{date.strftime("%y%m%d")}'


def change_to_directory():
    if not os.path.isdir(directory):
        os.mkdir(directory)
    os.chdir(directory)
    logger.info(f'changed to directory {directory}')


def prepare_data(name: str):
    global z_plot_data, am_plot_data, optical_plot_data
    logger.info('preparing scan data for GUI')
    filename = f'{directory}/{name}.h5'
    scan = Noise(file=File(filename, 'r'))
    scan.demod()
    z_plot_data.data = {
        'x': scan.z_frequencies,
        'y': scan.z_spectrum
    }
    am_plot_data.data = {
        'x': scan.amp_frequencies,
        'y': scan.amp_spectrum
    }
    optical_plot_data.data = {
        'x': scan.optical_frequencies,
        'sig_a': scan.optical_spectrum[0],
        'sig_b': scan.optical_spectrum[1]
    }

    message_box.text = name
    message_box.styles['background-color'] = '#03F934'


# CALLBACKS ############################################################################################################
def start():
    message_box.text = 'Scan started'
    message_box.styles['background-color'] = '#03F934'

    change_to_directory()
    scan = NoiseScan(sampling_seconds=sampling_s_input.value, signals=signals, setpoint=setpoint_input.value)
    scan.start()

    message_box.text = 'Scan complete'
    message_box.styles['background-color'] = '#FFFFFF'
    logger.info(f'scan name: {scan.name}')
    prepare_data(scan.name)


def delete_last():
    filename = f'{directory}/{message_box.text}.h5'
    logfile = f'{directory}/{message_box.text}.log'
    try:
        logger.info(f'removing file: {filename}')
        os.remove(filename)
        os.remove(logfile)
        message_box.text = 'last scan file deleted'
        message_box.styles['background-color'] = '#FFFFFF'
    except FileNotFoundError:
        err_msg = f'file not found: {filename}'
        logger.error(err_msg)
        message_box.text = err_msg
        message_box.styles['background-color'] = '#FF7777'


def elab_entry():
    message_box.text = 'not implemented yet'
    message_box.styles['background-color'] = '#FFFF99'


def stop():
    # ToDo some of these messages will not print (?!)
    message_box.text = 'server stopped'
    message_box.styles['background-color'] = '#FF7777'
    sys.exit()  # Stop the bokeh server


# WIDGETS ##############################################################################################################
start_button = Button(label='START')
start_button.on_click(start)

stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

delete_last_button = Button(label='delete last scan')
delete_last_button.on_click(delete_last)

elab_button = Button(label='make elab entry')
elab_button.on_click(elab_entry)

sampling_s_input = NumericInput(title='sampling time (s)', value=2, mode='float', low=.1, high=60, width=100)
setpoint_input = NumericInput(title='AFM setpoint', value=0.8, mode='float', low=0, high=1, width=80)

message_box = Div(text='message box')
message_box.styles = {
    'width': '280px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}

# SET UP PLOTS #########################################################################################################
cmap = cc.b_glasbey_category10


def setup_am_plot():
    init_data = {
        'x': np.linspace(0, 1, 100),
        'y': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM amplitude modulation', aspect_ratio=3, x_axis_type='log', y_axis_type='log')
    fig.xaxis.axis_label = 'frequency (Hz)'
    fig.line(x='x', y='y', source=plot_data)
    return fig, plot_data


def setup_z_plot():
    init_data = {
        'x': np.linspace(0, 1, 100),
        'y': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM z-piezo response', aspect_ratio=3, x_axis_type='log', y_axis_type='log')
    fig.xaxis.axis_label = 'frequency (Hz)'
    fig.line(x='x', y='y', source=plot_data)
    return fig, plot_data


def setup_optical_plot():
    init_data = {
        'x': np.linspace(0, 1, 100),
        'sig_a': np.ones(100),
        'sig_b': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='optical data', aspect_ratio=3, x_axis_type='log', y_axis_type='log')
    fig.xaxis.axis_label = 'frequency (Hz)'
    fig.line(x='x', y='sig_a', source=plot_data, legend_label='sig_a', line_color=cmap[0])
    fig.line(x='x', y='sig_b', source=plot_data, legend_label='sig_b', line_color=cmap[1])
    fig.legend.click_policy = 'hide'
    return fig, plot_data


# GUI OUTPUT ###########################################################################################################
am_plot, am_plot_data = setup_am_plot()
z_plot, z_plot_data = setup_z_plot()
optical_plot, optical_plot_data = setup_optical_plot()

controls_box = column([
    row([start_button, sampling_s_input, setpoint_input]),
    row([stop_server_button, delete_last_button, elab_button]),
    message_box
])

plot_box = gridplot([[am_plot], [optical_plot], [z_plot]], sizing_mode='stretch_width',
                    merge_tools=False)

gui = row([plot_box, controls_box], sizing_mode='stretch_width')

curdoc().add_root(gui)
curdoc().title = 'Noise Scan'
