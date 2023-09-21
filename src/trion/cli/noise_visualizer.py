# This is a Bokeh server app. To function, it must be run using the
# Bokeh server at the command line:
#
#     bokeh serve --show noise_visualizer.py
#
# Running "python noise_visualizer.py" will NOT work.
import numpy as np
import sys
import os
from time import perf_counter
from datetime import datetime
import logging

from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Toggle, Button, RadioButtonGroup, NumericInput, Div, LinearColorMapper
from bokeh.layouts import layout, column, row

from trion.analysis.signals import Signals
from trion.expt.scans import NoiseScan
from trion.analysis.experiment import load


signals = [
    Signals.sig_a,
    Signals.sig_b,
    Signals.tap_x,
    Signals.tap_y
]

date = datetime.now()
directory = f'Z:/data/_DATA/SNOM/{date.strftime("%Y")}/{date.strftime("%y%m%d")}'

filename = ''

def change_to_directory():
    if not os.path.isdir(directory):
        os.mkdir(directory)
    os.chdir(directory)
    logging.info(f'changed to {directory}')

class LogStream():
    def __init__(self):
        self.log = []

    def write(self, msg):
        self.log.append(msg)
        self.flush()

    def flush(self):
        message_box.text = self.log[-1]

    def close(self):
        self.flush()

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
logger.addHandler(logging.StreamHandler(stream=LogStream()))


# CALLBACKS ############################################################################################################
def start():
    global filename
    change_to_directory()
    scan = NoiseScan(sampling_seconds=sampling_s_input.value, signals=signals, npts=npts_input.value,
                     setpoint=setpoint_input.value)
    scan.start()
    logger.info(f'scan name: {scan.name}')
    filename = f'{directory}/{scan.name}.h5'

def stop(button):
    sys.exit()  # Stop the bokeh server

def update_message_box(msg: any, t: float = 0.):
    # if isinstance(msg, int):
    #     if msg == 0:
    #         dt = (perf_counter() - t) * 1e3  # ms
    #         message_box.styles['background-color'] = '#FFFFFF'
    #         message_box.text = f'demod and plotting: {dt:.0f} ms'
    #     elif msg % 10 == 1:
    #         message_box.styles['background-color'] = '#FF7777'
    #         message_box.text = 'empty bins detected'
    # else:
    #     message_box.styles['background-color'] = '#FF7777'
    #     message_box.text = msg
    pass

def delete_last():
    try:
        os.remove(filename)
    except FileNotFoundError:
        logger.info(f'file not found: {filename}')


# WIDGETS ##############################################################################################################
start_button = Button(label='START')
start_button.on_click(start)

stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

delete_last_button = Button(label='delete last scan')
delete_last_button.on_click(delete_last)

sampling_s_input = NumericInput(title='sampling time (s)', value=3, mode='float', low=.1, high=60, width=100)
npts_input = NumericInput(title='chunk size', value=5_000, mode='int', low=1000, high=100_000, width=100)
setpoint_input = NumericInput(title='AFM setpoint', value=0.8, mode='float', low=0, high=1, width=100)

message_box = Div(text='message box (log maybe?)')
message_box.styles = {
    'width': '200px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}


# SET UP PLOTS #########################################################################################################
def setup_am_plot():
    init_data = {
        'x': np.linspace(0, 1, 100),
        'y': np.sin(np.linspace(0, 2*np.pi, 100))
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM amplitude modulation', aspect_ratio=3)
    fig.line(x='x', y='y', source=plot_data)
    return fig, plot_data

def setup_z_plot():
    init_data = {
        'x': np.linspace(0, 1, 100),
        'y': np.cos(np.linspace(0, 2*np.pi, 100))
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM z-piezo response', aspect_ratio=3)
    fig.line(x='x', y='y', source=plot_data)
    return fig, plot_data

def setup_optical_plot():
    init_data = {
        'x': np.linspace(0, 1, 100),
        'sig_a': np.zeros(100),
        'sig_b': np.zeros(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='optical data', aspect_ratio=3)
    fig.line(x='x', y='sig_a', source=plot_data, legend_label='sig_a')
    fig.line(x='x', y='sig_b', source=plot_data, legend_label='sig_b')
    fig.legend.click_policy = 'hide'
    return fig, plot_data



# GUI OUTPUT ###########################################################################################################
am_plot, am_plot_data = setup_am_plot()
z_plot, z_plot_data = setup_z_plot()
optical_plot, optical_plot_data = setup_optical_plot()

controls_box = column([
    row([sampling_s_input, setpoint_input, npts_input]),
    row([start_button, delete_last_button, stop_server_button]),
    message_box
])

gui = layout(children=[
    [am_plot],
    [optical_plot],
    [z_plot],
    [controls_box]
], sizing_mode='stretch_width')

curdoc().add_root(gui)
curdoc().title = 'TRION visualizer'
