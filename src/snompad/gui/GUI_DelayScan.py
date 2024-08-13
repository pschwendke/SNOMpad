import numpy as np
import sys
import os
import colorcet as cc
from datetime import datetime
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Button, NumericInput, Div, RadioButtonGroup, Toggle, TextInput, Select
from bokeh.layouts import column, row, gridplot

from ..acquisition.scans import DelayScan
from ..analysis import load

import logging

if __name__ == '__main__':
    os.system('bokeh serve --show GUI_DelayScan.py')

logger = logging.getLogger()
logger.setLevel('INFO')

date = datetime.now()
directory = f'Z:/data/_DATA/SNOM/{date.strftime("%Y")}/{date.strftime("%y%m%d")}'

max_harm = 4


def change_to_directory():
    if not os.path.isdir(directory):
        os.mkdir(directory)
    os.chdir(directory)
    logger.info(f'changed to directory {directory}')


def prepare_data(name: str):
    global amp_plot_data, phase_plot_data, optical_plot_data
    logger.info('preparing scan data for GUI')
    filename = f'{directory}/{name}.h5'
    scan = load(filename)
    scan.demod()
    data = scan.demod_data

    amp_data = {'t': data['t'].values, 'amp': data['amp'].values}
    phase_data = {'t': data['t'].values, 'phase': data['phase'].values}
    optical_data = {'t': data['t'].values}
    for o in range(max_harm + 1):
        d = np.abs(data['optical'].sel(order=o).values)
        if d.max() > 0:
            d /= d.max()
        optical_data[str(o)] = d
    amp_plot_data.data = amp_data
    phase_plot_data.data = phase_data
    optical_plot_data.data = optical_data

    message_box.text = name
    message_box.styles['background-color'] = '#03F934'


# CALLBACKS ############################################################################################################
def start():
    logger.info('Started scan in GUI')
    message_box.text = 'Scan started'
    message_box.styles['background-color'] = '#03F934'

    change_to_directory()

    metadata = {
        'sample': sample_input.value,
        'user': user_input.value,
        'tip': tip_input.value,
        'tapping_amp_nm': amp_input.value,
        'tapping_frequency_Hz': freq_input.value,
        'light_source': laser_input.value,
        'probe_color_nm': probe_nm.value,
        'probe_power_mW': probe_mW.value,
        'probe_FWHM_nm': probe_FWHM.value,
        'pump_color_nm': pump_nm.value,
        'pump_power_mW': pump_mW.value,
        'pump_FWHM_nm': pump_FWHM.value
    }

    params = {
        'modulation': mod_button.labels[mod_button.active],
        'scan': scan_type_button.labels[scan_type_button.active],
        't_start': t_start_input.value,
        't_stop': t_stop_input.value,
        't_unit': t_unit_button.labels[t_unit_button.active],
        't0_mm': t0_input.value,
        'n_step': n_step_input.value,
        'scale': scale_input.labels[scale_input.active],
        'continuous': continuous_input.active,
        'setpoint': setpoint_input.value,
        'npts': npts_input.value,
        'metadata': metadata,
    }

    if params['scan'] == 'point':
        logger.info('Scan type: point scan')
        params.update({
            'x_target': x_target_input.value,
            'y_target': y_target_input.value,
            'in_contact': in_contact_input.active,
            'n': n_input.value,
        })

    scan = DelayScan(**params)
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

# scan parameters
parameter_title = Div(text='SCAN PARAMETERS')
parameter_title.styles = {'width': '400px', 'text-align': 'center', 'background-color': '#AAAAAA'}
scan_type_button = RadioButtonGroup(labels=['point'], active=0)
continuous_input = Toggle(label='continuous', active=True, width=100)
mod_button = RadioButtonGroup(labels=['none', 'shd', 'pshet'], active=1)

npts_input = NumericInput(title='chunk size', mode='int', value=5_000, low=0, high=200_000, width=80)
setpoint_input = NumericInput(title='AFM setpoint', value=0.8, mode='float', low=0, high=1, width=80)
t_start_input = NumericInput(title='t-start', value=None, mode='float', width=100)
t_stop_input = NumericInput(title='t-stop', value=None, mode='float', width=100)
n_step_input = NumericInput(title='steps', value=None, mode='int', low=0, width=100)
scale_input = RadioButtonGroup(labels=['lin', 'log'], active=0)
t_unit_button = RadioButtonGroup(labels=['mm', 'fs', 'ps'], active=0)
t0_input = NumericInput(title='t0 (mm)', mode='float', value=None, low=0, high=250, width=100)

sampling_ms_input = NumericInput(title='AFM sampling (ms)', value=80, mode='float', low=.1, high=60, width=100)

# specific for point scans
point_title = Div(text='POINT SCAN PARAMETERS')
point_title.styles = {'width': '400px', 'text-align': 'center', 'background-color': '#DDDDDD'}
x_target_input = NumericInput(title='x position (µm)', value=None, mode='float', low=0, high=100, width=100)
y_target_input = NumericInput(title='y position (µm)', value=None, mode='float', low=0, high=100, width=100)
n_input = NumericInput(title='pts per px', mode='int', value=30_000, low=0, high=200_000, width=80)
in_contact_input = Toggle(label='in contact', active=True, width=100)

# metadata
metadata_title = Div(text='METADATA')
metadata_title.styles = {'width': '400px', 'text-align': 'center', 'background-color': '#AAAAAA'}
sample_input = TextInput(title='sample name', value='', width=150)
user_input = TextInput(title='user name', value='', width=150)

tip_input = TextInput(title='AFM tip', value='', width=110)
amp_input = TextInput(title='tapping amp (nm)', value='', width=110)
freq_input = NumericInput(title='tapping freq (Hz)', value=None, low=0, high=500_000, mode='float', width=110)

laser_input = Select(title='probe light source', options=['HeNe', '3H', '2H'], value='HeNe', width=110)
probe_nm = NumericInput(title='probe color (nm)', value=None, low=0, high=1500, mode='float', width=110)
probe_mW = NumericInput(title='probe power (mW)', value=None, low=0, high=10, mode='float', width=110)
probe_FWHM = NumericInput(title='probe FWHM (nm)', value=None, low=0, high=100, mode='float', width=110)
pump_nm = NumericInput(title='pump color (nm)', value=None, low=0, high=1500, mode='float', width=110)
pump_mW = NumericInput(title='pump power (mW)', value=None, low=0, high=10, mode='float', width=110)
pump_FWHM = NumericInput(title='pump FWHM (nm)', value=None, low=0, high=100, mode='float', width=110)

message_box = Div(text='message box')
message_box.styles = {
    'width': '300px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}

# SET UP PLOTS #########################################################################################################
cmap = cc.b_glasbey_category10


def setup_amp_plot():
    init_data = {
        't': np.linspace(0, 1, 100),
        'amp': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM tapping amplitude', aspect_ratio=3)
    fig.xaxis.axis_label = 't (undefined unit)'
    fig.line(x='t', y='amp', source=plot_data)
    return fig, plot_data


def setup_phase_plot():
    init_data = {
        't': np.linspace(0, 1, 100),
        'phase': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM tapping phase', aspect_ratio=3)
    fig.xaxis.axis_label = 't (undefined unit)'
    fig.line(x='t', y='phase', source=plot_data)
    return fig, plot_data


def setup_optical_plot():
    init_data = {'t': np.linspace(0, 1, 100)}
    init_data.update({str(o): np.ones(100) for o in range(max_harm + 1)})
    plot_data = ColumnDataSource(init_data)

    fig = figure(title='optical data', aspect_ratio=3)
    fig.xaxis.axis_label = 't (undefined unit)'
    for o in range(max_harm + 1):
        line = fig.line(x='t', y=str(o), source=plot_data, legend_label=f'abs({o})', line_color=cmap[o])
        if o > 4:
            line.visible = False
    fig.legend.click_policy = 'hide'
    return fig, plot_data


# GUI OUTPUT ###########################################################################################################
amp_plot, amp_plot_data = setup_amp_plot()
phase_plot, phase_plot_data = setup_phase_plot()
optical_plot, optical_plot_data = setup_optical_plot()

controls_box = column([
    row([start_button, stop_server_button, delete_last_button, elab_button]),

    row([parameter_title]),
    row([scan_type_button, mod_button, continuous_input]),
    row([npts_input, setpoint_input, sampling_ms_input]),

    row([t_start_input, t_stop_input, t_unit_button]),
    row([n_step_input, scale_input, t0_input]),

    row([point_title]),
    row([x_target_input, y_target_input, n_input, in_contact_input]),

    row([metadata_title]),
    row([sample_input, user_input]),
    row([tip_input, amp_input, freq_input]),
    row([laser_input]),
    row([probe_nm, probe_mW, probe_FWHM]),
    row([pump_nm, pump_mW, pump_FWHM]),

    message_box
])

plot_box = gridplot([[amp_plot], [phase_plot], [optical_plot]], sizing_mode='stretch_width', merge_tools=False)

gui = row([plot_box, controls_box], sizing_mode='stretch_width')

curdoc().add_root(gui)
curdoc().title = 'Delay Scan'
