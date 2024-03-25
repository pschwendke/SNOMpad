import numpy as np
import sys
import os
from datetime import datetime
import colorcet as cc

from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Button, NumericInput, Div, RadioButtonGroup, Toggle, TextInput, Select
from bokeh.layouts import column, row, gridplot

from snompad.acquisition.scans import SteppedRetraction, ContinuousRetraction
from snompad.demodulation.measurements import load

import logging

if __name__ == '__main__':
    os.system('bokeh serve --show GUI_Retraction.py')

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

    amp_data = {'z': data['z'].values, 'amp': data['amp'].values}
    phase_data = {'z': data['z'].values, 'phase': data['phase'].values}
    optical_data = {'z': data['z'].values}
    for o in range(max_harm + 1):
        d = np.abs(data['optical'].sel(order=o).values)
        d /= d.max()
        optical_data[str(o)] = d
    amp_plot_data.data = amp_data
    phase_plot_data.data = phase_data
    optical_plot_data.data = optical_data

    message_box.text = name
    message_box.styles['background-color'] = '#03F934'


# CALLBACKS ############################################################################################################
def start():
    message_box.text = 'Scan started'
    message_box.styles['background-color'] = '#03F934'

    change_to_directory()
    scan_class = [SteppedRetraction, ContinuousRetraction][scan_type_button.active]
    modulation = mod_button.labels[mod_button.active]

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
        'modulation': modulation,
        'z_size': z_size_input.value,
        'z_res': z_res_input.value,
        'npts': npts_input.value,
        'x_target': x_target_input.value,
        'y_target': y_target_input.value,
        'setpoint': setpoint_input.value,
        'metadata': metadata,
        'ratiometry': ratiometry_button.active
    }

    if scan_type_button.active == 1:
        params.update({'afm_sampling_ms': sampling_ms_input.value})
    if pump_probe_button.active:
        params.update({'t': t_input.value, 't0_mm': t0_input.value, 'pump_probe': True,
                       't_unit': t_unit_button.labels[t_unit_button.active]})

    scan = scan_class(**params)
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
parameter_title.styles = {'width': '300px', 'text-align': 'center', 'background-color': '#AAAAAA'}
scan_type_button = RadioButtonGroup(labels=['stepped', 'continuous'], active=1)
mod_button = RadioButtonGroup(labels=['shd', 'pshet'], active=1)
pump_probe_button = Toggle(label='pump-probe', active=False, width=100)
ratiometry_button = Toggle(label='ratiometry', active=False, width=100)

z_size_input = NumericInput(title='z size (µm)', value=0.2, mode='float', low=0, high=1, width=80)
z_res_input = NumericInput(title='z resolution', value=200, mode='int', low=1, high=10000, width=80)
npts_input = NumericInput(title='npts', mode='int', value=5_000, low=0, high=200_000, width=80)
x_target_input = NumericInput(title='x position (µm)', value=None, mode='float', low=0, high=100, width=100)
y_target_input = NumericInput(title='y position (µm)', value=None, mode='float', low=0, high=100, width=100)
setpoint_input = NumericInput(title='AFM setpoint', value=0.8, mode='float', low=0, high=1, width=80)
t_input = NumericInput(title='delay time', value=None, mode='float', width=100)
t_unit_button = RadioButtonGroup(labels=['mm', 'fs', 'ps'], active=0)
t0_input = NumericInput(title='t0 (mm)', mode='float', value=None, low=0, high=250, width=100)
sampling_ms_input = NumericInput(title='AFM sampling (ms)', value=80, mode='float', low=.1, high=1000, width=100)

# metadata
metadata_title = Div(text='METADATA')
metadata_title.styles = {'width': '300px', 'text-align': 'center', 'background-color': '#AAAAAA'}
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
        'z': np.linspace(0, 1, 100),
        'amp': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM tapping amplitude', aspect_ratio=3)
    fig.xaxis.axis_label = 'dz (µm)'
    fig.line(x='z', y='amp', source=plot_data)
    return fig, plot_data


def setup_phase_plot():
    init_data = {
        'z': np.linspace(0, 1, 100),
        'phase': np.ones(100)
    }
    plot_data = ColumnDataSource(init_data)
    fig = figure(title='AFM tapping phase', aspect_ratio=3)
    fig.xaxis.axis_label = 'dz (µm)'
    fig.line(x='z', y='phase', source=plot_data)
    return fig, plot_data


def setup_optical_plot():
    init_data = {'z': np.linspace(0, 1, 100)}
    init_data.update({str(o): np.ones(100) for o in range(max_harm + 1)})
    plot_data = ColumnDataSource(init_data)

    fig = figure(title='optical data', aspect_ratio=3)
    fig.xaxis.axis_label = 'dz (µm)'
    for o in range(max_harm + 1):
        line = fig.line(x='z', y=str(o), source=plot_data, legend_label=f'abs({o})', line_color=cmap[o])
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
    row([scan_type_button, mod_button]),
    row([pump_probe_button, ratiometry_button]),
    row([z_size_input, z_res_input, npts_input, setpoint_input]),
    row([x_target_input, y_target_input, sampling_ms_input]),

    row([t_input, t_unit_button, t0_input]),

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
curdoc().title = 'Retraction Curve'
