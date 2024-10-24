# This is the SNOMpad GUI. It is a bokeh server.
# It can be run from the commandline as 'bokeh serve --show .'
# or as 'python launch_gui.py'
# Alternatively, launch_gui() can be imported from snompad.gui
import sys
import os
import numpy as np
import logging
from threading import Thread
from time import perf_counter, sleep
from scipy.stats import binned_statistic, binned_statistic_2d
from datetime import datetime

from bokeh.plotting import curdoc
from bokeh.models import Toggle, Button, NumericInput, TabPanel, Tabs, Div, RadioButtonGroup, Select, TextInput
from bokeh.layouts import layout, column, row, gridplot

# ToDo: configure logging
# from .utility.acquisition_logger import gui_logger
from snompad.gui.demod import demod_to_buffer
from snompad.gui.user_messages import error_message
from snompad.gui.utils import *
from snompad.gui.setup_plots import *

from snompad.drivers import DaqController
from snompad.acquisition.buffer import CircularArrayBuffer
from snompad.utility import Signals, plot_colors
from snompad.analysis.utils import fft_tip_frequency
from snompad.analysis.noise import phase_shifting
from snompad.acquisition.scans import SteppedRetraction, ContinuousRetraction

logger = logging.getLogger()
logger.setLevel('INFO')

# global variables
callback_period = 150  # ms  (time after which a periodic callback is started)
buffer_size = 200_000
max_harm = 8  # highest harmonics that is plotted, should be lower than 8 (because of display of signal/noise)
tip_frequency = 0.
signals = [Signals.sig_a, Signals.sig_b, Signals.tap_x, Signals.tap_y, Signals.ref_x, Signals.ref_y, Signals.chop]
acquisition_buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
go = False
err_code = 0

# variables for scanning
date = datetime.now()
data_dir = f'Z:/data/_DATA/SNOM/{date.strftime("%Y")}/{date.strftime("%y%m%d")}'
scan_daemon = None
current_name = '-no scan-'  # eg '240709-191412_continuous_retraction'
scanning = False  # False: idling, True: busy, 0: exception occured, 1: done waiting for next step


# DAEMON TARGETS #######################################################################################################
def acquisition_idle_loop():
    """ when 'GO' button is not active
    """
    while not go:
        sleep(0.01)
    acquisition_go_loop()


def acquisition_go_loop():
    """ when 'GO' button is active
    """
    global acquisition_buffer, err_code
    acquisition_buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
    daq = DaqController(dev='Dev1', clock_channel='pfi0', emulate=False)
    daq.setup(buffer=acquisition_buffer)
    try:
        daq.start()
        while go:
            sleep(.01)
            daq.reader.read()
    except Exception as e:
        if err_code // 100 % 10 == 0:
            err_code += 100
            print(e)
    finally:
        daq.close()
        acquisition_idle_loop()


def scanning_thread(scan, params):
    """ this executes in a daemon thread so that periodic callbacks don't pile up.
    """
    global scanning, current_name
    try:
        scanning = True
        task = scan(**params)
        task.start()
        current_name = task.name
        scanning = 1
    except Exception:
        scanning = 0
        raise


# PERIODIC CALLBACKS ###################################################################################################
def periodic_callback():
    global err_code
    t_start = perf_counter()
    if go:
        if tabs.active == 0:
            err_code += update_signal_tab(buffer=acquisition_buffer)
        elif tabs.active == 1:
            err_code += update_tuning_tab(buffer=acquisition_buffer)
        elif tabs.active == 2:
            err_code += update_noise_tab()
        elif tabs.active == 3:
            err_code += update_retraction_tab()
        elif tabs.active == 4:
            err_code += update_line_tab()
        elif tabs.active == 5:
            err_code += update_delay_tab()

    dt = (perf_counter() - t_start) * 1e3  # ms
    if err_code == 0:
        usr_msg = f'time to update: {int(dt)} ms'
    else:
        usr_msg = error_message(err_code=err_code)
        err_code = 0
    update_message_box(msg=usr_msg)
    update_refresh_time(dt)


def update_signal_tab(buffer) -> int:
    rtn_value = 0
    try:
        data = buffer.tail(n=npts_input.value)
        if len(data) < npts_input.value:
            return 30
        rtn_value, coefficients = demod_to_buffer(
            data=data,
            signals=signals,
            modulation=mod_button.labels[mod_button.active],
            tap=tap_input.value,
            ref=ref_input.value,
            chop=chop_button.active,
            ratiometry=ratiometry_button.active,
            abs_val=abs_button.active,
            max_harm=max_harm
        )
        if rtn_value == 0:
            coefficients = normalize_harmonics(coefficients)
            signal_buffer.put(data=coefficients[:max_harm + 1])
            update_signal_to_noise()
        if isinstance(rtn_value, str):
            print(rtn_value)
            rtn_value = 1
        update_phase_data(data=data)
    except Exception as e:
        print(e)
        rtn_value = 10
    return rtn_value


def update_tuning_tab(buffer) -> int:
    rtn_value = 0
    try:
        data = buffer.tail(n=npts_input.value)
        if len(data) < npts_input.value:
            return 3000
        tap_x = data[:, signals.index(Signals.tap_x)]
        tap_y = data[:, signals.index(Signals.tap_y)]
        tap_p = np.arctan2(tap_y, tap_x)
        if mod_button.labels[mod_button.active] == 'pshet':
            ref_x, ref_y = data[:, signals.index(Signals.ref_x)], data[:, signals.index(Signals.ref_y)]
            ref_p = np.arctan2(ref_y, ref_x)
        else:
            ref_x, ref_y, ref_p = None, None, None

        rtn_value = update_binning_plot(tap_p=tap_p, ref_p=ref_p)
        update_tip_frequency(tap_p)
        update_phase_shift_plots(tap_x, tap_y, ref_x, ref_y)
        update_phase_shift_stats(tap_x, tap_y, ref_x, ref_y)
    except Exception as e:
        print(e)
        rtn_value = 1000
        raise
    return rtn_value


def update_retraction_tab() -> int:
    global scanning, current_name, scan_daemon
    if scanning == False:
        return 0
    elif scanning == True:  # either busy scanning or doing demod
        return 20_000
    elif scanning == 0:  # exception occured
        if scan_daemon is not None:
            scan_daemon.join()
            scan_daemon = None
        return 10_000
    elif scanning == 1:  # done but need demod
        try:
            scan_daemon = None
            logger.info(f'scan complete: {current_name}')
            plot_retraction(current_name)
            scanning = True
            return 30_000
        except Exception as e:
            logger.error(str(e))
            scanning = 0
            return 10_000


def update_noise_tab() -> int:
    rtn_value = 0
    return rtn_value


def update_line_tab() -> int:
    rtn_value = 0
    return rtn_value


def update_delay_tab() -> int:
    rtn_value = 0
    return rtn_value


# GLOBAL WIDGET CALLBACKS ##############################################################################################
def stop(button):
    update_message_box(msg='Server stopped')
    sleep(0.5)
    sys.exit()  # Stop the bokeh server


def update_go(_):
    global go
    go = go_button.active


def update_refresh_time(dt):
    global callback_period
    if dt > callback_period:
        callback_period = dt // 10 * 10 + 10  # round up to next 10 ms
        renew_callback()


def input_callback_period(_, old, new):
    global callback_period
    callback_period = new
    renew_callback()


def renew_callback():
    global callback, callback_period
    curdoc().remove_periodic_callback(callback)
    callback = curdoc().add_periodic_callback(periodic_callback, callback_period)
    callback_input.value = callback_period


def update_message_box(msg: str):
    message_box.text = msg


# SIGNAL TAB FUNCTIONS AND CALLBACKS ###################################################################################
harm_plot_size = 40  # of values on x-axis when plotting harmonics
harm_scaling = np.zeros(max_harm + 1)  # normalization factors for plotted harmonics
signal_noise = {  # collection of signal-to-noise ratios
    h: Div(text='-', styles={'color': rgb_to_hex(plot_colors[h]), 'font-size': '200%'}) for h in range(8)
}
phase_plot_sample_size = 5000  # of diode data points to be plotted vs phase. every 10th point is plotted


def update_signal_to_noise():
    for h in signal_buffer.names:
        avg = signal_buffer.avg(key=h)
        std = signal_buffer.std(key=h)
        if std > 0:
            sn = np.abs(avg) / std
            signal_noise[int(h)].text = f'{sn:.1f}'


def update_phase_data(data):
    global phase_plot_data
    if mod_button.labels[mod_button.active] == 'no mod':
        return
    data_sample = data[-phase_plot_sample_size::20]
    new_data = {c.value: data_sample[:, signals.index(c)] for c in signals if c.value in ['sig_a', 'sig_b', 'chop']}
    if raw_theta_button.active == 1:  # 'raw vs theta_ref'
        new_data.update({'theta': np.arctan2(data_sample[:, signals.index(Signals.ref_y)],
                                             data_sample[:, signals.index(Signals.ref_x)])})
    else:  # 'raw vs theta_tap'
        new_data.update({'theta': np.arctan2(data_sample[:, signals.index(Signals.tap_y)],
                                             data_sample[:, signals.index(Signals.tap_x)])})
    phase_plot_data.data = new_data


def normalize_harmonics(coefficients):
    global harm_scaling
    if mod_button.labels[mod_button.active] != 'no mod':
        if harm_scaling[0] == 0:
            harm_scaling = np.ones(max_harm + 1) / np.abs(coefficients)
        for i, over_limit in enumerate(np.abs(coefficients * harm_scaling) > 1):
            if over_limit:
                signal_buffer.buffer.data[str(i)] /= np.abs(coefficients[i]) * harm_scaling[i]
                harm_scaling[i] = 1 / np.abs(coefficients[i])
        coefficients *= harm_scaling
    return coefficients


def reset_normalization(button):
    global harm_scaling
    harm_scaling = np.zeros(max_harm + 1)


# TUNING TAB FUNCTIONS AND CALLBACKS ###################################################################################
def update_binning_plot(tap_p, ref_p):
    global binning_data
    rtn_value = 0

    if ref_p is not None:  # pshet
        returns = binned_statistic_2d(x=tap_p, y=ref_p, values=None, statistic='count',
                                      bins=[tap_input.value, ref_input.value],
                                      range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        binned = returns.statistic
    else:
        returns = binned_statistic(x=tap_p, values=None, statistic='count',
                                   bins=tap_input.value, range=[-np.pi, np.pi])
        binned = returns.statistic[np.newaxis, :]
    new_data = {'binned': [binned]}
    binning_data.data = new_data

    if np.isnan(binned).any():
        rtn_value = 2000
    return rtn_value


def update_phase_shift_plots(tap_x, tap_y, ref_x, ref_y):
    global tap_shift_data, ref_shift_data
    new_tap_data = {'tap_x': tap_x[::200],
                    'tap_y': tap_y[::200]}
    tap_shift_data.data = new_tap_data
    if ref_x is not None and ref_y is not None:  # pshet
        new_ref_data = {'ref_x': ref_x[::200],
                        'ref_y': ref_y[::200]}
        ref_shift_data.data = new_ref_data


def update_phase_shift_stats(tap_x, tap_y, ref_x, ref_y):
    _, tap_phase_diff, tap_amp_diff, tap_amp, tap_std, tap_offset = phase_shifting(tap_x, tap_y)
    tap_err = tap_std / tap_amp
    if ref_x is not None and ref_y is not None:  # pshet
        _, ref_phase_diff, ref_amp_diff, ref_amp, ref_std, ref_offset = phase_shifting(ref_x, ref_y)
        ref_err = ref_std / ref_amp
    else:
        ref_amp_diff, ref_phase_diff, ref_err = 0., 0., 0.
    tap_amp_diff_txt.text = f'amp_x - amp_y: {tap_amp_diff:.2e} V'
    tap_phase_diff_txt.text = f'phi_x - phi_y: {tap_phase_diff:.2f} °'
    tap_err_txt.text = f'amplitude err: {tap_err:.2e} %'
    ref_amp_diff_txt.text = f'amp_x - amp_y: {ref_amp_diff:.2e} V'
    ref_phase_diff_txt.text = f'phi_x - phi_y: {ref_phase_diff:.2f} °'
    ref_err_txt.text = f'amplitude err: {ref_err:.2e} %'


def update_tip_frequency(tap_p):
    global tip_frequency
    tip_frequency = fft_tip_frequency(tap_p)
    tip_freq_stat.text = f'Assumed tip frequency:\n{tip_frequency/1000:.3f} kHz'


# RETRACTION TAB FUNCTIONS AND CALLBACKS ###############################################################################
def start_retraction():
    global scan_daemon
    update_message_box(msg='Retraction scan started')

    change_to_directory(data_dir)
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

    if scan_type_button.active == 1:  # continuous
        params.update({'afm_sampling_ms': sampling_ms_input.value})
    if pump_probe_button.active:
        params.update({'t': t_input.value, 't0_mm': t0_input.value, 'pump_probe': True,
                       't_unit': t_unit_button.labels[t_unit_button.active]})

    scan_daemon = Thread(target=scanning_thread, kwargs={'scan': scan_class, 'params': params}, daemon=True)
    scan_daemon.start()


def plot_retraction(scan):
    pass

# NOISE TAB FUNCTIONS AND CALLBACKS ####################################################################################
# LINE SCAN TAB FUNCTIONS AND CALLBACKS ################################################################################
# DELAY SCAN TAB FUNCTIONS AND CALLBACKS ###############################################################################
# GENERAL SCAN FUNCTIONS AND CALLBACKS #################################################################################
def delete_last():
    scanfile = f'{data_dir}/{current_name}.h5'
    logfile = f'{data_dir}/{current_name}.log'
    try:
        logger.info(f'removing file: {scanfile}')
        os.remove(scanfile)
        os.remove(logfile)
        update_message_box(msg='last scan file deleted')
    except FileNotFoundError:
        err_msg = f'file not found: {scanfile}'
        logger.error(err_msg)
        update_message_box(msg=err_msg)


def elab_entry():
    update_message_box(msg='not implemented yet')


def get_elab_metadata():
    update_message_box(msg='not implemented yet')


def stop_scan():
    if scan_daemon is not None:
        keyboard_interrupt_thread(thread=scan_daemon)


# WIDGETS ##############################################################################################################
# TOP PANEL
go_button = Toggle(label='GO', active=False, width=60)
go_button.on_click(update_go)
stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)
callback_input = NumericInput(title='callback period (ms)', value=callback_period, mode='int', width=200)
callback_input.on_change('value', input_callback_period)
message_box = Div(text='message box')
message_box.styles = {
    'width': '800px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}

# DEMOD PARAMS
tap_input = NumericInput(title='# tap bins', value=128, mode='int', low=16, high=256, width=90)
ref_input = NumericInput(title='# ref bins', value=84, mode='int', low=16, high=256, width=90)
npts_input = NumericInput(title='# of samples', value=50_000, mode='int', low=10_000, high=buffer_size, width=90)
mod_button = RadioButtonGroup(labels=['no mod', 'shd', 'pshet'], active=1)
chop_button = Toggle(label='chop', active=False, width=60)
abs_button = Toggle(label='abs', active=False, width=60)
ratiometry_button = Toggle(label='ratiometry', active=False, width=100)
raw_theta_button = RadioButtonGroup(labels=['raw vs theta_tap', 'raw vs theta_ref'], active=0)
tip_freq_stat = Div(text='Assumed tip frequency')
normalize_button = Button(label='reset norm')
normalize_button.on_click(reset_normalization)
noise_table = column(
    Div(text='Signal to noise'),
    row([signal_noise[h] for h in range(4)]),
    row([signal_noise[h] for h in range(4, 8)])
)

# SCAN PARAMETERS
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

# ACQUISITION METADATA
metadata_title = Div(text='METADATA')
metadata_title.styles = {'width': '300px', 'text-align': 'center', 'background-color': '#AAAAAA'}
sample_input = TextInput(title='sample name', value='', width=150)
user_input = TextInput(title='user name', value='', width=150)
tip_input = TextInput(title='AFM tip', value='', width=110)
amp_input = TextInput(title='tapping amp (nm)', value='', width=110)
freq_input = NumericInput(title='tapping freq (Hz)', value=tip_frequency, low=0, high=500_000, mode='float', width=110)
laser_input = Select(title='probe light source', options=['HeNe', '3H', '2H'], value='HeNe', width=110)
probe_nm = NumericInput(title='probe color (nm)', value=None, low=0, high=1500, mode='float', width=110)
probe_mW = NumericInput(title='probe power (mW)', value=None, low=0, high=10, mode='float', width=110)
probe_FWHM = NumericInput(title='probe FWHM (nm)', value=None, low=0, high=100, mode='float', width=110)
pump_nm = NumericInput(title='pump color (nm)', value=None, low=0, high=1500, mode='float', width=110)
pump_mW = NumericInput(title='pump power (mW)', value=None, low=0, high=10, mode='float', width=110)
pump_FWHM = NumericInput(title='pump FWHM (nm)', value=None, low=0, high=100, mode='float', width=110)

# SCAN WIDGETS
delete_last_button = Button(label='delete last scan')
delete_last_button.on_click(delete_last)
elab_button = Button(label='make elab entry')
elab_button.on_click(elab_entry)
metadata_button = Button(label='get metadata from elabFTW')
metadata_button.on_click(get_elab_metadata)
abort_button = Button(label='cancel current scan')
abort_button.on_click(stop_scan)

# TUNING TAB
tap_amp_diff_txt = Div(text=f'amp_x - amp_y: {0:.2e} V')
tap_phase_diff_txt = Div(text=f'phi_x - phi_y: {0:.2f} °')
tap_err_txt = Div(text=f'amplitude err: {0:.2e} %')
ref_amp_diff_txt = Div(text=f'amp_x - amp_y: {0:.2e} V')
ref_phase_diff_txt = Div(text=f'phi_x - phi_y: {0:.2f} °')
ref_err_txt = Div(text=f'amplitude err: {0:.2e} %')

# RETRACTION TAB
start_retraction_button = Button(label='START')
start_retraction_button.on_click(start_retraction)


# MAKE ALL THE PLOTS ###################################################################################################
# SIGNAL TAB
sig_fig, signal_buffer = make_sig_fig()
phase_fig, phase_plot_data = make_phase_fig()

# TUNING TAB
binning_fig, binning_data = make_binning_fig()
tap_shift_fig, tap_shift_data = make_tap_shift_fig()
ref_shift_fig, ref_shift_data = make_ref_shift_fig()

# RETRACTION TAB
retraction_amp_plot, retraction_amp_plot_data = setup_retr_amp_fig()
retraction_phase_plot, retraction_phase_plot_data = setup_retr_phase_fig()
retraction_sig_plot, retraction_sig_plot_data = setup_retr_optical_fig()


# MAKE ALL THE TAB LAYOUTS #############################################################################################
def make_signal_layout():
    controls_box = column([
        row([chop_button, mod_button]),
        row([tap_input, ref_input, npts_input]),
        row([raw_theta_button, abs_button]),
        row([ratiometry_button, normalize_button]),
        noise_table,
    ])
    return layout(children=[[sig_fig, [phase_fig, controls_box]]], sizing_mode='stretch_width')


def make_tuning_layout():
    controls_box = column([mod_button, npts_input, tap_input, ref_input, tip_freq_stat])
    phase_shift_box = column([
        row([tap_shift_fig, column([Div(text='tap phase shifting'),
                                    tap_amp_diff_txt,
                                    tap_phase_diff_txt,
                                    tap_err_txt])]),
        row([ref_shift_fig, column([Div(text='ref phase shifting'),
                                    ref_amp_diff_txt,
                                    ref_phase_diff_txt,
                                    ref_err_txt])])
    ])
    return layout(children=[[binning_fig, controls_box, phase_shift_box]])


def make_retraction_layout():
    controls_box = column([
    row([start_retraction_button, delete_last_button, elab_button, metadata_button]),

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
    ])
    plot_box = gridplot([[retraction_amp_plot], [retraction_phase_plot], [retraction_sig_plot]],
                        sizing_mode='stretch_width', merge_tools=False)
    return layout(children=[plot_box, controls_box])


def make_noise_layout():
    place_holder = Div(text='There is nothing here yet to see')
    return layout(children=[place_holder])


def make_line_layout():
    place_holder = Div(text='There is nothing here yet to see')
    return layout(children=[place_holder])


def make_delay_layout():
    place_holder = Div(text='There is nothing here yet to see')
    return layout(children=[place_holder])


sig_tab = TabPanel(child=make_signal_layout(), title='SNOM signals')
tuning_tab = TabPanel(child=make_tuning_layout(), title='SNOM tuning')
noise_tab = TabPanel(child=make_noise_layout(), title='Noise')
retraction_tab = TabPanel(child=make_retraction_layout(), title='Retraction Scan')
line_tab = TabPanel(child=make_line_layout(), title='Line Scan')
delay_tab = TabPanel(child=make_delay_layout(), title='Delay Scan')
tabs = Tabs(tabs=[sig_tab, tuning_tab, noise_tab, retraction_tab, line_tab, delay_tab], active=0)


# DEAMON THREADS #######################################################################################################
acquisition_daemon = Thread(target=acquisition_idle_loop, daemon=True)
acquisition_daemon.start()


# GLOBAL LAYOUT ########################################################################################################
gui_controls = row(children=[go_button, stop_server_button, callback_input, message_box], sizing_mode='stretch_width')
GUI_layout = layout(children=[
    gui_controls,
    tabs
], sizing_mode='stretch_both')
curdoc().add_root(GUI_layout)
curdoc().title = 'SNOMpad'
callback = curdoc().add_periodic_callback(periodic_callback, callback_period)
