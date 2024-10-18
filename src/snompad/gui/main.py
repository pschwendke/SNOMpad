# This is the SNOMpad GUI. It is a bokeh server.
# It can be run from the commandline as 'bokeh serve --show .'
# Alternatively, launch_gui() can be imported from snompad.gui
import sys
import numpy as np
import colorcet as cc
from threading import Thread
from time import perf_counter, sleep
from scipy.stats import binned_statistic, binned_statistic_2d

from bokeh.plotting import curdoc
from bokeh.models import Toggle, Button, NumericInput, TabPanel, Tabs, Div, RadioButtonGroup, LinearColorMapper
from bokeh.layouts import layout, column, row
from bokeh.plotting import figure, ColumnDataSource

# ToDo: configure logging
# from .utility.acquisition_logger import gui_logger
from snompad.gui.buffer import PlottingBuffer
from snompad.gui.demod import demod_to_buffer
from snompad.gui.user_messages import error_message
from snompad.drivers import DaqController
from snompad.acquisition.buffer import CircularArrayBuffer
from snompad.utility import Signals, plot_colors
from snompad.analysis.utils import tip_frequency
from snompad.analysis.noise import phase_shifting

signals = [
    Signals.sig_a,
    Signals.sig_b,
    Signals.tap_x,
    Signals.tap_y,
    Signals.ref_x,
    Signals.ref_y,
    Signals.chop
]

callback_period = 150  # ms  (time after which a periodic callback is started)
buffer_size = 200_000
acquisition_buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
go = False
err_code = 0


def rgb_to_hex(rgb: list) -> str:
    """ Just a conversion of color identifiers to use plot_colors with bokeh
    """
    r = round(rgb[0] * 255)
    g = round(rgb[1] * 255)
    b = round(rgb[2] * 255)
    out = f'#{r:02x}{g:02x}{b:02x}'
    return out


# ACQUISITION ##########################################################################################################
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


def update_noise_tab() -> int:
    rtn_value = 0
    return rtn_value


def update_retraction_tab() -> int:
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
max_harm = 7  # highest harmonics that is plotted, should be lower than 8 (because of display of signal/noise)
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


def make_sig_fig():
    # ToDo: can this go in another file?
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
    # ToDo: can this go in another file?
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
    f = tip_frequency(tap_p)
    tip_freq_stat.text = f'Assumed tip frequency:\n{f/1000:.3f} kHz'


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


# NOISE TAB FUNCTIONS AND CALLBACKS ####################################################################################
# RETRACTION TAB FUNCTIONS AND CALLBACKS ###############################################################################
# LINE SCAN TAB FUNCTIONS AND CALLBACKS ################################################################################
# DELAY SCAN TAB FUNCTIONS AND CALLBACKS ###############################################################################
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

# TUNING TAB
tap_amp_diff_txt = Div(text=f'amp_x - amp_y: {0:.2e} V')
tap_phase_diff_txt = Div(text=f'phi_x - phi_y: {0:.2f} °')
tap_err_txt = Div(text=f'amplitude err: {0:.2e} %')
ref_amp_diff_txt = Div(text=f'amp_x - amp_y: {0:.2e} V')
ref_phase_diff_txt = Div(text=f'phi_x - phi_y: {0:.2f} °')
ref_err_txt = Div(text=f'amplitude err: {0:.2e} %')


# MAKE ALL THE PLOTS ###################################################################################################
# signal tab
sig_fig, signal_buffer = make_sig_fig()
phase_fig, phase_plot_data = make_phase_fig()

# tuning tab
binning_fig, binning_data = make_binning_fig()
tap_shift_fig, tap_shift_data = make_tap_shift_fig()
ref_shift_fig, ref_shift_data = make_ref_shift_fig()


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


def make_noise_layout():
    place_holder = Div(text='There is nothing here yet to see')
    return layout(children=[place_holder])


def make_retraction_layout():
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


# START ACQUISITION ####################################################################################################
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
