# tab for the SNOMpad GUI to acquire, demod, and plot SNOM signals,
# as well as raw data of the photo diodes

import numpy as np
from time import perf_counter
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import Toggle, Button, RadioButtonGroup, NumericInput, Div, LinearColorMapper
from bokeh.layouts import layout, column, row

from . import buffer_size, harm_scaling, signals, max_harm
from ..utility import Signals, plot_colors
# from .acquisition import acquisition_buffer
# from .user_messages import err_code, err_msg
from .demod import demod_to_buffer
from .buffer import signal_buffer


def rgb_to_hex(rgb: list) -> str:
    r = round(rgb[0] * 255)
    g = round(rgb[1] * 255)
    b = round(rgb[2] * 255)
    out = f'#{r:02x}{g:02x}{b:02x}'
    return out


signal_noise = {  # collection of signal to noise ratios
    h: Div(text='-', styles={'color': rgb_to_hex(plot_colors[h]), 'font-size': '200%'}) for h in range(8)
}
diode_sample_size = 5000  # of diode data points to be plotted vs phase. every 10th point is plotted
# demod_time = 0


def update_signal_to_noise():
    for h in signal_buffer.names:
        avg = signal_buffer.avg(key=h)
        std = signal_buffer.std(key=h)
        if std > 0:
            sn = np.abs(avg) / std
            signal_noise[int(h)].text = f'{sn:.1f}'


def update_diode_data(data):
    global diode_plot_data
    if mod_button.labels[mod_button.active] == 'no mod':
        return
    # ToDo: restructure this. selection of data can be more efficient / better strucured
    theta_tap = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    theta_ref = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])

    # raw data
    new_data = {c.value: data[-diode_sample_size::20, signals.index(c)] for c in signals if c.value in ['sig_a', 'sig_b', 'chop']}
    if raw_theta_button.active == 1:  # 'raw vs theta_ref'
        new_data.update({'theta': theta_ref[-diode_sample_size::20]})
    else:  # 'raw vs theta_tap'
        new_data.update({'theta': theta_tap[-diode_sample_size::20]})
    diode_plot_data.data = new_data


# CALLBACKS ############################################################################################################
def update_signal_tab():
    # ToDo: figure out how to pass messages from here
    global err_code, err_msg
    try:
        t = perf_counter()
        data = data
        err_code = demod_to_buffer(
            data=data,
            modulation=mod_button.labels[mod_button.acitve],
            tap=tap_input.value,
            ref=ref_input.value,
            chop=chop_button.active,
            ratiometry=ratiometry_button.active,
            abs=abs_button.active
        )
        update_signal_to_noise()
        update_diode_data(data=data)
    except Exception as e:
        err_code = 1
        err_msg = 'update_signal_tab: ' + str(e)


def reset_normalization(button):
    global harm_scaling
    harm_scaling = np.zeros(max_harm + 1)


# SIGNAL PLOT ##########################################################################################################
default_visible = [3, 4, 5]
sig_fig = figure(aspect_ratio=3, tools='pan,ywheel_zoom,box_zoom,reset,save',
                 active_scroll='ywheel_zoom', active_drag='pan')
for h in range(max_harm+1):
    sig_fig.line(x='t', y=str(h), source=signal_buffer.buffer, line_color=rgb_to_hex(plot_colors[h]),
                 line_width=2, syncable=False, legend_label=f'{h}', visible=h in default_visible)
sig_fig.legend.location = 'center_left'
sig_fig.legend.click_policy = 'hide'


# DIODE PLOT ###########################################################################################################
channels = ['sig_a', 'sig_b', 'chop']
init_data = {c: np.zeros(diode_sample_size//20) for c in channels}
init_data.update({'theta': np.linspace(-np.pi, np.pi, diode_sample_size//20)})
diode_plot_data = ColumnDataSource(init_data)
diode_fig = figure(height=400, width=800, tools='pan,ywheel_zoom,box_zoom,reset,save',
                   active_scroll='ywheel_zoom', active_drag='pan')
for n, c in enumerate(channels):
    diode_fig.scatter(x='theta', y=c, source=diode_plot_data, marker='dot', line_color=rgb_to_hex(plot_colors[n]),
                      size=10, syncable=False, legend_label=c)
diode_fig.legend.location = 'top_right'
diode_fig.legend.click_policy = 'hide'


# WIDGETS ##############################################################################################################
mod_button = RadioButtonGroup(labels=['no mod', 'shd', 'pshet'], active=1)
chop_button = Toggle(label='chop', active=False, width=60)
abs_button = Toggle(label='abs', active=False, width=60)
ratiometry_button = Toggle(label='ratiometry', active=False, width=100)
normalize_button = Button(label='reset norm')
normalize_button.on_click(reset_normalization)

raw_theta_button = RadioButtonGroup(labels=['raw vs theta_tap', 'raw vs theta_ref'], active=0)

tap_input = NumericInput(title='# tap bins', value=128, mode='int', low=16, high=256, width=90)
ref_input = NumericInput(title='# ref bins', value=84, mode='int', low=16, high=256, width=90)
npts_input = NumericInput(title='# of samples', value=50_000, mode='int', low=10_000, high=buffer_size, width=90)

noise_table = column(
    Div(text='Signal to noise'),
    row([signal_noise[h] for h in range(4)]),
    row([signal_noise[h] for h in range(4, 8)])
)

# LAYOUT ###############################################################################################################
controls_box = column([
    row([chop_button, mod_button]),
    row([tap_input, ref_input, npts_input]),
    row([raw_theta_button, abs_button]),
    row([ratiometry_button, normalize_button]),
    noise_table,
])

sig_layout = layout(children=[
    [sig_fig],
    [diode_fig, controls_box]
], sizing_mode='stretch_width')
