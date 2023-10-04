import numpy as np
import sys
import os
import threading
from time import sleep, perf_counter
from scipy.stats import binned_statistic, binned_statistic_2d

import colorcet as cc
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Toggle, Button, RadioButtonGroup, NumericInput, Div, LinearColorMapper
from bokeh.layouts import layout, column, row

from trion.analysis.signals import Signals
from trion.analysis.demod import shd, pshet
from trion.expt.buffer import CircularArrayBuffer
from trion.expt.daq import DaqController

if __name__ == '__main__':
    os.system('bokeh serve --show GUI_Retraction.py')

callback_interval = 80  # ms
buffer_size = 200_000
harm_plot_size = 40  # number of values on x-axis when plotting harmonics
raw_plot_tail = 670  # number of raw data samples that are added every acquisition cycle (callback interval)
raw_plot_size = 2 * raw_plot_tail  # number of raw data samples that are displayed at one point in time
max_harm = 5  # highest harmonics that is plotted, should be lower than 8 (because of display of signal/noise)
buffer = None  # global variable, so that it can be shared across threads
signals = [
    Signals.sig_a,
    Signals.sig_b,
    Signals.tap_x,
    Signals.tap_y,
    Signals.ref_x,
    Signals.ref_y,
    Signals.chop
]
harm_scaling = np.zeros(max_harm + 1)  # scaling coefficients for plotting of harmonics
harm_colors = cc.b_glasbey_category10  # color scheme for plotting harmonics of optical signals
signal_noise = {  # collection of signal to noise ratios
    h: Div(text='-', styles={'color': harm_colors[h], 'font-size': '200%'}) for h in range(8)
}


# SIMPLE ACQUISITION IN BACKGROUND #####################################################################################
class Acquisitor:
    def __init__(self) -> None:
        self.waiting_loop()  # to make the thread 'start listening'

    def waiting_loop(self):
        """ when 'GO' button is not active
        """
        while not go_button.active:
            sleep(.01)
        self.acquisition_loop()

    def acquisition_loop(self):
        """ when 'GO' button is active
        """
        global buffer, harm_scaling
        harm_scaling = np.zeros(max_harm + 1)
        buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
        daq = DaqController(dev='Dev1', clock_channel='pfi0')
        daq.setup(buffer=buffer)
        try:
            daq.start()
            while go_button.active:
                sleep(.1)
                daq.reader.read()
        finally:
            daq.close()
        self.waiting_loop()


# CALLBACKS ############################################################################################################
def stop(button):
    sys.exit()  # Stop the bokeh server


def update():
    if go_button.active:
        try:
            tap = tap_input.value
            ref = ref_input.value
            t = perf_counter()
            data = buffer.tail(n=npts_input.value)
            rtn = update_harmonics(data=data, tap=tap, ref=ref)
            update_raw_and_phase(data=data, tap_res=tap, ref_res=ref)
            update_message_box(msg=rtn, t=t)
            update_signal_to_noise()
        except Exception as e:
            update_message_box(str(e))
            raise
        
    
def update_harmonics(data, tap, ref):
    global harm_scaling
    rtn_value = 0
    try:
        if pshet_button.active is True:
            coefficients = np.abs(pshet(data=data, signals=signals, tap_res=tap, ref_res=ref, binning='binning',
                                        chopped=chop_button.active, normalize=False))
        else:
            coefficients = shd(data=data, signals=signals, tap_res=tap,
                               chopped=chop_button.active, normalize=False)
        coefficients = np.abs(coefficients[: max_harm+1])  # add a button to toggle abs()

        if harm_scaling[0] == 0:
            harm_scaling = np.ones(max_harm+1) / coefficients
        for i, over_limit in enumerate(coefficients * harm_scaling > 1):
            if over_limit:
                harmonics_plot_data.data[str(i)] /= coefficients[i] * harm_scaling[i]
                harm_scaling[i] = 1 / coefficients[i]
        coefficients *= harm_scaling

        new_data = {str(i): np.array([coefficients[i]]) for i in range(max_harm+1)}
        new_data.update({'t': np.array([perf_counter()])})
        harmonics_plot_data.stream(new_data, rollover=harm_plot_size)

    except Exception as e:
        if 'empty bins' in str(e):
            rtn_value += 1  # empty bins
        else:
            rtn_value = str(e)
    return rtn_value


def update_raw_and_phase(data, tap_res, ref_res):
    theta_tap = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    theta_ref = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    # phase data
    if pshet_button.active is True:

        returns = binned_statistic_2d(x=theta_tap, y=theta_ref, values=None, statistic='count',
                                      bins=[tap_res, ref_res], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        binned = returns.statistic
    else:
        returns = binned_statistic(x=theta_tap, values=None, statistic='count',
                                   bins=tap_res, range=[-np.pi, np.pi])
        binned = returns.statistic[np.newaxis, :]
    new_data = {'binned': [binned]}
    phase_plot_data.data = new_data

    # raw data
    new_data = {c.value: data[-raw_plot_tail:, signals.index(c)] for c in signals if c.value in ['sig_a', 'sig_b', 'chop']}
    if raw_theta_button.active == 1:  # 'raw vs theta_ref'
        new_data.update({'theta': theta_ref[-raw_plot_tail:]})
    else:  # 'raw vs theta_tap'
        new_data.update({'theta': theta_tap[-raw_plot_tail:]})
    raw_plot_data.stream(new_data, rollover=raw_plot_size)


def update_message_box(msg: any, t: float = 0.):
    """ prints how many samples have been acquired and demodulated in how much time,
    and when empty bins occurred during binning
    """
    if isinstance(msg, int):
        if msg == 0:
            dt = (perf_counter() - t) * 1e3  # ms
            message_box.styles['background-color'] = '#FFFFFF'
            message_box.text = f'demod and plotting: {dt:.0f} ms'
        elif msg % 10 == 1:
            message_box.styles['background-color'] = '#FF7777'
            message_box.text = 'empty bins detected'
    else:
        message_box.styles['background-color'] = '#FF7777'
        message_box.text = msg


def update_signal_to_noise():
    for k in harmonics_plot_data.data.keys():
        if k != 't':
            avg = harmonics_plot_data.data[k].mean()
            std = harmonics_plot_data.data[k].std()
            if std > 0:
                sn = avg / std
                signal_noise[int(k)].text = f'{sn:.2}'


# WIDGETS ##############################################################################################################
stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

go_button = Toggle(label='GO', active=False, width=100)
pshet_button = Toggle(label='pshet', active=False, width=100)
chop_button = Toggle(label='chop', active=False, width=100)

raw_theta_button = RadioButtonGroup(labels=['raw vs theta_tap', 'raw vs theta_ref'], active=0)

tap_input = NumericInput(title='# tap bins', value=64, mode='int', low=16, high=256, width=100)
ref_input = NumericInput(title='# ref bins', value=64, mode='int', low=16, high=256, width=100)
npts_input = NumericInput(title='# of samples', value=50_000, mode='int', low=10_000, high=buffer_size, width=100)

message_box = Div(text='message box')
message_box.styles = {
    'width': '200px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}

noise_table = column(
    Div(text='Signal to noise'),
    row([signal_noise[h] for h in range(4)]),
    row([signal_noise[h] for h in range(4, 8)])
)


# SET UP PLOTS #########################################################################################################
def setup_harm_plot():
    init_data = {str(h): np.ones(harm_plot_size) for h in range(max_harm+1)}
    init_data.update({'t': np.zeros(harm_plot_size)})
    plot_data = ColumnDataSource(init_data)
    fig = figure(aspect_ratio=4)  # toolbar_location=None
    for h in range(max_harm+1):
        fig.line(x='t', y=str(h), source=plot_data, line_color=harm_colors[h], line_width=2,
                 syncable=False, legend_label=f'a{h:02}')
    fig.legend.location = 'center_left'
    fig.legend.click_policy = 'hide'
    return fig, plot_data


def setup_raw_plot():
    channels = ['sig_a', 'sig_b', 'chop']
    init_data = {c: np.zeros(raw_plot_tail) for c in channels}
    init_data.update({'theta': np.linspace(-np.pi, np.pi, raw_plot_tail)})
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=400, width=400)  # toolbar_location=None
    for c in channels:
        fig.scatter(x='theta', y=c, source=plot_data, marker='dot', size=10, syncable=False, legend_label=c)
    fig.legend.location = 'top_right'
    fig.legend.click_policy = 'hide'
    return fig, plot_data


def setup_phase_plot():
    # only display histogram
    init_data = {'binned': [np.random.uniform(size=(64, 64))]}
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=400, width=400, toolbar_location=None)
    cmap = LinearColorMapper(palette=cc.bgy, nan_color='#FF0000')
    plot = fig.image(image='binned', source=plot_data, color_mapper=cmap,
                     dh=2*np.pi, dw=2*np.pi, x=-np.pi, y=-np.pi, syncable=False)
    cbar = plot.construct_color_bar(padding=1)
    fig.add_layout(cbar, 'right')
    return fig, plot_data


########################################################################################################################
harmonics_plot, harmonics_plot_data = setup_harm_plot()
raw_plot, raw_plot_data = setup_raw_plot()
phase_plot, phase_plot_data = setup_phase_plot()

acquisition_loop = threading.Thread(target=Acquisitor, daemon=True)
acquisition_loop.start()


# GUI OUTPUT ###########################################################################################################
controls_box = column([
    row([go_button, pshet_button, chop_button]),
    row([tap_input, ref_input, npts_input]),
    raw_theta_button,
    stop_server_button,
    noise_table,
    message_box
])

gui = layout(children=[
    [harmonics_plot],
    [phase_plot, raw_plot, controls_box]
], sizing_mode='stretch_width')

curdoc().add_root(gui)
curdoc().add_periodic_callback(update, callback_interval)
curdoc().title = 'TRION visualizer'
