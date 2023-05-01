# This is a Bokeh server app. To function, it must be run using the
# Bokeh server at the command line:
#
#     bokeh serve --show TRION_visualizer.py
#
# Running "python TRION_visualizer.py" will NOT work.
import numpy as np
import sys
import threading
from time import sleep, perf_counter
from itertools import chain
from scipy.stats import binned_statistic, binned_statistic_2d

import colorcet as cc
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Toggle, Button, RadioButtonGroup, NumericInput, Div, LinearColorMapper
from bokeh.layouts import layout, column, row

from trion.analysis.signals import Demodulation, Signals, Detector, modulation_signals, detection_signals
from trion.analysis.demod import shd, pshet
from trion.expt.buffer import CircularArrayBuffer
from trion.expt.daq import DaqController

callback_interval = 200  # ms
buffer_size = 300_000
harm_plot_size = 50  # number of values on x-axis when plotting harmonics
raw_plot_size = 1_000  # number of raw data samples that are displayed at one point in time
raw_plot_tail = 300  # number of raw data samples that are added every acquisition cycle (callback interval)
max_harm = 8  # highest harmonics that is plotted
buffer = None  # global variable, so that it can be shared across threads


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
        global buffer
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
            tap_nbins = tap_input.value
            ref_nbins = ref_input.value
            t = perf_counter()
            data = buffer.get(n=npts_input.value)
            rtn = 0
            rtn += update_harmonics(data, tap_nbins=tap_nbins, ref_nbins=ref_nbins)
            rtn += update_raw_and_phase(data, tap_nbins=tap_nbins, ref_nbins=ref_nbins)
            update_message_box(rtn, t=t)
        except Exception as e:
            raise
            update_message_box(str(e))
        
    
def update_harmonics(data, tap_nbins, ref_nbins):
    global harm_scaling
    rtn_value = 0
    try:
        if pshet_button.active is False:
            coefficients = shd(data=data, signals=signals, tap_nbins=tap_nbins, chopped=chop_button.active)
        elif pshet_button.active is True:
            coefficients = np.abs(pshet(data=data, signals=signals, tap_nbins=tap_nbins, ref_nbins=ref_nbins,
                                        chopped=chop_button.active))
        coefficients = coefficients[: max_harm+1, 0]

        if harm_scaling[0] == 0:
            harm_scaling = np.ones(max_harm+1) / coefficients
        harm_scaling[coefficients > 1] = 1 / coefficients[coefficients > 1]
        coefficients *= harm_scaling  # ToDo: this does not affect already plotted data. Change this at some point

        new_data = {str(i): np.array([coefficients[i]]) for i in range(max_harm+1)}
        new_data.update({'t': np.array([perf_counter()])})
        harmonics_plot_data.stream(new_data, rollover=harm_plot_size)

    except Exception as e:
        if 'empty bins' in str(e):
            rtn_value += 1  # empty bins
        else:
            raise
            rtn_value += 10
    return rtn_value


def update_raw_and_phase(data, tap_nbins, ref_nbins):
    rtn_value = 0
    theta_tap = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    theta_ref = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    # phase data
    if pshet_button.active is False:
        returns = binned_statistic(x=theta_tap, values=None, statistic='count',
                                    bins=tap_nbins, range=[-np.pi, np.pi])
        binned = returns.statistic[np.newaxis, :]
    elif pshet_button.active is True:
        
        returns = binned_statistic_2d(x=theta_tap, y=theta_ref, values=None, statistic='count',
                                        bins=[tap_nbins, ref_nbins], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        binned = returns.statistic
    new_data = {'binned': [binned]}
    phase_plot_data.data = new_data

    # raw data
    new_data = {c.value: data[-raw_plot_tail:, signals.index(c)] for c in signals
                if c.value in ['sig_a', 'sig_b', 'chop']}
    if raw_theta_button.active == 0:  # 'raw vs theta_tap'
        new_data.update({'theta': theta_tap[-raw_plot_tail:]})
    elif raw_theta_button.active == 1:  # 'raw vs theta_ref'
        new_data.update({'theta': theta_ref[-raw_plot_tail:]})
    raw_plot_data.stream(new_data, rollover=raw_plot_size)

    return rtn_value


def update_message_box(msg: any, t: float = 0.):
    """ prints how many samples have been acquired and demodulated in how much time,
    and when empty bins occurred during binning
    """
    if isinstance(msg, int):
        if msg % 10 == 0:
            dt = (perf_counter() - t) * 1e3  # ms
            message_box.styles['background-color'] = '#FFFFFF'
            message_box.text = f'demod and plotting: {dt:.0f} ms'
        elif msg % 10 == 1:
            message_box.styles['background-color'] = '#FF7777'
            message_box.text = 'empty bins detected'
    else:
        message_box.styles['background-color'] = '#FF7777'
        message_box.text = msg


# INITIAL PARAMETERS / DATA ############################################################################################
signals = sorted(chain(detection_signals[Detector.dual],
                        modulation_signals[Demodulation.pshet]))
signals.append(Signals['chop'])

harm_scaling = np.zeros(max_harm+1)


# WIDGETS ##############################################################################################################
stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

go_button = Toggle(label='GO', active=False, width=100)
pshet_button = Toggle(label='pshet', active=False, width=100)
chop_button = Toggle(label='chop', active=False, width=100)

raw_theta_button = RadioButtonGroup(labels=['raw vs theta_tap', 'raw vs theta_ref'], active=0)
# phase_sampling_button = RadioButtonGroup(labels=['histogram', 'scatter'], active=1)

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

# SET UP PLOTS #########################################################################################################
def setup_harm_plot():
    # Currently, only sig_a is displayed
    init_data = {str(h): np.ones(harm_plot_size) for h in range(max_harm+1)}
    init_data.update({'t': np.zeros(harm_plot_size)})
    plot_data = ColumnDataSource(init_data)
    fig = figure(aspect_ratio=4)  # toolbar_location=None
    # p.x_range.follow = 'end' #  I think this is needed only when the number of points grows in total
    for h in range(max_harm+1):
        fig.line(x='t', y=str(h), source=plot_data, syncable=False, legend_label=f'O{h}A')
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
    plot = fig.image(image='binned', source=plot_data, color_mapper=cmap, dh=1., dw=1., x=0, y=0, syncable=False)
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
    message_box
])

gui = layout(children=[
    [harmonics_plot],
    [phase_plot, raw_plot, controls_box]
] , sizing_mode='stretch_width')

curdoc().add_root(gui)
curdoc().add_periodic_callback(update, callback_interval)
curdoc().title = 'TRION visualizer'
