# ToDo: normalizing/scaling the harmonics plot only works for positive numbers. insert abs somewhere.

import numpy as np
import sys
import os
import threading
# import logging
from time import sleep, perf_counter
# from scipy.stats import binned_statistic, binned_statistic_2d

import colorcet as cc
from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import Toggle, Button, RadioButtonGroup, NumericInput, Div, LinearColorMapper
from bokeh.layouts import layout, column, row, grid

from snompad.utility import Signals
from snompad.demodulation import shd, pshet
from snompad.demodulation.demod_utils import chop_pump_idx
from snompad.acquisition.buffer import CircularArrayBuffer
from snompad.drivers.daq_ctrl import DaqController

if __name__ == '__main__':
    os.system('bokeh serve --show GUI_Visualizer.py')

buffer_size = 200_000
refresh_rate = 100  # ms after which bokeh refreshes the plots
harm_plot_size = 40  # of values on x-axis when plotting harmonics
raw_sample_size = 5000  # size of slice of data to be displayed vs phase. every 10th point is plotted
max_harm = 7  # highest harmonics that is plotted, should be lower than 8 (because of display of signal/noise)
acquisition_buffer = None  # global variable, so that it can be shared across threads
return_msg = 'message box'  # message to be displayed in message box
# ToDo error code could be in binary like in delay stage. CHECK THIS
err_code = 0  # global error code
err_msg = ''
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

doc = curdoc()


# ACQUISITION DAEMON ###################################################################################################
class Acquisitor:
    def __init__(self) -> None:
        self.idle_loop()  # to make the thread 'start listening'

    def idle_loop(self):
        """ when 'GO' button is not active
        """
        while not go_button.active:
            sleep(.01)
        self.acquisition_loop()

    def acquisition_loop(self):
        """ when 'GO' button is active
        """
        global acquisition_buffer, harm_scaling, return_msg, err_code, err_msg
        harm_scaling = np.zeros(max_harm + 1)
        acquisition_buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
        daq = DaqController(dev='Dev1', clock_channel='pfi0')
        daq.setup(buffer=acquisition_buffer)
        try:
            daq.start()
            while go_button.active:
                sleep(.01)
                daq.reader.read()
        except Exception as e:
            # ToDo check if there are 'usual suspects' and catch them specifically
            if err_code == 0:
                err_code = 10
                err_msg = str(e)
            raise
        finally:
            daq.close()
            self.idle_loop()


# DEMODULATION DAEMON ##################################################################################################
class Demodulator:
    def __init__(self) -> None:
        self.idle_loop()  # to make the thread 'start listening'

    def idle_loop(self):
        """ when 'GO' button is not active
        """
        while not go_button.active:
            sleep(.01)
        self.demod_loop()

    def demod_loop(self):
        """ when 'GO' button is active
        """
        global acquisition_buffer, return_msg, err_code, err_msg
        try:
            sleep(.5)  # give the acquisition daemon a little head start
            while go_button.active:
                t = perf_counter()
                tap = tap_input.value
                ref = ref_input.value
                data = acquisition_buffer.tail(n=npts_input.value)
                rtn = update_harmonics(data=data, tap=tap, ref=ref)
                if isinstance(rtn, int):
                    if rtn == 0:
                        return_msg = f'time for demodulation: {(perf_counter() - t) * 1e3:.0f} ms'
                    elif rtn % 10 == 1:
                        return_msg = 'empty bins detected'
                else:
                    return_msg = rtn
        except Exception as e:
            # ToDo check if there are 'usual suspects' and catch them specifically
            if err_code == 0:
                err_code = 20
                err_msg = str(e)
            raise
        finally:
            self.idle_loop()


# CALLBACKS ############################################################################################################
def stop(button):
    sys.exit()  # Stop the bokeh server


def reset_normalization(button):
    global harm_scaling
    harm_scaling = np.zeros(max_harm + 1)


def update():
    global err_code, err_msg
    if go_button.active:
        try:
            # t = perf_counter()
            data = acquisition_buffer.tail(n=npts_input.value)
            # rtn = update_harmonics(data=data, tap=tap, ref=ref)
            update_raw_and_phase(data=data)
            update_signal_to_noise()
            # update_message_box(msg=rtn, dt=(perf_counter() - t) * 1e3)
            update_message_box(msg=return_msg)
        except Exception as e:
            if err_code == 0:
                err_code = 30
                err_msg = str(e)
            raise
    if err_code != 0:
        # ToDo: manage error messages
        go_button.active = False


async def update_harmonics(data, tap, ref):
    global harm_scaling, err_code, err_msg
    rtn_value = 0
    try:
        # if pshet:
        if await mod_button.labels[mod_button.active] == 'pshet':
            coefficients = np.abs(pshet(data=data, signals=signals, tap_res=tap, ref_res=ref, binning='binning',
                                        chopped=chop_button.active, pshet_demod='sidebands',
                                        ratiometry=ratiometry_button.active))
        # if shd:
        elif await mod_button.labels[mod_button.active] == 'shd':
            coefficients = shd(data=data, signals=signals, tap_res=tap, binning='binning',
                               chopped=chop_button.active, ratiometry=ratiometry_button.active)
        
        # if no mod:
        elif await mod_button.labels[mod_button.active] == 'no mod':
            coefficients = np.zeros(max_harm+1)
            if await chop_button.active:
                chopped_idx, pumped_idx = chop_pump_idx(data[:, signals.index(Signals.chop)])
                chopped = data[chopped_idx, signals.index(Signals.sig_a)].mean()
                pumped = data[pumped_idx, signals.index(Signals.sig_a)].mean()
                pump_probe = (pumped - chopped)  # / chopped
                coefficients[0] = chopped
                coefficients[1] = pumped
                coefficients[2] = pump_probe
                rtn_value = '(0) probe only -- (1) pump only -- (2) pump-probe'
            else:
                coefficients[0] = data[:, signals.index(Signals.sig_a)].mean()
                coefficients[1] = data[:, signals.index(Signals.sig_b)].mean()
                rtn_value = '(0) sig_a -- (1) sig_b'

        if await abs_button.active:
            coefficients = np.abs(coefficients[: max_harm+1])
        else:
            coefficients = np.real(coefficients[: max_harm+1])

        # normalize modulated data
        if await mod_button.labels[mod_button.active] != 'no mod':
            if harm_scaling[0] == 0:
                harm_scaling = np.ones(max_harm+1) / np.abs(coefficients)
            for i, over_limit in enumerate(np.abs(coefficients * harm_scaling) > 1):
                if over_limit:
                    harmonics_plot_data.buffer.data[str(i)] /= np.abs(coefficients[i]) * harm_scaling[i]
                    harm_scaling[i] = 1 / np.abs(coefficients[i])
            coefficients *= harm_scaling

        doc.add_next_tick_callback(harmonics_plot_data.put(data=coefficients[:max_harm + 1]))

    except Exception as e:
        if 'empty bins' in str(e):
            rtn_value += 1  # empty bins
            err_code = 41
        else:
            if err_code == 0:
                err_code = 40
                err_msg = str(e)
            raise
    return rtn_value


def update_raw_and_phase(data):
    if mod_button.labels[mod_button.active] == 'no mod':
        return
    # ToDo: restructure this. selection of data can be more efficient / better strucured
    theta_tap = np.arctan2(data[:, signals.index(Signals.tap_y)], data[:, signals.index(Signals.tap_x)])
    theta_ref = np.arctan2(data[:, signals.index(Signals.ref_y)], data[:, signals.index(Signals.ref_x)])
    # phase data
    # if raw_theta_button.active == 1:  # 'raw vs theta_ref'
    #     returns = binned_statistic_2d(x=theta_tap, y=theta_ref, values=None, statistic='count',
    #                                   bins=[tap_res, ref_res], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    #     binned = returns.statistic
    # else:
    #     returns = binned_statistic(x=theta_tap, values=None, statistic='count',
    #                                bins=tap_res, range=[-np.pi, np.pi])
    #     binned = returns.statistic[np.newaxis, :]
    # new_data = {'binned': [binned]}
    # phase_plot_data.data = new_data

    # raw data
    new_data = {c.value: data[-raw_sample_size::10, signals.index(c)] for c in signals if c.value in ['sig_a', 'sig_b', 'chop']}
    if raw_theta_button.active == 1:  # 'raw vs theta_ref'
        new_data.update({'theta': theta_ref[-raw_sample_size::10]})
    else:  # 'raw vs theta_tap'
        new_data.update({'theta': theta_tap[-raw_sample_size::10]})
    raw_plot_data.stream(new_data, rollover=raw_sample_size//10)


def update_message_box(msg: any, dt: float = 0.):
    """ prints how many samples have been acquired and demodulated in how much time,
    and when empty bins occurred during binning
    """
    message_box.styles['background-color'] = '#FFFFFF'
    message_box.text = msg
    if 'empty bins' in msg:
        message_box.styles['background-color'] = '#FF7777'


def update_signal_to_noise():
    for h in harmonics_plot_data.names:
        avg = harmonics_plot_data.avg(key=h)
        std = harmonics_plot_data.std(key=h)
        if std > 0:
            sn = np.abs(avg) / std
            signal_noise[int(h)].text = f'{sn:.1f}'


# WIDGETS ##############################################################################################################
stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

go_button = Toggle(label='GO', active=False, width=60)
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

message_box = Div(text='message box')
message_box.styles = {
    'width': '300px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}

noise_table = grid(sizing_mode='stretch_width', children=column(
    Div(text='Signal to noise'),
    row([signal_noise[h] for h in range(4)]),
    row([signal_noise[h] for h in range(4, 8)])
))


# SET UP PLOTS #########################################################################################################
class PlottingBuffer:
    def __init__(self, n: int, names: list):
        """ simple wrapper around a bokeh ColumnDataSource for easier access
        """
        init_data = {h: np.ones(n) for h in names}
        init_data.update({'t': np.arange(-n + 1, 1)})
        self.buffer = ColumnDataSource(init_data)
        self.names = names

    def put(self, data: np.ndarray):
        for n, key in enumerate(self.names):
            channel = self.buffer.data[key]
            channel[:-1] = channel[1:]
            channel[-1] = data[n]
            self.buffer.data[key] = channel

    def avg(self, key):
        return self.buffer.data[key].mean()

    def std(self, key):
        return self.buffer.data[key].std()


def setup_harm_plot(buffer: ColumnDataSource):
    default_vis = [3, 4, 5]
    fig = figure(aspect_ratio=3, tools='pan,ywheel_zoom,box_zoom,reset,save',
                 active_scroll='ywheel_zoom', active_drag='pan')
    for h in range(max_harm+1):
        fig.line(x='t', y=str(h), source=buffer, line_color=harm_colors[h], line_width=2,
                 syncable=False, legend_label=f'{h}', visible=h in default_vis)
    fig.legend.location = 'center_left'
    fig.legend.click_policy = 'hide'
    return fig


def setup_raw_plot():
    channels = ['sig_a', 'sig_b', 'chop']
    init_data = {c: np.zeros(raw_sample_size//10) for c in channels}
    init_data.update({'theta': np.linspace(-np.pi, np.pi, raw_sample_size//10)})
    plot_data = ColumnDataSource(init_data)
    fig = figure(height=400, width=800, tools='pan,ywheel_zoom,box_zoom,reset,save',
                 active_scroll='ywheel_zoom', active_drag='pan')
    for i, c in enumerate(channels):
        fig.scatter(x='theta', y=c, source=plot_data, marker='dot', line_color=harm_colors[i],
                    size=10, syncable=False, legend_label=c)
    fig.legend.location = 'top_right'
    fig.legend.click_policy = 'hide'
    return fig, plot_data


# def setup_phase_plot():
#     init_data = {'binned': [np.random.uniform(size=(64, 64))]}
#     plot_data = ColumnDataSource(init_data)
#     fig = figure(height=400, width=400, toolbar_location=None)
#     cmap = LinearColorMapper(palette=cc.bgy, nan_color='#FF0000')
#     plot = fig.image(image='binned', source=plot_data, color_mapper=cmap,
#                      dh=2*np.pi, dw=2*np.pi, x=-np.pi, y=-np.pi, syncable=False)
#     cbar = plot.construct_color_bar(padding=1)
#     fig.add_layout(cbar, 'right')
#     return fig, plot_data


# SET UP ROUTINE #######################################################################################################
harmonics_plot_data = PlottingBuffer(n=harm_plot_size, names=[str(h) for h in range(max_harm + 1)])
harmonics_plot = setup_harm_plot(buffer=harmonics_plot_data.buffer)
raw_plot, raw_plot_data = setup_raw_plot()
# phase_plot, phase_plot_data = setup_phase_plot()

acquisition_loop = threading.Thread(target=Acquisitor, daemon=True)
acquisition_loop.start()

demod_loop = threading.Thread(target=Demodulator, daemon=True)
demod_loop.start()


# GUI OUTPUT ###########################################################################################################
controls_box = column([
    row([go_button, chop_button, mod_button]),
    row([tap_input, ref_input, npts_input]),
    row([raw_theta_button, abs_button]),
    row([stop_server_button, ratiometry_button, normalize_button]),
    noise_table,
    message_box
])

gui = layout(children=[
    [harmonics_plot],
    [raw_plot, controls_box]
], sizing_mode='stretch_width')

doc.add_root(gui)
idle_callback = doc.add_periodic_callback(update, refresh_rate)
doc.title = 'SNOMpad visualizer'
