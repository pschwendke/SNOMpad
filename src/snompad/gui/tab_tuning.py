# tab for the SNOMpad GUI to acquire, demod, and plot parameters concerning
# experiment performance, calibration, or tuning
import numpy as np
import colorcet as cc
from scipy.stats import binned_statistic, binned_statistic_2d

from bokeh.layouts import layout, column, row
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import RadioButtonGroup, NumericInput, Div, LinearColorMapper

from . import buffer_size, signals
from ..utility import Signals
from ..analysis.utils import tip_frequency
from ..analysis.noise import phase_shifting


# CALLLBACKS ###########################################################################################################
def update_tuning_tab(buffer):
    rtn_value = 0
    try:
        data = buffer.tail(n=npts_input.value)
        tap_x, tap_y = data[:, signals.index(Signals.tap_x)], data[:, signals.index(Signals.tap_y)]
        tap_p = np.arctan2(tap_y, tap_x)
        if mod_button.active == 1:  # pshet
            ref_x, ref_y = data[:, signals.index(Signals.ref_x)], data[:, signals.index(Signals.ref_y)]
            ref_p = np.arctan2(ref_y, ref_x)
        else:
            ref_x, ref_y, ref_p = None, None, None

        rtn_value += update_binning_plot(tap_p, ref_p)
        update_tip_frequency(tap_p)
        update_phase_shift_plots(tap_x, tap_y, ref_x, ref_y)
        update_phase_shift_stats(tap_x, tap_y, ref_x, ref_y)
    except Exception as e:
        print(e)
        rtn_value += 1000
    return rtn_value


def update_binning_plot(tap_p, ref_p):
    rtn_value = 0

    if ref_p is not None:  # pshet
        returns = binned_statistic_2d(x=tap_p, y=ref_p, values=None, statistic='count',
                                      bins=[tap_input.value, ref_input.value], range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        binned = returns.statistic
    else:
        returns = binned_statistic(x=tap_p, values=None, statistic='count',
                                   bins=tap_input.value, range=[-np.pi, np.pi])
        binned = returns.statistic[np.newaxis, :]

    new_data = {'binned': [binned]}
    binning_data.data = new_data

    if any(np.isnan(binned)):
        rtn_value += 2000
    return rtn_value


def update_phase_shift_plots(tap_x, tap_y, ref_x, ref_y):
    new_tap_data = {'tap_x': tap_x[::200],
                    'tap_y': tap_y[::200]}
    tap_shift_data.data = new_tap_data
    if ref_x is not None and ref_y is not None:  # pshet
        new_ref_data = {'ref_x': ref_x[::200],
                        'ref_y': ref_y[::200]}
        ref_shift_data.data = new_ref_data


def print_shift_stats(tap_amp_diff, tap_phase_diff, tap_err,
                      ref_amp_diff, ref_phase_diff, ref_err):
    global tap_shift_stats, ref_shift_stats
    tap_shift_stats = [
        Div(text=f'amp_x - amp_y: {tap_amp_diff:.2e} V'),
        Div(text=f'phi_x - phi_y: {tap_phase_diff:.2f} °'),
        Div(text=f'amplitude err: {tap_err:.2e} %')
    ]
    ref_shift_stats = [
        Div(text=f'amp_x - amp_y: {ref_amp_diff:.2e} V'),
        Div(text=f'phi_x - phi_y: {ref_phase_diff:.2f} °'),
        Div(text=f'amplitude err: {ref_err:.2e} %')
    ]


def update_phase_shift_stats(tap_x, tap_y, ref_x, ref_y):
    _, tap_phase_diff, tap_amp_diff, tap_amp, tap_std, tap_offset = phase_shifting(tap_x, tap_y)
    tap_err = tap_std / tap_amp
    if ref_x is not None and ref_y is not None:  # pshet
        _, ref_phase_diff, ref_amp_diff, ref_amp, ref_std, ref_offset = phase_shifting(ref_x, ref_y)
        ref_err = ref_std / ref_amp
    else:
        ref_amp_diff, ref_phase_diff, ref_err = 0., 0., 0.

    print_shift_stats(tap_amp_diff, tap_phase_diff, tap_err,
                      ref_amp_diff, ref_phase_diff, ref_err)


def update_tip_frequency(tap_p):
    f = tip_frequency(tap_p)
    tip_freq_stat.text = f'Assumed tip frequency: {f/1000:.3f} kHz'


# WIDGETS ##############################################################################################################
mod_button = RadioButtonGroup(name='modulation', labels=['shd', 'pshet'], active=1)
tap_input = NumericInput(title='# tap bins', value=128, mode='int', low=16, high=256, width=90)
ref_input = NumericInput(title='# ref bins', value=84, mode='int', low=16, high=256, width=90)
npts_input = NumericInput(title='# of samples', value=50_000, mode='int', low=10_000, high=buffer_size, width=90)

tip_freq_stat = Div(text='Assumed tip frequency')

tap_shift_stats = []
ref_shift_stats = []
print_shift_stats(0., 0., 0., 0., 0., 0.)


# BINNING PLOTS ########################################################################################################
init_data = {'binned': [np.random.uniform(size=(64, 64))]}
binning_data = ColumnDataSource(init_data)
binning_fig = figure(height=600, width=600, toolbar_location=None)
cmap = LinearColorMapper(palette=cc.bgy, nan_color='#FF0000')
plot = binning_fig.image(image='binned', source=binning_data, color_mapper=cmap,
                         dh=2*np.pi, dw=2*np.pi, x=-np.pi, y=-np.pi, syncable=False)
cbar = plot.construct_color_bar(padding=1)
binning_fig.add_layout(cbar, 'right')


# PHASE SHIFT PLOTS ####################################################################################################
init_data = {'tap_x': [np.zeros(1)], 'tap_y': [np.zeros(1)]}
tap_shift_data = ColumnDataSource(init_data)
tap_shift_fig = figure(height=300, width=300, toolbar_location=None)
tap_shift_fig.scatter(x='tap_x', y='tap_y', source=tap_shift_data,
                      line_color='blue', marker='dot', size=10, syncable=False)

init_data = {'ref_x': [np.zeros(1)], 'ref_y': [np.zeros(1)]}
ref_shift_data = ColumnDataSource(init_data)
ref_shift_fig = figure(height=300, width=300, toolbar_location=None)
ref_shift_fig.scatter(x='ref_x', y='ref_y', source=ref_shift_data,
                      line_color='blue', marker='dot', size=10, syncable=False)


# LAYOUT ###############################################################################################################
controls_box = row([mod_button, npts_input, tap_input, ref_input, tip_freq_stat])
phase_shift_box = column([
    row([tap_shift_fig, column([Div(text='tap phase shifting')] + [s for s in tap_shift_stats])]),
    row([ref_shift_fig, column([Div(text='ref phase shifting')] + [s for s in ref_shift_stats])])
])

tuning_layout = layout(children=[
    controls_box,
    [binning_fig, phase_shift_box]
], sizing_mode='stretch_width')
