# This is the SNOMpad GUI. It is a bokeh server.
# It can be run from the commandline as 'bokeh serve --show .'
# Alternatively, launch_gui() can be imported from snompad.gui
import sys
from time import perf_counter

# from .utility.acquisition_logger import gui_logger
from .gui.tab_signal import sig_layout, update_signal_tab
from .gui.tab_tuning import tuning_layout, update_tuning_tab
from .gui.user_messages import message_box, update_message_box

from bokeh.plotting import curdoc
from bokeh.layouts import layout, row
from bokeh.models import Toggle, Button, NumericInput, TabPanel, Tabs


callback_period = 150  # ms
go = False

err_code = 0
err_msg = ''
usr_msg = 'SNOMpad GUI'


# CALLLBACKS ###########################################################################################################
def stop(button):
    sys.exit()  # Stop the bokeh server


def periodic_callback():
    global usr_msg
    t_start = perf_counter()
    if go:
        if tabs.active == 0:  # signal tab
            update_signal_tab()
        elif tabs.active == 1:  # tuning tab
            update_tuning_tab()
    if err_code != 0:
        usr_msg = err_msg
    update_message_box(msg=usr_msg)
    update_refresh_time(t_start)


def update_refresh_time(t):
    global callback_period
    dt = (perf_counter() - t) * 1e3  # ms
    if dt > callback_period:
        callback_period = dt // 10 * 10 + 10  # round up to next 10 ms
        renew_callback()


def input_callback_period(_, old, new):
    global callback_period
    callback_period = new
    renew_callback()


def renew_callback():
    global callback
    curdoc().remove_periodic_callback(callback)
    callback = curdoc().add_periodic_callback(periodic_callback, callback_period)


def update_go(_):
    global go
    go = go_button.active


# WIDGETS ##############################################################################################################
go_button = Toggle(label='GO', active=False, width=60)
go_button.on_click(update_go)

stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

callback_input = NumericInput(title='Periodic callback period (ms)', value=callback_period, mode='int', width=120)
callback_input.on_change('value', input_callback_period)


# LAYOUT ###############################################################################################################
sig_tab = TabPanel(child=sig_layout, title='SNOM signals')
tuning_tab = TabPanel(child=tuning_layout, title='SNOM tuning')
tabs = Tabs(tabs=[sig_tab, tuning_tab], active=0)

gui_controls = row(children=[go_button, stop_server_button, callback_input, message_box])

GUI_layout = layout(children=[
    gui_controls,
    tabs
])
curdoc().add_root(GUI_layout)


########################################################################################################################
curdoc().title = 'SNOMpad'
callback = curdoc().add_periodic_callback(periodic_callback, callback_period)
