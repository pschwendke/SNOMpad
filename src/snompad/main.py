# This is the SNOMpad GUI. It is a bokeh server.
# It can be run from the commandline as 'bokeh serve --show .'
# Alternatively, launch_gui() can be imported from snompad.gui
import sys

# from .utility.acquisition_logger import gui_logger
from .gui.tab_signal import sig_layout, update_signal_tab, demod_time
from .gui.tab_tuning import tuning_layout, update_tuning_tab
from .gui.user_messages import message_box, update_message_box

from bokeh.plotting import curdoc
from bokeh.layouts import layout, column, row
from bokeh.models import Toggle, Button, RadioButtonGroup, NumericInput, Div, TabPanel, Tabs

refresh_time = 150  # ms
go = False


# CALLLBACKS ###########################################################################################################
def stop(button):
    sys.exit()  # Stop the bokeh server


def periodic_callback():
    pass
    # if go:
    #     if tabs.active == 0:  # signal tab
    #         update_signal_tab()
    #     elif tabs.active == 1:  # tuning tab
    #         update_tuning_tab()
    # update_message_box()
    # update_refresh_time()


# def update_refresh_time():
#     global refresh_time
#     if demod_time > refresh_time:
#         refresh_time = demod_time // 10 * 10 + 10  # round up to next 10 ms
#         callback._period = refresh_time
#
#
# def input_refresh_time(input, old, new):
#     global refresh_time
#     refresh_time = new
#
#
# def update_go(button):
#     global go
#     go = go_button.active


# WIDGETS ##############################################################################################################
go_button = Toggle(label='GO', active=False, width=60)
# go_button.on_change(update_go)

stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

callback_input = NumericInput(title='Periodic callback interval (ms)', value=150, mode='int', width=90)
# callback_input.on_change(input_refresh_time)


# LAYOUT ###############################################################################################################
sig_tab = TabPanel(child=sig_layout, title='SNOM signals')
tuning_tab = TabPanel(child=tuning_layout, title='SNOM tuning')
tabs = Tabs(tabs=[sig_tab, tuning_tab], active=0)

gui_controls = row(children=[go_button, stop_server_button, callback_input, message_box])

GUI_layout = layout(children=[
    gui_controls,
    # tabs
])
curdoc().add_root(GUI_layout)


########################################################################################################################
curdoc().title = 'SNOMpad'
# callback = curdoc().add_periodic_callback(periodic_callback, refresh_time)
