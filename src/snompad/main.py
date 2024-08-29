# This is the SNOMpad GUI. It is a bokeh server.
# It can be run from the commandline as 'bokeh serve --show .'
# Alternatively, launch_gui() can be imported from snompad.gui
import sys
from time import perf_counter, sleep

# from .utility.acquisition_logger import gui_logger
from .gui import buffer_size, signals
from .gui.tab_signal import sig_layout, update_signal_tab
from .gui.tab_tuning import tuning_layout, update_tuning_tab
from .gui.user_messages import error_message
from .drivers import DaqController
from .acquisition.buffer import CircularArrayBuffer

from bokeh.plotting import curdoc
from bokeh.layouts import layout, row
from bokeh.models import Toggle, Button, NumericInput, TabPanel, Tabs, Div

acquisition_buffer = None
callback_period = 150  # ms
go = False
err_code = 0


# ACQUISITION ##########################################################################################################
class Acquisitor:
    def __init__(self) -> None:
        self.idle_loop()  # to make the thread 'start listening'

    def idle_loop(self):
        """ when 'GO' button is not active
        """
        while not go:
            sleep(.01)
        self.acquisition_loop()

    def acquisition_loop(self):
        """ when 'GO' button is active
        """
        global acquisition_buffer, err_code
        acquisition_buffer = CircularArrayBuffer(vars=signals, size=buffer_size)
        daq = DaqController(dev='Dev1', clock_channel='pfi0')
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
            self.idle_loop()


# CALLLBACKS ###########################################################################################################
def stop(button):
    sys.exit()  # Stop the bokeh server


def periodic_callback():
    global err_code
    t_start = perf_counter()
    if go:
        if tabs.active == 0:  # signal tab
            err_code += update_signal_tab(buffer=acquisition_buffer)
        elif tabs.active == 1:  # tuning tab
            update_tuning_tab(buffer=acquisition_buffer)
    dt = (perf_counter() - t_start) * 1e3  # ms
    if err_code == 0:
        usr_msg = f'time to update: {int(dt)} ms'
    else:
        usr_msg = error_message(err_code=err_code)
        err_code = 0
    update_message_box(msg=usr_msg)
    update_refresh_time(dt)


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
    global callback
    curdoc().remove_periodic_callback(callback)
    callback = curdoc().add_periodic_callback(periodic_callback, callback_period)


def update_go(_):
    global go
    go = go_button.active


def update_message_box(msg: str):
    message_box.text = msg


# WIDGETS ##############################################################################################################
go_button = Toggle(label='GO', active=False, width=60)
go_button.on_click(update_go)

stop_server_button = Button(label='stop server')
stop_server_button.on_click(stop)

callback_input = NumericInput(title='Periodic callback period (ms)', value=callback_period, mode='int', width=120)
callback_input.on_change('value', input_callback_period)

message_box = Div(text='message box')
message_box.styles = {
    'width': '300px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}

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
callback = curdoc().add_periodic_callback(periodic_callback, callback_period)
