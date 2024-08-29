# error codes and user messages used in SNOMpad GUI
import numpy as np
from bokeh.models import Div


def digits(n):
    out = []
    length = int(np.log10(n) // 1) + 1
    for i in range(length):
        out.append(n // 10**i % 10)
    return out


# ERROR CODES ##########################################################################################################
messages = [
    {  # 10 ** 0
        1: 'exception during demodulation',
        2: 'empty bins in binned data'
    },
    {1: 'exception during updating of signal tab'},  # 10 ** 1
    {1: 'exception in acquisition loop'},  # 10 ** 2
]


def error_message(err_code):
    if err_code == 0:
        return 'no error'
    else:
        err_msg = ''
        for i, d in enumerate(digits(err_code)):
            err_msg += messages[i][d] + '\n'
    return err_msg


# CALLLBACKS ###########################################################################################################
def update_message_box(msg: str):
    message_box.text = msg


# WIDGETS ##############################################################################################################
message_box = Div(text='message box')
message_box.styles = {
    'width': '300px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}
