# error codes and user messages used in SNOMpad GUI

from bokeh.models import Div

err_code = 0
err_msg = ''
usr_msg = 'message box'


# CALLLBACKS ###########################################################################################################
def update_message_box():
    pass


# WIDGETS ##############################################################################################################
message_box = Div(text=usr_msg)
message_box.styles = {
    'width': '300px',
    'border': '1px solid#000',
    'text-align': 'center',
    'background-color': '#FFFFFF'
}
