# error codes and user messages used in SNOMpad GUI

from bokeh.models import Div

# ERROR CODES ##########################################################################################################
# ToDo: this


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
