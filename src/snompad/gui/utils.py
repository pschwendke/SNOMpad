import os
import logging
import ctypes
from typing import Type

logger = logging.getLogger(__name__)


def rgb_to_hex(rgb: list) -> str:
    """ Just a conversion of color identifiers to use plot_colors with bokeh
    """
    r = round(rgb[0] * 255)
    g = round(rgb[1] * 255)
    b = round(rgb[2] * 255)
    out = f'#{r:02x}{g:02x}{b:02x}'
    return out


def change_to_directory(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    os.chdir(dir)
    logger.info(f'changed to directory {dir}')


def keyboard_interrupt_thread(thread):
    exc = ctypes.py_object(Type[KeyboardInterrupt])
    id = thread.native_id
    ctypes.pythonapi.PyThreadState_SetAsyncExc(id, exc)

