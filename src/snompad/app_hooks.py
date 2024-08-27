# This file is only relevant for the GUI. Threads are set up here.
from threading import Thread

from .gui.acquisition import Acquisitor


def on_server_loaded(server_context):
    acquisition_daemon = Thread(target=Acquisitor, daemon=True)
    acquisition_daemon.start()


def on_server_unloaded(server_context):
    pass
