# trion.expt.gui.core
# core elements for TRION GUI

import logging
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
import pyqtgraph as pg
from .daq import  DaqPanel

logger = logging.getLogger(__name__)


class DataWindow(pg.GraphicsLayoutWidget):
    """
    Window for data visualization.

    The displayed data can be in multiple modes:
    1. raw (a time series of data points)
    2. XY plot
    3. ...
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # start with a single window
        self.addPlot(row=0, col=0, name="main")


class TRIONMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *a, daq=None, **kw):
        super().__init__(*a, **kw)

        # store references to Trion objects

        # setup threads

        # data view window (central widget)
        self.data_view = DataWindow(parent=self)
        self.setCentralWidget(self.data_view)

        # setup control panels
        # DAQ control panel
        self.daq_panel = DaqPanel("Acquisition", daq=daq, parent=self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.daq_panel)
        # create actions
        # create statusbar
        # create menubar
        # create toolbar

        # connect actions

        # finalize connections
        self.setWindowTitle("TRION Experimental controller")

