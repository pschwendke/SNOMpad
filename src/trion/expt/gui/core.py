# trion.expt.gui.core
# core elements for TRION GUI

import logging
from PySide2 import QtWidgets
from PySide2.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QGridLayout,
                               QDockWidget)
from PySide2.QtCore import Qt
import pyqtgraph as pg
from nidaqmx.system import System

from .qdaq import DaqPanel, ExpConfig, ExpPanel, AcquisitionController
from ..daq import DaqController

logger = logging.getLogger(__name__)


class ViewPanel(QtWidgets.QDockWidget):
    """
    Panel for control of the Data window
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        back_panel = QWidget(parent=self)
        self.setWidget(back_panel)
        main_layout = QVBoxLayout()
        back_panel.setLayout(main_layout)

        # self.view_type = QComboBox(parent=self) # will need this

        grid_layout = QGridLayout()
        back_panel.setLayout(grid_layout)



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
    def __init__(self, *a, daq=None,  **kw):
        super().__init__(*a, **kw)

        # Setup UI elements
        self.data_view = DataWindow(parent=self)
        self.setCentralWidget(self.data_view)

        # setup control panels
        # Experimental control panel
        self.expt_panel = ExpPanel("Experiment", parent=self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.expt_panel)

        # DAQ control panel
        self.daq_panel = DaqPanel("Acquisition", parent=self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.daq_panel)

        # store references to Trion objects
        self.daq = daq or DaqController(dev=System().devices.device_names[0])
        self.acq_cntrl = AcquisitionController(
            daq=self.daq,
            expt_panel=self.expt_panel,
            daq_panel=self.daq_panel,
        )

        # setup threads

        # data view window (central widget)
        # create actions
        # create statusbar
        # create menubar
        # create toolbar

        # connect actions

        # finalize connections
        self.setWindowTitle("TRION Experimental controller")

