# trion.expt.gui.core
# core elements for TRION GUI

import logging
from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QStatusBar
from nidaqmx.system import System

from .data_window import RawView, ViewPanel, DisplayController, DataWindow
from .log import QPopupLogDlg
from .qdaq import DaqPanel, ExpPanel
from .acq_ctrl import AcquisitionController
from ..daq import DaqController

logger = logging.getLogger(__name__)


class TRIONMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *a, daq=None,  **kw):
        super().__init__(*a, **kw)

        # Setup UI elements
        self.data_view = DataWindow(parent=self)#, size=(1200, 800))

        # setup control panels
        # Experimental control panel
        self.expt_panel = ExpPanel("Experiment", parent=self)
        # DAQ control panel

        self.daq_panel = DaqPanel("Acquisition", parent=self)

        # Display control panel
        self.view_panel = ViewPanel("Data display", parent=self)

        # store references to Trion objects
        self.daq = daq or DaqController(dev=System().devices.device_names[0])
        self.acq_cntrl = AcquisitionController(
            daq=self.daq,
            expt_panel=self.expt_panel,
            daq_panel=self.daq_panel,
            data_window=self.data_view,
        )
        self.display_cntrl = DisplayController(
            data_window=self.data_view,
            display_panel=self.view_panel,
        )

        # setup threads

        # create actions
        # create statusbar
        self.setStatusBar(TrionsStatusBar())
        # create menubar
        # create toolbar

        # create log popup dialog
        self.log_popup = QPopupLogDlg(self)

        # connect actions

        # finalize connections

        # build layout
        self.setCentralWidget(self.data_view)
        self.addDockWidget(Qt.RightDockWidgetArea, self.view_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.expt_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.daq_panel)

        self.setWindowTitle("TRION Experimental controller")
        self.resize(800, 480)


class TrionsStatusBar(QStatusBar):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def showLog(self, msg):
        self.showMessage(msg, 2E3)