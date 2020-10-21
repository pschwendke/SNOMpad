# trion.expt.gui.core
# core elements for TRION GUI

import logging
from types import SimpleNamespace
from os.path import expanduser
import os.path as pth

from PySide2 import QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QStatusBar, QMessageBox, QLabel, QAction, \
    QToolBar, QFileDialog
from nidaqmx.system import System

from qtlets.qtlets import HasQtlets

from .data_window import RawView, ViewPanel, DisplayController, DataWindow
from .log import QPopupLogDlg
from .qdaq import DaqPanel, ExpPanel
from .acq_ctrl import AcquisitionController, DaqController

logger = logging.getLogger(__name__)


class TRIONMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *a, daq=None, default_dir=None,
                 **kw):
        super().__init__(*a, **kw)
        if default_dir is None:
            default_dir = expanduser("~")
        self.dir = default_dir
        # Setup graphical elements
        # ========================
        # vizualization window
        self.data_view = DataWindow(parent=self)#, size=(1200, 800))
        # Experimental control panel
        self.expt_panel = ExpPanel("Experiment", parent=self)
        # DAQ control panel
        self.daq_panel = DaqPanel("Acquisition", parent=self)
        # Display control panel
        self.view_panel = ViewPanel("Data display", parent=self)

        # setup threads

        # Create actions
        # ==============
        # Most actions get connected by the controllers.
        self.act = SimpleNamespace()  # simple container to organize the actions

        # Application
        # -----------
        self.act.exit = QAction("E&xit", self)
        self.act.exit.setStatusTip("Close the application")

        # DAQ
        # ---
        self.act.run_cont = QAction("Acquire continuous", self)
        self.act.run_cont.setStatusTip("Acquire data continuously")
        self.act.run_finite = QAction("Acquire finite", self)
        self.act.run_finite.setStatusTip("Acquire a finite number of samples")
        self.act.run_finite.setVisible(False)
        self.act.stop = QAction("Stop", self)
        self.act.stop.setStatusTip("Stop current acquisition")
        self.act.self_cal = QAction("Self-calibrate", self)
        self.act.self_cal.setStatusTip("Self calibrate DAQ")

        # Data handling
        # -------------
        self.act.export = QAction("Export data", self)
        self.act.export.setStatusTip("Export raw data to file")

        # Create auxiliary window elements
        # ================================
        # create statusbar
        self.setStatusBar(TrionsStatusBar())

        # create menubar
        app_menu = self.menuBar().addMenu("App")
        app_menu.addAction(self.act.exit)

        acq_menu = self.menuBar().addMenu("DAQ")
        acq_menu.addAction(self.act.self_cal)
        acq_menu.addSeparator()
        acq_menu.addAction(self.act.run_finite)
        acq_menu.addAction(self.act.run_cont)
        acq_menu.addAction(self.act.stop)

        data_menu = self.menuBar().addMenu("Data")
        data_menu.addAction(self.act.export)

        # create toolbar
        self.toolbar = QToolBar("toolbar", self)
        self.addToolBar(self.toolbar)
        self.toolbar.setFloatable(False)
        self.toolbar.setMovable(False)
        self.toolbar.addAction(self.act.self_cal)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act.run_finite)
        self.toolbar.addAction(self.act.run_cont)
        self.toolbar.addAction(self.act.stop)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act.export)

        # create log popup dialog
        self.log_popup = QPopupLogDlg(self)


        # Setup controllers
        # =================
        self.daq = daq or DaqController(dev=System().devices.device_names[0])

        self.display_cntrl = DisplayController(
            parent=self,
            data_window=self.data_view,
            view_panel=self.view_panel,
        )
        self.acq_cntrl = AcquisitionController(
            parent=self,
            daq=self.daq,
            expt_panel=self.expt_panel,
            daq_panel=self.daq_panel,
            display_controller=self.display_cntrl
        )


        # finalize connections
        self.act.exit.triggered.connect(self.close)
        self.act.export.triggered.connect(self.on_export)
        self.display_cntrl.update_fps.connect(self.statusBar().update_fps)

        # build layout
        self.setCentralWidget(self.data_view)
        self.addDockWidget(Qt.RightDockWidgetArea, self.view_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.expt_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.daq_panel)

        self.setWindowTitle("TRION Experimental controller")
        self.resize(800, 480)

    def on_export(self):
        filename, _ = QFileDialog.getSaveFileName(
            parent=self,
            dir=pth.join(self.dir, "trions.csv"),
            filter="csv (*.csv);;npz archive (*.npz);;any (*.*)"
        )
        logger.debug("export filename="+repr(filename))
        if filename:
            self.acq_cntrl.export.emit(filename)


    def shutdown(self):
        # stop acquisition
        self.acq_cntrl.stop()
        # stop threads
        # wait threads
        # close objects
        self.daq.close()
        logger.debug("Main window shutdown complete.")

    def closeEvent(self, event):
        dlg = QMessageBox.warning(
            self,
            "Confirm Exit",
            "Are you sure you wish to quit the TRION experimental controller?",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if dlg == QMessageBox.Ok:
            event.accept()
        else:
            event.ignore()


class TrionsStatusBar(QStatusBar):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fps_indicator = QLabel("fps: -")
        self.addPermanentWidget(self.fps_indicator)

    def showLog(self, msg):
        self.showMessage(msg, 2E3)

    def update_fps(self, value):
        self.fps_indicator.setText(f"fps: {value:.02f}")
