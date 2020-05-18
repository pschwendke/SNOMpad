# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging
import numpy as np
from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QLabel, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QGroupBox, QHBoxLayout, QComboBox
)
from PySide2.QtGui import (
    QDoubleValidator, QIntValidator
)
from PySide2.QtCore import Qt, QStateMachine, QState, QObject, QTimer

from ..buffer import CircularArrayBuffer
from ..daq import DaqController, System
from ...analysis.signals import Scan, Acquisition, Detector, Experiment
from .utils import ToggleButton, add_grid, enum_to_combo, FloatEdit, IntEdit

logger = logging.getLogger(__name__)


class AcquisitionController(QObject):
    def __init__(self, *a, daq=None, expt_panel=None, daq_panel=None,
                 data_window=None,
                 display_dt=0.02, read_dt=0.01,
                 **kw):
        """
        This object handle the control of the acquisition.

        Parameters
        ----------
        daq : DaqController
            The Daq controller
        display_dt : float
            update time in s. Defaults to 0.01. Note that QTimer internally
            uses ms.

        Tracks the acquisition process and wraps the DaqController and
        aquisition buffer.

        Currently supports only continuous acquisition in memory.
        """
        # This can also support our acquisition state machine...
        super().__init__(*a, **kw)
        self.expt_panel = expt_panel
        self.daq_panel = daq_panel
        self.data_view = data_window
        self.daq = daq
        self.buffer = None
        self.display_timer = QTimer()
        self.display_timer.setInterval(display_dt * 1000)
        self.read_timer = QTimer()
        self.read_timer.setInterval(read_dt * 1000)

        # TODO connect the display timer somewhere...
        self.read_timer.timeout.connect(self.read_next)
        self.display_timer.timeout.connect(self.refresh_display)

        self.refresh_controls()

        self.daq_panel.dev_name.activated.connect(
            lambda: self.set_device(self.daq_panel.dev_name.currentText())
        )
        self.daq_panel.sample_clock.editingFinished.connect(
            lambda: self.acq_ctrl.set_clock_channel(self.daq_panel.sample_clock.text())
        )
        self.daq_panel.sample_rate.valueEdited.connect(self.set_sample_rate)
        self.daq_panel.sig_range.valueEdited.connect(self.set_sig_range)
        self.daq_panel.phase_range.valueEdited.connect(self.set_phase_range)
        self.daq_panel.refresh_btn.clicked.connect(self.refresh_controls)

        self.daq_panel.go_btn.toggled.connect(self.on_go)

    def on_go(self, go):
        if go:
            self.setup()
            self.start()
        else:
            self.stop()

    def close(self):
        self.daq.close()

    def setup(self, **kw):
        """
        Prepare the acquisition. Sets up the buffer and prepares the controller.


        Parameters
        ----------
        signals : list of str
            List of required signal types.
        buf_size : int
            Size of the internal buffer.

        Other keyword arguments set the properties of the DaqController.
        """
        logger.debug("Setting up experiment")
        # get experiment
        exp = self.expt_panel.experiment()
        signals = exp.signals()
        # get buffer size
        buf_size = self.daq_panel.buffer_size.value()
        self.buffer = CircularArrayBuffer(vars=signals, max_size=buf_size)
        for k, v in kw.items():
            setattr(self.daq, k, v)
        self.daq.setup(buffer=self.buffer)

    def start(self):
        logger.debug("Starting update")
        self.prepare_display()
        self.daq.start()
        self.display_timer.start()
        self.read_timer.start()

    def stop(self):
        logger.debug("Stopping update")
        self.daq.stop()
        self.read_timer.stop()
        self.display_timer.stop()
        self.display_timer.timeout.emit()  # fire once more, to be sure.

    def read_next(self):
        n = self.daq.reader.read()

    def set_device(self, name):
        self.daq.dev = name

    def set_clock_channel(self, chan):
        self.daq.clock_channel = chan

    def set_sample_rate(self, rate):
        logger.debug("setting sample_rate to: %s", rate)
        self.daq.sample_rate = rate

    def set_sig_range(self, value):
        self.daq.sig_range = float(value)

    def set_phase_range(self, value):
        self.daq.phase_range = float(value)

    def refresh_controls(self):
        self.daq_panel.dev_name.setCurrentIndex(
            self.daq_panel.dev_name.findText(self.daq.dev)
        )
        self.daq_panel.sample_clock.setText(self.daq.clock_channel)
        self.daq_panel.sample_rate.setValue(self.daq.sample_rate)
        self.daq_panel.sig_range.setValue(self.daq.sig_range)
        self.daq_panel.phase_range.setValue(self.daq.phase_range)

    def refresh_display(self):
        "pass the data from the buffer to the view."
        # TODO: we probably will want to handle some display parameters.
        names = self.buffer.vars
        y = self.buffer.get(self.buffer.size)
        self.data_view.plot(y, names)


    def prepare_display(self):
        self.data_view.prepare_plots(self.buffer.vars)
        # TODO: prepare the graphics layout with the correct number of plots


class ExpConfig(QWidget):
    """
    Widget for holding the experimental configuration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scan_type = enum_to_combo(Scan)
        self.acquisition_type = enum_to_combo(Acquisition)
        self.detector_type = enum_to_combo(Detector)

        self.setLayout(QGridLayout())
        grid = [
            ["Scan type:", self.scan_type],
            ["Acquisition:", self.acquisition_type],
            ["Detector:", self.detector_type]

        ]

        add_grid(grid, self.layout())
        self.layout().setRowStretch(self.layout().rowCount(), 1)

    def experiment(self):
        exp = Experiment(
            self.scan_type.currentData(),
            self.acquisition_type.currentData(),
            self.detector_type.currentData()
        )
        logger.debug("Experiment is: %s", exp)
        return exp


class ExpPanel(QDockWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setWidget(ExpConfig())
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

    def experiment(self): # Not sure I like this...
        return self.widget().experiment()


class DaqPanel(QDockWidget):
    def __init__(self, *args,
                 daq: DaqController = None,
                 acq_ctrl=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: state machine, somewhere...
        self.daq = daq
        self.acq_ctrl = acq_ctrl
        back_panel = QWidget()
        self.setWidget(back_panel)
        back_panel.setLayout(QVBoxLayout())
        grid_layout = QGridLayout()
        back_panel.layout().addLayout(grid_layout)

        grid = []
        # show device name,
        self.dev_name = QComboBox()
        self.dev_name.addItems(System().devices.device_names)
        grid.append([
            "Device", self.dev_name
        ])
        # clock channel
        self.sample_clock = QLineEdit("", parent=self) # TODO: use a dropdown
        grid.append([
            "Sample clock", self.sample_clock
        ])
        # TODO: check if we can get the limits of the validators by inspection
        # sample rate
        self.sample_rate = FloatEdit(200_000, bottom=0, top=1E6, parent=self)
        grid.append([
            "Sample rate", self.sample_rate, "Hz"
        ])
        # signal range
        self.sig_range = FloatEdit(1.0, bottom=-10, top=10, parent=self)
        grid.append([
            "Signal range", self.sig_range, "V"
        ])
        # phase range
        self.phase_range = FloatEdit(1, bottom=-10, top=10, parent=self)
        grid.append([
            "Modulation range", self.phase_range, "V"
        ])
        self.buffer_size = IntEdit(200_000, bottom=0, parent=self)
        grid.append([
            "Buffer size", self.buffer_size
        ])

        add_grid(grid, grid_layout)

        grid_layout.setRowStretch(grid_layout.rowCount(), 1)
        grid_layout.setColumnStretch(1, 1)

        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

        btn_layout = QHBoxLayout()
        back_panel.layout().addLayout(btn_layout)
        self.go_btn = ToggleButton("Start", alt_text="Stop")
        self.refresh_btn = QPushButton("Refresh")
        btn_layout.addStretch()
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.go_btn)

        # Connect stuff
        # THIS IS A LOT OF BOILERPLATE...
        # I should probably make typed QEdits...
        # The "right thing" is a metaclass.... or a class factory?
        # or traits...
        # TODO: move connections to controller.
        # self.dev_name.activated.connect(
        #     lambda: self.acq_ctrl.set_device(self.dev_name.currentText())
        # )
        # self.sample_clock.editingFinished.connect(
        #     lambda: self.acq_ctrl.set_clock_channel(self.sample_clock.text())
        # )
        # self.sample_rate.valueEdited.connect(self.acq_ctrl.set_sample_rate)
        # self.sig_range.valueEdited.connect(self.acq_ctrl.set_sig_range)
        # self.phase_range.valueEdited.connect(self.acq_ctrl.set_phase_range)
        # self.refresh_btn.clicked.connect(self.refresh)

    def on_go_btn(self):
        """
        Start acquisition
        """
        raise NotImplementedError()






