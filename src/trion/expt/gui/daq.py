# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging
from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QLabel, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QGroupBox, QHBoxLayout, QComboBox
)
from PySide2.QtGui import (
    QDoubleValidator
)
from PySide2.QtCore import Qt, QStateMachine, QState, QObject, QTimer

from ..buffer import CircularArrayBuffer
from ..daq import DaqController, System
from ...analysis.signals import Scan, Acquisition, Detector
from .utils import ToggleButton, add_grid, enum_to_combo

logger = logging.getLogger(__name__)


class AcquisitionController(QObject):
    def __init__(self, daq: DaqController, *a, update_dt=0.01, **kw):
        """
        This object handle the control of the acquisition.

        It tracks the acquisition process and wraps the DaqController and
        aquisition buffer.

        Currently supports only continuous acquisition in memory.


        """
        super().__init__(*a, **kw)
        self.daq = daq
        self.buffer = None
        self.read_timer = QTimer()

    def close(self):
        self.daq.close()

    def setup(self, signals, buf_size, **kw):
        """
        Prepare the acquisition. Sets up the buffer and prepares the controller.


        Parameters
        ----------
        signals : list of str
            List of required signal types.
        buf_size : int
            Size of the internal buffer.

        Other keyword arguments set the properties of the Daqcontroller.
        """
        self.buffer = CircularArrayBuffer(vars=signals, max_size=buf_size)
        for k, v in kw.items():
            setattr(self.daq, k, v)
        self.daq.setup(buffer=self.buffer)

    def start(self):
        self.daq.start()

    def stop(self):
        self.daq.stop()


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


class ExpPanel(QDockWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setWidget(ExpConfig())
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )


class DaqPanel(QDockWidget):
    def __init__(self, *args, daq=None, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: state machine, somewhere...


        self.daq = daq
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
        self.clock_chan = QLabel("", parent=self) # TODO: use a dropdown
        grid.append([
            "Sample clock", self.clock_chan
        ])
        # TODO: check if we can get the limits of the validators by inspection
        # sample rate
        self.sample_rate = QLineEdit("200 000", parent=self)
        vld = QDoubleValidator(0, 1E6, 100)
        self.sample_rate.setValidator(vld)
        grid.append([
            "Sample rate", self.sample_rate, "Hz"
        ])
        # signal range
        self.sig_range = QLineEdit("1", parent=self)
        self.sig_range.setValidator(QDoubleValidator(-10, 10, 100))
        grid.append([
            "Signal range", self.sig_range, "V"
        ])
        # phase range
        self.phase_range = QLineEdit("1", parent=self)
        self.phase_range.setValidator(QDoubleValidator(-10, 10, 0))
        grid.append([
            "Modulation range", self.phase_range, "V"
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
        btn_layout.addStretch()
        btn_layout.addWidget(self.go_btn)

        # Connect stuff


    def setup_daq(self, daq):
        """
        Setup values from the configuration of the controller
        """
        raise NotImplementedError()

    def on_go_btn(self):
        """
        Start acquisition
        """
        raise NotImplementedError()






