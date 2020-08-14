# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging
from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox
)
from PySide2.QtCore import Qt

from ..daq import DaqController, System
from ...analysis.signals import Scan, Acquisition, Detector, Experiment
from .utils import ToggleButton, add_grid, enum_to_combo, FloatEdit, IntEdit

logger = logging.getLogger(__name__)


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


    def on_go_btn(self):
        """
        Start acquisition
        """
        raise NotImplementedError()






