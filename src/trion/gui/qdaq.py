# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging
from types import SimpleNamespace

from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QStackedWidget, QCheckBox,
)
from PySide2.QtCore import Qt, QStateMachine, QState, QEvent, \
    QAbstractTransition

from qtlets.widgets import StrEdit, FloatEdit, IntEdit, ValuedComboBox

from trion.expt.daq import System
from trion.analysis.signals import Scan, Acquisition, Detector, Experiment
from trion.expt.buffer.factory import BackendType
from .buffer import MemoryBufferPanel, H5BufferPanel
from .utils import ToggleButton, add_grid, enum_to_combo

logger = logging.getLogger(__name__)


class ExpPanel(QDockWidget):
    """
    Widget for holding the experimental configuration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        back_panel = QWidget()
        self.setWidget(back_panel)
        back_panel.setLayout(QGridLayout())

        self.scan_type = enum_to_combo(Scan)
        self.acquisition_type = enum_to_combo(Acquisition)
        self.detector_type = enum_to_combo(Detector)

        self.nreps = IntEdit(1, bottom=1)
        self.nreps.setStatusTip(
            "Number of repetitions")
        self.buffer_type_combo = ValuedComboBox()
        self.npts = IntEdit(200_000, bottom=10)
        self.npts.setStatusTip("Number of points. Ignored for continuous acquisition.")
        self.continous = QCheckBox("")
        self.continous.setStatusTip("Acquire continuously. "
                                    "Not possible for all buffer types.")
        self.filename = StrEdit("trions.h5")
        file_edit = QWidget()
        file_edit.setLayout(QHBoxLayout())
        file_edit.layout().setContentsMargins(0,0,0,0)
        file_edit.layout().addWidget(self.filename)
        self.open_btn = QPushButton("-")
        self.open_btn.setFixedSize(23, 23)
        file_edit.layout().addWidget(self.open_btn)
        grid = [
            ["Scan type:", self.scan_type],
            ["Acquisition:", self.acquisition_type],
            ["Detector:", self.detector_type],
            ["Buffer:", self.buffer_type_combo],
            ["Filename:", file_edit],
            ["N reps:", self.nreps],
            ["N pts:", self.npts],
            ["Continuous?", self.continous],
        ]

        for name, widget, backend in [
            ("Memory", None, BackendType.numpy),
            ("H5 file", None, BackendType.hdf5),
        ]:
            self.buffer_type_combo.addItem(name, backend)

        add_grid(grid, back_panel.layout())
        back_panel.layout().setRowStretch(back_panel.layout().rowCount(), 1)

        self.enabled_on_numpy = [
            (self.filename, False),
            (self.open_btn, False),
            (self.nreps, False),
            (self.continous, True),
        ]
        # finalize layout

        self.buffer_type_combo.valueEdited.connect(self.on_buffer_type_changed)
        self.buffer_type_combo.valueEdited.emit(self.buffer_type_combo.currentData())

        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

    def experiment(self): # This is experimental logic in a gui element...
        exp = Experiment(
            self.scan_type.currentData(),
            self.acquisition_type.currentData(),
            self.detector_type.currentData(),
        )
        logger.debug("Experiment is: %s", exp)
        return exp

    def on_buffer_type_changed(self, backend):
        # I tried using a statemachine but it crashed.
        flip = backend != BackendType.numpy
        for widget, status in self.enabled_on_numpy:
            widget.setEnabled(status ^ flip)


# class ExpPanel(QDockWidget):
#     def __init__(self, *a, **kw):
#         super().__init__(*a, **kw)
#         self.setWidget(ExpConfig())
#         self.setAllowedAreas(Qt.LeftDockWidgetArea)
#         self.setFeatures(
#             QDockWidget.DockWidgetMovable |
#             QDockWidget.DockWidgetFloatable
#         )
#
#     def experiment(self): # Not sure I like this...
#         return self.widget().experiment()


class DaqPanel(QDockWidget):
    def __init__(self, *args,
                 acq_ctrl=None, buffer_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.acq_ctrl = acq_ctrl
        self.buffer_cfg = buffer_cfg

        back_panel = QWidget()
        self.setWidget(back_panel)
        back_panel.setLayout(QVBoxLayout())
        daq_layout = QGridLayout()
        back_panel.layout().addLayout(daq_layout)

        # Populate DAQ group
        grid = []
        # show device name,
        self.dev_name = QComboBox()
        self.dev_name.addItems(System().devices.device_names)
        grid.append([
            "Device", self.dev_name
        ])
        # clock channel
        self.sample_clock = StrEdit("", parent=self)  # TODO: use a dropdown
        grid.append([
            "Sample clock", self.sample_clock
        ])
        # sample rate
        self.sample_rate = FloatEdit(200_000, bottom=0, top=1E6, parent=self)
        grid.append([
            "Sample rate", self.sample_rate, "Hz"
        ])
        # signal range
        # TODO: use a dropdown
        self.sig_range = FloatEdit(1.0, bottom=-10, top=10, parent=self)
        grid.append([
            "Signal range", self.sig_range, "V"
        ])
        # phase range
        # TODO: use a dropdown
        self.phase_range = FloatEdit(1, bottom=-10, top=10, parent=self)
        grid.append([
            "Modulation range", self.phase_range, "V"
        ])

        add_grid(grid, daq_layout)

        # Button at the end
        btn_layout = QHBoxLayout()
        back_panel.layout().addLayout(btn_layout)
        back_panel.layout().addStretch(1)
        # TODO: these all need to be made into actions!
        self.go_btn = ToggleButton("Start", alt_text="Stop")
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.go_btn)


        # Finalize
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )
