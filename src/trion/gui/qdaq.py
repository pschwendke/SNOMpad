# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging

from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QStackedWidget,
)
from PySide2.QtCore import Qt

from qtlets.widgets import StrEdit, FloatEdit, IntEdit, ValuedComboBox

from trion.expt.daq import System
from trion.analysis.signals import Scan, Acquisition, Detector, Experiment
from trion.expt.buffer.factory import BackendType
from .buffer import MemoryBufferPanel, H5BufferPanel
from .utils import ToggleButton, add_grid, enum_to_combo

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
                 acq_ctrl=None, buffer_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.acq_ctrl = acq_ctrl
        self.buffer_cfg = buffer_cfg

        back_panel = QWidget()
        self.setWidget(back_panel)
        back_panel.setLayout(QVBoxLayout())
        daq_group = QGroupBox("DAQ")
        daq_group.setFlat(True)
        back_panel.layout().addWidget(daq_group)
        buffer_group = QGroupBox("Buffer")
        buffer_group.setFlat(True)
        back_panel.layout().addWidget(buffer_group)

        # Populate DAQ group
        daq_group.setLayout(QGridLayout())
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

        add_grid(grid, daq_group.layout())

        # Populate buffer group
        buffer_group.setLayout(QGridLayout())
        grid = []
        self.buffer_type_combo = ValuedComboBox()

        grid.append(["Type", self.buffer_type_combo])
        add_grid(grid, buffer_group.layout())

        self.memory_buffer = MemoryBufferPanel()
        self.h5_buffer = H5BufferPanel()

        self.buffer_stack = QStackedWidget()
        for name, widget, backend in [
            ("Memory", self.memory_buffer, BackendType.numpy),
            ("H5 file", self.h5_buffer, BackendType.hdf5),
        ]:
            self.buffer_type_combo.addItem(name, backend)
            self.buffer_stack.addWidget(widget)

        buffer_group.layout().addWidget(
            self.buffer_stack,
            1,
            0, 1, buffer_group.layout().columnCount()
        )
        self.buffer_stack.setCurrentIndex(0)

        self.buffer_cfg.link_widget(self.buffer_type_combo, "backend")
        self.buffer_cfg.link_widget(self.memory_buffer.buffer_size, "size")
        self.buffer_cfg.link_widget(self.h5_buffer.buffer_size, "size")
        self.buffer_cfg.link_widget(self.h5_buffer.filename, "fname")

        for grp in [daq_group, buffer_group]:
            grp.layout().setRowStretch(grp.layout().rowCount(), 1)
            grp.layout().setColumnStretch(1, 1)


        # Button at the end
        btn_layout = QHBoxLayout()
        back_panel.layout().addLayout(btn_layout)
        # TODO: these all need to be made into actions!
        self.go_btn = ToggleButton("Start", alt_text="Stop")
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.go_btn)

        self.buffer_type_combo.currentIndexChanged.connect(self.on_buffer_type_changed)

        # Finalize
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

    def on_buffer_type_changed(self, idx: int):
        self.buffer_stack.setCurrentIndex(idx)
