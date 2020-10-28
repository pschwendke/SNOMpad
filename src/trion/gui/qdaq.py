# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging

from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox, )
from PySide2.QtCore import Qt

from qtlets.widgets import StrEdit, FloatEdit

from trion.expt.daq import System
from .utils import ToggleButton, add_grid

logger = logging.getLogger(__name__)


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
