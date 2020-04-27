# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging
from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QLabel, QPushButton, QLineEdit, QWidget,
    QVBoxLayout,
)
from PySide2.QtGui import (
    QDoubleValidator
)
from PySide2.QtCore import Qt
from ..daq import DaqController

logger = logging.getLogger(__name__)

class DaqPanel(QDockWidget):

    def __init__(self, *args, daq=None, **kwargs):
        super().__init__(*args, **kwargs) # TODO: update title to something nicer
        self.daq = daq
        back_panel = QWidget()
        self.setWidget(back_panel)
        grid_layout = QGridLayout()
        back_panel.setLayout(grid_layout)

        grid = []
        # show device name,
        self.dev_name = QLabel("", parent=self)
        self.change_dev = QPushButton("Change", parent=self)
        grid.append([
            "Device", self.dev_name, self.change_dev
        ])
        # clock channel
        self.clock_chan = QLabel("", parent=self) # TODO: use a dropdown
        grid.append([
            "Sample clock", self.clock_chan
        ])
        # TODO: check if we can get the limits of the validators by inspection
        # sample rate
        self.sample_rate = QLineEdit("", parent=self)
        vld = QDoubleValidator(0, 1E6, 100)
        self.sample_rate.setValidator(vld)
        grid.append([
            "Sample rate", self.sample_rate, "Hz"
        ])
        # signal range
        self.sig_range = QLineEdit("", parent=self)
        self.sig_range.setValidator(QDoubleValidator(-10, 10, 100))
        grid.append([
            "Signal range", self.sig_range, "V"
        ])
        # phase range
        self.phase_range = QLineEdit("", parent=self)
        self.phase_range.setValidator(QDoubleValidator(-10, 10, 0))
        grid.append([
            "Modulation range", self.phase_range, "V"
        ])
        # buttons: channel map
        # self.channel_map = QPushButton("Channel map...", parent=self)
        # self.channel_map.setDisabled(True)

        for i, line in enumerate(grid):
            for j, elem in enumerate(line):
                if elem is None:  continue
                if isinstance(elem, str):
                    elem = QLabel(elem)
                grid_layout.addWidget(elem, i, j)

        grid_layout.setRowStretch(grid_layout.rowCount(), 1)
        grid_layout.setColumnStretch(1, 1)
        # main_layout.addWidget(self.channel_map,
        #     main_layout.rowCount(), main_layout.columnCount()-2, 2, 1)

        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

    def connect_daq(self, daq):
        """
        Setup values from the configuration of the controller
        """
        raise NotImplementedError()




