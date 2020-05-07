# trion.expt.gui.daq
# GUI elements related to the DAQ

import logging
from PySide2.QtWidgets import (
    QDockWidget, QGridLayout, QLabel, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QGroupBox, QHBoxLayout
)
from PySide2.QtGui import (
    QDoubleValidator
)
from PySide2.QtCore import Qt, QStateMachine, QState, QObject, QTimer

from ..buffer import CircularArrayBuffer
from ..daq import DaqController
from .utils import ToggleButton

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

class DaqPanel(QDockWidget):

    def __init__(self, *args, daq=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_machine = QStateMachine()
        # TODO: offline state
        self.idle = QState()
        #self.busy = QState()
        self.acquiring = QState()
        self.state_machine.addState(self.idle)
        #self.state_machine.addState(self.busy)
        self.state_machine.addState(self.acquiring)


        self.daq = daq
        back_panel = QWidget()
        self.setWidget(back_panel)
        back_panel.setLayout(QVBoxLayout())
        grid_layout = QGridLayout()
        back_panel.layout().addLayout(grid_layout)

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

        btn_layout = QHBoxLayout()
        back_panel.layout().addLayout(btn_layout)
        self.go_btn = ToggleButton("Start", alt_text="Stop")
        btn_layout.addStretch()
        btn_layout.addWidget(self.go_btn)

        # Connect stuff

        self.state_machine.start()

    def setup_daq(self, daq):
        """
        Setup values from the configuration of the controller
        """
        raise NotImplementedError()

    def on_go_btn(self):
        """
        Start acquisition
        """






