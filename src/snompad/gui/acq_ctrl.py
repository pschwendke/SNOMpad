import logging
import time
from types import SimpleNamespace
import os.path as pth

from PySide2.QtCore import QObject, QTimer, QStateMachine, QState, Signal, Qt
from PySide2.QtWidgets import QFileDialog, QDockWidget, QWidget, QVBoxLayout, QGridLayout, QComboBox, QHBoxLayout, \
    QPushButton, QCheckBox
from nidaqmx import DaqError
from nidaqmx.system import System
from qtlets.qtlets import HasQtlets
from qtlets.widgets import StrEdit, FloatEdit

from trion.analysis.experiment import Experiment as OriginalExperiment

from trion.expt.buffer.factory import prepare_buffer, BackendType
from trion.expt.buffer.factory import BufferConfig as OriginalBufferConfig
from trion.expt.daq import DaqController as OriginalDaqController
from trion.gui.utils import add_grid, ToggleButton

logger = logging.getLogger(__name__)


class Experiment(HasQtlets, OriginalExperiment):
    pass


class DaqController(HasQtlets, OriginalDaqController):
    pass


class BufferConfig(HasQtlets, OriginalBufferConfig): pass


class AcquisitionController(QObject):
    export = Signal(str)

    def __init__(self, *a, daq=None, expt_panel=None, daq_panel=None,
                 display_controller=None,
                 buffer_cfg: BufferConfig=None,
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
        acquisition buffer.

        Currently supports only continuous acquisition in memory.
        """
        # This can also support our acquisition state machine...
        super().__init__(*a, **kw)
        self.expt_panel = expt_panel
        self.daq_panel = daq_panel
        self.display_cntrl = display_controller
        self.buffer_cfg = buffer_cfg
        self.daq = daq
        self.buffer = None
        self.exp = Experiment()
        self.current_exp = None
        self.act = self.parent().act
        self.n_acquired = 0  # we should instead use buffer.n_written. (or written_current_frame ? vs written_total?)

        # setup state machine
        self.statemachine = QStateMachine()
        self.states = SimpleNamespace()
        self.states.idle = QState()
        self.states.running = QState()

        # state behavior
        self.states.idle.entered.connect(self.stop)
        self.states.running.entered.connect(self.start)

        # state properties
        for obj, attr, status_on_idle in [
            (self.daq_panel.go_btn, "checked", False),
            (self.act.run_cont, "enabled", True),
            (self.act.stop, "enabled", False),
            (self.act.self_cal, "enabled", True),
            (self.act.export, "enabled", True),
        ]:
            self.states.idle.assignProperty(obj, attr, status_on_idle)
            self.states.running.assignProperty(obj, attr, not status_on_idle)

        # state transitions
        for sig in [
            self.daq_panel.go_btn.clicked,
            self.act.run_cont.triggered,
        ]:
            self.states.idle.addTransition(sig, self.states.running)
        for sig in [
            self.daq_panel.go_btn.clicked,
            self.act.stop.triggered,
        ]:
            self.states.running.addTransition(sig, self.states.idle)

        # finalize state machine
        self.statemachine.addState(self.states.idle)
        self.statemachine.addState(self.states.running)
        self.statemachine.setInitialState(self.states.idle)

        self.read_timer = QTimer()
        self.read_timer.setInterval(read_dt * 1000)

        self.read_timer.timeout.connect(self.read_next)

        self.act.self_cal.triggered.connect(self.on_self_calibrate)
        self.export.connect(self.on_export)

        # TODO: have typed comboboxes
        self.daq_panel.dev_name.activated.connect(
            lambda: self.set_device(self.daq_panel.dev_name.currentText())
        )
        self.daq.link_widget(self.daq_panel.sample_clock, "clock_channel")
        self.daq.link_widget(self.daq_panel.sample_rate, "sample_rate")
        self.daq.link_widget(self.daq_panel.sig_range, "sig_range")
        self.daq.link_widget(self.daq_panel.phase_range, "phase_range")
        self.daq.link_widget(self.daq_panel.emulate_data, "emulate")

        # It can be dangerous to change this during the acquisition.
        # It will probably be necessary disable some of these during acquisition
        self.exp.link_widget(self.expt_panel.scan_type, "scan")
        self.exp.link_widget(self.expt_panel.acquisition_type, "acquisition")
        self.exp.link_widget(self.expt_panel.detector_type, "detector")
        self.exp.link_widget(self.expt_panel.npts, "npts")
        self.exp.link_widget(self.expt_panel.frame_reps, "frame_reps")
        self.exp.link_widget(self.expt_panel.continuous, "continuous")
        # continuous not yet supported I think

        self.buffer_cfg.link_widget(self.expt_panel.buffer_type_combo, "backend")
        self.buffer_cfg.link_widget(self.expt_panel.npts, "size")
        self.buffer_cfg.link_widget(self.expt_panel.filename, "fname")
        self.expt_panel.open_btn.clicked.connect(self.on_select_file)

        self.statemachine.start()

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

        self.current_exp = Experiment(**self.exp.to_dict())
        self.current_exp.continuous = (self.current_exp.continuous
                           and self.buffer_cfg.backend is BackendType.numpy)

        # get buffer size
        self.buffer = prepare_buffer(self.current_exp, self.buffer_cfg)
        self.display_cntrl.prepare_plots(self.buffer)
        for k, v in kw.items():
            setattr(self.daq, k, v)
        self.daq.setup(buffer=self.buffer)
        self.n_acquired = 0

    def start(self):
        try:
            self.setup()
            logger.debug("Starting update")
            self.daq.start()
            self.read_timer.start()
            self.display_cntrl.start()
        except Exception:
            self.act.stop.trigger()
            raise

    def stop(self):
        logger.debug("Stopping update")
        self.daq.stop()
        self.read_timer.stop()
        self.display_cntrl.stop()

    def read_next(self):
        # A lot is going on here...
        try:
            n = self.daq.read()

        except (Exception, DaqError):
            # stop otherwise we will raise this infinitely
            self.act.stop.trigger()
            raise
        else:
            self.n_acquired += n
        # i think we should replace this by a `something.is_frame_done()`
        if not self.current_exp.continuous and self.n_acquired >= self.current_exp.npts:
            # this frame is complete
            # i think we should replace this by a `something.is_acquisition_complete()`
            if self.current_exp.frame_reps > 1 and self.buffer.frame_idx < self.current_exp.frame_reps:
                logger.info("Preparing next frame.")
                self.prepare_next_frame()
            else:
                # we're done
                self.act.stop.trigger()
                logger.info("Acquisition complete.")

    def prepare_next_frame(self):
        self.read_timer.stop()
        # it works as long as we're just incrementing a frame_idx
        self.buffer.next_frame()
        self.n_acquired = 0
        self.read_timer.start()

    def set_device(self, name):
        self.daq.dev = name

    def on_self_calibrate(self):
        self.daq.self_calibrate()

    def on_export(self, filename):
        if self.buffer is None:
            raise RuntimeError("Cannot export empty buffer.")
        self.buffer.export(filename)

    def on_select_file(self):
        filename, _ = QFileDialog.getSaveFileName(
            parent=self.parent(),
            dir=pth.join(
                pth.dirname(self.buffer_cfg.fname),
                "trions.h5"),
            filter="hdf5 (*.h5);;any (*.*)"
        )
        logger.debug("File selected: " + repr(filename))
        if filename:
            self.buffer_cfg.fname = filename


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
        self.emulate_data = QCheckBox("")
        self.emulate_data.setChecked(False)
        self.emulate_data.setStatusTip("Emulate data in software")
        grid.append([
            "Emulate?", self.emulate_data, "",
        ])

        add_grid(grid, daq_layout)

        # Button at the end
        btn_layout = QHBoxLayout()
        back_panel.layout().addLayout(btn_layout)
        back_panel.layout().addStretch(1)
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