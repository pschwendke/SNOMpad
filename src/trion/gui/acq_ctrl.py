import logging
from types import SimpleNamespace
import os.path as pth
from copy import deepcopy

import attr

from PySide2.QtCore import QObject, QTimer, QStateMachine, QState, Signal
from PySide2.QtWidgets import QFileDialog
from qtlets.qtlets import HasQtlets

from trion.analysis.signals import Scan, Acquisition, Detector
from trion.analysis.signals import Experiment as OriginalExperiment
from trion.expt.buffer import CircularArrayBuffer
from trion.expt.buffer.factory import prepare_buffer, BackendType
from trion.expt.daq import DaqController as OriginalDaqController

logger = logging.getLogger(__name__)


class Experiment(HasQtlets, OriginalExperiment):
    pass


class DaqController(HasQtlets, OriginalDaqController):
    pass


@attr.s(order=False)
class BufferConfig(HasQtlets):
    """Holds the configuration of the buffer."""
    fname: str = attr.ib(default="")
    backend: BackendType = attr.ib(default=BackendType.numpy)
    size: int = attr.ib(default=200_000)
    continuous: bool = attr.ib(default=True)
    returnable = ["fname", "backend", "size", "continuous"]

    def __attrs_post_init__(self):
        super().__init__()  # tsk tsk tsk...

    def config(self):
        return {k: getattr(self, k) for k in self.returnable}


class AcquisitionController(QObject):
    export = Signal(str)
    def __init__(self, *a, daq=None, expt_panel=None, daq_panel=None,
                 display_controller: DaqController=None,
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
        self.n_acquired = 0

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

        self.display_timer = QTimer()
        self.display_timer.setInterval(display_dt * 1000)
        self.read_timer = QTimer()
        self.read_timer.setInterval(read_dt * 1000)

        self.read_timer.timeout.connect(self.read_next)
        self.display_timer.timeout.connect(self.refresh_display)

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

        # It can be dangerous to change this during the acquisition.
        # It will probably be necessary disable some of these during acquisition
        self.exp.link_widget(self.expt_panel.scan_type, "scan")
        self.exp.link_widget(self.expt_panel.acquisition_type, "acquisition")
        self.exp.link_widget(self.expt_panel.detector_type, "detector")
        self.exp.link_widget(self.expt_panel.npts, "npts")
        self.exp.link_widget(self.expt_panel.nreps, "nreps")
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
        config = self.exp.__dict__.copy()
        try:
            del config["qtlets"]
        except KeyError:
            print(type(self.exp))
            raise
        self.current_exp = OriginalExperiment(**config)
        self.current_exp.continuous = (self.current_exp.continuous
                           and self.buffer_cfg.backend is BackendType.numpy)
        signals = self.current_exp.signals()
        # get buffer size
        buf_cfg = self.buffer_cfg.config()
        self.buffer = prepare_buffer(vars=signals, **buf_cfg)
        for k, v in kw.items():
            setattr(self.daq, k, v)
        self.daq.setup(buffer=self.buffer)
        self.n_acquired = 0

    def start(self):
        self.setup()
        logger.debug("Starting update")
        self.prepare_display()
        self.daq.start()
        self.display_timer.start()
        self.read_timer.start()
        self.display_cntrl.fps_updt_timer.start()

    def stop(self):
        logger.debug("Stopping update")
        self.daq.stop()
        self.read_timer.stop()
        self.display_timer.stop()
        self.display_timer.timeout.emit()  # fire once more, to be sure.
        self.display_cntrl.fps_updt_timer.stop()

    def read_next(self):
        try:
            n = self.daq.reader.read()
        except Exception:
            self.act.stop.trigger()
            raise
        else:
            self.n_acquired += n
        if not self.current_exp.continuous and self.n_acquired >= self.current_exp.npts:
            self.act.stop.trigger()

    def set_device(self, name):
        self.daq.dev = name

    def refresh_display(self):
        "pass the data from the buffer to the display controller."
        try:
            names = self.buffer.vars
        except AttributeError:
            if self.buffer is not None:
                raise
            # otherwise we just don't have a buffer yet...
        else:
            y = self.buffer.tail(self.buffer.size)
            self.display_cntrl.plot(y, names)

    def prepare_display(self):
        self.display_cntrl.prepare_plots(self.buffer.vars)

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

