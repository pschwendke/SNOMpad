import logging
from types import SimpleNamespace

from PySide2.QtCore import QObject, QTimer, QStateMachine, QState, Signal
from PySide2.QtWidgets import QFileDialog
from qtlets.qtlets import HasQtlets
from trion.expt.buffer import CircularArrayBuffer
from trion.expt.daq import DaqController as OriginalDaqController

logger = logging.getLogger(__name__)


class DaqController(HasQtlets, OriginalDaqController):
    pass


class AcquisitionController(QObject):
    export = Signal(str)
    def __init__(self, *a, daq=None, expt_panel=None, daq_panel=None,
                 display_controller: DaqController=None,
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
        self.display_cntrl = display_controller
        self.daq = daq
        self.buffer = None
        self.act = self.parent().act

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

        #self.act.run_cont.toggled.connect(self.toggle_acquisition)

        # TODO: have typed comboboxes
        self.daq_panel.dev_name.activated.connect(
            lambda: self.set_device(self.daq_panel.dev_name.currentText())
        )
        self.daq.link_widget(self.daq_panel.sample_clock, "clock_channel")
        self.daq.link_widget(self.daq_panel.sample_rate, "sample_rate")
        self.daq.link_widget(self.daq_panel.sig_range, "sig_range")
        self.daq.link_widget(self.daq_panel.phase_range, "phase_range")

        self.daq_panel.refresh_btn.clicked.connect(self.refresh_controls)

        #self.daq_panel.go_btn.clicked.connect(self.on_go_btn)

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
        exp = self.expt_panel.experiment()
        signals = exp.signals()
        # get buffer size
        buf_size = self.daq_panel.buffer_size.value()
        self.buffer = CircularArrayBuffer(vars=signals, size=buf_size)
        for k, v in kw.items():
            setattr(self.daq, k, v)
        self.daq.setup(buffer=self.buffer)

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

