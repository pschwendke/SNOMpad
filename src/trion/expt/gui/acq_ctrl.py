from PySide2.QtCore import QObject, QTimer

from trion.expt.buffer import CircularArrayBuffer
from trion.expt.gui.qdaq import logger


class AcquisitionController(QObject):
    def __init__(self, *a, daq=None, expt_panel=None, daq_panel=None,
                 data_window=None,
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
        self.data_view = data_window
        self.daq = daq
        self.buffer = None
        self.display_timer = QTimer()
        self.display_timer.setInterval(display_dt * 1000)
        self.read_timer = QTimer()
        self.read_timer.setInterval(read_dt * 1000)

        self.read_timer.timeout.connect(self.read_next)
        self.display_timer.timeout.connect(self.refresh_display)

        self.refresh_controls()

        self.daq_panel.dev_name.activated.connect(
            lambda: self.set_device(self.daq_panel.dev_name.currentText())
        )
        self.daq_panel.sample_clock.editingFinished.connect(
            lambda: self.acq_ctrl.set_clock_channel(self.daq_panel.sample_clock.text())
        )
        self.daq_panel.sample_rate.valueEdited.connect(self.set_sample_rate)
        self.daq_panel.sig_range.valueEdited.connect(self.set_sig_range)
        self.daq_panel.phase_range.valueEdited.connect(self.set_phase_range)
        self.daq_panel.refresh_btn.clicked.connect(self.refresh_controls)

        self.daq_panel.go_btn.toggled.connect(self.on_go)

    def on_go(self, go):
        if go:
            self.setup()
            self.start()
        else:
            self.stop()

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
        self.buffer = CircularArrayBuffer(vars=signals, max_size=buf_size)
        for k, v in kw.items():
            setattr(self.daq, k, v)
        self.daq.setup(buffer=self.buffer)

    def start(self):
        logger.debug("Starting update")
        self.prepare_display()
        self.daq.start()
        self.display_timer.start()
        self.read_timer.start()

    def stop(self):
        logger.debug("Stopping update")
        self.daq.stop()
        self.read_timer.stop()
        self.display_timer.stop()
        self.display_timer.timeout.emit()  # fire once more, to be sure.

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
        "pass the data from the buffer to the view."
        names = self.buffer.vars
        y = self.buffer.get(self.buffer.size)
        self.data_view.plot(y, names)

    def prepare_display(self):
        self.data_view.prepare_plots(self.buffer.vars)