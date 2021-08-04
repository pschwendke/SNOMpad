from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDockWidget, QWidget, QGridLayout, QCheckBox, \
    QHBoxLayout, QPushButton
from qtlets.widgets import IntEdit, ValuedComboBox, StrEdit

from trion.analysis.signals import Scan, Demodulation, Detector
from trion.analysis.experiment import Experiment
from trion.expt.buffer.factory import BackendType
from trion.gui.qdaq import logger
from trion.gui.utils import enum_to_combo, add_grid


class ExpPanel(QDockWidget):
    """
    Widget for holding the experimental configuration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        back_panel = QWidget()
        self.setWidget(back_panel)
        back_panel.setLayout(QGridLayout())

        self.scan_type = enum_to_combo(Scan, ValuedComboBox)
        self.acquisition_type = enum_to_combo(Demodulation, ValuedComboBox)
        self.detector_type = enum_to_combo(Detector, ValuedComboBox)

        self.frame_reps = IntEdit(1, bottom=1)
        self.frame_reps.setStatusTip(
            "Number of repetitions")
        self.buffer_type_combo = ValuedComboBox()
        self.npts = IntEdit(200_000, bottom=10)
        self.npts.setStatusTip("Number of points. Ignored for continuous acquisition.")
        self.continuous = QCheckBox("")
        self.continuous.setChecked(True)
        self.continuous.setStatusTip("Acquire continuously. "
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
            ["Demodulation:", self.acquisition_type],
            ["Detector:", self.detector_type],
            ["Buffer:", self.buffer_type_combo],
            ["Filename:", file_edit],
            ["N reps:", self.frame_reps],
            ["N pts:", self.npts],
            ["Continuous?", self.continuous],
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
            (self.frame_reps, False),
            (self.continuous, True),
        ]
        # finalize layout

        self.buffer_type_combo.valueEdited.connect(self.on_buffer_type_changed)
        self.buffer_type_combo.valueEdited.emit(self.buffer_type_combo.currentData())

        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )

    # def experiment(self): # This is experimental logic in a gui element...
    #     exp = Experiment(
    #         self.scan_type.currentData(),
    #         self.acquisition_type.currentData(),
    #         self.detector_type.currentData(),
    #         nreps=self.nreps.value(),
    #         npts=self.npts.value(),
    #         continuous=self.continous.isChecked(),
    #     )
    #     logger.debug("Experiment is: %s", exp)
    #     return exp

    def on_buffer_type_changed(self, backend):
        # I tried using a statemachine but it crashed.
        flip = backend != BackendType.numpy
        for widget, status in self.enabled_on_numpy:
            widget.setEnabled(status ^ flip)