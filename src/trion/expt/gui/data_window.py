from itertools import repeat, cycle

import numpy as np
import pyqtgraph as pg
from enum import IntEnum, auto
from PySide2.QtCore import QSize, QObject
from PySide2.QtWidgets import QStackedWidget, QDockWidget, QWidget, QVBoxLayout, \
    QGridLayout
from .utils import enum_to_combo, IntEdit, add_grid


class DisplayMode(IntEnum):
    raw = 0 # values indidcate the order in the stackwidget


class RawCntrl(QWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        lyt = QGridLayout()
        self.setLayout(lyt)
        grid = []

        add_grid(grid, lyt)
        lyt.setColumnStretch(1,1)
        lyt.setRowStretch(lyt.rowCount(), 1)


class ViewPanel(QDockWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        panel = QWidget()
        lyt = QGridLayout()
        panel.setLayout(lyt)
        self.setWidget(panel)

        self.display_combo = enum_to_combo(DisplayMode)
        self.display_combo.setDisabled(True)
        lyt.addWidget(self.display_combo, 0, 0)

        self.panels = {}
        self.panels[DisplayMode.raw] = RawCntrl()

        self.stack = QStackedWidget()
        lyt.addWidget(self.stack, 1, 0, 1, 2)
        self.stack.addWidget(self.panels[DisplayMode.raw])

        lyt.setColumnStretch(1,1)
        lyt.setRowStretch(lyt.rowCount(), 1)


class DataWindow(pg.GraphicsLayoutWidget):
    """
    Window for data visualization.

    The displayed data can be in multiple modes:
    1. raw (a time series of data points)
    2. XY plot
    3. ...
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # start with a single window
        self.addPlot(row=0, col=0, name="main")
        self.color_cycle = [
            [31, 119, 180, 255],
            [255, 127, 14, 255],
            [44, 160, 44, 255],
            [214, 39, 40, 255],
            [148, 103, 189, 255],
            [140, 86, 75, 255],
            [227, 119, 194, 255],
            [127, 127, 127, 255],
            [188, 189, 34, 255],
            [23, 190, 207, 255],
        ]

        self.curves = {}
        for plot in self.ci.items:
            plot.setDownsampling(mode="subsample", auto=True)
            plot.setClipToView(True)
            plot.enableAutoRange("xy", False)
            plot.setYRange(-2, 2)
            plot.setXRange(0, 200_000)
            plot.showAxis("top")
            plot.showAxis("right")

    def plot(self, data, names): # this is heavy WIP

        #data[~np.isfinite(data)] = 0
        x = np.arange(data.shape[0])
        for i, n in enumerate(names):
            y = data[:,i]
            m = np.isfinite(y)
            self.curves[n].setData(x[m], y[m], connect="finite")

    def prepare_plots(self, names):
        """Prepares windows and pens for the different curves"""
        for item in self.ci.items:
            item.clear()
            item.addLegend()

        cyc = cycle(self.color_cycle)
        for n in names:
            item = self.getItem(0, 0)
            pen = item.plot(pen=next(cyc), name=n.value)
            self.curves[n] = pen

    def minimumSizeHint(self):
        return QSize(400, 400)


class DisplayController(QObject):
    def __init__(self, *a, data_window: DataWindow=None,
                 display_panel: ViewPanel = None,
                 **kw):
        super().__init__(*a, **kw)
        self.data_view = data_window
        self.display_panel = display_panel

        # TODO will need to handle the different display modes. (state machine?)

        #self.display_panel.panels[DisplayMode.raw].downsampling.valueEdited.connect(self.on_downsampling_edited)


