from itertools import repeat, cycle

import logging
import numpy as np
import pyqtgraph as pg
from enum import IntEnum, auto
from PySide2.QtCore import QSize, QObject, QStateMachine, QState, Signal
from PySide2.QtWidgets import QStackedWidget, QDockWidget, QWidget, QVBoxLayout, \
    QGridLayout, QTabWidget
from bidict import bidict
from pyqtgraph import mkPen, mkBrush

from .utils import enum_to_combo, IntEdit, add_grid
from ...analysis import signals
from ...analysis.signals import signal_colormap, Signals

logger = logging.getLogger(__name__)

class DisplayMode(IntEnum):
    raw = 0 # values indidcate the order in the stackwidget
    phase = 1


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
        # TODO: populate this using a QStacked Widget.
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

        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )


class RawView(pg.GraphicsLayoutWidget):
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

        self.cmap = signal_colormap()

        self.curves = {}
        for plot in self.ci.items:
            plot.setDownsampling(mode="subsample", auto=True)
            plot.setClipToView(False)
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

        for n in names:
            item = self.getItem(0, 0)
            pen = item.plot(pen=self.cmap[n], name=n.value)
            self.curves[n] = pen

    def minimumSizeHint(self):
        return QSize(400, 400)


class PhaseView(pg.GraphicsLayoutWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # start with a single window
        self.addPlot(row=0, col=0, name="tap phase")

        self.cmap = signal_colormap()

        # index of columns for data processing
        self.x_idx = None
        self.y_idx = None
        self.columns = {}
        self.curves = {} # merge with above?

        for plot in self.ci.items:
            plot.setDownsampling(mode="subsample", auto=False, ds=20)
            plot.setClipToView(False)
            plot.enableAutoRange("xy", False)
            plot.setYRange(-2, 2)
            plot.setXRange(-np.pi, np.pi)
            plot.showAxis("top")
            plot.showAxis("right")

    def prepare_plots(self, names):
        """Prepares windows and pens for the different curves"""
        for item in self.ci.items:
            item.clear()
            item.addLegend()

        self.x_idx = names.index(Signals.tap_x)
        self.y_idx = names.index(Signals.tap_y)
        self.columns = {
            n: names.index(n) for n in names if n in signals.all_detector_signals
        }
        for n in self.columns:
            item = self.getItem(0, 0)
            pen = mkPen(color=self.cmap[n])
            brush = mkBrush(None)
            crv = item.plot(
                pen=None,
                symbolPen=pen,
                symbolBrush=brush,
                name=n.value,
                symbolSize=3,
                pxMode=True,
            )
            self.curves[n] = crv

    def plot(self, data, names):
        #data[~np.isfinite(data)] = 0
        x = np.arctan2(data[:,self.y_idx], data[:,self.x_idx])
        for n, i in self.columns.items():
            y = data[:,i]
            m = np.isfinite(y)
            self.curves[n].setData(x[m], y[m])


class DataWindow(QTabWidget):
    """Container for main TabWidget."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Create pages
        self.raw_view = RawView()
        self.phase_view = PhaseView()

        self.tab_map = bidict({
            DisplayMode.raw: self.addTab(self.raw_view, "Raw"),
            DisplayMode.phase: self.addTab(self.phase_view, "Phase")
        })

        self.setTabPosition(QTabWidget.South)



class DisplayController(QObject):
    """
    Handles connection between the display windows and the other elements.
    """
    view_changed = Signal()
    def __init__(self, *a, data_window: DataWindow=None,
                 display_control: ViewPanel = None,
                 **kw):
        super().__init__(*a, **kw)
        self.data_view = data_window
        self.display_control = display_control
        self.mode_map = self.data_view.tab_map
        self.current = self.data_view.tab_map.inverse[self.data_view.currentIndex()]
        self.plot_cache = ((),{})
        self.view_changed.connect(self.onViewChanged)
        self.data_view.currentChanged.connect(self.onTabIndexChanged)

    def onTabIndexChanged(self, idx):
        logger.info("Tab changed to "+str(idx))
        self.view_changed.emit()

    def onViewChanged(self):
        self.data_view.currentWidget().plot(*self.plot_cache[0],
                                            **self.plot_cache[1])
        pass

    def prepare_plots(self, buf):
        for i in range(self.data_view.count()):
            self.data_view.widget(i).prepare_plots(buf)

    def plot(self, *a, **kw):
        self.plot_cache = (a, kw)
        self.data_view.currentWidget().plot(*a, **kw)


