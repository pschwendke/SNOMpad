from abc import ABC
from itertools import repeat, cycle, product

import logging
import numpy as np
import pyqtgraph as pg
from enum import IntEnum, auto
from PySide2.QtCore import QSize, QObject, QStateMachine, QState, Signal
from PySide2.QtWidgets import QStackedWidget, QDockWidget, QWidget, QVBoxLayout, \
    QGridLayout, QTabWidget
from bidict import bidict
from pyqtgraph import mkPen, mkBrush

from .utils import enum_to_combo, IntEdit, add_grid, FloatEdit
from ..buffer import CircularArrayBuffer
from ...analysis import signals
from ...analysis.signals import signal_colormap, Signals

logger = logging.getLogger(__name__)

class DisplayMode(IntEnum): # TODO: remove
    raw = 0 # values indidcate the order in the stackwidget
    phase = 1
    fourier = 2

class RawCntrl(QWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.title = "Raw data view"
        lyt = QGridLayout()
        self.setLayout(lyt)

        self.downsampling = IntEdit(1, None)
        grid = []

        grid.append(
            ["Downsampling", self.downsampling]
        )
        add_grid(grid, lyt)
        lyt.setColumnStretch(1,1)
        lyt.setRowStretch(lyt.rowCount(), 1)

class PhaseCntrl(QWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.title = "Phase view"
        lyt = QGridLayout()
        self.setLayout(lyt)

        self.downsampling = IntEdit(1, None)
        grid = []

        grid.append(
            ["Downsampling", self.downsampling]
        )
        add_grid(grid, lyt)
        lyt.setColumnStretch(1, 1)
        lyt.setRowStretch(lyt.rowCount(), 1)


class FourierCntrl(QWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.title = "Fourier view"


class ViewPanel(QDockWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        panel = QWidget()
        lyt = QGridLayout()
        panel.setLayout(lyt)
        self.setWidget(panel)

        self.panels = {}
        self.raw_cntrl =  RawCntrl()
        self.phase_cntrl = PhaseCntrl()
        self.fourier_cntrl = FourierCntrl()
        self.panels[DisplayMode.raw] = self.raw_cntrl
        self.panels[DisplayMode.phase] = self.phase_cntrl
        self.panels[DisplayMode.fourier] = self.fourier_cntrl


        self.stack = QStackedWidget()
        lyt.addWidget(self.stack, 1, 0, 1, 2)
        for idx, panel in self.panels.items():
            self.stack.addWidget(panel)

        lyt.setColumnStretch(1,1)
        lyt.setRowStretch(lyt.rowCount(), 1)

        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )
        self.setWindowTitle(self.current_widget.title)

    @property
    def current_widget(self):
        return self.stack.currentWidget()

    def onTabIndexChanged(self, idx):
        self.stack.setCurrentIndex(idx)
        self.setWindowTitle(self.current_widget.title)


class BaseView(pg.GraphicsLayoutWidget):
    downsampling_cntrl_updated = Signal()

    def connect_signals(self):
        for plot in self.ci.items:
            plot.ctrl.downsampleSpin.valueChanged.connect(
                self.downsampling_cntrl_updated
            )
    def set_downsampling(self, value):
        """Set identical downsampling for all plots"""
        logger.debug("Setting downsampling to: "+repr(value))
        for plot in self.ci.items:
            plot.setDownsampling(ds=value, mode="subsample", auto=False)

    def get_downsampling(self):
        """
        Get downsampling factor for all plots.

        If mode is "auto" or downsampling factor is not the same for all plots,
        return False.
        """
        ds_config = [it.downsampleMode() for it in self.ci.items]
        automode = any(cfg[1] for cfg in ds_config)
        if automode:
            return False
        ds = [cfg[0] for cfg in ds_config]
        if any(v != ds[0] for v in ds):
            return False
        assert type(ds[0]) is int
        return ds[0]

    def minimumSizeHint(self):
        return QSize(400, 400)


class RawView(BaseView):
    """
    Plotting of the raw data stream.

    The displayed data has multiple Axes:
    - optical signals
    - tapping modulation
    - TODO: reference modulation
    - TODO: chopping
    """
    plot_indices = {
        Signals.sig_A: 0,
        Signals.sig_B: 0,
        Signals.sig_s: 0,
        Signals.sig_d: 0,
        Signals.tap_x: 1,
        Signals.tap_y: 1,
    }
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # start with a single window
        p0 = self.addPlot(row=0, col=0, name="Optical")
        self.addPlot(row=1, col=0, name="Tapping")

        self.cmap = signal_colormap() # TODO: move to base class

        self.curves = {}
        for plot in self.ci.items:
            #plot.setDownsampling(mode="subsample", auto=True)
            plot.setClipToView(False)
            plot.enableAutoRange("xy", False)
            plot.setYRange(-1, 1)
            plot.setXRange(0, 200_000)
            plot.showAxis("top")
            plot.showAxis("right")
            if plot is not p0:
                plot.setXLink(p0)

        self.set_downsampling(200)
        self.connect_signals()

    def prepare_plots(self, names):
        """Prepares windows and pens for the different curves"""
        logger.debug("Raw plot names:" + repr(names))
        for item in self.ci.items:
            item.clear()
            item.addLegend()

        for n in names:
            idx = self.plot_indices[n]
            item = self.getItem(idx, 0)
            pen = item.plot(pen=self.cmap[n], name=n.value)
            self.curves[n] = pen
        #logger.debug("Optical Y lim:" + repr(self.optical_ylim))
        self.connect_signals()

    def plot(self, data, names):
        """Plot data. Names are the associated columns."""
        x = np.arange(data.shape[0])
        for i, n in enumerate(names):
            y = data[:,i]
            m = np.isfinite(y)
            self.curves[n].setData(x[m], y[m], connect="finite")

    # @property
    # def optical_ylim(self):
    #     plot = self.getItem(0, 0)
    #     return plot.viewRange()[1]
    #
    # def set_optical_ylim(self, vmin, vmax):
    #     plot = self.getItem(0,0)
    #     plot.setYRange(vmin, vmax, padding=0)
    #
    # @property
    # def modulation_ylim(self):
    #     plot = self.getItem(1,0)
    #     return plot.viewRange()[1]
    #
    # def set_modulation_ylim(self, vmin, vmax):
    #     plot = self.getItem(1,0)
    #     plot.setYRange(vmin, vmax, padding=0)



class PhaseView(BaseView):
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


class FourierView(BaseView):
    """
    View the Fourier components, similar to optical alignment in NeaScan.

    The fourier components are computed from the signal amplitudes and the
    modulation phases. The fourier components are computed over the last
    `win_len` data points. The last `bufsize` values are kept for display in
    a rotating buffer.
    """
    def __init__(self, *a, bufsize=250, win_len=50_000, max_order=4, **kw):
        super().__init__(*a, **kw)
        # this guy has an internal buffer to track history
        self.buf = None
        self.bufsize = bufsize
        self.x_idx = None
        self.y_idx = None
        self.columns = {}  # signal name to index mapping
        self.input_indices = []  # indices of the signals in the input data
        self.win_len = win_len
        self.orders = np.arange(max_order + 1) # can't change yet...
        self.curves = {}
        self.cmap = signal_colormap()

        # Try a sparkline mode where each each order has its' plot, rescaled individually
        for idx in self.orders:
            self.addPlot(row=idx, col=0, name=f"order {idx}")

        p0 = self.getItem(0, 0)
        for plot in self.ci.items: # TODO: condense this in a common setup method
            plot.setDownsampling(mode="subsample", auto=False, ds=1)
            plot.setClipToView(False)
            plot.enableAutoRange(y=True)
            #plot.setYRange(0, 2)
            plot.setXRange(0, bufsize)
            #plot.showAxis("bottom", False)
            plot.hideAxis("bottom")
            plot.showAxis("right")
            for ax in ["left", "right"]:
                plot.getAxis(ax).setWidth(60)
            if plot is not p0:
                plot.setXLink(p0)
        self.getItem(0,0).showAxis("top")
        self.getItem(len(self.ci.items)-1,0).showAxis("bottom")
        self.ci.setContentsMargins(0, 10, 0, 10)
        self.set_downsampling(1)
        #self.ci.setBorder((50, 50, 100))
        self.connect_signals()

    def prepare_plots(self, names):
        for i, item in enumerate(self.ci.items): # TODO: this loop can be refactored into a base class
            item.clear()
            if i == 0:
                item.addLegend()

        self.x_idx = names.index(Signals.tap_x)
        self.y_idx = names.index(Signals.tap_y)
        self.columns = {
            n: names.index(n) for n in names if n in signals.all_detector_signals
        } # mapping for input data indices.
        self.input_indices = list(self.columns.values())
        self.buf = CircularArrayBuffer(
            max_size=self.bufsize,
            vars=list(product(self.orders, self.columns))
        )
        self.curves = {}
        for order, n in self.buf.vars:
            item = self.getItem(order, 0)
            pen = item.plot(pen=self.cmap[n], name=n.value)
            self.curves[order, n] = pen

    def compute_components(self, data, names):
        phi = np.arctan2(data[-self.win_len:, self.y_idx],
                         data[-self.win_len:, self.x_idx])
        data = data.take(self.input_indices, axis=1)
        # data has shape (N_pts, N_sig)
        # phi has shape (N_pts,)
        # orders has shape (N_orders,)
        # final size should be (N_orders, N_sig)
        demod = np.mean(
            data[-self.win_len:,None,:] * np.exp(
                -1j * self.orders[None,:,None] * phi[:,None,None]
            ),
            axis = 0, # for optimization this should be -1.
        )
        #assert demod.shape == (len(self.orders), len(self.input_indices))
        self.buf.put(np.abs(demod).reshape(1,-1))

    def plot(self, data, names):
        self.compute_components(data, names)
        y = self.buf.get(self.bufsize)
        vars = self.buf.vars
        x = np.arange(y.shape[0])
        m = np.isfinite(y[:,0])
        for i, v in enumerate(vars):
            self.curves[v].setData(x[m], y[:,i].compress(m, axis=0))


class DataWindow(QTabWidget):
    """Container for main TabWidget."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Create pages
        self.raw_view = RawView()
        self.phase_view = PhaseView()
        self.fourier_view = FourierView()

        self.tab_map = bidict({
            DisplayMode.raw: self.addTab(self.raw_view, "Raw"),
            DisplayMode.phase: self.addTab(self.phase_view, "Phase"),
            DisplayMode.fourier: self.addTab(self.fourier_view, "Fourier")
        })

        self.setTabPosition(QTabWidget.South)


class DisplayController(QObject):
    """
    Handles connection between the display windows and the other elements.
    """
    view_changed = Signal()
    def __init__(self, *a, data_window: DataWindow=None,
                 view_panel: ViewPanel = None,
                 acquisition_controller = None,
                 **kw):
        super().__init__(*a, **kw)
        self.data_view = data_window
        self.view_panel = view_panel
        self.mode_map = self.data_view.tab_map
        self.current = self.data_view.tab_map.inverse[self.data_view.currentIndex()]
        self.acquisition_controller = acquisition_controller
        self.plot_cache = None #((),{})
        self.view_changed.connect(self.onViewChanged)
        self.data_view.currentChanged.connect(self.onTabIndexChanged)
        self.data_view.currentChanged.connect(self.view_panel.onTabIndexChanged)

        for cntrl, view in [
            (self.view_panel.raw_cntrl, self.data_view.raw_view),
            (self.view_panel.phase_cntrl, self.data_view.phase_view),
            ]:
            cntrl.downsampling.valueEdited.connect(
                view.set_downsampling
            )
            view.downsampling_cntrl_updated.connect(
                self.update_ui
            )
        self.update_ui()

    def onTabIndexChanged(self, idx):
        logger.debug("Tab changed to "+str(idx))
        self.view_changed.emit()

    def onViewChanged(self):
        if self.plot_cache is not None:
            self.currentView.plot(*self.plot_cache[0],
                                  **self.plot_cache[1])

    @property
    def currentView(self):
        return self.data_view.currentWidget()

    @property
    def views(self):
        return [self.data_view.widget(i) for i in range(self.data_view.count())]

    def prepare_plots(self, buf):
        for view in self.views:
            view.prepare_plots(buf)

    def plot(self, *a, **kw):
        self.plot_cache = (a, kw)
        self.currentView.plot(*a, **kw)

    def update_ui(self):
        """Update UI elements to sync their values"""
        # signal to sync this is missing. I should probably change the
        # behavior of the right-click menu.
        for cntrl, view in [
            (self.view_panel.raw_cntrl, self.data_view.raw_view),
            (self.view_panel.phase_cntrl, self.data_view.phase_view),
            ]:
            cntrl.downsampling.setValue(
                view.get_downsampling()
            )
