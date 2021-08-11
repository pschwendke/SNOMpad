from abc import ABC
from collections import deque
from itertools import repeat, cycle, product

import logging
from time import time

import numpy as np
import pyqtgraph as pg
from enum import IntEnum, auto
from PySide2.QtCore import QSize, QObject, Signal, QTimer
from PySide2.QtWidgets import QStackedWidget, QDockWidget, QWidget, \
    QGridLayout, QTabWidget, QSpinBox
from bidict import bidict
from pyqtgraph import mkPen, mkBrush
from qtlets import HasQtlets
import attr

from .utils import add_grid
from qtlets.widgets import IntEdit
from trion.expt.buffer import CircularArrayBuffer
from trion.analysis import signals
from trion.analysis.signals import signal_colormap, Signals
from ..analysis.demod import dft_naive

logger = logging.getLogger(__name__)

# TODO: move all of these objects to their own thread.

class DisplayMode(IntEnum):
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
        self.display_size = IntEdit(200_000)
        grid = []
        grid.append(["Display pts:", self.display_size])
        grid.append(
            ["Downsampling:", self.downsampling]
        )
        add_grid(grid, lyt)
        lyt.setColumnStretch(1,1)
        lyt.setRowStretch(lyt.rowCount(), 1)
        lyt.setContentsMargins(0,0,0,0)


class PhaseCntrl(QWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.title = "Phase view"
        lyt = QGridLayout()
        self.setLayout(lyt)

        self.downsampling = IntEdit(1, None)
        self.display_size = IntEdit(200_000)
        grid = []
        grid.append(["Display pts:", self.display_size])
        grid.append(
            ["Downsampling:", self.downsampling]
        )
        add_grid(grid, lyt)
        lyt.setColumnStretch(1, 1)
        lyt.setRowStretch(lyt.rowCount(), 1)
        lyt.setContentsMargins(0,0,0,0)


class FourierCntrl(QWidget):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.title = "Fourier view"
        self.window_len = QSpinBox()
        self.window_len.setMinimum(10)
        self.window_len.setMaximum(2_000_000)
        lyt = QGridLayout()
        self.setLayout(lyt)
        grid = []
        grid.append(["Window size:", self.window_len])

        add_grid(grid, lyt)
        lyt.setColumnStretch(1, 1)
        lyt.setRowStretch(lyt.rowCount(), 1)
        lyt.setContentsMargins(0,0,0,0)



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
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cmap = signal_colormap()


    def clear_plots(self):
        """Clear the contents of all plots"""
        for i, item in enumerate(self.ci.items):
            item.clear()

    def minimumSizeHint(self):
        return QSize(400, 400)


class RawView(BaseView):
    """
    Plotting of the raw data stream.

    The displayed data has multiple Axes:
    - optical signals
    - tapping modulation
    - (future) reference modulation
    - (future) chopping
    """
    plot_indices = {
        Signals.sig_a: 0,
        Signals.sig_b: 0,
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

        self.curves = {}
        for plot in self.ci.items:
            plot.setClipToView(False)
            plot.enableAutoRange("xy", False)
            plot.setYRange(-1, 1)
            plot.setXRange(0, 200_000)
            plot.showAxis("top")
            plot.showAxis("right")
            plot.addLegend(offset=(2,2))
            if plot is not p0:
                plot.setXLink(p0)

    def prepare_plots(self, names):
        """Prepares windows and pens for the different curves"""
        logger.debug("Raw plot names:" + repr(names))
        self.clear_plots()

        for n in names:
            idx = self.plot_indices[n]
            item = self.getItem(idx, 0)
            pen = item.plot(pen=self.cmap[n], name=n.value)
            self.curves[n] = pen

    def plot(self, data, names, **kwargs):
        """Plot data. Names are the associated columns."""
        ds = kwargs["downsample"]
        x = np.arange(0, data.shape[0])[::ds]
        for i, n in enumerate(names):
            y = data[::ds,i]
            m = np.isfinite(y)
            self.curves[n].setData(x[m], y[m], connect="finite")


class PhaseView(BaseView):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # start with a single window
        self.addPlot(row=0, col=0, name="tap phase")

        # index of columns for data processing
        self.x_idx = None
        self.y_idx = None
        self.columns = {}
        self.curves = {}

        for plot in self.ci.items:
            plot.setClipToView(False)
            plot.enableAutoRange("xy", False)
            plot.setYRange(-2, 2)
            plot.setXRange(-np.pi, np.pi)
            plot.addLegend(offset=(2,2))
            plot.showAxis("top")
            plot.showAxis("right")

    def prepare_plots(self, names):
        """Prepares windows and pens for the different curves"""
        self.clear_plots()

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

    def plot(self, data, names, **kwargs):
        #data[~np.isfinite(data)] = 0
        ds = kwargs["downsample"]
        x = np.arctan2(data[::ds,self.y_idx], data[::ds,self.x_idx])
        for n, i in self.columns.items():
            y = data[::ds,i]
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
    def __init__(self, *a, bufsize=250, max_order=4, **kw):
        super().__init__(*a, **kw)
        # this guy has an internal buffer to track history
        self.buf = None
        self.bufsize = bufsize
        self.x_idx = None
        self.y_idx = None
        self.columns = {}  # signal name to index mapping
        self.input_indices = []  # indices of the signals in the input data
        self.orders = np.arange(max_order + 1)
        self.curves = {}

        # Try a sparkline mode where each each order has its' plot, rescaled individually
        for idx in self.orders:
            self.addPlot(row=idx, col=0, name=f"order {idx}")

        p0 = self.getItem(0, 0)
        for plot in self.ci.items: # TODO: condense this in a common setup method
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
        self.getItem(0,0).addLegend(offset=(2,2))
        self.getItem(len(self.ci.items)-1,0).showAxis("bottom")
        self.ci.setContentsMargins(0, 10, 0, 10)

        #self.ci.setBorder((50, 50, 100))

    def prepare_plots(self, names):
        self.clear_plots()

        self.x_idx = names.index(Signals.tap_x)
        self.y_idx = names.index(Signals.tap_y)
        self.columns = {
            n: names.index(n) for n in names if n in signals.all_detector_signals
        } # mapping for input data indices.
        self.input_indices = list(self.columns.values())
        self.buf = CircularArrayBuffer(
            size=self.bufsize,
            vars=list(product(self.orders, self.columns))
        )
        self.curves = {}
        for order, n in self.buf.vars:
            item = self.getItem(order, 0)
            pen = item.plot(pen=self.cmap[n], name=n.value)
            self.curves[order, n] = pen

    def compute_components(self, data, names, **kwargs):
        win_len = kwargs["window_size"]
        phi = np.arctan2(data[-win_len:, self.y_idx],
                         data[-win_len:, self.x_idx])
        data = data.take(self.input_indices, axis=1)
        # data has shape (N_pts, N_sig)
        # phi has shape (N_pts,)
        # orders has shape (N_orders,)
        # final size should be (N_orders, N_sig)
        demod = dft_naive(phi, data.T, orders=self.orders)
        self.buf.put(np.abs(demod).reshape(1, -1))

    def plot(self, data, names, **kwargs):
        self.compute_components(data, names, **kwargs)
        y = self.buf.tail(self.bufsize)
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



@attr.s(auto_attribs=True)
class DisplayConfig(HasQtlets):
    display_size: int = 200_000
    downsample: int = 200
    window_size: int = 2000

    def __attrs_post_init__(self):
        super().__init__()


class DisplayController(QObject):
    """
    Handles connection between the display windows and the other elements.
    """
    # TODO: move display timer here, include an "on_start" and "on_stop" slot
    view_changed = Signal()
    update_fps = Signal(float)

    def __init__(self, *a, data_window: DataWindow=None,
                 view_panel: ViewPanel=None,
                 acquisition_controller=None,
                 buffer=None,
                 display_dt=0.02,
                 display_config = None,
                 **kw):
        super().__init__(*a, **kw)
        self.data_view = data_window
        self.view_panel = view_panel
        self.buffer = buffer
        self.mode_map = self.data_view.tab_map
        self.current = self.data_view.tab_map.inverse[self.data_view.currentIndex()]
        self.acquisition_controller = acquisition_controller
        self.display_cfg = display_config or DisplayConfig()
        self.plot_cache = None #((),{})
        self.view_changed.connect(self.onViewChanged)
        self.data_view.currentChanged.connect(self.onTabIndexChanged)
        self.data_view.currentChanged.connect(self.view_panel.onTabIndexChanged)
        self._frame_log = deque([], 10)  # record of recent frames, to compute average framerate.
        self.fps_updt_timer = QTimer()
        self.fps_updt_timer.setInterval(500)
        self.display_timer = QTimer()
        self.display_timer.setInterval(display_dt * 1000)

        for cntrl in [self.view_panel.raw_cntrl, self.view_panel.phase_cntrl]:
            self.display_cfg.link_widget(cntrl.downsampling, "downsample")
            self.display_cfg.link_widget(cntrl.display_size, "display_size")
        self.display_cfg.link_widget(self.view_panel.fourier_cntrl.window_len,
                                     "window_size")

        self.fps_updt_timer.timeout.connect(self.on_update_fps)
        self.display_timer.timeout.connect(self.refresh_display)

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
        self.buffer = buf
        for view in self.views:
            view.prepare_plots(buf.vars)

    def refresh_display(self):
        "pass the data from the buffer to the display controller."
        try:
            names = self.buffer.vars
        except AttributeError:
            if self.buffer is not None:
                raise
            # otherwise we just don't have a buffer yet...
        else:
            try:
                y = self.buffer.tail(self.display_cfg.display_size)
                self.plot(y, names, **attr.asdict(self.display_cfg))
            except Exception:
                if self.acquisition_controller is not None:
                    self.acquisition_controller.act.stop.trigger()
                self.display_timer.stop() # we'll fall out of sync...
                raise

    def plot(self, *a, **kw):
        self.plot_cache = (a, kw)
        self.currentView.plot(*a, **kw)
        self._frame_log.append(time())

    def fps(self):
        """Estimate current fps from the last 10 frames."""
        if len(self._frame_log) < 2:
            return None
        else:
            return len(self._frame_log)/(self._frame_log[-1] - self._frame_log[0])

    def on_update_fps(self):
        self.update_fps.emit(self.fps())

    def start(self):
        self.fps_updt_timer.start()
        self.display_timer.start()

    def stop(self):
        self.display_timer.stop()
        self.display_timer.timeout.emit()  # fire once more, to be sure.
        self.fps_updt_timer.stop()
