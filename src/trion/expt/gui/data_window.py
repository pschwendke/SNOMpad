import pyqtgraph as pg
from enum import IntEnum, auto
from PySide2.QtCore import QSize
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

        self.num_pts = IntEdit(100_000, bottom=0)
        grid.append(
            ["Display points", self.num_pts]
        )
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

        self.stack = QStackedWidget()
        lyt.addWidget(self.stack, 1, 0, 1, 2)
        self.stack.addWidget(RawCntrl())

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
        self.curves = {}
        self.autoscale_next = True # that stinks...

    def plot(self, data, names): # this is heavy WIP
        #data[~np.isfinite(data)] = 0
        for i, n in enumerate(names):
            y = data[:,i]
            self.curves[n].setData(y)
        if self.autoscale_next:
            self.autoscale_next = False
            self.getItem(0,0).enableAutoRange('xy', False)

    def prepare_plots(self, names):
        """Prepares windows and pens for the different curves"""
        item = self.getItem(0, 0)
        item.clear()
        for n in names:
            item = self.getItem(0, 0)
            pen = item.plot()
            self.curves[n] = pen
        self.autoscale_next = True

    def minimumSizeHint(self):
        return QSize(400, 400)
