import logging
from enum import IntEnum

from PySide2.QtWidgets import (QWidget, QGridLayout, QPushButton)
from PySide2.QtCore import Qt
from qtlets.widgets import IntEdit, StrEdit

from trion.gui.utils import add_grid

logger = logging.getLogger(__name__)

# class BufferPanels(IntEnum):
#     Memory = 0


class MemoryBufferPanel(QWidget):
    def __init__(self, *a, **kw):
        """
        Control panel for in-memory buffer (ie: ArrayBuffer)
        """
        super().__init__(*a, **kw)
        self.setLayout(QGridLayout())
        grid = []
        # TODO: there bug when buffer size is too small. Should be fixed.
        self.buffer_size = IntEdit(200_000, bottom=10_000, parent=self)

        grid.append(["Buffer size", self.buffer_size])
        add_grid(grid, self.layout())

        self.layout().setRowStretch(self.layout().rowCount(), 1)
        self.layout().setColumnStretch(1, 1)
        self.layout().setContentsMargins(0,0,0,0)


class H5BufferPanel(QWidget):
    def __init__(self, *a, **kw):
        """
        Control panel for H5py buffer
        """
        super().__init__(*a, **kw)
        self.setLayout(QGridLayout())
        grid = []

        self.filename = StrEdit("trions.h5")
        self.open_btn = QPushButton("-")
        grid.append(["File", self.filename, self.open_btn])

        self.buffer_size = IntEdit(200_000, bottom=0, parent=self)
        grid.append(["Size", self.buffer_size])
        add_grid(grid, self.layout())
        self.layout().setRowStretch(self.layout().rowCount(), 1)
        self.layout().setColumnStretch(1, 1)
        self.layout().setContentsMargins(0,0,0,0)
