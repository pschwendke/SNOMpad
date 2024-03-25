# gui utils

import logging

from PySide2.QtCore import Signal
from PySide2.QtGui import QIntValidator, QDoubleValidator
from PySide2.QtWidgets import QPushButton, QLabel, QComboBox, QLineEdit

logger = logging.getLogger(__name__)

def add_grid(grid, layout, row_offset = 0, column_offset=0):
    """Add list of lists to gridlayout"""
    for i, row in enumerate(grid):
        for j, elem in enumerate(row):
            if elem is None: continue
            if isinstance(elem, str):
                elem = QLabel(elem)
            layout.addWidget(elem, i+row_offset, j+column_offset)


def enum_to_combo(enum, cls=QComboBox):
    combo = cls()
    for member in enum:
        combo.addItem(member.name, member)
    return combo


class ToggleButton(QPushButton):
    """
    A toggle button.

    Parameters
    ----------
    alt_text : str, kw only
        Text in 'checked' state.

    Other arguments are passed to QPushButton.
    """
    def __init__(self, *a, **kw):
        self.alt_text = kw.pop("alt_text", None)
        super().__init__(*a, **kw)
        self.org_text = self.text()
        self.setCheckable(True)
        self.toggled.connect(self.on_toggle)

    def on_toggle(self, checked):
        if checked:
            self.setText(self.alt_text or self.org_text)
        else:
            self.setText(self.org_text)
