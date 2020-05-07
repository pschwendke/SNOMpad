# gui utils

import logging
from PySide2.QtWidgets import QPushButton


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

