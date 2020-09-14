#

import sys
from random import randint

from PySide2.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication
from traitlets import HasTraits, Integer

from trion.expt.gui.utils import IntEdit
from trion.expt.gui.qtlets import Qtlet, IntQtlet, HasQtlets

class Data(HasQtlets):
    value = Integer(default_value=1, min=0, max=10)

class Form(QWidget):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.data = data
        self.edit = IntEdit("...")

        self.otheredit = IntEdit("???")
        #self.otheredit.setEnabled(False)
        self.button = QPushButton("Roll!")

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.otheredit)
        layout.addWidget(self.button)
        self.setLayout(layout)

        data.link_widget(self.edit, "value")
        data.link_widget(self.otheredit, "value")

        # set the initial value, streamline this at initial connection

        self.button.clicked.connect(self.on_btn_click)
        self.setWindowTitle("Directional connection")

    def on_btn_click(self):
        print("Roll!!")
        # this is done in the calling thread.
        # We're not exploiting Qt's queued events in this direction
        self.data.value = randint(0, 10)


def update_cb(change):
    print(f"{change.old} -> {change.new}")


if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    d = Data(value=3)
    d.observe(update_cb, names="value")
    form = Form(data=d)
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
