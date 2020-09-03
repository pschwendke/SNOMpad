import sys
from random import randint
from traitlets import HasTraits, Integer
from PySide2.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, \
    QVBoxLayout, QWidget
from PySide2.QtCore import QObject, Signal
from trion.expt.gui.utils import IntEdit


class Data(HasTraits):
    value = Integer(default_value=1)

class Qtlet(QObject):
    value_updated = Signal(int)

class Form(QWidget):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.data = data
        self.edit = IntEdit("...")

        self.otheredit = IntEdit("???")
        self.otheredit.setEnabled(False)
        self.button = QPushButton("Roll!")

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.otheredit)
        layout.addWidget(self.button)
        self.setLayout(layout)

        # set the initial value, streamline this at initial connection
        self.edit.setValue(data.value)
        self.otheredit.setValue(data.value)

        # this needs to be collected in the controller
        self.data.ctrl.value_updated.connect(self.edit.setValue)
        self.data.ctrl.value_updated.connect(self.otheredit.setValue)
        self.edit.valueEdited.connect(self.on_manual_edit)

        self.button.clicked.connect(self.on_btn_click)
        self.setWindowTitle("Directional connection")

    def on_btn_click(self):
        print("Roll!!")
        self.data.value = randint(0, 10)

    def on_manual_edit(self, value):
        self.data.value = value

datum = Data(value=3)
datum.ctrl = Qtlet()

# this goes in the controller
def update_cb(change):
    print(f"{change.old} -> {change.new}")
    datum.ctrl.value_updated.emit(change.new)


datum.observe(update_cb, names=["value"])


if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    form = Form(data=datum)
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())

