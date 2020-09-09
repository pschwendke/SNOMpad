# explicitely create a qtlet, and assign it.
import sys
from random import randint

from PySide2.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication
from traitlets import HasTraits, Integer

from trion.expt.gui.utils import IntEdit
from trion.expt.gui.qtlets import Qtlets, IntQtlet


class Data(HasTraits):
    value = Integer(default_value=1)


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

        self.button.clicked.connect(self.on_btn_click)
        self.setWindowTitle("Directional connection")

    def on_btn_click(self):
        print("Roll!!")
        # this is done in the widget thread.
        # we want to do this in the qtlet thread.
        self.data.value = randint(0, 10)

        # this works as we wish, the change is made from a signal received by
        # the qtlet object.
        # NOPE! this only propagate to the widgets, no to data.value.
        #self.data.qtlet.data_changed.emit(randint(0, 10))
        # we need a signal for the Qtlet in the other direction.
        # the we can just link the widget signals to the qtlet signal.

        # will this work?


        # we could do:
        #
        # some access attribute magic, such as:
        # note in this case we need to put the signal indirection in the magic
        # attribute access... bad idea
        #    self.data.qtlets["value"] = randint(0,10)
        # or set_qtlet(self.data, "value", randint(0,10))
        # or self.data.set_qtl(name, value)
        #
        # some signal connection, such as:
        #    self.data.qtlets["value"].data_changed.emit(randint(0,10))
        # or signals(self.data, "value").emit(randint(0,10))
        # or

def update_cb(change):
    print(f"{change.old} -> {change.new}")

if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    d = Data(value=3)
    d.observe(update_cb, names="value")
    qtl = IntQtlet(d, "value")
    d.qtlet = qtl
    form = Form(data=d)
    qtl.link_widget(form.edit)
    qtl.link_widget(form.otheredit)
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())


