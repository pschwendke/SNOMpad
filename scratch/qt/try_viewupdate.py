import sys
from random import randint
from traitlets import HasTraits, Integer, link
from PySide2.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, \
    QVBoxLayout, QWidget
from PySide2.QtCore import QObject, Signal
from trion.expt.gui.utils import IntEdit


# from http://code.activestate.com/recipes/510402-attribute-proxy-forwarding-attribute-access/
class forwardTo(object):
    """
    A descriptor based recipe that makes it possible to write shorthands
    that forward attribute access from one object onto another.

    >>> class C(object):
    ...     def __init__(self):
    ...         class CC(object):
    ...             def xx(self, extra):
    ...                 return 100 + extra
    ...             foo = 42
    ...         self.cc = CC()
    ...
    ...     localcc = forwardTo('cc', 'xx')
    ...     localfoo = forwardTo('cc', 'foo')
    ...
    >>> print C().localcc(10)
    110
    >>> print C().localfoo
    42

    Arguments: objectName - name of the attribute containing the second object.
               attrName - name of the attribute in the second object.
    Returns:   An object that will forward any calls as described above.
    """
    def __init__(self, objectName, attrName):
        self.objectName = objectName
        self.attrName = attrName

    def __get__(self, instance, owner=None):
        return getattr(getattr(instance, self.objectName), self.attrName)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.objectName), self.attrName, value)

    def __delete__(self, instance):
        delattr(getattr(instance, self.objectName), self.attrName)


class Data(HasTraits):
    value = Integer(default_value=1)


# Do we need one of these per instance, or per attribute?
# What does traitlets and spectate do?
#   Traitlets works "per instance". Notification is actually done in
#   `_notify_observers`, by looking up matching elements in the
#   `obj._trait_notifiers` attribute.
#   Spectate also uses per-instance dictionnary of callbacks.
#
# Per instance:
#   - less of them, simpler to to moveToThread
#   - can have global signals when any value is changed
#   - need to do bookkeeping on the callbacks
#   - there will be multiple widgets per attribute.
# Per attribute
#   - need to do bookkeeping at the instance level, having multiple controllers
#
# should we rewrite this as a full-blown descriptor? I feel like I'm strapping a pig onto something...
class Qtlet(QObject):
    value_updated = Signal(int)
    def __init__(self, inst, attr):
        super().__init__()
        self.widgets = []
        self.inst = inst
        self.attr = attr
        #self.value = forwardTo(inst, attr) # is this ok? nope!
        inst.observe(self.on_data_changed, names=attr)

    def update_all_widgets(self):  # into
        # for all widgets, set the value
        v = getattr(self.inst, self.attr)
        for w in self.widgets:
            self.update_single_widget(w, v)

    @staticmethod
    def update_single_widget(widget, value):
        # todo: move to a function, using single dispatch for the different widget types!
        widget.setValue(value)

    def on_widget_edited(self, value):
        """Called when the value from the widget is edited."""
        # tood: check if we can use functools.patrial setattr
        setattr(self.inst, self.attr, value)

    def on_data_changed(self, value):
        """Called when the data is changed"""
        print(f"data changed to {value}")
        self.value_updated.emit(value.new)

    def link_widget(self, widget):
        # Todo: can we again use single dispatch to get the correct slot?
        # todo: we would need an "unlink"
        self.update_single_widget(widget, getattr(self.inst, self.attr))
        widget.valueEdited.connect(self.on_widget_edited)
        self.value_updated.connect(widget.setValue)

        self.widgets.append(widget)


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

        self.ctrl = Qtlet(data, "value")
        self.ctrl.link_widget(self.edit)
        self.ctrl.link_widget(self.otheredit)

        # set the initial value, streamline this at initial connection

        self.button.clicked.connect(self.on_btn_click)
        self.setWindowTitle("Directional connection")

    def on_btn_click(self):
        print("Roll!!")
        self.data.value = randint(0, 10)

    def on_manual_edit(self, value):
        self.data.value = value

datum = Data(value=3)
#datum.ctrl = Qtlet()

# this goes in the controller
def update_cb(change):
    print(f"{change.old} -> {change.new}")
#

datum.observe(update_cb, names=["value"])


if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    form = Form(data=datum)
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())

