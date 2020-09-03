# qtlets.py
# try to remove boilerplate from QT by using taitlets' observation behavior
# what we need is essentially an adapter between traits' observe and Qt signals
# We don't need to change the keys ever, so we'll use instances.

from abc import abstractmethod
from traitlets import HasTraits
from PySide2.QtCore import QObject, Signal



# How do we structure this?
# The instance will be a subclass of the HasQtlet object. We will interact with
# it. The instance will be used for `register_widget`, `sync_widgets`, `movetothread`
#
# The instance will have a the qtlets in a mapping similar to the traits.
#
# Mixins with Qt is painful, but that's a lot of Qtlets... I guess that's fine
#

class Qtlets(QObject):
    """Adapter between `traitlets` notification and Qt Signals and Slots"""
    # define custom signal per type?
    data_changed = Signal(object)  # fallback
    # maybe we need another metaclass ... Due to Qt's constraints, we can't
    # do otherwise...
    def __init__(self, inst, attr, *a, **kw):
        super().__init__(*a, **kw)
        self.widgets = []
        self.inst = inst # or use this to set attribute of the attribute proxy descriptor?
        self.attr = attr
        inst.observe(self.notify_widgets, names=attr)

    # we're building a proxy to act as a descriptor... can we grab the attribute
    # directly?
    @property
    def value(self):
        return getattr(self.inst, self.attr)

    @value.setter
    def value(self, value):
        setattr(self.inst, self.attr, value)

    def on_widget_edited(self, value):  # this is a slot
        """
        Update the attribute to given value.
        """
        # note this is exactly the same as @value.setter...
        self.value = value

    def notify_widgets(self, change):
        """
        Update the widgets to reflect a change in the underlying data.
        """
        # todo: find a more elegant way to pick the signal. Maybe it's ok?
        self.data_changed.emit(change.new)

    def sync_widgets(self):
        """Force the update of all linked widgets with the current value."""
        self.data_changed.emit(self.value)

    def link_widget(self, widget): # todo: add options for read and write?
        """Link a widget to the trait."""
        # todo: use a function and dispatch to get the two methods (widget.valueEdited, widget.setValue)
        widget.valueEdited.connect(self.on_widget_edited)
        self.value_updated.connect(widget.setValue)
        self.widgets.append(widget)
        self.value_updated.emit(self.value)


class IntQtlet(Qtlets):
    data_changed = Signal(int)


class HasQtlet(HasTraits):  # or MetaHasTraits?
    raise NotImplementedError