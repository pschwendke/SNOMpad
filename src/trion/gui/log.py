import logging
import traceback
from string import capwords

from PySide2 import QtGui
from PySide2.QtCore import QObject, Signal
from PySide2.QtWidgets import QMessageBox, QSpacerItem, QSizePolicy

logger = logging.getLogger(__name__)

class BriefFormatter(logging.Formatter):
    """Brief formatter that shortens traceback info."""
    def format(self, record):
        if record.exc_info:
            # force new exc_text
            record.exc_text = ''
        s = super(BriefFormatter, self).format(record)
        if record.exc_info:
            record.exc_text = ''  # avoid caching this result
        return s

    def formatException(self, ei):
        return "See log file for detailed error information."


class QtLogHandler(logging.Handler): # mixin may not work, or yield name collisions
    """
    Base class for handlers based on Qt Handlers.

    Logging is carried out by displaying the message on a target widget. The
    specifics are implemented by the `display` method.

    The handlers can finalize their setup after main window creation using the
    `setup` call.
    """
    def __init__(self, widget=None, **kw):
        super(QtLogHandler, self).__init__(**kw)
        self.widget = widget

    def setup(self, mw):
        """Setup connection between logging.Handler and target UI object."""
        pass

    def emit(self, record):
        msg = self.format(record)
        if self.widget is not None:
            self.display(msg, record)

    def display(self, msg, record):
        """
        Actual displaying the message.

        To avoid ugly crashes, this should emit a Qt signal
        """
        raise NotImplementedError

    def write(self, m):
        pass  # ???


class _QtLogSignals(QObject):
    logMessage = Signal(str)
    logRecord = Signal(str, logging.LogRecord)


QtLogSignals = _QtLogSignals()


class StatusBarLogger(QtLogHandler):
    """
    Logging handler displaying to statusbar.
    """
    logMessage = QtLogSignals.logMessage

    def setup(self, mw):
        self.widget = mw.statusBar()
        self.logMessage.connect(self.widget.showLog)

    def display(self, msg, record):
        self.logMessage.emit(msg)


class PopupLogger(QtLogHandler):
    """
    Logging handler displaying a message as popup.
    """
    icons = {'DEBUG': QMessageBox.Information,
             'INFO': QMessageBox.Information,
             'WARNING': QMessageBox.Warning,
             'ERROR': QMessageBox.Warning,
             'CRITICAL': QMessageBox.Critical}
    logRecord = QtLogSignals.logRecord

    def setup(self, mw):
        self.widget = mw.log_popup
        self.logRecord.connect(self.widget.showLog)

    def display(self, msg, record):
        self.logRecord.emit(msg, record)


class QPopupLogDlg(QMessageBox):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        lyout = self.layout()
        spacer = QSpacerItem(400, 0,
                                   hPolicy=QSizePolicy.Minimum,
                                   vPolicy=QSizePolicy.MinimumExpanding)
        lyout.addItem(spacer, lyout.rowCount(), 0, 1, lyout.columnCount())

    def showLog(self, msg, record):
        """Show the popup window."""
        lvl = capwords(record.levelname)
        icon = PopupLogger.icons[lvl.upper()]
        self.setIcon(icon)
        self.setText(lvl)
        self.setInformativeText(msg)
        if record.exc_info:
            self.setDetailedText(''.join(traceback.format_exception(*record.exc_info)))
        else:
            self.setDetailedText('')
        self.show()
