# launch_trion.py: launch the trion experiment GUI

import sys
import logging
from PySide2 import QtGui, QtWidgets

from trion.expt.gui.core import TRIONMainWindow

logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)
# load logging config. Try to load a toml...

logger = logging.getLogger()


# setup sys.excepthook

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    win = TRIONMainWindow()

    win.show()
    retcode = app.exec_()
    sys.exit(retcode)
