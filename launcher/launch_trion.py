# launch_trion.py: launch the trion experiment GUI

import sys
import os
import os.path
import logging
import logging.config
from PySide2 import QtGui, QtWidgets
import toml

from trion.expt.gui.core import TRIONMainWindow

if not os.path.exists("./log/"):
    os.mkdir("log")
logging.captureWarnings(True)
with open("log_cfg.toml", "r") as f:
    logging.config.dictConfig(toml.load(f))
logger = logging.getLogger("root")


# setup sys.excepthook

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    win = TRIONMainWindow()

    win.show()
    retcode = app.exec_()

    logger.info("Shutting down")
    sys.exit(retcode)
