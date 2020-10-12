# launch_trion.py: launch the trion experiment GUI

import sys
import os
import os.path
import logging
import logging.config
from PySide2 import QtGui, QtWidgets
import toml

from trion.expt.gui.core import TRIONMainWindow
from trion.expt.gui.log import QtLogHandler

if not os.path.exists("./log/"):
    os.mkdir("log")
logging.captureWarnings(True)
with open("log_cfg.toml", "r") as f:
    logging.config.dictConfig(toml.load(f))
logger = logging.getLogger("root")

def log_except(*exc_info):
    """Catches all exceptions, and logs."""
    logger.error("Unexpected Error!", exc_info=exc_info)

sys.excepthook = log_except

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    win = TRIONMainWindow()
    # Finalize Qt Log Handlers
    handlers = [l for l in logger.root.handlers if isinstance(l, QtLogHandler)]
    for h in handlers:
        logger.debug("Setting up logger: %s", h)
        h.setup(win)

    app.aboutToQuit.connect(win.shutdown)
    win.show()
    retcode = app.exec_()

    # disconnect Qt handlers
    for h in handlers:
        h.close()
        logger.root.removeHandler(h)

    logger.info("Shutdown complete")
    sys.exit(retcode)
