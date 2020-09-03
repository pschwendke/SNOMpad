# obs.py
# try to remove boilerplate from QT by using taitlets' observation behavior
# what we need is essentially an adapter between traits' observe and Qt signals
# We don't need to change the keys ever, so we'll use instances.

import traitlets

