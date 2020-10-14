# check csv output is to our taste

import numpy as np
from trion.analysis.io import export_csv

hdr = ["tap_x", "tap_y", "sig_A"]
data = np.random.randn(100, 3)

export_csv("test.csv", hdr, data)
