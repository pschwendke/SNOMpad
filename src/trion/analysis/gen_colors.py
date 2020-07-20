# generate colors

import numpy as np
from matplotlib.colors import to_rgba_array
from matplotlib.pyplot import get_cmap
from trion.analysis.signals import Signals
from pprint import pprint
import toml

tab20 = np.array(np.array(get_cmap("tab20").colors)*255, dtype=int).tolist()

map = {
    Signals.sig_A: tab20[0],
    Signals.sig_B: tab20[1],
    Signals.sig_d: tab20[2],
    Signals.sig_s: tab20[3],
    Signals.tap_x: tab20[4],
    Signals.tap_y: tab20[5],
    Signals.tap_p: tab20[4],
}

map = {k.name: v for k, v in map.items()}

#pprint(map)

with open("signal_colors.toml", "w") as f:
    toml.dump(map, f)
