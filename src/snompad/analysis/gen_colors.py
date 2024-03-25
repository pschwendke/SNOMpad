# generate colors

import numpy as np
from matplotlib.pyplot import get_cmap
from trion.utility.signals import Signals
import toml

tab20 = np.array(np.array(get_cmap("tab20").colors)*255, dtype=int).tolist()

map = {
    Signals.sig_a: tab20[0],
    Signals.sig_b: tab20[1],
    Signals.sig_d: tab20[2],
    Signals.sig_s: tab20[3],
    Signals.tap_x: tab20[4],
    Signals.tap_y: tab20[5],
    Signals.tap_p: tab20[4],
    Signals.ref_x: tab20[6],
    Signals.ref_y: tab20[7],
    Signals.ref_p: tab20[6],
}

map = {k.name: v for k, v in map.items()}

#pprint(map)

with open("signal_colors.toml", "w") as f:
    toml.dump(map, f)
