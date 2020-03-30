import numpy as np

n_chan, n_samp = 4, 100
c = np.zeros((n_chan, n_samp), order="C") # default order
assert c.flags.c_contiguous
c_cols = c[:,10:15]
assert not c_cols.flags.c_contiguous
