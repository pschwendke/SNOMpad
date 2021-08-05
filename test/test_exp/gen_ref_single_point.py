# generate reference files for single point

import numpy as np
from common import exp_configs, n_samples
from itertools import product

from trion.expt.acquisition import single_point


def make_fname(exp, n):
    return "ref-sp-"+"-".join(s.name for s in exp.signals())+f"-n{n:04d}.npy"


def make_refs(exp, n):
    data = single_point("DevT", exp.signals(), n, truncate=True)
    fn = make_fname(exp, n)
    np.save(fn, data, allow_pickle=False)


if __name__ == "__main__":
    for e, n in product(exp_configs, n_samples):
        make_refs(e, n)
