from itertools import takewhile
import pandas as pd
import re

fname = "2022-02-04 173054 PH AC WS2 on Si ret.txt"

with open(fname, encoding="utf-8") as f:
    meta = list(takewhile(lambda ln: ln.startswith("#"), f))
# meta = [ln.strip().lstrip("# ") for ln in meta if ":" in ln]
# meta = dict([split_metadata_entry(ln) for ln in meta])
meta = "".join(meta)
# meta = dict([ln for ln in meta if len(ln)==2 else ln + [None]])
frame = pd.read_table(fname, comment="#").dropna(axis="columns")
frame.attrs["metadata"] = meta
from pprint import pprint