import pandas as pd
import numpy as np


def empty_bins_in(df: pd.DataFrame) -> bool:
    """Checks binned 1D or 2D DataFrame for empty bins in index (1D) and column names (2D).
    Returns True if empty bins are detected."""

    # check for missing tap bins
    for n in range(df.shape[0]):
        if df.index[n] != n:
            return True

    # check for missing pshet bins
    if isinstance(df.columns, pd.MultiIndex):
        for channel in df.columns.get_level_values(0).drop_duplicates():
            for m in range(df[channel].shape[1]):
                if df[channel].columns[m] != m:
                    return True

    # check for empty (NAN) bins
    if np.isnan(df).any(axis=None):
        return True
    return False
