# io.py: read and write utility function
from os.path import splitext

import numpy as np
import pandas as pd
from .signals import Signals


def export_npz(filename, data, header, compress=True):
    """
    Export a single point data table as an npz archive.

    Parameters
    ----------
    filename: str
        Output file name
    data: np.ndarray of shape (N, M)
        Experimental data, containing N data points in M columns.
    header: list of str, length `M`
        name of columns in data
    """
    payload = dict(zip(header, data.T))
    if compress:
        np.savez_compressed(filename, **payload)
    else:
        np.savez(filename, **payload)


def export_csv(filename, data, header):
    """
    Export a single point data table as a csv file.

    Uses pandas.

    Parameters
    ----------
    filename: str
        Output file name. Also accepts a buffer string.
    data: np.ndarray of shape (N, M)
        Experimental data, containing N data points in M columns.
    header: list of str, length `M`
        name of columns in data
    """

    df = pd.DataFrame(data, columns=header)
    df.to_csv(filename, index=False)


writers = {
    ".npz": export_npz,
    ".csv": export_csv,
}


def export_data(filename, data, header):
    """
    Export data to file. Exact export behavior depends on data and file extension.

    """
    ext = splitext(filename)[1]
    if ext not in writers:
        raise ValueError(f"Unsupported extension: {ext}. Should be one of: "+", ".join(writers))
    if not data.ndim == 2:
        raise ValueError(f"Can only save 2d arrays as .npz, but shape is {data.shape}")
    if len(header) != data.shape[1]:
        raise ValueError(f"Mismatched header and data. header: {len(header)}, data: {data.shape[1]}")
    for i, v in enumerate(header):
        if isinstance(v, Signals):
            header[i] = v.value
    writer = writers[ext]
    writer(filename, data, header)
