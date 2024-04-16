# utility functions for analysis of scan data
import numpy as np
import xarray as xr
from sys import platform
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, set_start_method

from . import shd, pshet
from .scan_demod_base import BaseScanDemod
from ..utility.signals import Scan


# CONTINUOUS LINE-BASED SCANS ##########################################################################################
def sort_lines(scan: BaseScanDemod, trim_ratio: float = .05, plot=False):
    """ The function takes a line-based scan (LineScan, Image), and defines left and right turning points of the lines.
    The data taken in between is sorted into 'scan' 'rescan' and given a line number.
    The convention is that the fast axis is the x-axis, defined by x_res and x_size.
    The slow axis is the y-axis, defined by y_res and y_size.

    Parameters:
    __________
    scan: BaseScanDemod
        Should work for Linescan and Image objects
    trim_ratio: float
        defines radius around line turning points that are cut off (pixels that are rejected) by the ratio to
        the overall length of the line.
    plot: bool
        if true, a figure illustrating the sorting into approach, turning points and lines is plotted
    """
    # ToDo: for images, make this the distance from left and right side of the scan frame
    xy_coords = np.vstack([scan.afm_data.x.values, scan.afm_data.y.values]).T
    start_coord = np.array([scan.metadata['x_start'], scan.metadata['y_start']])
    stop_coord = np.array([scan.metadata['x_stop'], scan.metadata['y_stop']])
    start_dist = np.linalg.norm(start_coord - xy_coords, axis=1)
    stop_dist = np.linalg.norm(stop_coord - xy_coords, axis=1)
    if 'x_size' not in scan.metadata.keys() and scan.mode == Scan.continuous_line:  # this is more of a legacy thing
        dx = scan.metadata['x_stop'] - scan.metadata['x_start']
        dy = scan.metadata['y_stop'] - scan.metadata['y_start']
        scan.metadata['x_size'] = np.sqrt(dx ** 2 + dy ** 2)  # for images this is the diagonal

    status = 'approach'
    line_counter = -1
    line_labels = []
    for i, xy in enumerate(xy_coords):
        if status == 'approach':
            if start_dist[i] < trim_ratio * scan.metadata['x_size']:
                status = 'left trim'
        elif status == 'left trim':
            if start_dist[i] > trim_ratio * scan.metadata['x_size']:
                status = 'scan'
                line_counter += 1
        elif status == 'scan':
            if stop_dist[i] < trim_ratio * scan.metadata['x_size']:
                status = 'right trim'
        elif status == 'right trim':
            if stop_dist[i] > trim_ratio * scan.metadata['x_size']:
                status = 'rescan'
        elif status == 'rescan':
            if start_dist[i] < trim_ratio * scan.metadata['x_size']:
                status = 'left trim'
        if line_counter > scan.metadata['y_res'] - 1:
            status = 'tail'
        line_labels.append([line_counter, status])

    scan.afm_data['line'] = xr.DataArray(data=np.array([l[0] for l in line_labels]), dims='idx')
    scan.afm_data['direction'] = xr.DataArray(data=np.array([l[1] for l in line_labels]), dims='idx')
    scan.afm_data.attrs['trim_ratio'] = trim_ratio

    if plot:
        import matplotlib.pyplot as plt
        cmap = {
            'approach': 'grey',
            'scan': 'C00',
            'rescan': 'C01',
            'left trim': 'red',
            'right trim': 'green',
            'tail': 'grey'
        }
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].scatter(scan.afm_data.idx.values, scan.afm_data.x.values,
                      color=[cmap[d] for d in scan.afm_data.direction.values], marker='.')
        ax[0].set_ylabel('x /um')
        ax[1].scatter(scan.afm_data.idx.values, scan.afm_data.y.values,
                      color=[cmap[d] for d in scan.afm_data.direction.values], marker='.')
        ax[1].set_ylabel('y /um')
        ax[2].plot(scan.afm_data.idx.values, scan.afm_data.line.values)
        ax[2].set_ylabel('line #')
        ax[2].set_xlabel('idx')
        plt.show()


def bin_line(scan: BaseScanDemod, res: int = 100, direction: str = 'scan', line_no: list = []):
    """ Constructs a line on which measured data points should sit. Measured data points are selected via line number
    and scan direction, and binned onto constructed line of given resolution.
    Averaged AFM data and a list of indices of daq data is returned for every pixel on line.

    Parameters:
    __________
    scan: BaseScanDemod
        Should work for Linescan and Image objects
    res: int
        resolution of constructed line
    direction: 'scan' or 'rescan'
        scan direction of selected data
    line_no: list
        line numbers that should be evaluated. When [] is passed, all line numbers are evalueted.
    """
    dir_idx = scan.afm_data.idx[scan.afm_data.direction.values == direction].values
    if line_no:
        line_idx = []
        for l in line_no:
            line_idx.append(scan.afm_data.idx[scan.afm_data.line.values == l].values)
        idx = np.intersect1d(dir_idx, np.hstack(line_idx))
    else:
        idx = dir_idx

    x = scan.afm_data.sel(idx=idx).x.values
    y = scan.afm_data.sel(idx=idx).y.values

    coeff_matrix = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(coeff_matrix, y, rcond=None)[0]

    dist = (m * x - y + b) / np.sqrt(m ** 2 + 1)  # ToDo I don't trust this model. REDO THIS
    avg = dist.mean()
    std = dist.std()

    x_out = np.linspace(x[0], x[-1], res, endpoint=True)
    y_out = m * x_out + b

    dist_criterium = .5  # part of nearest neighbour distance
    length = scan.metadata['x_size'] - (scan.metadata['x_size'] * scan.afm_data.attrs['trim_ratio'] * 2)
    nn_dist = length / (res - 1)
    cutoff = nn_dist * dist_criterium

    r = []
    z = []
    amp = []
    phase = []
    idxs = []
    for i in range(res):
        coord = np.sqrt((x_out[0] - x_out[i]) ** 2 + (y_out[0] - y_out[i]) ** 2)
        r.append(coord)
        nearest = (np.linalg.norm([x_out[i], y_out[i]] - np.vstack([x, y]).T, axis=1) < cutoff)
        z.append(scan.afm_data.sel(idx=idx).z[nearest].values.mean())
        amp.append(scan.afm_data.sel(idx=idx).amp[nearest].values.mean())
        phase.append(scan.afm_data.sel(idx=idx).phase[nearest].values.mean())
        idxs.append(scan.afm_data.sel(idx=idx).idx[nearest].values)

    idxs = np.array(idxs, dtype=object)
    npts_per_idx = np.array([len(i) for i in idxs]) * scan.metadata['npts']

    ds = xr.Dataset()
    ds['z'] = xr.DataArray(data=z, dims='r', coords={'r': r})
    ds['amp'] = xr.DataArray(data=amp, dims='r')
    ds['phase'] = xr.DataArray(data=phase, dims='r')
    ds['idx'] = xr.DataArray(data=idxs, dims='r')
    ds.attrs['line ft: avg dist (m)'] = avg * 1e-6
    ds.attrs['line fit: std (m)'] = std * 1e-6
    ds.attrs['npts per pixel: avg'] = npts_per_idx.mean()
    ds.attrs['npts per pixel: min'] = npts_per_idx.min()
    ds.attrs['npts per pixel: max'] = npts_per_idx.max()
    return ds


# BATCH DEMODULATION USING MULTIPROCESSING #############################################################################
if platform != 'win32':
    set_start_method('fork')


def demod_worker(job_idx, measurement: BaseScanDemod, func, **kwargs):
    idx = measurement.demod_data.idx[job_idx].item()
    data = np.vstack([measurement.daq_data[i] for i in idx])
    harm = func(data=data, **kwargs)
    return harm


def batch_demod_1d(measurement: BaseScanDemod, processes: int = None, **kwargs):
    """ Demodulates all optical data in an instance of BaseScanDemod. measurement object needs an 'demod_data'
    attribute. All pixels are demodulated using multiprocessing if available.
    Only does 1D (line scans, retraction) at the moment..
    """
    func = [shd, pshet][['shd', 'pshet'].index(measurement.metadata['modulation'])]
    if platform in ['darwin', 'linux', 'linux2'] and processes != 1:  # with multiprocessing
        print('Batch demodulation (shd) with multiprocessing')
        worker_kwargs = {'measurement': measurement, 'func': func}
        worker_kwargs.update(**kwargs)
        job_idx = np.arange(len(measurement.demod_data.idx.values))
        with Pool(processes=processes) as pool:
            demod_data = pool.map(func=partial(demod_worker, **worker_kwargs), iterable=job_idx)
    else:  # no multiprocessing
        print('Batch demodulation (shd) without multiprocessing:')
        demod_data = []
        for i, val in tqdm(enumerate(measurement.demod_data.idx), total=len(measurement.demod_data.idx)):
            idx = val.values.item()
            data = np.vstack([measurement.daq_data[i] for i in idx])
            demod_data.append(func(data=data, **kwargs))
    demod_data = np.array(demod_data)
    measurement.demod_data['optical'] = xr.DataArray(data=demod_data, dims=('r', 'order'),
                                                     coords={'order': np.arange(demod_data.shape[1])})
