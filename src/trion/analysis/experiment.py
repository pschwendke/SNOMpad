import numpy as np
import xarray as xr
import attr
import colorcet as cc
import h5py

from tqdm import tqdm
from itertools import chain
from typing import Iterable
from abc import ABC, abstractmethod

from trion.analysis.signals import Scan, Demodulation, Detector, Signals, detection_signals, modulation_signals
from trion.analysis.demod import shd, pshet


class Measurement(ABC):
    """ Base class to load, demodulate and export acquired data.
    """
    def __init__(self, file: h5py.File):
        self.file = file
        self.name = file.attrs['name']
        self.afm_data = None
        self.nea_data = None
        self.demod_data = None
        self.demod_filename = None
        self.mode = Scan[file.attrs['acquisition_mode']]
        self.signals = [Signals[s] for s in file.attrs['signals']]
        self.modulation = Demodulation[file.attrs['modulation']]
        self.chopped = file.attrs['chopped']

        if self.file['afm_data'].keys():
            self.afm_data = self.h5_to_xr_dataset('afm_data')
        if self.file['nea_data'].keys():
            self.nea_data = self.h5_to_xr_dataset('nea_data')

    def h5_to_xr_dataset(self, group_name: str):
        group = self.file[group_name]
        ds = xr.Dataset()
        for ch, dset in group.items():
            if dset.dims[0].keys():
                values = np.array(dset)
                dims = [d.keys()[0] for d in dset.dims]
                da = xr.DataArray(data=values, dims=dims, coords={d: np.array(group[d]) for d in dims})
                da.attrs = dset.attrs
                ds[ch] = da
        ds.attrs = group.attrs
        return ds

    @abstractmethod
    def demod(self):
        """ Collect and demodulate raw data, to produce NF images, retraction curves, spectra etc.
        """
    @abstractmethod
    def plot(self):
        """ Plot (and save) demodulated data, e.g. images or curves
        """
    @abstractmethod
    def export(self):
        """ Export demodulated data as xarray dataset, csv or gwyddion file
        """


def load(filename: str) -> Measurement:
    """ Loads scan from .h5 file and returns Measurement object. The function is agnostic to scan type.
    """
    file = h5py.File(filename, 'r')
    scan_type = file.attrs['acquisition_mode']

    if scan_type in ['stepped_retraction', 'continuous_retraction']:
        return Retraction(file)
    elif scan_type in ['stepped_image', 'continuous_image']:
        return Image(file)
    elif scan_type in ['stepped_line', 'continuous_line']:
        return Line(file)
    # elif scan_type == 'noise_sampling':
    #     return Noise(file)
    # elif scan_type == 'delay_scan':
    #     return DelayScan(file)
    else:
        raise NotImplementedError


class Retraction(Measurement):
    def __init__(self, file: h5py.File):
        super().__init__(file)

    def demod(self, tap_res: int = 64, ref_res: int = 64, npts=None, demod_filename=None, method='binning', **kwargs):
        """ Creates xr.Dataset in self.demod_data. Dimensions are 'z_target' for stepped retraction and 'z' for
        continuous retraction. Dataset contains one DataArray for every tracked channel, and one complex-valued
        DataArray for demodulated harmonics, with extra dimension 'order'.
        When demod_filename is a str, a demod file is created in the given filename. When filename is '', the file is
        saved in the '_ANALYSIS' directory on the server.
        When the demod file already exists, demod data will be added to it. In case demodulation with given parameters
        is already present in demod file, it will be read from the file.

        Parameters
        ----------
        tap_res: int
            resolution of discrete uniform phase domain on theta_tap axis
        ref_res: int
            resolution of discrete uniform phase domain on theta_ref axis
        npts: int
            min number of samples to demodulate for every pixel. Only whole chunks saved in daq_data are combined
        demod_filename: str
            path and filename of demod file
        method: str in ['binning', 'kernel']
            method to average over phase domain and find harmonic contributions
        kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if not npts and self.mode == Scan.stepped_retraction:
            chunk_size = 1  # one chunk of DAQ data per demodulated pixel
        elif not npts and self.mode == Scan.continuous_retraction:
            chunk_size = 5
        else:
            chunk_size = npts // self.file.attrs['npts'] + 1
        max_order = int(tap_res // 2 + 1)
        n = len(self.file['daq_data'].keys())
        z_res = n // chunk_size
        self.demod_data = xr.Dataset()

        z = self.afm_data['z'].values
        idx = self.afm_data['idx'].values
        amp = self.afm_data['amp'].values
        phase = self.afm_data['phase'].values
        idx = idx[:z_res*chunk_size].reshape((z_res, chunk_size))
        z = z[:z_res * chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        amp = amp[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        phase = phase[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        self.demod_data['amp'] = xr.DataArray(data=amp, dims='z', coords={'z': z})
        self.demod_data['phase'] = xr.DataArray(data=phase, dims='z', coords={'z': z})
        if self.mode == Scan.stepped_retraction:
            z_target = self.afm_data['z_target'].values
            z_target = z_target[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
            self.demod_data['z_target'] = xr.DataArray(data=z_target, dims='z', coords={'z': z})

        # demodulate DAQ data for every pixel
        harmonics = np.zeros((z_res, max_order), dtype=complex)
        for p, h in tqdm(enumerate(harmonics)):
            data = np.vstack([np.array(self.file['daq_data'][str(i)]) for i in idx[p]])
            if self.modulation == Demodulation.shd:
                coefficients = shd(data=data, signals=self.signals, tap_nbins=tap_res, **kwargs)
            elif self.modulation == Demodulation.pshet:
                coefficients = pshet(data=data, signals=self.signals, tap_nbins=tap_res, ref_nbins=ref_res, **kwargs)
            else:
                raise NotImplementedError
            h = coefficients
        self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('z', 'order'),
                                                  coords={'z': z, 'order': np.arange(max_order)})

        # ToDo: Metadata
        # ToDo: create demod file
        #  and read from demod file

    def plot(self, max_order: int = 4, orders=None, afm_amp=False, afm_phase=False, grid=True, show=True, save=False):
        import matplotlib.pyplot as plt
        if self.demod_data is None:
            self.demod()
        if not (hasattr(orders, '__iter__') and all([type(h) is int for h in orders])):
            orders = np.arange(max_order)

        fig, ax1 = plt.subplots()
        cmap = cc.glasbey_category10
        x = (self.demod_data['z'].values - self.demod_data['z'].values.min())

        if afm_phase:
            ax2 = ax1.twinx()
            y = self.demod_data['phase'].values / 2 / np.pi * 360  # degrees
            ax2.plot(x, y, color='gray', marker='.', ms=3, lw=.5)
            ax2.set_ylabel('AFM phase (degrees)', color='gray')

        if afm_amp:
            ax3 = ax1.twinx()
            y = self.demod_data['amp'].values / self.demod_data['amp'].values.max()
            ax3.plot(x, y, marker='.', ms=3, lw=.5, label='AFM amplitude (scaled)', color='darkblue', alpha=.5)
            ax3.tick_params(right=False, labelright=False)
            ax3.legend(loc='upper right')

        for o in orders:
            y = np.real(self.demod_data['optical'].values[:, o].squeeze())
            y /= np.abs(y).max()
            ax1.plot(x, y, marker='.', lw=1, label=str(o), color=cmap[o])

        ax1.grid(visible=grid, which='major', axis='both')
        ax1.set_ylabel('optical amplitude (normalized)')
        ax1.set_xlabel('dz (nm)')
        ax1.legend(loc='lower right')

        if save:
            plt.savefig(f'{self.afm_data.attrs["name"]}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.afm_data.attrs["name"]}.svg')
        if show:
            plt.show()
        plt.close()

    def export(self):
        raise NotImplementedError


class Image(Measurement):
    def __init__(self, file: h5py.File):
        super().__init__(file)
        raise NotImplementedError

    def demod(self):
        pass

    def plot(self):
        pass

    def export(self):
        pass


class Line(Measurement):
    def __init__(self, file:h5py.File):
        super().__init__(file)
        raise NotImplementedError

    def demod(self):
        pass

    def plot(self):
        pass

    def export(self):
        pass

# class Noise(Measurement):
#     def __init__(self, file: h5py.File):
#         super().__init__(file)
#         self.z_spectrum = None
#         self.z_frequencies = None
#         self.amp_spectrum = None
#         self.amp_frequencies = None
#         self.optical_spectrum = None
#         self.optical_frequencies = None
#
#     def demod(self):
#         nea_data = self.afm_data['Z'].values.flatten()
#         nea_data = nea_data[~np.isnan(nea_data)]  # for some reason NANs appear at the end
#         integration_seconds = self.afm_data.attrs['afm_sampling_milliseconds'] * 1e-3
#         n = len(nea_data)
#         self.z_spectrum = np.abs(np.fft.rfft(nea_data, n))
#         self.z_frequencies = np.fft.rfftfreq(n, integration_seconds)
#
#         signals = [Signals[s] for s in self.afm_data.attrs['signals']]
#         if Signals.tap_x in signals:
#             npz = np.load(self.directory + self.afm_data.attrs['daq_data_filename'])
#             daq_data = np.vstack([i for i in npz.values()]).T
#             tap_x = daq_data[:, signals.index(Signals.tap_x)]
#             tap_y = daq_data[:, signals.index(Signals.tap_y)]
#             tap_p = np.arctan2(tap_y, tap_x)
#             amplitude = tap_x / np.cos(tap_p)
#             n = len(amplitude)
#             dt_seconds = 5e-6  # time between DAQ samples
#             self.amp_spectrum = np.abs(np.fft.rfft(amplitude, n))
#             self.amp_frequencies = np.fft.rfftfreq(n, dt_seconds)
#
#             if Signals.sig_a in signals:
#                 sig_a = daq_data[:, signals.index(Signals.sig_a)]
#                 self.optical_spectrum = np.abs(np.fft.rfft(sig_a, n))
#                 self.optical_frequencies = np.fft.rfftfreq(n, dt_seconds)
#
#     def plot(self):
#         pass
#
#     def export(self):
#         if self.z_spectrum is not None:
#             filename = self.afm_data.attrs['name'] + '.npz'
#             data = np.vstack([self.z_frequencies, self.z_spectrum]).T
#             export_data(filename, data, ['t', 'z'])
#         if self.amp_spectrum is not None:
#             filename = self.afm_data.attrs['name'] + '.npz'
#             data = np.vstack([self.amp_frequencies, self.amp_spectrum]).T
#             export_data(filename, data, ['t', 'amplitude'])
#         if self.optical_spectrum is not None:
#             filename = self.afm_data.attrs['name'] + '.npz'
#             data = np.vstack([self.optical_frequencies, self.optical_spectrum]).T
#             export_data(filename, data, ['t', 'sig_a'])


# TODO: Should probably factor this out into an object to specify the acquisition (roles + channels.. the channel map),
#   and one to specify the analysis (ie: so we can acquire pshet channels, but analyze only the sHD part...
#   The channel map is enough only for single pixels. For scans, we need to add more info. The detector bit is also part
#   of the channel map...
@attr.s(order=False)
class Experiment:
    """
    Describes a TRION experimental configuration.

    Parameters
    ----------
    scan: Scan
        AFM scan protocol, such as single-point, approach curve or AFM
    acquisition: Demodulation
        Interferometric Near field acquisition modes: self-homodyne, pshet, etc.
    detector: Detector
        Optical detector configuration
    nreps: int
        Number of repetitions. Default `1`
    npts: int
        Number of points per frame. Ignored if `continuous`. Default 200_000.
    continuous:
        Continous acquisition. Default True.
    """
    scan: Scan = attr.ib(default=Scan.point)
    acquisition: Demodulation = attr.ib(default=Demodulation.shd)
    detector: Detector = attr.ib(default=Detector.single)
    frame_reps: int = attr.ib(default=1)
    npts: int = attr.ib(default=200_000)
    continuous: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        super().__init__()

    def signals(self) -> Iterable[Signals]:
        """
        Lists the signals for the current Experimental configuration.
        """
        return sorted(chain(
            detection_signals[self.detector],
            modulation_signals[self.acquisition]
        ))

    def is_valid(self) -> bool:
        """
        Check if the experimental configuration is valid.
        """
        # Currently this is a bit basic, but this will get more complicated
        return (
                type(self.scan) is Scan and
                type(self.acquisition) is Demodulation and
                type(self.detector) is Detector
        )

    def axes(self) -> list:
        if self.frame_reps > 1:
            ax = ["frame_rep"]
        else:
            ax = []
        return ax

    def to_dict(self):
        return attr.asdict(self)

    @classmethod
    def from_dict(cls, **cfg):
        cfg = cfg.copy()
        for key, enum in [("scan", Scan),
                          ("demodulation", Demodulation),
                          ("detector", Detector),
                          ]:
            v = cfg["key"]
            if isinstance(v, str):
                cfg[key] = enum[cfg[key]]
        return cls(**cfg)
