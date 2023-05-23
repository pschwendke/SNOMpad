import numpy as np
import xarray as xr
import attr
import colorcet as cc

from tqdm import tqdm
from itertools import chain
from typing import Iterable
from abc import ABC, abstractmethod

from trion.analysis.signals import Scan, Demodulation, Detector, Signals, detection_signals, modulation_signals
from trion.analysis.io import export_data
from trion.analysis.demod import shd, pshet


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


class Measurement(ABC):
    """ Base class to load, demodulate and export acquired data.
    """
    def __init__(self, ds: xr.Dataset, directory):
        self.afm_data = ds
        self.directory = directory
        try:
            self.afm_data = self.afm_data['__xarray_dataarray_variable__']
        except KeyError:
            pass
        self.signals = [Signals[s] for s in self.afm_data.attrs['signals']]
        self.modulation = Demodulation[self.afm_data.attrs['modulation']]

        # TODO delete this when merged into develop (should be set in BaseScan)
        if 'name' not in self.afm_data.attrs.keys():
            name = self.afm_data.attrs['date'].replace('-', '').replace('T', '-').replace(':', '')
            name += '_' + self.afm_data.attrs['acquisition_mode']
            self.afm_data.attrs['name'] = name

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


class Image(Measurement):
    def __init__(self, ds: xr.Dataset, directory=''):
        super().__init__(ds, directory)

    def demod(self):
        pass

    def plot(self):
        pass

    def export(self):
        pass


class Retraction(Measurement):
    def __init__(self, ds: xr.Dataset, directory=''):
        super().__init__(ds, directory)
        self.harmonics = None
        self.z = None
        self.m1a = None
        self.m1p = None

        # TODO delete this when merged into develop (should be set in BaseScan)
        if 'name' not in self.afm_data.attrs.keys():
            name = self.afm_data.attrs['date'].replace('T', '_').replace(':', '').replace('-', '')
            name += '_' + self.afm_data.attrs['acquisition_mode']
            self.afm_data.attrs['name'] = name

    def demod(self, tap_nbins: int = 64, ref_nbins: int = 64, combine_px=None):
        if not combine_px and self.afm_data.attrs['acquisition_mode'] == 'stepped_retraction':
            combine_px = 1
        elif not combine_px and self.afm_data.attrs['acquisition_mode'] == 'continuous_retraction':
            combine_px = 8
        max_order = int(tap_nbins // 2 + 1)
        z_res = len(self.afm_data['px']) // combine_px
        self.harmonics = np.zeros((z_res, max_order, 1), dtype=complex)

        # read tracked AFM data
        if self.afm_data.attrs['acquisition_mode'] == 'stepped_retraction':
            self.z = self.afm_data.values[:, list(self.afm_data['ch'].values).index('z_target')]  # single_point() is buggy
            self.m1a = (self.afm_data.values[:, list(self.afm_data['ch'].values).index('M1A_pre')] +
                        self.afm_data.values[:, list(self.afm_data['ch'].values).index('M1A_post')]) / 2
            self.m1p = (self.afm_data.values[:, list(self.afm_data['ch'].values).index('M1P_pre')] +
                        self.afm_data.values[:, list(self.afm_data['ch'].values).index('M1P_post')]) / 2
        elif self.afm_data.attrs['acquisition_mode'] == 'continuous_retraction':
            self.z = self.afm_data.values[:, list(self.afm_data['ch'].values).index('Z')]
            self.m1a = self.afm_data.values[:, list(self.afm_data['ch'].values).index('M1A')]
            self.m1p = self.afm_data.values[:, list(self.afm_data['ch'].values).index('M1P')]
        else:
            raise NotImplementedError
        # and reshape to given z-resolution
        self.z = self.z[:z_res * combine_px].reshape((z_res, combine_px)).mean(axis=1)
        self.m1a = self.m1a[:z_res*combine_px].reshape((z_res, combine_px)).mean(axis=1)
        self.m1p = self.m1p[:z_res*combine_px].reshape((z_res, combine_px)).mean(axis=1)

        data_buffer = []
        for p in tqdm(self.afm_data['px'].values):
            filename = f'{self.afm_data.attrs["data_folder"]}/pixel_{p:05d}.npz'
            npz = np.load(self.directory + filename)
            data_buffer.append(np.vstack([i for i in npz.values()]).T)
            if (p+1) % combine_px == 0:
                data = np.vstack(data_buffer)
                if self.modulation == Demodulation.shd:
                    coefficients = shd(data=data, signals=self.signals, tap_nbins=tap_nbins)
                elif self.modulation == Demodulation.pshet:
                    coefficients = pshet(data=data, signals=self.signals, tap_nbins=tap_nbins, ref_nbins=ref_nbins)
                else:
                    raise NotImplementedError
                self.harmonics[int((p+1) / combine_px) - 1] = coefficients
                data_buffer = []

    def plot(self, max_order: int = 4, orders=None, afm_amp=False, afm_phase=False, grid=True, show=True, save=True):
        import matplotlib.pyplot as plt
        if self.harmonics is None:
            self.demod()
        if not (hasattr(orders, '__iter__') and all([type(h) is int for h in orders])):
            orders = np.arange(max_order)

        fig, ax1 = plt.subplots()
        cmap = cc.glasbey_category10
        x = (self.z - self.z.min()) * 1e9

        if afm_phase:
            ax2 = ax1.twinx()
            y = self.m1p / 2 / np.pi * 360  # degrees
            ax2.plot(x, y, color='gray', marker='.', ms=3, lw=.5)
            ax2.set_ylabel('AFM phase (degrees)', color='gray')

        if afm_amp:
            ax3 = ax1.twinx()
            y = self.m1a / self.m1a.max()
            ax3.plot(x, y, marker='.', ms=3, lw=.5, label='AFM amplitude (scaled)', color='darkblue', alpha=.5)
            ax3.tick_params(right=False, labelright=False)
            ax3.legend(loc='upper right')

        for o in orders:
            y = np.real(self.harmonics[:, o].squeeze())
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
        pass


class Noise(Measurement):
    def __init__(self, ds: xr.Dataset, directory=''):
        super().__init__(ds, directory)
        self.z_spectrum = None
        self.z_frequencies = None
        self.amp_spectrum = None
        self.amp_frequencies = None
        self.optical_spectrum = None
        self.optical_frequencies = None

    def demod(self):
        nea_data = self.afm_data['Z'].values.flatten()
        nea_data = nea_data[~np.isnan(nea_data)]  # for some reason NANs appear at the end
        integration_seconds = self.afm_data.attrs['afm_sampling_milliseconds'] * 1e-3
        n = len(nea_data)
        self.z_spectrum = np.abs(np.fft.rfft(nea_data, n))
        self.z_frequencies = np.fft.rfftfreq(n, integration_seconds)

        signals = [Signals[s] for s in self.afm_data.attrs['signals']]
        if Signals.tap_x in signals:
            npz = np.load(self.directory + self.afm_data.attrs['daq_data_filename'])
            daq_data = np.vstack([i for i in npz.values()]).T
            tap_x = daq_data[:, signals.index(Signals.tap_x)]
            tap_y = daq_data[:, signals.index(Signals.tap_y)]
            tap_p = np.arctan2(tap_y, tap_x)
            amplitude = tap_x / np.cos(tap_p)
            n = len(amplitude)
            dt_seconds = 5e-6  # time between DAQ samples
            self.amp_spectrum = np.abs(np.fft.rfft(amplitude, n))
            self.amp_frequencies = np.fft.rfftfreq(n, dt_seconds)

            if Signals.sig_a in signals:
                sig_a = daq_data[:, signals.index(Signals.sig_a)]
                self.optical_spectrum = np.abs(np.fft.rfft(sig_a, n))
                self.optical_frequencies = np.fft.rfftfreq(n, dt_seconds)

    def plot(self):
        pass

    def export(self):
        if self.z_spectrum is not None:
            filename = self.afm_data.attrs['name'] + '.npz'
            data = np.vstack([self.z_frequencies, self.z_spectrum]).T
            export_data(filename, data, ['t', 'z'])
        if self.amp_spectrum is not None:
            filename = self.afm_data.attrs['name'] + '.npz'
            data = np.vstack([self.amp_frequencies, self.amp_spectrum]).T
            export_data(filename, data, ['t', 'amplitude'])
        if self.optical_spectrum is not None:
            filename = self.afm_data.attrs['name'] + '.npz'
            data = np.vstack([self.optical_frequencies, self.optical_spectrum]).T
            export_data(filename, data, ['t', 'sig_a'])


def load(filename: str) -> Measurement:
    """ Loads scan from .nc file and returns Measurement object. The function is agnostic to scan type.
    """
    scan = xr.load_dataset(filename)
    directory = '/'.join(filename.split('/')[:-1]) + '/'
    # directory = directory[1:]  # ToDo: does this work on a windows machine ???
    try:
        scan_type = Scan[scan.attrs['acquisition_mode']]
    except KeyError:
        scan_type = Scan[scan['__xarray_dataarray_variable__'].attrs['acquisition_mode']]

    if scan_type in [Scan.stepped_retraction, Scan.continuous_retraction]:
        return Retraction(scan, directory)
    elif scan_type in [Scan.stepped_image, Scan.continuous_image]:
        return Image(scan, directory)
    elif scan_type == Scan.noise_sampling:
        return Noise(scan, directory)
    else:
        raise NotImplementedError
