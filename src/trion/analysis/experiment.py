import numpy as np
import xarray as xr
import attr

from itertools import chain
from typing import Iterable
from abc import ABC, abstractmethod

from trion.analysis.signals import Scan, Demodulation, Detector, Signals, \
    detection_signals, modulation_signals, all_modulation_signals, \
    all_detector_signals
from trion.analysis.io import export_data

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
    def __init__(self, ds: xr.Dataset):
        self.afm_data = ds

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
    def __init__(self, ds: xr.Dataset):
        super().__init__(ds)

    def demod(self):
        pass

    def plot(self):
        pass

    def export(self):
        pass


class Retraction(Measurement):
    def __init__(self, ds: xr.Dataset):
        super().__init__(ds)

    def demod(self):
        pass

    def plot(self):
        pass

    def export(self):
        pass


class Noise(Measurement):
    def __init__(self, ds: xr.Dataset):
        super().__init__(ds)
        self.z_spectrum = None
        self.z_frequencies = None
        self.amp_spectrum = None
        self.amp_frequencies = None
        self.optical_spectrum = None
        self.optical_frequencies = None

    def demod(self):
        nea_data = self.afm_data.values().flatten()
        integration_seconds = self.afm_data.attrs['afm_sampling_milliseconds'] * 1e-3
        n = len(nea_data)
        self.z_spectrum = np.fft.rfft(nea_data, n)
        self.z_frequencies = np.fft.rfftfreq(n, integration_seconds)

        signals = [Signals[s] for s in self.afm_data['signals']]
        if Signals.tap_x in signals:
            npz = np.load(self.afm_data.attrs['daq_data_filename'])
            daq_data = np.vstack([i for i in npz.items()]).T
            tap_x = daq_data[:, signals.index(Signals.tap_x)]
            tap_y = daq_data[:, signals.index(Signals.tap_y)]
            tap_p = np.arctan2(tap_y, tap_x)
            amplitude = tap_x / np.cos(tap_p)
            n = len(amplitude)
            dt_seconds = 5e-6  # time between DAQ samples
            self.amp_spectrum = np.fft.rfft(amplitude, n)
            self.amp_frequencies = np.fft.rfft(n, dt_seconds)

            if Signals.sig_a in signals:
                sig_a = daq_data[:, signals.index(Signals.sig_a)]
                self.optical_spectrum = np.fft.rfft(sig_a, n)
                self.optical_frequencies = np.fft.rfftfreq(n, dt_seconds)

    def plot(self):
        pass

    def export(self):
        if self.z_spectrum is not None:
            filename = f'{self.afm_data.attrs["date"]}_z_spectrum.npz'
            data = np.vstack([self.z_frequencies, self.z_spectrum]).T
            export_data(filename, data, ['t', 'z'])
        if self.amp_spectrum is not None:
            filename = f'{self.afm_data.attrs["date"]}_afm_amplitude_spectrum.npz'
            data = np.vstack([self.amp_frequencies, self.amp_spectrum]).T
            export_data(filename, data, ['t', 'amplitude'])
        if self.optical_spectrum is not None:
            filename = f'{self.afm_data.attrs["date"]}_optical_spectrum.npz'
            data = np.vstack([self.optical_frequencies, self.optical_spectrum]).T
            export_data(filename, data, ['t', 'sig_a'])


# def load_image(filename: str) -> Image:
#     """ Loads stepped or continuous images from .nc files saved after acquisition, and returns an Image object
#     """
#     pass
#
#
# def load_retraction(filename: str) -> Retraction:
#     """ Loads stepped or continuous retraction curves from .nc files saved after acquisition,
#     and returns a Retraction object
#     """
#     pass
#
#
# def load_noise(filename: str) -> Noise:
#     """ Loads AFM noise sampling from .nc files saved after acquisition, and returns a Noise object
#     """
#     afm_data = xr.load_dataset(filename)


def load(filename: str) -> Measurement:
    """ Loads scan from .nc file and returns Measurement object. The function is agnostic to scan type.
    """
    scan = xr.load_dataset(filename)
    scan_type = Scan[scan.attrs['acquisition_mode']]
    if scan_type in [Scan.stepped_retraction, Scan.continuous_retraction]:
        return Retraction(scan)
    elif scan_type in[Scan.stepped_image, Scan.continuous_image]:
        return Image(scan)
    elif scan_type == Scan.noise_sampling:
        return Noise(scan)
    else:
        raise NotImplementedError
