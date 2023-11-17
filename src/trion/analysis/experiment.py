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
from trion.analysis.demod import shd, pshet, sort_chopped
from trion.analysis.io import export_gwy


def h5_to_xr_dataset(group: h5py.Group):
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

        self.metadata = {}
        for k, v in file.attrs.items():
            self.metadata[k] = v

        if self.file['afm_data'].keys():
            self.afm_data = h5_to_xr_dataset(group=self.file['afm_data'])
        if self.file['nea_data'].keys():
            self.nea_data = h5_to_xr_dataset(group=self.file['nea_data'])

    def __str__(self) -> str:
        ret = 'Metadata:\n'
        for k, v in self.metadata.items():
            ret += f'{k}: {v}\n'
        if self.afm_data is not None:
            ret += '\nAFM data:\n'
            ret += self.afm_data.__str__()
        if self.nea_data is not None:
            ret += '\nNeaScan data:\n'
            ret += self.nea_data.__str__()
        return ret
    
    def __repr__(self) -> str:
        return f'<TRION measurement: {self.name}>'

    def to_h5(self):
        """ Write hdf5 demod file in standard directory
        """

    def load_h5(self):
        """ Load hdf5 demod file
        """

    @abstractmethod
    def demod(self):
        """ Collect and demodulate raw data, to produce NF images, retraction curves, spectra etc.
        """
    @abstractmethod
    def plot(self):
        """ Plot (and save) demodulated data, e.g. images or curves
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
    elif scan_type == 'noise_sampling':
        return Noise(file)
    elif scan_type == 'delay_collection':
        return Delay(file)
    elif scan_type == 'point':
        return Point(file)
    else:
        raise NotImplementedError
    

class Point(Measurement):
    def demod(self, max_order: int = 5, demod_filename=None, **kwargs):
        """ Creates a dictionary in self.demod_data. dict contains one np.ndarray for every tracked channel,
        and one complex-valued xr.DataArray for demodulated harmonics, with dimension 'order'.
        When demod_filename is a str, a demod file is created in the given filename. When filename is 'auto',
        the file is saved in the '_ANALYSIS' directory on the server.
        When the demod file already exists, demod data will be added to it. In case demodulation with given parameters
        is already present in demod file, it will be read from the file.

        Parameters
        ----------
        max_order: int
            max harmonic order that should be returned.
        demod_filename: str or 'auto'
            path and filename of demod file
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        self.demod_data = {
            'x': self.afm_data['x'].values.mean(),
            'y': self.afm_data['y'].values.mean(),
            'z': self.afm_data['z'].values.mean(),
            'amp': self.afm_data['amp'].values.mean(),
            'phase': self.afm_data['phase'].values.mean() / 2 / np.pi * 360
        }
        if 't' in self.afm_data.dims:
            self.demod_data['t'] = self.afm_data['t'].values[0]

        # ToDo: check for demod file

        idx = self.afm_data['idx'].values
        # harmonics = np.zeros(max_order + 1, dtype=complex)
        # for px, _ in tqdm(enumerate(harmonics)):
        data = np.vstack([np.array(self.file['daq_data'][str(i)]) for i in idx])
        if self.modulation == Demodulation.shd:
            coefficients = shd(data=data, signals=self.signals, **kwargs)
        elif self.modulation == Demodulation.pshet:
            coefficients = pshet(data=data, signals=self.signals, **kwargs)
        elif self.modulation == Demodulation.none:
            coefficients = np.zeros(max_order+1)
            if self.metadata['pump_probe']:
                chopped_idx, pumped_idx = sort_chopped(data[:, self.signals.index(Signals.chop)])
                chopped = data[chopped_idx, self.signals.index(Signals.sig_a)].mean()
                pumped = data[pumped_idx, self.signals.index(Signals.sig_a)].mean()
                pump_probe = (pumped - chopped) / chopped
                coefficients[0] = pump_probe
            else:
                coefficients[0] = data[:, self.signals.index(Signals.sig_a)].mean()
        else:
            raise NotImplementedError
        harmonics = coefficients[:max_order + 1]
        self.demod_data['optical'] = xr.DataArray(data=harmonics, dims='order',
                                                  coords={'order': np.arange(max_order + 1)})

        # ToDo: create or write to demod file

    def plot(self):
        raise NotImplementedError


class Retraction(Measurement):
    def demod(self, max_order: int = 5, demod_npts=None, demod_filename=None, **kwargs):
        """ Creates xr.Dataset in self.demod_data. Dataset contains one DataArray for every tracked channel,
        and one complex-valued DataArray for demodulated harmonics, with extra dimension 'order'.
        When demod_filename is a str, a demod file is created in the given filename. When filename is 'auto',
        a filename is generated and saved in the working directory
        When the demod file already exists, demod data will be added to it. In case demodulation with given parameters
        is already present in demod file, it will be read from the file.

        Parameters
        ----------
        max_order: int
            max harmonic order that should be returned.
        demod_npts: int
            min number of samples to demodulate for every pixel. Only whole chunks saved in daq_data are combined
        demod_filename: str or 'auto'
            path and filename of demod file
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if not demod_npts and self.mode == Scan.stepped_retraction:
            chunk_size = 1  # one chunk of DAQ data per demodulated pixel
        elif not demod_npts and self.mode == Scan.continuous_retraction:
            chunk_size = 5
        else:
            chunk_size = demod_npts // self.file.attrs['npts'] + 1
        n = len(self.file['daq_data'].keys())
        z_res = n // chunk_size
        self.demod_data = xr.Dataset()

        # reshape data to new z resolution (create new pixels)
        z = self.afm_data['z'].values
        z = z[:z_res * chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        if self.mode == Scan.stepped_retraction:
            z_target = self.afm_data['z_target'].values
            z_target = z_target[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
            self.demod_data['z_target'] = xr.DataArray(data=z_target, dims='z', coords={'z': z})

        amp = self.afm_data['amp'].values
        amp = amp[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        self.demod_data['amp'] = xr.DataArray(data=amp, dims='z', coords={'z': z})
        phase = self.afm_data['phase'].values
        phase = phase[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        phase = phase / 2 / np.pi * 360
        self.demod_data['phase'] = xr.DataArray(data=phase, dims='z', coords={'z': z})

        idx = self.afm_data['idx'].values
        idx = idx[:z_res*chunk_size].reshape((z_res, chunk_size))

        # ToDo: check for demod file

        # demodulate DAQ data for every pixel
        harmonics = np.zeros((z_res, max_order + 1), dtype=complex)
        for px, _ in tqdm(enumerate(harmonics)):
            data = np.vstack([np.array(self.file['daq_data'][str(i)]) for i in idx[px]])
            if self.modulation == Demodulation.shd:
                coefficients = shd(data=data, signals=self.signals, **kwargs)
            elif self.modulation == Demodulation.pshet:
                coefficients = pshet(data=data, signals=self.signals, **kwargs)
            else:
                raise NotImplementedError
            harmonics[px] = coefficients[:max_order + 1]
        self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('z', 'order'),
                                                  coords={'z': z, 'order': np.arange(max_order + 1)})
        self.demod_data.attrs = self.metadata

        # ToDo: create or write to demod file

    def plot(self, max_order: int = 4, orders=None, grid=False, show=True, save=False):
        import matplotlib.pyplot as plt
        if self.demod_data is None:
            self.demod()
        if not (hasattr(orders, '__iter__') and all([type(h) is int for h in orders])):
            orders = np.arange(max_order + 1)

        if self.modulation == Demodulation.pshet:
            fig, ax = plt.subplots(3, 1, sharex='col', figsize=(8, 8))
        else:
            fig, ax = plt.subplots(2, 1, sharex='col', figsize=(8, 8))
        harm_cmap = cc.glasbey_category10

        z = (self.demod_data['z'].values - self.demod_data['z'].values.min()) * 1e3  # nm

        a = self.demod_data['amp'].values / self.demod_data['amp'].values.max()
        ax[0].plot(z, a, marker='.', lw=1, color='C00')
        ax[0].set_ylabel('AFM amplitude (scaled)', color='C00')
        ax[0].grid(visible=grid, which='major', axis='both')

        ax_phase = ax[0].twinx()
        p = self.demod_data['phase'].values
        ax_phase.plot(z, p, color='C01', marker='.', lw=1)
        ax_phase.set_ylabel('AFM phase (degrees)', color='C01')

        for o in orders:
            sig = np.abs(self.demod_data['optical'].values[:, o])
            sig /= np.abs(sig).max()
            ax[1].plot(z, sig, marker='.', lw=1, label=f'abs({o})', color=harm_cmap[o])
        ax[1].grid(visible=grid, which='major', axis='both')
        ax[1].set_ylabel('optical amplitude (scaled)')
        ax[1].legend(loc='upper right')

        if self.modulation == Demodulation.pshet:
            for o in orders:
                sig = np.angle(self.demod_data['optical'].values[:, o])
                ax[2].plot(z, sig, marker='.', lw=1, label=f'phase({o})', color=harm_cmap[o])
            ax[2].grid(visible=grid, which='major', axis='both')
            ax[2].set_ylabel('optical phase')
            ax[2].set_xlabel('dz (nm)')
            ax[2].legend(loc='upper right')
        ax[-1].set_xlabel('dz (nm)')

        if save:
            plt.savefig(f'{self.name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.name}.svg', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


class Image(Measurement):
    def demod(self, max_order: int = 5, demod_filename=None, **kwargs):
        """ Creates xr.Dataset in self.demod_data. Dataset contains one DataArray for every tracked channel,
        and one complex-valued DataArray for demodulated harmonics, with extra dimension 'order'.
        When demod_filename is a str, a demod file is created in the given filename. When filename is 'auto',
        a filename is generated and saved in the working directory.
        When the demod file already exists, demod data will be added to it. In case demodulation with given parameters
        is already present in demod file, it will be read from the file.

        Parameters
        ----------
        max_order: int
            max harmonic order that should be returned.
        demod_filename: str or 'auto'
            path and filename of demod file
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.mode == Scan.continuous_image:
            raise NotImplementedError
        self.demod_data = xr.Dataset()
        x = self.afm_data.x_target.values
        y = self.afm_data.y_target.values
        z = self.afm_data.z.values
        amp = self.afm_data.amp.values
        phase = self.afm_data.phase.values

        self.demod_data['z'] = xr.DataArray(data=z, dims=('y', 'x'), coords={'x': x, 'y': y})
        self.demod_data['z'].attrs['z_unit'] = 'um'
        self.demod_data['amp'] = xr.DataArray(data=amp, dims=('y', 'x'), coords={'x': x, 'y': y})
        self.demod_data['amp'].attrs['z_unit'] = 'nm'
        self.demod_data['phase'] = xr.DataArray(data=phase, dims=('y', 'x'), coords={'x': x, 'y': y})
        self.demod_data['phase'].attrs['z_unit'] = 'rad'

        # ToDo: check for demod file

        harmonics = np.zeros((len(y), len(x), max_order + 1), dtype=complex)
        #  this indexing needs to be redone when demodulating continuous images
        for i in tqdm(self.afm_data.idx.values.flatten()):
            data = np.array(self.file['daq_data'][str(i)])
            if self.modulation == Demodulation.shd:
                coefficients = shd(data=data, signals=self.signals, **kwargs)
            elif self.modulation == Demodulation.pshet:
                coefficients = pshet(data=data, signals=self.signals, **kwargs)
            else:
                raise NotImplementedError

            y_idx, x_idx = np.where(self.afm_data.idx.values == i)
            harmonics[y_idx, x_idx] = coefficients[:max_order + 1]
        self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('y', 'x', 'order'),
                                                  coords={'x': x, 'y': y, 'order': np.arange(max_order + 1)})
        self.demod_data.attrs = self.metadata

        # ToDo: create or write to demod file

    def plot(self, max_order: int = 4, orders=None, show=True, save=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        if self.demod_data is None:
            self.demod()
        if not (hasattr(orders, '__iter__') and all([type(h) is int for h in orders])):
            orders = np.arange(max_order + 1)

        cmap_topo = cc.m_dimgray
        cmap_phase = cc.CET_C3[128:] + cc.CET_C3[:128]
        cmap_amp = cc.m_gray
        cmap_phase = LinearSegmentedColormap.from_list('cyclic', cmap_phase)
        cmap_opt = cc.m_fire
        lim = np.array([self.demod_data.x.values.min(), self.demod_data.x.values.max(),
                        self.demod_data.y.values.max(), self.demod_data.y.values.min()])  # um

        # ToDo make a decent layout for this
        #  implement show and save
        sig = self.demod_data.z.values * 1e3  # nm
        sig -= sig.min()
        plt.imshow(sig, extent=lim, cmap=cmap_topo)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.colorbar(label='topography (nm)')
        plt.show()

        sig = self.demod_data.phase / 2 / np.pi * 360  # deg
        plt.imshow(sig, extent=lim, cmap=cmap_phase)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.colorbar(label=' AFM phase (degrees)')
        plt.show()

        sig = self.demod_data.amp  # nm
        plt.imshow(sig, extent=lim, cmap=cmap_amp)
        plt.xlabel('x (um)')
        plt.ylabel('y (um)')
        plt.colorbar(label='AFM amplitude (nm)')
        plt.show()

        for o in orders:
            sig = self.demod_data.optical.sel(order=o)
            plt.imshow(np.abs(sig), extent=lim, cmap=cmap_opt)
            plt.xlabel('x (um)')
            plt.ylabel('y (um)')
            plt.colorbar(label=f'optical amplitude ({o}. harm) (arb. units)')
            plt.show()

            plt.imshow(np.angle(sig), extent=lim, cmap=cmap_phase)
            plt.xlabel('x (um)')
            plt.ylabel('y (um)')
            plt.colorbar(label=f'optical phase ({o}. harm) (arb. units)')
            plt.show()

    def to_gwy(self, filename=None):
        if self.demod_data is None:
            self.demod()
        if filename is None:
            filename = self.name + '.gwy'

        out_data = xr.Dataset()
        out_data['z'] = self.demod_data['z']
        out_data['amp'] = self.demod_data['amp']
        out_data['phase'] = self.demod_data['phase']

        for o in self.demod_data.order.values:
            out_data[f'optical amplitude {o}'] = np.abs(self.demod_data.optical.sel(order=o))
            out_data[f'optical amplitude {o}'].attrs['z_unit'] = 'V'
            out_data[f'optical phase {o}'] = xr.ufuncs.angle(self.demod_data.optical.sel(order=o))
            out_data[f'optical phase {o}'].attrs['z_unit'] = 'rad'

        out_data.attrs = self.demod_data.attrs

        export_gwy(filename=filename, data=out_data)


class Line(Measurement):
    def demod(self, max_order: int = 5, demod_npts=None, demod_filename=None, **kwargs):
        """ Creates xr.Dataset in self.demod_data. The dimension along line scan is called 'r'.
        Dataset contains one DataArray for every tracked channel, and one complex-valued DataArray for demodulated
        harmonics, with extra dimension 'order'.
        When demod_filename is a str, a demod file is created in the given filename. When filename is 'auto',
        a filename is generated and saved in the working directory
        When the demod file already exists, demod data will be added to it. In case demodulation with given parameters
        is already present in demod file, it will be read from the file.

        Parameters
        ----------
        max_order: int
            max harmonic order that should be returned.
        demod_npts: int
            min number of samples to demodulate for every pixel. Only whole chunks saved in daq_data are combined
        demod_filename: str or 'auto'
            path and filename of demod file
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if not demod_npts and self.mode == Scan.stepped_line:
            chunk_size = 1  # one chunk of DAQ data per demodulated pixel
        elif not demod_npts and self.mode == Scan.continuous_line:
            chunk_size = 5
        else:
            chunk_size = demod_npts // self.file.attrs['npts'] + 1
        n = len(self.file['daq_data'].keys())
        r_res = n // chunk_size
        self.demod_data = xr.Dataset()

        # reshape data to new r resolution (create new pixels)
        x = self.afm_data['x'].values
        x = x[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
        y = self.afm_data['y'].values
        y = y[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
        z = self.afm_data['z'].values
        z = z[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
        r = np.sqrt(x**2 + z**2)
        r -= r.min()
        self.demod_data['x'] = xr.DataArray(data=x, dims='r', coords={'r': r})
        self.demod_data['y'] = xr.DataArray(data=y, dims='r', coords={'r': r})
        self.demod_data['z'] = xr.DataArray(data=z, dims='r', coords={'r': r})
        if self.mode == Scan.stepped_line:
            x_target = self.afm_data['x_target'].values
            x_target = x_target[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            self.demod_data['x_target'] = xr.DataArray(data=x_target, dims='r', coords={'r': r})
            y_target = self.afm_data['y_target'].values
            y_target = y_target[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            self.demod_data['y_target'] = xr.DataArray(data=y_target, dims='r', coords={'r': r})

        amp = self.afm_data['amp'].values
        amp = amp[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
        self.demod_data['amp'] = xr.DataArray(data=amp, dims='r', coords={'r': r})
        phase = self.afm_data['phase'].values
        phase = phase[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
        phase = phase / 2 / np.pi * 360
        self.demod_data['phase'] = xr.DataArray(data=phase, dims='r', coords={'r': r})

        idx = self.afm_data['idx'].values
        idx = idx[:r_res * chunk_size].reshape((r_res, chunk_size))

        # ToDo: check for demod file

        # demodulate DAQ data for every pixel
        harmonics = np.zeros((r_res, max_order + 1), dtype=complex)
        for px, _ in tqdm(enumerate(harmonics)):
            data = np.vstack([np.array(self.file['daq_data'][str(i)]) for i in idx[px]])
            if self.modulation == Demodulation.shd:
                coefficients = shd(data=data, signals=self.signals, **kwargs)
            elif self.modulation == Demodulation.pshet:
                coefficients = pshet(data=data, signals=self.signals, **kwargs)
            else:
                raise NotImplementedError
            harmonics[px] = coefficients[:max_order + 1]
        self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('r', 'order'),
                                                  coords={'r': r, 'order': np.arange(max_order + 1)})
        self.demod_data.attrs = self.metadata

        # ToDo: create or write to demod file

    def plot(self, max_order: int = 4, orders=None, grid=False, show=True, save=False):
        import matplotlib.pyplot as plt
        if self.demod_data is None:
            self.demod()
        if not (hasattr(orders, '__iter__') and all([type(h) is int for h in orders])):
            orders = np.arange(max_order + 1)

        if self.modulation == Demodulation.pshet:
            fig, ax = plt.subplots(3, 1, sharex='col', figsize=(8, 8))
        else:
            fig, ax = plt.subplots(2, 1, sharex='col', figsize=(8, 8))
        harm_cmap = cc.glasbey_category10

        r = self.demod_data['r'].values * 1e3  # nm

        z = (self.demod_data['z'].values - self.demod_data['z'].values.min()) * 1e3  # nm
        ax[0].plot(r, z, marker='.', lw=1, color='C00')
        ax[0].set_ylabel('AFM topography (nm)', color='C00')
        ax[0].grid(visible=grid, which='major', axis='both')

        ax_amp = ax[0].twinx()
        a = self.demod_data['amp'].values
        ax_amp.plot(r, a, color='C01', marker='.', lw=1, label='AFM amplitude')
        ax_amp.tick_params(right=False, labelright=False)
        ax_amp.legend(loc='upper right')

        ax_phase = ax[0].twinx()
        p = self.demod_data['phase'].values
        ax_phase.plot(r, p, color='C02', marker='.', lw=1)
        ax_phase.set_ylabel('AFM phase (degrees)', color='C02')

        for o in orders:
            sig = np.abs(self.demod_data['optical'].values[:, o])
            sig /= np.abs(sig).max()
            ax[1].plot(r, sig, marker='.', lw=1, label=f'abs({o})', color=harm_cmap[o])
        ax[1].grid(visible=grid, which='major', axis='both')
        ax[1].set_ylabel('optical amplitude (scaled)')
        ax[1].legend(loc='upper right')

        if self.modulation == Demodulation.pshet:
            for o in orders:
                sig = np.angle(self.demod_data['optical'].values[:, o])
                ax[2].plot(r, sig, marker='.', lw=1, label=f'phase({o})', color=harm_cmap[o])
            ax[2].grid(visible=grid, which='major', axis='both')
            ax[2].set_ylabel('optical phase')
            ax[2].set_xlabel('dz (nm)')
            ax[2].legend(loc='upper right')
        ax[-1].set_xlabel('r (nm)')

        if save:
            plt.savefig(f'{self.name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.name}.svg', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


class Noise(Measurement):
    def __init__(self, file: h5py.File):
        super().__init__(file=file)
        self.z_spectrum = None
        self.z_frequencies = None
        self.amp_spectrum = None
        self.amp_frequencies = None
        self.optical_spectrum = None
        self.optical_frequencies = None

    def demod(self):
        nea_data = self.nea_data['Z'].values.flatten()
        nea_data = nea_data[~np.isnan(nea_data)]  # cut away any NANs
        integration_seconds = self.metadata['afm_sampling_ms'] * 1e-3
        n = len(nea_data)
        self.z_spectrum = np.abs(np.fft.rfft(nea_data, n))
        self.z_frequencies = np.fft.rfftfreq(n, integration_seconds)

        signals = [Signals[s] for s in self.metadata['signals']]

        daq_data = np.array(self.file['daq_data']['1'])
        tap_x = daq_data[:, signals.index(Signals.tap_x)]
        tap_y = daq_data[:, signals.index(Signals.tap_y)]
        tap_p = np.arctan2(tap_y, tap_x)
        amplitude = tap_x / np.cos(tap_p)
        n = len(amplitude)
        dt_seconds = 5e-6  # time between DAQ samples
        self.amp_spectrum = np.abs(np.fft.rfft(amplitude, n))
        self.amp_frequencies = np.fft.rfftfreq(n, dt_seconds)

        sig_a = daq_data[:, signals.index(Signals.sig_a)]
        sig_b = daq_data[:, signals.index(Signals.sig_b)]
        self.optical_spectrum = np.vstack([np.abs(np.fft.rfft(sig_a, n)), np.abs(np.fft.rfft(sig_b, n))])
        self.optical_frequencies = np.fft.rfftfreq(n, dt_seconds)

    def plot(self, save=False, show=True):
        import matplotlib.pyplot as plt
        if self.z_spectrum is None:
            self.demod()

        fig, ax = plt.subplots(3, 1, sharex='col', figsize=(8, 8))

        f = self.z_frequencies
        sig = self.z_spectrum
        ax[0].plot(f, sig, lw=1)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('z piezo movement')

        f = self.amp_frequencies
        sig = self.amp_spectrum
        ax[1].plot(f, sig, lw=1)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_ylabel('deflection laser')

        f = self.optical_frequencies
        sig_a = self.optical_spectrum[0]
        sig_b = self.optical_spectrum[1]
        ax[2].plot(f, sig_a, lw=1, label='sig_a')
        ax[2].plot(f, sig_b, lw=1, label='sig_b')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].set_xlabel('frequency (Hz)')
        ax[2].set_ylabel('probe scatter')

        plt.legend()

        if save:
            plt.savefig(f'{self.name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.name}.svg', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


class Delay(Measurement):
    def demod(self, max_order: int = 5, demod_filename=None, **kwargs):
        if self.metadata['scan_type'] == 'point':
            meas_class = Point
        else:
            raise NotImplementedError
            # Todo: extend for other scan types

        group_names = []  # I have the feeling this is more complicated than it should
        for k, v in self.file.items():
            if 't' in v.attrs.keys():
                group_names.append(k)
        group_names_sorted = np.array(group_names)
        group_names_sorted[[int(g.split('_')[-1]) for g in group_names]] = group_names
        delay_positions = []
        for n in tqdm(group_names_sorted):
            pos = meas_class(self.file[n])
            pos.demod(max_order=max_order, **kwargs)
            delay_positions.append(pos.demod_data)
            # Todo: collect demod files

        t = np.array([pos['t'] for pos in delay_positions])
        optical = np.vstack([pos['optical'].values for pos in delay_positions])

        self.demod_data = xr.Dataset({
            # ToDo: does this also work for retraction, line, and images?
            'x': xr.DataArray(data=np.array([pos['x'] for pos in delay_positions]), dims='t', coords={'t': t}),
            'y': xr.DataArray(data=np.array([pos['y'] for pos in delay_positions]), dims='t', coords={'t': t}),
            'z': xr.DataArray(data=np.array([pos['z'] for pos in delay_positions]), dims='t', coords={'t': t}),
            'amp': xr.DataArray(data=np.array([pos['amp'] for pos in delay_positions]), dims='t', coords={'t': t}),
            'phase': xr.DataArray(data=np.array([pos['phase'] for pos in delay_positions]), dims='t',
                                  coords={'t': t}),
            'optical': xr.DataArray(data=optical, dims=('t', 'order'),
                                    coords={'t': t, 'order': np.arange(max_order + 1)})
        })

    def plot(self):
        raise NotImplementedError


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
