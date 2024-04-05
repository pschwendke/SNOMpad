# measurement classes corresponding to acquisition scan classes.
# classes hold methods to read, demodulate, save, and plot SNOM data.
import numpy as np
import xarray as xr
import colorcet as cc
from tqdm import tqdm

from . import shd, pshet
from ..utility.signals import Scan, Demodulation, Signals
from ..demodulation.demod_utils import chop_pump_idx
from .scan_demod_base import BaseScanDemod
from .scan_demod_utils import sort_lines, bin_line
from ..file_handlers.gwyddion import export_gwy


def load(filename: str) -> BaseScanDemod:
    """ Loads scan from .h5 file and returns BaseScanDemod object. The function is agnostic to scan type.
    """
    if 'retraction' in filename:
        return Retraction(filename)
    elif 'image' in filename:
        return Image(filename)
    elif 'line' in filename:
        return Line(filename)
    elif 'noise' in filename:
        return Noise(filename)
    elif 'delay' in filename:
        return Delay(filename)
    elif 'point' in filename:
        return Point(filename)
    else:
        raise NotImplementedError


class Point(BaseScanDemod):
    def demod(self, max_order: int = 5, **kwargs):
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
        data = np.vstack([self.scan.daq_data[i] for i in idx])
        if self.modulation == Demodulation.shd:
            # ToDo This was a quick patch. Revisit this. Is there a better solution, or should this go everywhere?
            try:
                coefficients = shd(data=data, signals=self.signals, **kwargs)
            except ValueError as e:
                if "empty bins" in str(e):
                    coefficients = np.full((max_order + 1,), np.nan)
                else: raise
        elif self.modulation == Demodulation.pshet:
            coefficients = pshet(data=data, signals=self.signals, **kwargs)
        elif self.modulation == Demodulation.none:
            coefficients = np.zeros(max_order+1)
            if self.metadata['pump_probe']:
                chopped_idx, pumped_idx = chop_pump_idx(data[:, self.signals.index(Signals.chop)])
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


class Retraction(BaseScanDemod):
    def reshape(self, demod_npts: int = None):
        """ Defines and brings AFM data to a specific shape (spatial resolution), on which pixels the optical data
        is then demodulated.

        Parameters
        ----------
        demod_npts: int
            number of samples to demodulate for every pixel. Only whole chunks saved in daq_data are combined
        """
        self.demod_data = xr.Dataset()
        self.demod_data.attrs = self.metadata

        if not demod_npts:
            chunk_size = 1  # one chunk of DAQ data per demodulated pixel
            demod_npts = self.scan.metadata['npts']
        else:
            chunk_size = demod_npts // self.scan.metadata['npts'] + 1
            demod_npts = self.scan.metadata['npts'] * chunk_size
        n = len(self.daq_data.keys())
        z_res = n // chunk_size
        self.demod_data.attrs['demod_params'] = {'demod_npts': demod_npts, 'z_res': z_res}

        z = self.afm_data['z'].values.squeeze()
        z = z[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        if self.mode == Scan.stepped_retraction:
            z_target = self.afm_data['z_target'].values.squeeze()
            z_target = z_target[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
            self.demod_data['z_target'] = xr.DataArray(data=z_target, dims='z', coords={'z': z})

        amp = self.afm_data['amp'].values.squeeze()
        amp = amp[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        self.demod_data['amp'] = xr.DataArray(data=amp, dims='z', coords={'z': z})
        phase = self.afm_data['phase'].values.squeeze()
        phase = phase[:z_res*chunk_size].reshape((z_res, chunk_size)).mean(axis=1)
        phase = phase / 2 / np.pi * 360
        self.demod_data['phase'] = xr.DataArray(data=phase, dims='z')
        idx = self.afm_data['idx'].values.squeeze()
        idx = idx[:z_res * chunk_size].reshape((z_res, chunk_size))
        idx = [i for i in idx] + ['']  # this is a truly ugly workaround to make an array of arrays
        idx = np.array(idx, dtype=object)
        self.demod_data['idx'] = xr.DataArray(data=idx[:-1], dims='z')

    def demod(self, max_order: int = 5, **kwargs):
        """ Demodulates optical data for every pixel along dimension z. A complex-valued DataArray for demodulated
        harmonics with extra dimension 'order' is created in self.demod_data, and to self.demod_cache.

        Parameters
        ----------
        max_order: int
            max harmonic order that should be returned.
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.demod_data is None:
            self.reshape()
        demod_params = self.demod_params(old_params=self.demod_data.attrs['demod_params'], kwargs=kwargs)
        hash_key = str(demod_params['hash_key'])
        if hash_key in self.demod_cache.keys() and self.demod_cache[hash_key].attrs['demod_params'] == demod_params:
            self.demod_data = self.demod_cache[hash_key]
        else:
            harmonics = np.zeros((demod_params['z_res'], max_order + 1), dtype=complex)
            for n, px in tqdm(enumerate(self.demod_data.z.values)):
                data = np.vstack([self.scan.daq_data[i] for i in self.demod_data.sel(z=px).idx.item()])
                if self.modulation == Demodulation.shd:
                    coefficients = shd(data=data, signals=self.signals, **kwargs)
                elif self.modulation == Demodulation.pshet:
                    coefficients = pshet(data=data, signals=self.signals, max_order=max_order, **kwargs)
                else:
                    raise NotImplementedError
                harmonics[n] = coefficients[:max_order + 1]
            self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('z', 'order'),
                                                      coords={'order': np.arange(max_order + 1)})
            self.demod_data.attrs['demod_params'] = demod_params
            self.demod_cache[hash_key] = self.demod_data
            self.cache_to_file()

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
            plt.savefig(f'{self.name}.png', dpi=100, bbox_inches='tight')
            # plt.savefig(f'{self.name}.svg', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


class Image(BaseScanDemod):
    def demod(self, max_order: int = 5, **kwargs):
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
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.mode == Scan.continuous_image:
            raise NotImplementedError
        self.demod_data = xr.Dataset()
        x = self.afm_data.x_target.values.squeeze()
        y = self.afm_data.y_target.values.squeeze()
        z = self.afm_data.z.values.squeeze()
        amp = self.afm_data.amp.values.squeeze()
        phase = self.afm_data.phase.values.squeeze()

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
            data = self.scan.daq_data[i]
            if self.modulation == Demodulation.shd:
                coefficients = shd(data=data, signals=self.signals, **kwargs)
            elif self.modulation == Demodulation.pshet:
                coefficients = pshet(data=data, signals=self.signals, max_order=max_order, **kwargs)

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
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        plt.colorbar(label='topography (nm)')
        plt.show()

        sig = self.demod_data.phase / 2 / np.pi * 360  # deg
        plt.imshow(sig, extent=lim, cmap=cmap_phase)
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        plt.colorbar(label=' AFM phase (degrees)')
        plt.show()

        sig = self.demod_data.amp  # nm
        plt.imshow(sig, extent=lim, cmap=cmap_amp)
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        plt.colorbar(label='AFM amplitude (nm)')
        plt.show()

        for o in orders:
            sig = self.demod_data.optical.sel(order=o)
            plt.imshow(np.abs(sig), extent=lim, cmap=cmap_opt)
            plt.xlabel(r'x ($\mu$m)')
            plt.ylabel(r'y ($\mu$m)')
            plt.colorbar(label=f'optical amplitude ({o}. harm) (arb. units)')
            plt.show()

            plt.imshow(np.angle(sig), extent=lim, cmap=cmap_phase)
            plt.xlabel(r'x ($\mu$m)')
            plt.ylabel(r'y ($\mu$m)')
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


class Line(BaseScanDemod):
    def reshape(self, demod_npts: int = None, r_res: int = 100, direction: str = 'scan', line_no: list = None,
                plot: bool = False, trim_ratio: float = .05):
        """ Defines and brings AFM data to a specific shape (spatial resolution), on which pixels the optical data
        is then demodulated.

        Parameters
        ----------
        demod_npts: int
            min number of samples to demodulate for every pixel.
        r_res: int
            resolution in scan direction r of line scan.
        plot: bool
            if True, sorting of lines into approach, scan, rescan, and line numbers will be plotted.
        trim_ratio: float
            ratio of total line width that is cut off as turning points
        direction: 'scan' or 'rescan'
            defines points in which scan direction will be demodulated
        line_no: list
            number of lines that are demodulated together. When [] is passed, all lines in specified direction.
        """
        if self.mode == Scan.stepped_line:
            self.demod_data = xr.Dataset()
            self.demod_data.attrs = self.metadata

            if not demod_npts:
                chunk_size = 1  # one chunk of DAQ data per demodulated pixel
                demod_npts = self.scan.metadata['npts']
            else:
                chunk_size = demod_npts // self.scan.metadata['npts'] + 1
                demod_npts = self.scan.metadata['npts'] * chunk_size
            n = len(self.daq_data.keys())
            r_res = n // chunk_size
            self.demod_data.attrs['demod_params'] = {'demod_npts': demod_npts, 'r_res': r_res}

            x = self.afm_data['x'].values.squeeze()
            x = x[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            y = self.afm_data['y'].values.squeeze()
            y = y[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            z = self.afm_data['z'].values.squeeze()
            z = z[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            r = np.sqrt(x ** 2 + y ** 2)
            r -= r.min()
            self.demod_data['x'] = xr.DataArray(data=x, dims='r', coords={'r': r})
            self.demod_data['y'] = xr.DataArray(data=y, dims='r')
            self.demod_data['z'] = xr.DataArray(data=z, dims='r')
            if self.mode == Scan.stepped_line:
                x_target = self.afm_data['x_target'].values.squeeze()
                x_target = x_target[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
                self.demod_data['x_target'] = xr.DataArray(data=x_target, dims='r')
                y_target = self.afm_data['y_target'].values.squeeze()
                y_target = y_target[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
                self.demod_data['y_target'] = xr.DataArray(data=y_target, dims='r')
            amp = self.afm_data['amp'].values.squeeze()
            amp = amp[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            self.demod_data['amp'] = xr.DataArray(data=amp, dims='r')
            phase = self.afm_data['phase'].values.squeeze()
            phase = phase[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            phase = phase / 2 / np.pi * 360
            self.demod_data['phase'] = xr.DataArray(data=phase, dims='r')
            idx = self.afm_data['idx'].values.squeeze()
            idx = idx[:r_res * chunk_size].reshape((r_res, chunk_size))
            idx = [i for i in idx] + ['']  # this is a truly ugly workaround to make an array of arrays
            idx = np.array(idx, dtype=object)
            self.demod_data['idx'] = xr.DataArray(data=idx[:-1], dims='r')

        elif self.mode == Scan.continuous_line:
            if line_no is None:
                line_no = []
            sort_lines(scan=self, trim_ratio=trim_ratio, plot=plot)  # ToDo: check if this has been done before
            self.demod_data = bin_line(scan=self, res=r_res, direction=direction, line_no=line_no)
            self.demod_data.attrs.update(self.metadata)
            self.demod_data.attrs['demod_params'] = {'r_res': r_res, 'trim_ratio': trim_ratio,
                                                     'direction': direction, 'line_no': line_no}
        else:
            raise NotImplementedError

    def demod(self, max_order: int = 5, **kwargs):
        """ Demodulates optical data for every pixel along dimension r. A complex-valued DataArray for demodulated
        harmonics with extra dimension 'order' is created in self.demod_data, and saved to self.demod_cache.

        Parameters
        ----------
        max_order: int
            max harmonic order that should be returned.
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.demod_data is None:
            self.reshape()
        demod_params = self.demod_params(old_params=self.demod_data.attrs['demod_params'], kwargs=kwargs)
        hash_key = str(demod_params['hash_key'])
        if hash_key in self.demod_cache.keys() and self.demod_cache[hash_key].attrs['demod_params'] == demod_params:
            self.demod_data = self.demod_cache[hash_key]
        else:
            harmonics = np.zeros((demod_params['r_res'], max_order + 1), dtype=complex)
            for n, px in tqdm(enumerate(self.demod_data.r.values)):
                data = np.vstack([self.scan.daq_data[i] for i in self.demod_data.sel(r=px).idx.item()])
                if self.modulation == Demodulation.shd:
                    coefficients = shd(data=data, signals=self.signals, **kwargs)
                elif self.modulation == Demodulation.pshet:
                    coefficients = pshet(data=data, signals=self.signals, max_order=max_order, **kwargs)
                else:
                    raise NotImplementedError
                harmonics[n] = coefficients[:max_order + 1]
            self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('r', 'order'),
                                                      coords={'order': np.arange(max_order + 1)})
            self.demod_data.attrs['demod_params'] = demod_params
            self.demod_cache[hash_key] = self.demod_data
            self.cache_to_file()

    def plot(self, max_order: int = 4, orders=None, grid=False, show=True, save=False):
        """ Plots line scan.

        Parameters
        ----------
        max_order: int
            max order that will be plotted. This value is ignored when orders are passed specifically.
        orders: iterable
            list of orders to be plotted. When this is passed, max_order is ignored
        grid: bool
            When True, a grid is plotted as well.
        show: bool
            When True, the figure is shown via plt.show()
        save: bool
            When True, the figure is saved with a generated file name.
        """
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
        ax_phase.set_ylabel('AFM phase (rad)', color='C02')

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


class Noise(BaseScanDemod):
    def __init__(self, filename: str):
        super().__init__(filename)
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

        daq_data = np.array(self.scan.daq_data[0])
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


# ToDo this needs more cleaning up after redoing DelayScan class
class Delay(BaseScanDemod):
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
