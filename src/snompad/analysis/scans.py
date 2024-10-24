# measurement classes corresponding to acquisition scan classes.
# classes hold methods to read, demodulate, save, and plot SNOM data.
""" Classes all have a similar structure: First the raw data is put to a specific shape, i.e. indexed pixels with
associated chunks of DAQ data, and averaged AFM data.
These indexed pixels are then demodulated, and the optical data stored together with AFM data in the xr.Dataset
demod_data.
demod_data is as well stored in the demod_cache, to avoid an unnecessary repetition of the demodulation.
The demod_cache can be written to and read from demod_file, which represents hdf5 storage on the hard drive.
"""
import numpy as np
import xarray as xr
import colorcet as cc
from tqdm import tqdm

from snompad.analysis.base import BaseScanDemod
from snompad.analysis.utils import sort_lines, bin_line
from snompad.analysis.image_corrections import subtract_linear
from snompad.demodulation import shd, pshet
from snompad.demodulation.utils import chop_pump_idx
from snompad.utility import Scan, Demodulation, Signals, c_air
from snompad.file_handlers.gwyddion import export_gwy


class Retraction(BaseScanDemod):
    def reshape(self, demod_npts: int = None):
        """ Defines and brings AFM data to a specific shape (spatial resolution), on which pixels the optical data
        is then demodulated. The resulting xr.Dataset is located at self.demod_data.

        PARAMETERS
        ----------
        demod_npts: int
            minimum number of samples to demodulate for every pixel. Only whole chunks saved in daq_data are combined
        """
        self.demod_data = xr.Dataset()
        self.demod_data.attrs = self.metadata

        chunk_size = 1  # one chunk of DAQ data per demodulated pixel
        if isinstance(demod_npts, int):
            chunk_size = demod_npts // self.metadata['npts'] + 1
            chunk_size -= int(demod_npts % self.metadata['npts'] == 0)  # subtract extra chunk for exact matches
        demod_npts = self.metadata['npts'] * chunk_size
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
        self.demod_data['phase'] = xr.DataArray(data=phase, dims='z', coords={'z': z})
        idx = self.afm_data['idx'].values.squeeze()
        idx = idx[:z_res * chunk_size].reshape((z_res, chunk_size))
        idx = [i for i in idx] + ['']       # this is a truly ugly workaround to make an array of arrays of possibly
        idx = np.array(idx, dtype=object)   # different lengths to save as xr.DataArray
        self.demod_data['idx'] = xr.DataArray(data=idx[:-1], dims='z', coords={'z': z})

    def demod(self, save: bool = False, **kwargs):
        """ Demodulates optical data for every pixel along dimension z. A complex-valued DataArray for demodulated
        harmonics with extra dimension 'order' is created in self.demod_data, and to self.demod_cache.

        PARAMETERS
        ----------
        save: bool
            if True, demod data will be stored in, and read from a file if available
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.demod_data is None:
            self.reshape()
        if save and self.demod_file is None:
            self.open_demod_file()
        demod_func = [shd, pshet][[Demodulation.shd, Demodulation.pshet].index(self.modulation)]
        demod_params = self.demod_params(params=self.demod_data.attrs['demod_params'], kwargs=kwargs)
        demod_key = ''.join(sorted([k + str(v) for k, v in demod_params.items()]))
        if demod_key in self.demod_cache.keys():
            self.demod_data = self.demod_cache[demod_key]
        else:
            harmonics = []
            for idx in tqdm(self.demod_data.idx):
                data = np.vstack([self.daq_data[i] for i in idx.item()])
                harmonics.append(demod_func(data=data, signals=self.signals, **kwargs))
            harmonics = np.array(harmonics)
            self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('z', 'order'),
                                                      coords={'order': np.arange(harmonics.shape[1]),
                                                              'z': self.demod_data['z'].values})
            self.demod_data.attrs['demod_params'] = demod_params
            self.demod_cache[demod_key] = self.demod_data
            if save:
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
        amp_color = cc.glasbey_category10[92]
        phase_color = cc.glasbey_category10[94]

        z = (self.demod_data['z'].values - self.demod_data['z'].values.min()) * 1e3  # nm

        a = self.demod_data['amp'].values
        ax[0].plot(z, a, marker='.', lw=1, color=amp_color)
        ax[0].set_ylabel('AFM amplitude (nm)', color=amp_color)
        ax[0].grid(visible=grid, which='major', axis='both')

        ax_phase = ax[0].twinx()
        p = self.demod_data['phase'].values
        ax_phase.plot(z, p, color=phase_color, marker='.', lw=1)
        ax_phase.set_ylabel('AFM phase (degrees)', color=phase_color)

        for o in orders:
            sig = np.abs(self.demod_data['optical'].values[:, o])
            sig /= np.abs(sig).max()
            ax[1].plot(z, sig, marker='.', lw=1, label=r'$s_{{{o}}}$'.format(o=str(o)), color=harm_cmap[o])
        ax[1].grid(visible=grid, which='major', axis='both')
        ax[1].set_ylabel('optical amplitude (scaled)')
        ax[1].legend(loc='upper right')

        if self.modulation == Demodulation.pshet:
            for o in orders:
                sig = np.angle(self.demod_data['optical'].values[:, o])
                ax[2].plot(z, sig, marker='.', lw=1, label=r'$\varphi_{{{o}}}$'.format(o=str(o)), color=harm_cmap[o])
            ax[2].grid(visible=grid, which='major', axis='both')
            ax[2].set_ylabel('optical phase')
            ax[2].set_xlabel('dz (nm)')
            ax[2].legend(loc='upper right')
        ax[-1].set_xlabel('dz (nm)')

        if save:
            plt.savefig(f'{self.name}.png', dpi=100, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def to_csv(self, filename: str = None):
        """ Simply exports self.demod_data to a csv file.
        """
        if self.demod_data is None:
            self.demod()
        if filename is None:
            filename = f'{self.name}.csv'
        col_names = ['z', 'amp', 'phase']
        out = np.vstack([self.demod_data[c].values for c in col_names])
        optical = []
        for o in self.demod_data.order.values:
            col_names.append(f's_{o}')
            optical.append(np.abs(self.demod_data.optical.sel(order=o).values))
            if self.modulation == Demodulation.pshet:
                col_names.append(f'phi_{o}')
                optical.append(np.angle(self.demod_data.optical.sel(order=o).values))
        optical = np.array(optical)
        out = np.vstack([out, optical]).T
        col_names = ','.join(col_names)
        np.savetxt(filename, out, delimiter=',', header=col_names)


class Image(BaseScanDemod):
    def demod(self, max_order: int = 5, **kwargs):
        """ Creates xr.Dataset in self.demod_data. Dataset contains one DataArray for every tracked channel,
        and one complex-valued DataArray for demodulated harmonics, with extra dimension 'order'.
        When demod_filename is a str, a demod file is created in the given filename. When filename is 'auto',
        a filename is generated and saved in the working directory.
        When the demod file already exists, demod data will be added to it. In case demodulation with given parameters
        is already present in demod file, it will be read from the file.

        PARAMETERS
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
            data = self.daq_data[i]
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

        PARAMETERS
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

            chunk_size = 1  # one chunk of DAQ data per demodulated pixel
            if isinstance(demod_npts, int):
                chunk_size = demod_npts // self.metadata['npts'] + 1
                chunk_size -= int(
                    demod_npts % self.metadata['npts'] == 0)  # subtract extra chunk for exact matches
            demod_npts = self.metadata['npts'] * chunk_size
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
            self.demod_data['y'] = xr.DataArray(data=y, dims='r', coords={'r': r})
            self.demod_data['z'] = xr.DataArray(data=z, dims='r', coords={'r': r})
            if self.mode == Scan.stepped_line:
                x_target = self.afm_data['x_target'].values.squeeze()
                x_target = x_target[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
                self.demod_data['x_target'] = xr.DataArray(data=x_target, dims='r', coords={'r': r})
                y_target = self.afm_data['y_target'].values.squeeze()
                y_target = y_target[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
                self.demod_data['y_target'] = xr.DataArray(data=y_target, dims='r', coords={'r': r})
            amp = self.afm_data['amp'].values.squeeze()
            amp = amp[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            self.demod_data['amp'] = xr.DataArray(data=amp, dims='r', coords={'r': r})
            phase = self.afm_data['phase'].values.squeeze()
            phase = phase[:r_res * chunk_size].reshape((r_res, chunk_size)).mean(axis=1)
            phase = phase / 2 / np.pi * 360
            self.demod_data['phase'] = xr.DataArray(data=phase, dims='r', coords={'r': r})
            idx = self.afm_data['idx'].values.squeeze()
            idx = idx[:r_res * chunk_size].reshape((r_res, chunk_size))
            idx = [i for i in idx] + ['']       # this is a truly ugly workaround to make an array of arrays of possibly
            idx = np.array(idx, dtype=object)   # different lengths to save as xr.DataArray
            self.demod_data['idx'] = xr.DataArray(data=idx[:-1], dims='r', coords={'r': r})

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

    def demod(self, save: bool = False, **kwargs):
        """ Demodulates optical data for every pixel along dimension r. A complex-valued DataArray for demodulated
        harmonics with extra dimension 'order' is created in self.demod_data, and saved to self.demod_cache.

        PARAMETERS
        ----------
        save: bool
            if True, demod data will be stored in, and read from a file if available
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.demod_data is None:
            self.reshape()
        if save and self.demod_file is None:
            self.open_demod_file()
        demod_func = [shd, pshet][[Demodulation.shd, Demodulation.pshet].index(self.modulation)]
        demod_params = self.demod_params(params=self.demod_data.attrs['demod_params'], kwargs=kwargs)
        demod_key = ''.join(sorted([k + str(v) for k, v in demod_params.items()]))
        if demod_key in self.demod_cache.keys():
            self.demod_data = self.demod_cache[demod_key]
        else:
            harmonics = []
            for idx in tqdm(self.demod_data.idx):
                data = np.vstack([self.daq_data[i] for i in idx.item()])
                harmonics.append(demod_func(data=data, signals=self.signals, **kwargs))
            harmonics = np.array(harmonics)
            self.demod_data['optical'] = xr.DataArray(data=harmonics, dims=('r', 'order'),
                                                      coords={'order': np.arange(harmonics.shape[1]),
                                                              'r': self.demod_data['r'].values})
            self.demod_data.attrs['demod_params'] = demod_params
            self.demod_cache[demod_key] = self.demod_data
            if save:
                self.cache_to_file()

    def plot(self, max_order: int = 4, orders=None, grid=False, show=True, save=False):
        """ Plots line scan.

        PARAMETERS
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
        z_color = cc.glasbey_category10[95]
        amp_color = cc.glasbey_category10[92]
        phase_color = cc.glasbey_category10[94]

        r = self.demod_data['r'].values * 1e3  # nm

        # z = (self.demod_data['z'].values - self.demod_data['z'].values.min()) * 1e3  # nm
        z = self.demod_data['z'].values * 1e3  # nm
        z = subtract_linear(line=z)
        ax[0].plot(r, z, marker='.', lw=1, color=z_color)
        ax[0].set_ylabel('AFM topography, corrected linear (nm)', color=z_color)
        ax[0].grid(visible=grid, which='major', axis='both')

        ax_amp = ax[0].twinx()
        a = self.demod_data['amp'].values
        ax_amp.plot(r, a, color=amp_color, marker='.', lw=1, label='AFM amplitude', alpha=0.5)
        ax_amp.tick_params(right=False, labelright=False)
        ax_amp.legend(loc='upper right')

        ax_phase = ax[0].twinx()
        p = self.demod_data['phase'].values
        ax_phase.plot(r, p, color=phase_color, marker='.', lw=1)
        ax_phase.set_ylabel('AFM phase (degrees)', color=phase_color)

        for o in orders:
            sig = np.abs(self.demod_data['optical'].values[:, o])
            sig /= np.abs(sig).max()
            ax[1].plot(r, sig, marker='.', lw=1, label=r'$s_{{{o}}}$'.format(o=str(o)), color=harm_cmap[o])
        ax[1].grid(visible=grid, which='major', axis='both')
        ax[1].set_ylabel('optical amplitude (scaled)')
        ax[1].legend(loc='upper right')

        if self.modulation == Demodulation.pshet:
            for o in orders:
                sig = np.angle(self.demod_data['optical'].values[:, o])
                ax[2].plot(r, sig, marker='.', lw=1, label=r'$\varphi_{{{o}}}$'.format(o=str(o)), color=harm_cmap[o])
            ax[2].grid(visible=grid, which='major', axis='both')
            ax[2].set_ylabel('optical phase')
            ax[2].set_xlabel('dz (nm)')
            ax[2].legend(loc='upper right')
        ax[-1].set_xlabel('r (nm)')

        if save:
            plt.savefig(f'{self.name}.png', dpi=100, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def to_csv(self, filename: str = None):
        """ Simply exports self.demod_data to a csv file.
        """
        if self.demod_data is None:
            self.demod()
        if filename is None:
            filename = f'{self.name}.csv'
        col_names = ['r', 'z', 'amp', 'phase']
        out = np.vstack([self.demod_data[c].values for c in col_names])
        optical = []
        for o in self.demod_data.order.values:
            col_names.append(f's_{o}')
            optical.append(np.abs(self.demod_data.optical.sel(order=o).values))
            if self.modulation == Demodulation.pshet:
                col_names.append(f'phi_{o}')
                optical.append(np.angle(self.demod_data.optical.sel(order=o).values))
        optical = np.array(optical)
        out = np.vstack([out, optical]).T
        col_names = ','.join(col_names)
        np.savetxt(filename, out, delimiter=',', header=col_names)


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

        daq_data = np.array(self.daq_data[0])
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


class Delay(BaseScanDemod):
    def reshape(self, demod_npts: int = None, t0_mm: float = None):
        """ Defines and brings AFM data to a specific shape (temporal resolution), on which pixels the optical data
        is then demodulated. The resulting xr.Dataset is located at self.demod_data.

        PARAMETERS
        ----------
        demod_npts: int
            minimum number of samples to demodulate for every pixel. Only whole chunks saved in daq_data are combined
        t0_mm: float
            position for t_0 on the delay stage. Only used to calculate time delay t in self.demod_data.
        """
        self.demod_data = xr.Dataset()
        self.demod_data.attrs = self.metadata

        chunk_size = 1  # one chunk of DAQ data per demodulated pixel
        if isinstance(demod_npts, int):
            chunk_size = demod_npts // self.metadata['npts'] + 1
            chunk_size -= int(demod_npts % self.metadata['npts'] == 0)  # subtract extra chunk for exact matches
        demod_npts = self.metadata['npts'] * chunk_size
        n = len(self.daq_data.keys())
        t_res = n // chunk_size
        self.demod_data.attrs['demod_params'] = {'demod_npts': demod_npts, 't_res': t_res}

        t_pos = self.afm_data['t_pos'].values.squeeze()
        t_pos = t_pos[:t_res * chunk_size].reshape((t_res, chunk_size)).mean(axis=1)

        t_target = self.afm_data['t_target'].values.squeeze()
        t_target = t_target[:t_res * chunk_size].reshape((t_res, chunk_size)).mean(axis=1)
        self.demod_data['t_target'] = xr.DataArray(data=t_target, dims='t_pos', coords={'t_pos': t_pos})
        if 't0_mm' in self.metadata.keys() or t0_mm is not None:
            if t0_mm is not None:
                t = (t0_mm - t_pos) * 2e-3 / c_air  # s
            else:
                t = (self.metadata['t0_mm'] - t_pos) * 2e-3 / c_air  # s
            self.demod_data['t'] = xr.DataArray(data=t, dims='t_pos', coords={'t_pos': t_pos})

        x = self.afm_data['x'].values.squeeze()
        x = x[:t_res * chunk_size].reshape((t_res, chunk_size)).mean(axis=1)
        self.demod_data['x'] = xr.DataArray(data=x, dims='t_pos', coords={'t_pos': t_pos})
        y = self.afm_data['y'].values.squeeze()
        y = y[:t_res * chunk_size].reshape((t_res, chunk_size)).mean(axis=1)
        self.demod_data['y'] = xr.DataArray(data=y, dims='t_pos', coords={'t_pos': t_pos})
        z = self.afm_data['z'].values.squeeze()
        z = z[:t_res * chunk_size].reshape((t_res, chunk_size)).mean(axis=1)
        self.demod_data['z'] = xr.DataArray(data=z, dims='t_pos', coords={'t_pos': t_pos})
        amp = self.afm_data['amp'].values.squeeze()
        amp = amp[:t_res*chunk_size].reshape((t_res, chunk_size)).mean(axis=1)
        self.demod_data['amp'] = xr.DataArray(data=amp, dims='t_pos', coords={'t_pos': t_pos})
        phase = self.afm_data['phase'].values.squeeze()
        phase = phase[:t_res*chunk_size].reshape((t_res, chunk_size)).mean(axis=1)
        phase = phase / 2 / np.pi * 360
        self.demod_data['phase'] = xr.DataArray(data=phase, dims='t_pos', coords={'t_pos': t_pos})

        idx = self.afm_data['idx'].values.squeeze()
        idx = idx[:t_res * chunk_size].reshape((t_res, chunk_size))
        idx = [i for i in idx] + ['']       # this is a truly ugly workaround to make an array of arrays of possibly
        idx = np.array(idx, dtype=object)   # different lengths to save as xr.DataArray
        self.demod_data['idx'] = xr.DataArray(data=idx[:-1], dims='t_pos', coords={'t_pos': t_pos})
        
    def demod(self, save: bool = False, **kwargs):
        """ Demodulates optical data for every pixel along dimension t_pos. A complex-valued DataArray for demodulated
        harmonics with extra dimension 'order' is created in self.demod_data, and to self.demod_cache.

        PARAMETERS
        ----------
        save: bool
            if True, demod data will be stored in, and read from a file if available
        **kwargs
            keyword arguments that are passed to demod function, i.e. shd() or pshet()
        """
        if self.demod_data is None:
            self.reshape()
        if save and self.demod_file is None:
            self.open_demod_file()
        demod_func = [shd, pshet][[Demodulation.shd, Demodulation.pshet].index(self.modulation)]
        demod_params = self.demod_params(params=self.demod_data.attrs['demod_params'], kwargs=kwargs)
        demod_key = ''.join(sorted([k + str(v) for k, v in demod_params.items()]))
        if demod_key in self.demod_cache.keys():
            self.demod_data = self.demod_cache[demod_key]
        else:
            harmonics_pumped = []
            harmonics_chopped = []
            for idx in tqdm(self.demod_data.idx):
                data = np.vstack([self.daq_data[i] for i in idx.item()])
                chop_idx, pump_idx = chop_pump_idx(data[:, self.signals.index(Signals.chop)])
                harmonics_pumped.append(demod_func(data=data[pump_idx], signals=self.signals, **kwargs))
                harmonics_chopped.append(demod_func(data=data[chop_idx], signals=self.signals, **kwargs))
            harmonics_pumped = np.array(harmonics_pumped)
            harmonics_chopped = np.array(harmonics_chopped)
            pp_harmonics = harmonics_pumped / harmonics_chopped - 1

            self.demod_data['optical_pumped'] = xr.DataArray(data=harmonics_pumped, dims=('t_pos', 'order'),
                                                             coords={'t_pos': self.demod_data['t_pos'],
                                                                     'order': np.arange(harmonics_pumped.shape[1])})
            self.demod_data['optical_chopped'] = xr.DataArray(data=harmonics_chopped, dims=('t_pos', 'order'),
                                                              coords={'t_pos': self.demod_data['t_pos'],
                                                                      'order': np.arange(harmonics_chopped.shape[1])})
            self.demod_data['optical_pump-probe'] = xr.DataArray(data=pp_harmonics, dims=('t_pos', 'order'),
                                                                 coords={'t_pos': self.demod_data['t_pos'],
                                                                         'order': np.arange(pp_harmonics.shape[1])})
            self.demod_data.attrs['demod_params'] = demod_params
            self.demod_cache[demod_key] = self.demod_data
            if save:
                self.cache_to_file()

    def plot(self, max_order: int = 4, orders=None, grid=False, show=True, save=False):
        """ Plots delay scan

        PARAMETERS
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
        # ToDo: redo this at some point
        import matplotlib.pyplot as plt
        if self.demod_data is None:
            self.demod()
        if not (hasattr(orders, '__iter__') and all([type(h) is int for h in orders])):
            orders = np.arange(max_order + 1)
        harm_cmap = cc.glasbey_category10

        fig, ax = plt.subplots(figsize=(6, 12))
        fontsize = 14

        if 't' in self.demod_data.variables:
            t = self.demod_data['t'].values
        else:
            t = self.demod_data['t_pos'].values
        t *= 1e12
        for o in orders:
            sig = self.demod_data['optical_pump-probe'][:, o]
            sig /= np.abs(sig).max()
            sig += o
            ax.plot(t, sig, marker='', lw=1, ls='-', color=harm_cmap[o])
            ax.axhline(o, color='grey', lw=.5)

        ax.set_yticks(np.arange(6))

        ax.set_ylabel(r'harmonics of pump-probe signal $s_\mathrm{pp}$ (scaled and shifted)', fontsize=fontsize)
        ax.set_xlabel('delay (ps)', fontsize=fontsize)

        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)

        if save:
            plt.savefig(f'{self.name}.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def to_csv(self, filename: str = None):
        # ToDo: this
        pass
