# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
Universidad de Chile - 2021
mail: santiago.parraguez@ug.uchile.cl

=============
 KINTUN WENU
=============
  - GES DISC

Provides classes to handle different type of datasets retrieved from https://disc.gsfc.nasa.gov
"""
__all__ = ['GesDiscProduct', 'GesDiscDataset']
# ============= IMPORTS ===============
import os
import numpy as np
import pandas as pd

from netCDF4 import Dataset
from .processing import _generate_regular_grid_points, regular_grid_data
from .scrap import download_files
from ._utils import _smoothn_fill, _standardise_unit_string
# =================================================================================
class GesDiscProduct:
    __module__ = 'kintunwenu'
    """
    Class for manage .nc products from GesDisc, extracting its metadata.

    Supports:
        - S5P_L2__NO2
    """
    # =======================================
    def __init__(self, product):
        """
        Parameters
        ----------
        product : Dataset
        """
        self.summary = product.summary
        self.algorithm_version = product.algorithm_version
        self.processor_version = product.processor_version
        self.product_version = product.product_version
        self.spatial_resolution = product.spatial_resolution
        self.time_start = product.time_coverage_start
        self.time_end = product.time_coverage_end

        self.longitude = None
        self.latitude = None
        self.time_utc = None
        self.qa_value = None
        self.data = None
        self.units = None
        self.product = None

        variables = product.groups['PRODUCT'].variables
        attrs = {}
        for k, v in variables.items():
            if k.lower().endswith('_column'):
                attrs.update({'data': v[0], 'units': _standardise_unit_string(v.units), 'product': k})
            elif k in ['latitude', 'longitude', 'time_utc', 'qa_value']:
                attrs.update({k: v[0]})
        for key, value in attrs.items():
            setattr(self, key, value[:])

    # =======================================
    @property
    def max(self):
        return self.data.max().astype(np.float32)

    @property
    def min(self):
        return self.data.min().astype(np.float32)

    @property
    def shape(self):
        return self.data.shape

    # =======================================
    def regular_grid(self, lon_lim, lat_lim, grid_space, method='nearest'):
        """
        Returns the coordinates and data of the product in a regular grid.
        It allows the grid to have a different step of longitude and latitude.

        Parameters
        ----------
        lon_lim : tuple, list
            Iterable with longitude limits of the grid with shape (2,)
        lat_lim : tuple, list
            Iterable with latitude limits of the grid with shape (2,)
        grid_space : float, tuple, list
            A single value or an iterable with grid steps along each axis.
            If a single value is given, it assumes same step for longitude and latitude.
        method : str
            Interpolation method, can be 'nearest', 'linear' or 'cubic'.
        """
        return regular_grid_data(self.longitude, self.latitude, self.data, lon_lim, lat_lim, grid_space, method)

    # =======================================
    def qa_filter(self, qa_value=0.75, copy=True):
        """
        Mask data in product where qa_value is lower than threshold.

        Parameters
        ----------
        qa_value : float
            Minimum accepted value.
        copy : bool
            If True (default) make a copy of data in the result. If False modify data in place and return a view.
        """
        return np.ma.masked_where(self.qa_value < qa_value, self.data, copy=copy)

# =================================================================================
class GesDiscDataset:
    __module__ = 'kintunwenu'
    """
    Class for manage a dataset of .nc products from GesDisc based in the ktw.GesDiscProduct class.
    """
    # =======================================
    def __init__(self, coordinates, resolution,
                 interpolation='nearest', units='umol/m2', qa_filter=None,
                 **kwargs):
        """
        Parameters
        ----------
        coordinates : tuple of float, list of float
            Limits of the dataset in coordinates, with shape (4,).
            Must be ordered by longitude first and latitude last.
        resolution : float, tuple of float, list of float
            A single value or an iterable with grid steps along each axis.
            If a single value is given, it assumes same step for longitude and latitude.
        interpolation : str
            Interpolation method, can be 'nearest', 'linear' or 'cubic'.
        units : str
            Indicates the units to use in the dataset.
            If products are given in other units, they will be converted.
        qa_filter : float
            Value of qa_value used to filter invalid data. If None (default), no values are filtered.
        """
        if len(coordinates) != 4:
            raise AssertionError(f'An iterable with 4 values must be given for '
                                 f'coordinates limits ({len(coordinates)} given).')
        if (coordinates[1] <= coordinates[0]) | (coordinates[3] <= coordinates[2]):
            raise AssertionError(f'Coordinates must be ordered, with longitudes first and latitudes last.')
        self.lon_lim = tuple(coordinates[:2])
        self.lat_lim = tuple(coordinates[2:])

        self.grid_resolution = (resolution, resolution) if isinstance(resolution, (float, int)) else tuple(resolution)
        if (self.grid_resolution[0] <= 0) | (self.grid_resolution[1] <= 0):
            raise AssertionError('Given grid resolution must be greater than 0.')

        if interpolation in ['nearest', 'linear', 'cubic']:
            self.interpolation = interpolation
        else:
            raise AssertionError('Interpolation method must be "nearest", "linear" or "cubic".')

        supported_units = ['umol/m2']
        if units not in supported_units:
            raise NotImplementedError(f'Units not implemented. Currently implemented: {supported_units}')
        self.units = units

        self.qa_filter = qa_filter

        self.longitude, self.latitude = _generate_regular_grid_points(self.lon_lim, self.lat_lim, self.grid_resolution)
        self.data = None

        self._fill_data = kwargs.pop('fill_data', True)
        self._fill_tolerance = kwargs.pop('fill_tolerance', 5)
        self._smoothing_factor = kwargs.pop('s', None)
        self._robust = kwargs.pop('robust', False)
        self._datetime_index = None

    # =======================================
    @property
    def max(self):
        return self.data.max().astype(np.float32)

    @property
    def min(self):
        return self.data.min().astype(np.float32)

    @property
    def shape(self):
        return self.data.shape

    @property
    def time_stamp(self):
        return pd.Series(self._datetime_index.index, index=self._datetime_index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    # =======================================
    def mean(self, axis=0, *kwargs):
        """
        Compute the arithmetic mean along the specified axis. Based on np.mean().

        Parameters
        ----------
        axis : None, int, tuple of ints
        """
        return self.data.mean(axis, *kwargs)

    # =======================================
    def var(self, axis=None, *kwargs):
        """
        Compute the variance along the specified axis. Based on np.var().

        Parameters
        ----------
        axis : None, int, tuple of ints
        """
        return self.data.var(axis, *kwargs)

    # =======================================
    def clear_data(self):
        """
        Erase memory of the object
        """
        self.data = None
        self._datetime_index = None
        return None

    # =======================================
    def load_products(self, product):
        """
        Update the dataset by adding new products.
        Can manage a single GesDiscProduct product or an iterable containing several of them.

        Parameters
        ----------
        product : GesDiscProduct or list of GesDiscProduct
            A single product or a list of them. The products will be added to the dataset.
        """
        if isinstance(product, GesDiscProduct):
            product_list = [product]
        elif hasattr(product, '__iter__') and not isinstance(product, str):
            if not all(isinstance(p, GesDiscProduct) for p in product):
                raise AssertionError('All objects to be appended must be ktw.GesDiscProduct.')
            product_list = list(product)
        else:
            raise AssertionError('Must give a single ktw.GesDiscProduct or a list with them.')

        new_data, new_time_stamp = [], []
        for product in product_list:
            z = product.qa_filter(self.qa_filter, copy=True) if self.qa_filter is not None else product.data
            _, _, z = regular_grid_data(product.longitude, product.latitude, z,
                                        self.lon_lim, self.lat_lim, self.grid_resolution, self.interpolation)
            if z.count() == 0:
                continue
            z = self._transform_units(z, product.units)
            if self._fill_data is True:
                z = _smoothn_fill(z, tolerance=self._fill_tolerance,
                                  s=self._smoothing_factor, robust=self._robust, grid=self.grid_resolution)

            new_data.append(z)
            new_time_stamp.append(product.time_start)

        if new_data:
            new_data = np.ma.stack(new_data, axis=0)
            new_time_stamp = pd.to_datetime(new_time_stamp)

            if self.data is None:
                self.data = new_data
                self._datetime_index = pd.Series(np.arange(len(new_time_stamp)), index=new_time_stamp)
            else:
                self.data = np.ma.concatenate([self.data, new_data], axis=0)
                current_len = len(self._datetime_index)
                new_datetime_index = pd.Series(np.arange(current_len, current_len+len(new_time_stamp)),
                                               index=new_time_stamp)
                self._datetime_index = pd.concat([self._datetime_index, new_datetime_index], axis=0, ignore_index=False)

        return None

    # =======================================
    def download_products(self, urls, batch_size=None, verbose=1, attempts=1) -> None:
        """
        Download and load into the dataset products from a list of urls. It doesn't save the files into disk.
        Using the batch_size can save memory during the process.

        Parameters
        ----------
        urls : list, tuple of str
            Iterable with links to get the files.
        batch_size : int
            Number of files downloaded and processed at the same time.
        verbose : int
            Show information during running.
        attempts: int
            Number of times each file will be attempted to download.
        """
        n_files = len(urls)
        if batch_size is not None:
            n_batchs = (n_files + batch_size - 1) // batch_size
        else:
            n_batchs = 1
            batch_size = n_files

        if verbose > 0:
            print(f'Start download of {n_files} netCDF files in {n_batchs} batches')

        for batch in range(n_batchs):
            if verbose > 0:
                print(f'Downloading batch {batch + 1}/{n_batchs} of files...', end='\r')

            batch_urls, urls = urls[:batch_size], urls[batch_size:]
            products = download_files(urls=batch_urls, verbose=0, save_files=False, attempts=attempts)

            if verbose > 0:
                print(f'Processing and loading batch {batch + 1}/{n_batchs} into dataset...', end='\r')

            self.load_products([GesDiscProduct(prod) for prod in products])
            for prod in products:
                prod.close()

        if verbose > 0:
            print(f'Completed download of {n_files} files')

        return None

    # =======================================
    def read_products(self, path, batch_size=None, verbose=1) -> None:
        """
        Load netCDF files from a given directory.
        Using the batch_size can save memory during the process.

        Parameters
        ----------
        path : str
            Directory from which files are loaded.
        batch_size : int
            Number of files loaded and processed at the same time.
        verbose : int
            Show information during running.
        """
        list_dir = [f for f in os.listdir(path) if f.split('.')[-1] in ['nc', 'nc4', 'h5', 'he5']]
        n_files = len(list_dir)
        if path is None:
            path = ''

        if batch_size is not None:
            n_batchs = (n_files + batch_size - 1) // batch_size
        else:
            n_batchs = 1
            batch_size = n_files

        n_succ, n_fail = 0, 0

        for batch in range(n_batchs):
            if verbose > 0:
                print(f'Loading batch {batch + 1}/{n_batchs} of files...', end='\r')

            batch_files, list_dir = list_dir[:batch_size], list_dir[batch_size:]
            products = []
            for f in batch_files:
                try:
                    products.append(Dataset(path + '/' + f))
                    n_succ += 1
                except ValueError:
                    n_fail += 1

            self.load_products([GesDiscProduct(prod) for prod in products])

        print(f'Read {n_succ}/{n_files} successfully. Failed to read {n_fail} files.')
        return None

    # =======================================
    def create_dataset(self, shape, step=None):
        """
        Extract a dataset of images with specified shape from the products processed.

        Parameters
        ----------
        shape : tuple of int, list of int
            Shape of the output images.
        step : tuple of int, list of int, int
            Step along each axis between one retrieved image and the next one.
        """
        if self.data is None:
            print('Warning: no data loaded to create dataset.')
            return None

        if step is None:
            step = (shape[0]//2, shape[1]//2)
        elif isinstance(step, (float, int)):
            step = (step, step)

        strides = np.lib.stride_tricks.sliding_window_view(self.data.filled(np.nan), shape, axis=(1, 2))
        m, n = step if not isinstance(step, int) else (step, step)
        strides = strides[:, ::n, ::m, :, :].reshape(-1, *shape)
        strides = strides[~np.isnan(strides).any(axis=(1, 2)), :, :]

        return strides

    # =======================================
    def _transform_units(self, data, dunits):
        """
        Transform data from its original units to the dataset units.

        Parameters
        ----------
        data : np.ma.MaskedArray
            Data to be transformed.
        dunits : str
            Original units of the data.
        """
        num, den = self.units.split('/')
        dnum, dden = dunits.split('/')

        if num == 'mol' and dnum == 'molec':  # molec -> mol
            data /= 6.02214e23
            dnum = 'mol'
        if num == 'umol' and dnum == 'mol':  # mol -> umol
            data *= 1e6
            dnum = 'umol'
        if den == 'm2' and dden == 'cm2':  # /cm2 -> /m2
            data *= 1e4
            dden = 'm2'

        if num != dnum or den != dden:
            raise NotImplementedError(f'Unable to convert units. {dunits} -> {self.units}')

        return data
