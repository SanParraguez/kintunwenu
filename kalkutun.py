# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> KALKUTUN

Provides classes to handle different type of satellite datasets.
"""

__all__ = [
    'Kalkutun',
    'GridCrafter'
]

# ============= IMPORTS ===============================
import logging
import shapely
import numpy as np

from contextlib import nullcontext
from netCDF4 import Dataset
from pathlib import Path
from .geodata import create_geo_dataset, are_over_pole
from .grid import create_grid, weighted_regrid
from .polygons import get_corners_from_grid, split_anomaly_polygons
from .scrap import download_ncfile
from .units import standardise_unit_string, convert_units

# =================================================================================

class Kalkutun:
    """
    A class to handle satellite datasets for the TROPOMI sensor.

    This class supports the following products:
    > TROPOMI
        - S5P_OFFL_L2__NO2
        - S5P_RPRO_L2__NO2
        - S5P TROPOMI/WFMD

    The class provides an interface to easily access and manipulate the product data. It also includes methods to
    convert units, retrieve polygon data, and create a dataframe of the polygon data.

    Kalkutun: from Mapuzungún, means "Do witchcraft"

    Attributes
    ----------
    longitude : numpy.ndarray
        The longitudes of the product.
    latitude : numpy.ndarray
        The latitudes of the product.
    tracer : str
        The tracer gas of the product (e.g., "NO2").
    product : str
        The name of the product (e.g., "nitrogendioxide_tropospheric_column").
    units : str
        The unit of the product data.
    data : numpy.ndarray
        The product data.

    Methods
    -------
    __init__(self, dataset)
        Initializes the Kalkutun object.

    copy(self, )

    convert_units(self, to_unit)
        Converts the units of the product data.

    qa_filter(self, )

    get_polygons(self, return_data=True)
        Returns a list of Polygon objects created from the product's longitude and latitude.

    get_polygon_dataframe(self)
        Returns a Pandas dataframe containing the polygon data.

    Raises
    ------
    NotImplementedError
        If the sensor is not supported.
    """
    __module__ = 'kintunwenu'

    def __init__(self, dataset, tracer=None, kw_vars=None):
        """
        Initializes the Kalkutun object.

        Parameters
        ----------
        dataset : netCDF4.Dataset or string
            The netCDF4 dataset containing the product.
        tracer : string (optional)
            Indicates which tracer to retrieve in case the file contains more than one.
        kw_vars : dict or list or tuple (optional)
            If provided, it will iterate and get the specified variables from the dataset.

        Raises
        ------
        NotImplementedError
            If the sensor is not supported.
        """
        # New products should be added here as an 'if' condition.
        # Ensure that every variable has dimensions (time, scanline, pixel, ...) so the gridding routines
        #   can properly read and process the data.

        to_context = True
        if isinstance(dataset, (str, Path)):
            try:
                dataset = Dataset(dataset)
            except OSError:
                dataset = download_ncfile(dataset)
        elif isinstance(dataset, Dataset):
            # Do not close the dataset after reading, assuming that it could be used inside
            # another 'with' statement outside this class.
            to_context = False
        else:
            raise NotImplementedError('Kalkutun must be created from a path to file, an url or an actual Dataset')

        # Initialize class attributes
        self.data = np.ma.empty(0)
        self.units = None
        self.product = None
        self.tracer = None
        self.longitude = np.empty(0)
        self.latitude = np.empty(0)
        self.longitude_corners = None
        self.latitude_corners = None
        self.qa_value = None
        self.time_utc = None
        self.format = None      # polygons, centers or corners
        self.variables = {}

        # Initialize context to read values from dataset
        with dataset if to_context else nullcontext(dataset) as ds:
            # -----------------------------------------------
            #  General
            # -----------------------------------------------
            # Attempt to retrieve id information from product
            try:
                self.id = ds.id
            except AttributeError:
                raise NotImplementedError('Unable to retrieve id information from product.')

            # Attempt to retrieve sensor information from product
            try:
                self.sensor = ds.sensor.lower()
            except AttributeError:
                raise NotImplementedError('Unable to retrieve sensor information from product.')

            # -----------------------------------------------
            #  TROPOMI NO2 L2
            # -----------------------------------------------
            if 'TROPOMI/S5P NO2 1-Orbit L2 Swath' in ds.title:

                id_tracer = ds.id.split('__')[1].upper()
                if tracer and id_tracer != tracer.upper():
                    logging.info(f'Warning: retrieved {id_tracer} but {tracer.upper()} was given.')

                # Retrieve attributes from variables in product
                variables = ds.groups['PRODUCT'].variables
                product_name = [key for key in variables.keys() if key.endswith('_column')]
                product_name = product_name[0] if len(product_name) == 1 \
                    else AttributeError('Found more tan one column in data')

                # Collect attributes
                attrs = {
                    'data': variables[product_name][:],
                    'units': standardise_unit_string(variables[product_name].units),
                    'product': product_name,
                    'tracer': id_tracer,
                    'longitude': variables['longitude'][:],
                    'latitude': variables['latitude'][:],
                    'qa_value': variables['qa_value'][:],
                    'time_utc': np.array(variables['time_utc'][:], dtype='datetime64[ns]'),
                    'format': 'centers',
                }

                self.variables.update({
                    'avg_kernel': variables['averaging_kernel'][:],
                    'std_deviation': variables[product_name+'_precision'][:],
                })

            # -----------------------------------------------
            #  TROPOMI WFMD IUP CH4/C0 v1.8
            # -----------------------------------------------
            elif ds.title == 'TROPOMI/WFMD XCH4 and XCO':

                # Check if tracer is provided
                if tracer is None:
                    raise AssertionError('Given dataset contains both CH4 and CO data, a tracer has to be provided.')
                else:
                    tracer = tracer.upper()

                # Get product based on tracer
                if tracer == 'CH4':
                    tracer_name = 'xch4'
                    product_name = ds.variables[tracer_name].standard_name
                elif tracer == 'CO':
                    tracer_name = 'xco'
                    product_name = 'dry_atmosphere_mole_fraction_of_carbon_monoxide'
                else:
                    raise AttributeError(f'Tracer provided {tracer} not found in dataset')

                # Collect attributes
                attrs = {
                    'data': ds.variables[tracer_name][:],
                    'units': standardise_unit_string(ds.variables[tracer_name].units),
                    'product': product_name,
                    'tracer': tracer,
                    'longitude': ds.variables['longitude'][:],
                    'latitude': ds.variables['latitude'][:],
                    'longitude_corners': ds.variables['longitude_corners'][:],
                    'latitude_corners': ds.variables['latitude_corners'][:],
                    'qa_value': ds.variables[f'{tracer_name}_quality_flag'][:],
                    'time_utc': np.array(ds.variables['time'][:], dtype='datetime64[s]'),
                    'format': 'polygons',
                }

                self.variables.update({
                    'avg_kernel': ds.variables[f'{tracer_name}_averaging_kernel'][:],
                })

            # Product not recognized
            else:
                raise NotImplementedError(f"Product '{ds.title}' has not been implemented.")
            # -----------------------------------------------

        # Set attributes
        for key, value in attrs.items():
            setattr(self, key, value)

    # -----------------------------------------------------------------------------
    def max(self, *args, **kwargs):
        """This property returns the maximum value of the data array."""
        return self.data.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        """This property returns the minimum value of the data array."""
        return self.data.min(*args, **kwargs)

    def mean(self, *args, **kwargs):
        """This property returns the mean value of the data array."""
        return self.data.mean(*args, **kwargs)

    def count(self, *args, **kwargs):
        """This property returns the count value of the data array."""
        return self.data.count(*args, **kwargs)

    @property
    def size(self):
        """This property returns the size of the data array."""
        return self.data.size

    @property
    def shape(self):
        """This property returns the shape of the data array."""
        return self.data.shape

    # -----------------------------------------------------------------------------
    def __eq__(self, other):
        """
        Compare if self is equal to another object by checking all its variables.
        """
        if isinstance(other, Kalkutun):
            for my_key, other_key in zip(self.__dict__, other.__dict__):
                if my_key == other_key:
                    if isinstance(getattr(self, my_key), np.ndarray):
                        if not np.array_equal(getattr(self, my_key), getattr(other, other_key)):
                            return False
                    else:
                        if getattr(self, my_key) != getattr(other, other_key):
                            return False
                else:
                    return False
            return True
        else:
            return False

    # -----------------------------------------------------------------------------
    def copy(self):
        """
        Create a copy of the current object.

        Returns
        -------
        Kalkutun
            A new instance of the current object.
        """
        new_object = self.__class__.__new__(self.__class__)
        new_dict = {k: (v.copy() if callable(getattr(v, 'copy', None)) else v) for k, v in self.__dict__.items()}
        new_object.__dict__.update(new_dict)
        return new_object

    # -----------------------------------------------------------------------------
    def convert_units(self, to_unit, var_list=None):
        """
        Change units of the product data.

        Parameters
        ----------
        to_unit : str
            The desired new unit of the data.
        var_list : list
            Which variables to include from the product

        Returns
        -------
        None
        """
        to_unit = standardise_unit_string(to_unit)
        self.data = convert_units(self.data, from_unit=self.units, to_unit=to_unit, species=self.tracer)

        if not var_list: var_list = []
        for var in var_list:
            self.variables[var] = convert_units(self.variables[var], from_unit=self.units,
                                                to_unit=to_unit, species=self.tracer)

        self.units = to_unit

        return

    # -----------------------------------------------------------------------------
    def qa_filter(self, min_value, inplace=False):
        """
        Filter the data array based on a quality assurance (QA) value threshold.

        Parameters
        ----------
        min_value : float
            The minimum QA value to retain data.
        inplace : bool, optional
            If True, the function applies the masking operation on the data array in-place.
            If False, the function returns a new masked array (default).

        Returns
        -------
            If inplace is False, A masked array with values below the minimum QA value masked.
            None otherwise.
        """
        if (self.qa_value == 0.0).all():
            logging.info('Warning: Product seems to not have any valid quality value')
            return self.data if not inplace else None
        if inplace:
            self.data = np.ma.masked_where(self.qa_value < min_value, self.data)
        else:
            return np.ma.masked_where(self.qa_value < min_value, self.data)

    # -----------------------------------------------------------------------------
    def coordinates_filter(self, *args, inplace=False, **kwargs):
        """

        Parameters
        ----------
        args : iterable
            Coordinate borders to use as filter (min lon, max lon, min lat, max lat).
        inplace : bool
        kwargs

        Returns
        -------

        """
        if len(args) in [1, 4]:
            lon_min, lon_max, lat_min, lat_max = args
        elif len(args) == 2:
            (lon_min, lon_max), (lat_min, lat_max) = args
        elif kwargs:
            lon_min = kwargs.pop('lon_min', None)
            lon_max = kwargs.pop('lon_max', None)
            lat_min = kwargs.pop('lat_min', None)
            lat_max = kwargs.pop('lat_max', None)
        else:
            raise ValueError('Coordinates limits should be provided by one or all together.')

        lon_min = lon_min if lon_min else -180
        lon_max = lon_max if lon_max else 180
        lat_min = lat_min if lat_min else -90
        lat_max = lat_max if lat_max else 90

        if lon_min >= lon_max or lon_min > 180 or lon_max < -180:
            raise AssertionError(f'Longitude {lon_min} has to be smaller than {lon_max}')
        if lat_min >= lat_max or lat_min > 90 or lat_max < -90:
            raise AssertionError(f'Latitude {lat_min} has to be smaller than {lat_max}')

        new_data = self.data.copy()
        if self.longitude.shape == self.data.shape:
            if lon_min:
                new_data = np.ma.masked_where(self.longitude < lon_min, new_data)
            if lon_max:
                new_data = np.ma.masked_where(self.longitude > lon_max, new_data)
            if lat_min:
                new_data = np.ma.masked_where(self.latitude < lat_min, new_data)
            if lat_max:
                new_data = np.ma.masked_where(self.latitude > lat_max, new_data)
        else:
            raise NotImplementedError('Filter not implemented for corner points, try filtering polygons.')

        if inplace:
            self.data = new_data
        else:
            return new_data

    # -----------------------------------------------------------------------------
    def get_polygons(self):
        """
        Returns a list of Polygon objects created from the product's longitude and latitude.

        Returns
        -------
        np.ndarray
            An array of Polygon objects created from the product's longitude and latitude.
        """
        if self.longitude_corners is not None and self.latitude_corners is not None:
            coords = np.moveaxis(np.asarray([self.longitude_corners, self.latitude_corners]), 0, -1)
        else:
            coords = [get_corners_from_grid(lon, lat, mode=self.format) for lon, lat
                      in zip(self.longitude, self.latitude)]
        return [shapely.polygons(crds) for crds in coords]

    # -----------------------------------------------------------------------------
    def get_polygon_dataframe(self, var_list=None, reset_index=True,
                              split_antimeridian=True,
                              drop_poles=False, drop_masked=True, drop_invalid=True,
                              workers=None, geod=None):
        """
        Generates a GeoDataFrame from the object coordinates and data.

        Parameters
        ----------
        var_list : list
            List of extra variables to be gridded
        reset_index : bool, optional
            Indicates if index should be reset before return or not.
        split_antimeridian : bool, optional
            If True (default), splits those polygons crossing the antimeridian.
        drop_poles : bool, optional
            If True, drops all geometries over the poles.
        drop_masked : bool, optional
            If True (default), drops any rows with masked values in the returned DataFrame.
        drop_invalid : bool, optional
            If True (default), drops all invalid geometries.
        workers : int, optional
            The number of worker processes to use for parallelization when checking geometries over the poles.
        geod : Geodesic, optional
            A `Geodesic` object to use for the calculations when checking geometries over the poles.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the polygons of each cell and its actual value.
            The DataFrame has two columns:
                - 'value': The data values associated with each polygon.
                - 'polygon': The Shapely Polygon objects representing geographical polygons.
        """
        if var_list is None:
            var_list = []
        elif isinstance(var_list, str):
            var_list = [var_list]

        # ToDo: Check if this is still neccesary, might be better to just remove it
        if self.time_utc.ndim == self.data.ndim - 1:
            time_utc = np.repeat(self.time_utc[..., None], self.data.shape[-1], axis=-1)
        else:
            time_utc = self.time_utc

        if self.format == 'centers':
            old_shape = self.data.shape
            new_shape = (old_shape[0], (old_shape[1] - 2) * (old_shape[2] - 2), *old_shape[3:])
            data = self.data[:, 1:-1, 1:-1].reshape(new_shape)
            time_utc = time_utc[:, 1:-1, 1:-1].reshape(new_shape)
            kw_vars = {}
            for k in var_list:
                if k in self.variables.keys():
                    old_shape = self.variables[k].shape
                    new_shape = (old_shape[0], (old_shape[1] - 2) * (old_shape[2] - 2), *old_shape[3:])
                    kw_vars[k] = self.variables[k][:, 1:-1, 1:-1].reshape(new_shape)

        elif self.format in ['corners', 'polygons']:
            old_shape = self.data.shape
            new_shape = (old_shape[0], old_shape[1] * old_shape[2], *old_shape[3:])
            data = self.data.reshape(new_shape)
            time_utc = time_utc.reshape(new_shape)
            kw_vars = {k: self.variables.get(k).reshape(new_shape) for k in var_list if k in self.variables.keys()}
        else:
            raise AssertionError('Format of the data not recognized')

        logging.debug("Retrieve polygons from dataset")
        polygons_array = self.get_polygons()
        logging.debug(f"  Created {len(polygons_array)} polygons")
        df = create_geo_dataset(polygons_array, data=data, timestamp=time_utc, **kw_vars)

        if drop_masked and np.ma.is_masked(data):
            df = [df[i].iloc[~di.mask] for i, di in enumerate(data)]
            logging.debug(f"  Dropped masked to {len(df)} geometries")
            if len(df) == 0:
                return df

        if drop_invalid or drop_poles or split_antimeridian:
            is_valid = [shapely.is_valid(dfi['geometry']) for dfi in df]
            logging.debug(f"  Dropped {np.sum([len(df[i])-is_valid[i].sum() for i in range(len(df))])} invalid geometries")
            df = [dfi[is_valid[i]] for i, dfi in enumerate(df)]

        if drop_poles:
            over_pole = [are_over_pole(dfi['geometry'], geod=geod, workers=workers) for dfi in df]
            logging.debug(f"  Dropped {np.sum((opi.sum() for opi in over_pole))} geometries over poles")
            df = [dfi[~over_pole[i]] for i, dfi in enumerate(df)]

        if split_antimeridian:
            df = [split_anomaly_polygons(dfi, to_dataframe=True) for dfi in df]
            logging.debug(f"  Split into {len(df)} polygons")

        if reset_index:
            df = [dfi.reset_index(drop=True) for dfi in df]

        return df if len(df) > 1 else df[0]

# =================================================================================

class GridCrafter:
    """
    Adapting data to its new grid like a master of disguise.
    Basically, just creates a regular grid, further capabilities are desired.

    Parameters
    ----------
    longitudes, latitudes : np.ndarray
        Grid corners to be used.
    interpolation : str
        Method of interpolation to be used.
    min_fill : float
        Fraction of the grid cell that has to be filled to be valid. If not provided, 0.0 by default.
    qa_filter : float
        Minimum value of the quality flag necessary to consider that measurement.
    units : str
        Desired output units of the main data of the product. It will try to convert both the data and its
        standard deviation.
    geod : pyproj.Geod
        Geodetic object to be used for calculating areas of cells. Assumes Earth if none is provided.
    """
    __module__ = 'kintunwenu'

    def __init__(
            self,
            grid_lons, grid_lats,
            interpolation='weighted',
            min_fill=None, qa_filter=None,
            units=None, geod=None,
            **kwargs
    ):
        self.lon_lim = np.min(grid_lons), np.max(grid_lons)
        self.lat_lim = np.min(grid_lats), np.max(grid_lats)

        if interpolation in ['weighted']:
            self.interpolation = interpolation
        else:
            raise AssertionError("Interpolation method must be 'weighted'.")

        self.min_fill = min_fill
        self.units = units
        self.geod = geod
        self.qa_filter = qa_filter
        self.lat_filter = kwargs.pop('lat_filter', None)

        self.lons, self.lats = np.asarray(grid_lons), np.asarray(grid_lats)

        if self.lons.ndim != self.lats.ndim:
            raise ValueError(f"Grid dimensions have to be the same for latitudes and longitudes")
        if self.lons.ndim > 2 or self.lats.ndim > 2:
            raise ValueError(f"Grid dimensions must be 1 or 2, not {self.lons.ndim}")
        if self.lons.ndim == 2:
            if self.lons.shape != self.lats.shape:
                raise ValueError(f"Grid longitudes and latitudes must have same shape. "
                                 f"({self.lons.shape}) and ({self.lats.shape}) found.")

    # -----------------------------------------------------------------------------
    @classmethod
    def from_grid(cls, grid_lons, grid_lats, **kwargs):
        """
        Creates an instance of GridCrafter but providing a grid.
        Useful for non-monotonous grids.

        Parameters
        ----------
        grid_lons, grid_lats : np.ndarray
            Grid corners to be used.
        """
        return cls(grid_lons, grid_lats, **kwargs)

    # -----------------------------------------------------------------------------
    @classmethod
    def from_size(cls, grid_size, lon_lim=(-180, 180), lat_lim=(-90, 90), method='corners', **kwargs):
        """
        Creates an instance of GridCrafter but providing coordinates and a grid resolution.
        Useful for monotonous grids.

        Parameters
        ----------
        grid_size : float or tuple[float, float]
            Size of the grid cells, if a float is given it will assume regular grid.
        lon_lim : tuple[float, float]
            Longitude limits of the grid, included.
        lat_lim : tuple[float, float]
            Latitude limits of the grid, included.
        method : str
            Indicates if the points are the corners or the centers of the grid. Default: 'corners'.
        """
        grid_lons, grid_lats = create_grid(grid_size, lon_lim, lat_lim, method)

        return cls(grid_lons, grid_lats, **kwargs)

    # -----------------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.regrid(*args, **kwargs)

    # -----------------------------------------------------------------------------
    def regrid(self, product, qa_filter=None, coord_filter=None, drop_negatives=False, **kwargs):
        """
        Perform a regridding over a product

        Parameters
        ----------
        product : Dataset or Kalkutun
            Product to be regridded.
        qa_filter : float, optional
            Quality assurance filter to be applied.
        coord_filter : tuple
            Constrain the domain by masking values outside the limits given.
        drop_negatives : bool, optional
            Indicates if negative values should be considered or not.

        Returns
        -------
        np.ndarray
            2D-array with the weighted values.
        """
        kprod = product.copy() if isinstance(product, Kalkutun) else Kalkutun(product)

        drop_poles = kwargs.pop('drop_poles', False)
        var_list = kwargs.pop('var_list', list(kprod.variables.keys()))

        if coord_filter is not None:
            kprod.coordinates_filter(coord_filter, inplace=True)
        else:
            kprod.coordinates_filter(self.lon_lim, self.lat_lim, inplace=True)

        if self.units is not None:
            kprod.convert_units(self.units)

        if qa_filter is not None:
            kprod.qa_filter(qa_filter, inplace=True)
        elif self.qa_filter is not None:
            kprod.qa_filter(self.qa_filter, inplace=True)

        df_obs = kprod.get_polygon_dataframe(
            var_list=var_list, drop_masked=True, drop_invalid=True,
            drop_poles=drop_poles, split_antimeridian=True, reset_index=True
        )

        if drop_negatives is True:
            df_obs = df_obs[df_obs['data'] > 0.0]

        if len(df_obs) == 0:
            logging.warning(f"    No polygons to regrid for {product}, returning None (might check masked data)")
            return None

        if self.interpolation == 'weighted':
            regrid = weighted_regrid(
                self.lons, self.lats, df_obs['geometry'], df_obs.drop('geometry', axis=1),
                min_fill=self.min_fill, geod=self.geod, **kwargs
            )
        else:
            raise NotImplementedError('Interpolation type not implemented')

        logging.debug(f"  Finished gridding of product ({next(iter(regrid.values())).count()} valid values)")
        return regrid

# =================================================================================
