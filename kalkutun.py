# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================

KALKUTUN
--------

Provides classes to handle different type of satellite datasets.
"""

__all__ = [
    'Kalkutun'
]

# ============= IMPORTS ===============================

import logging
import shapely
import numpy as np

from contextlib import nullcontext
from netCDF4 import Dataset

from .netcdf import get_netcdf_var
from .units import standardise_unit_string, convert_units

# =================================================================================

class Kalkutun:
    """
    A class for handling satellite netCDF datasets.

    This class provides an interface to access and manipulate product data from satellite netCDF datasets.
    It offers methods to convert units, retrieve polygon data, and create a DataFrame of the polygon data.

    "Kalkutun" is derived from Mapuzungún, meaning "Do witchcraft".

    grid_format : str
        String that indicates the format of data provided to create the grid.
        It could only be 'corners' for now.

    kw_vars : str
        Dictionary with sub-dictionaries for each variable to be stored. Each entrance has to have either
        a 'path' to the variable in the netCDF file or a 'formula' to calculate it.

        Also, the following options are supported:
            - units     : string indicating units to set the variable into.
            - getattr   : list with attribute names to be retrieved from variable.
            - setattr   : dictionary with extra attributes to be set.

        Example
        -------
        {
            'no2' : {
                'name'      :  'nitrogendioxide_tropospheric_column',
                'path'      :  '/PRODUCT/nitrogendioxide_tropospheric_column',
                'units'     :  'umol/m2',
                'getattr'   :  ['long_name', 'standard_name'],
                'setattr'   :  {'tracer' : 'NO2'}
            }
        }


    Attributes
    ----------
    dimensions : dict
        A dictionary containing information about the dimensions of the dataset.
    variables : dict
        A dictionary containing information about the variables in the dataset.
    grid_format : str
        The format of the grid coordinates, if available.
    grid_vars : dict
        A dictionary containing information about the grid variables, if available.


    Methods
    -------
    copy()
        Creates a deep copy of the current object.

    convert_units(varname, to_unit)
        Converts the units of the specified variable to the given unit.

    get_polygons()
        Retrieves an array of Polygon objects representing the grid cells of the dataset.

    filter

    minmax_filter

    qa_filter

    coordinates_filter(self, var, *args, inplace=False, **kwargs)
        Filters the variable array based on coordinate borders.

    """

    __module__ = 'kintunwenu'

    def __init__(self, dataset, grid_format=None, kw_grid=None, kw_attrs=None, kw_vars=None):
        """
        Initializes the Kalkutun object.

        Parameters
        ----------
        dataset : netCDF4.Dataset or str
            The netCDF4 dataset containing the product.
        grid_format : str, optional
            Format of the grid (default: None).
        kw_grid : dict, optional
            Dictionary containing grid information (default: None).
        kw_attrs : dict, optional
            Dictionary containing attribute information (default: None).
        kw_vars : dict, optional
            Dictionary containing variable information (default: None).
        """
        # Initialize parameters
        kw_grid = kw_grid or {}
        kw_attrs = kw_attrs or {}
        kw_vars = kw_vars or {}

        # *** Initialize dataset ***
        # Do not close the dataset after reading, assuming that it could be used inside
        # another 'with' statement outside this class.
        to_context = False
        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset)
            to_context = True

        # Initialize class attributes
        self._dimensions = {}
        self._variables = {}
        self._grid_format = grid_format
        self._grid_vars = {}
        self._polygons = None
        self._from_formula = None

        # Open file and retrieve information
        with dataset if to_context else nullcontext(dataset) as ds:
            self._process_variables(ds, kw_vars)
            self._process_grid(ds, grid_format, kw_grid)

        # ToDo: remove extra logging
        # logging
        for k, v in self._variables.items():
            vtypes = {ki: type(vi).__name__ for ki, vi in v.items()}
            logging.info(f"{k[:25]:25}: {vtypes}")
            for vi in v:
                if vi == 'values':
                    logging.info(f"  {vi:15} : [{v[vi].min()}, {v[vi].max()}]")
                else:
                    logging.info(f"  {vi:15} : {v[vi]}")

        logging.info(f"To formula: {self._from_formula}")

        logging.info(f"{self._grid_vars['corner_dim']}")
        logging.info(f"Latitude dims  : {self._grid_vars['latitude']['dims']}")
        logging.info(f"Longitude dims : {self._grid_vars['longitude']['dims']}")

        logging.info(self.polygons.flatten()[:3])

    # -----------------------------------------------------------------------------

    def _process_variables(self, dataset, kw_vars):
        """
        Process variables specified in kw_vars dictionary.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            The netCDF4 dataset containing the product.
        kw_vars : dict
            Dictionary containing variable information.
        """
        # store which variables will be calculated by formula
        to_formula = self._from_formula or []

        # iterate over variables
        for k, var in kw_vars.items():

            name = var.get('name', k)
            formula = var.get('formula', None)
            path = var.get('path', None)

            # raise if both formula and path are provided
            if path is not None and formula is not None:
                raise ValueError(f'Either path or formula for variable has to be provided, both found for {k}')
            # skip if formula is provided
            if formula is not None:
                to_formula.append(k)
                continue

            # get variable from file
            retr_var = get_netcdf_var(dataset, var['path'])

            # get dimensions
            var_dims = retr_var.dimensions

            # store values and dimensions
            self._variables[name] = {
                'values' : retr_var[:],
                'dims'   : var_dims[:],
                'shape'  : retr_var[:].shape,
                'attrs'  : {},
            }

            # get attributes
            to_get = var.get('getattr')
            for j in to_get:
                self._variables[var['name']]['attrs'][j] = getattr(retr_var, j)

            # set attributes
            to_set = var.get('setattr')
            for j in to_set:
                self._variables[name]['attrs'][j] = to_set[j]

            # get units if available
            self._variables[name]['units'] = getattr(retr_var, 'units', None)

            # convert units
            to_unit = var.get('units', None)
            if to_unit is not None:
                self.convert_units(name, to_unit)

        self._from_formula = to_formula

    # -----------------------------------------------------------------------------

    def _process_grid(self, dataset, grid_format, kw_grid):
        """
        Process grid information specified in kw_grid dictionary.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            The netCDF4 dataset containing the product.
        grid_format : str
            Format of the grid.
        kw_grid : dict
            Dictionary containing grid information.
        """
        # get grid definition
        if grid_format == 'corners':

            dim = kw_grid['dimension']
            lat_var = get_netcdf_var(dataset, kw_grid['latitude']['path'])
            lon_var = get_netcdf_var(dataset, kw_grid['longitude']['path'])

            if lat_var.dimensions != lon_var.dimensions:
                raise ValueError(f"Variables of longitudes and latitudes have different dimensions: "
                                 f"{lon_var.dimensions} != {lat_var.dimensions}")

            self._grid_vars = {
                'corner_dim': dim,
                'grid_dim'  : tuple(x for x in lat_var.dimensions if x != dim),
                'latitude'  : {'dims': lat_var.dimensions, 'values':  lat_var[:]},
                'longitude' : {'dims': lon_var.dimensions, 'values':  lon_var[:]},
            }

        elif grid_format is not None:
            raise NotImplementedError(f'Reading of coordinates format {grid_format} not implemented')

    # -----------------------------------------------------------------------------

    def max(self, var: str, *args, **kwargs):
        """This property returns the maximum value of the data array."""
        return self.variables[var]['values'].max(*args, **kwargs)

    def min(self, var: str, *args, **kwargs):
        """This property returns the minimum value of the data array."""
        return self.variables[var]['values'].min(*args, **kwargs)

    def mean(self, var: str, *args, **kwargs):
        """This property returns the mean value of the data array."""
        return self.variables[var]['values'].mean(*args, **kwargs)

    def count(self, var: str, *args, **kwargs):
        """This property returns the count value of the data array."""
        return self.variables[var]['values'].count(*args, **kwargs)

    def size(self, var: str) -> int:
        """This property returns the size of the data array."""
        return self.variables[var]['values'].size

    def shape(self, var: str) -> tuple:
        """This property returns the shape of the data array."""
        return self.variables[var]['values'].shape

    # -----------------------------------------------------------------------------

    @property
    def dimensions(self) -> dict:
        """Returns the product dimensions stored as a dictionary."""
        return self._dimensions

    @property
    def variables(self) -> dict:
        """Returns the product variables stored as a dictionary."""
        return self._variables

    @property
    def grid_format(self) -> str:
        """Returns the format in which the grid cells were defined."""
        return self._grid_format

    @property
    def grid_dimensions(self) -> tuple:
        """Returns the dimensions of the horizontal grid."""
        return self._grid_vars['grid_dim']

    # -----------------------------------------------------------------------------

    def __eq__(self, other):
        """
        Compare if self is equal to another object by checking all its variables.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if isinstance(other, Kalkutun):
            for my_key, other_key in zip(self.__dict__, other.__dict__):
                if my_key != other_key:
                    return False
                if isinstance(getattr(self, my_key), np.ndarray):
                    if not np.array_equal(getattr(self, my_key), getattr(other, other_key)):
                        return False
                elif getattr(self, my_key) != getattr(other, other_key):
                    return False
            return True
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

    def convert_units(self, varname, to_unit):
        """
        Change units of the product data variable.

        Parameters
        ----------
        varname : str
            Name of variable to convert
        to_unit : str
            The desired new unit of the data.

        Returns
        -------
        None
        """
        to_unit = standardise_unit_string(to_unit)

        self.variables[varname]['values'] = convert_units(
            self.variables[varname]['values'],
            from_unit=self.variables[varname]['units'],
            to_unit=to_unit
        )

        self.variables[varname]['units'] = to_unit
        return

    # -----------------------------------------------------------------------------

    @property
    def polygons(self):
        """
        Array of polygons representing each grid cell.

        Returns
        -------
        np.ndarray
            An array of Polygon objects representing each grid cell.
        """
        if self._polygons is None:
            self.update_polygons()
        return self._polygons

    @polygons.deleter
    def polygons(self):
        """Deletes the stored polygons."""
        self._polygons = None

    # -----------------------------------------------------------------------------

    def update_polygons(self):
        """Creates and stores polygons from the current grid."""
        self._polygons = self.get_polygons()

    # -----------------------------------------------------------------------------

    def get_polygons(self):
        """
        Returns an array of Polygon objects created from the product's longitude and latitude.

        Returns
        -------
        np.ndarray
            An array of Polygon objects created from the product's longitude and latitude.
        """
        if self._grid_format is None:
            raise ValueError('grid_format has not been specified, cannot retrieve polygons')

        if self._grid_format == 'corners':
            lon_idx = self._grid_vars['longitude']['dims'].index(self._grid_vars['corner_dim'])
            lat_idx = self._grid_vars['latitude']['dims'].index(self._grid_vars['corner_dim'])
            lons = np.moveaxis(self._grid_vars['longitude']['values'], lon_idx, -1)
            lats = np.moveaxis(self._grid_vars['latitude']['values'], lat_idx, -1)
            return shapely.polygons(np.stack((lons, lats), -1))

        else:
            raise ValueError(f'Format for creation of polygons {self._grid_format} not recognized')

    # -----------------------------------------------------------------------------

    def filter(self, var, mask, inplace=False):
        """
        Filter function to mask values of a variable

        Parameters
        ----------
        var : str
            Name of the variable to filter
        mask : array-like
            Mask to use for filtering
        inplace : bool, optional
            Indicates if the operation should be performed in place (default: False)

        Returns
        -------
        None or np.ma.MaskedArray
            If inplace is True, None is returned, otherwise a masked array is returned
        """
        if not isinstance(inplace, bool):
            raise ValueError(f'Inplace should be a bool, {inplace} received')

        if isinstance(var, tuple):
            raise TypeError("Filter method does not support multiple variables. Use a single variable name.")

        if mask.ndim != self.variables[var]['values'].ndim:
            if self.variables[var]['values'].shape[:mask.ndim] == mask.shape:
                extra_dims = self.variables[var]['values'].ndim - mask.ndim
                mask = np.expand_dims(mask, axis=tuple(range(mask.ndim, mask.ndim+extra_dims)))
                mask = np.broadcast_to(mask, self.variables[var]['values'].shape)

        if inplace:
            self.variables[var]['values'] = np.ma.masked_where(mask, self.variables[var]['values'])
            return None

        return np.ma.masked_where(mask, self.variables[var]['values'])

    # -----------------------------------------------------------------------------

    def minmax_filter(self, var, min_value=None, max_value=None, from_var=None, inplace=False):
        """
        General filter function that masks values in variable based on min and max.
        If from_var is provided, that variable is checked for the condition.

        Parameters
        ----------
        var : str or tuple
            Variable name or tuple of variable names to filter
        min_value : float or int, optional
            Minimum value threshold
        max_value : float or int, optional
            Maximum value threshold
        from_var : str, optional
            Variable to use to check values for masking
        inplace : bool, optional
            Indicates if the operation should be performed in place (default: False)

        Returns
        -------
        None or np.ma.MaskedArray or tuple of np.ma.MaskedArray
            If inplace is True, None is returned, otherwise a tuple of (or single) masked arrays is returned
        """
        if min_value is None and max_value is None:
            raise ValueError('Either min_value or max_value has to be provided')

        if from_var is not None:
            mask = self.variables[from_var]['values']
        else:
            mask = self.variables[var]['values']

        if min_value is not None and max_value is not None:
            mask = (mask < min_value) | (mask > max_value)
        elif min_value is not None:
            mask = mask < min_value
        elif max_value is not None:
            mask = mask > max_value

        if isinstance(var, str):
            return self.filter(var, mask, inplace)

        return tuple(self.filter(v, mask, inplace) for v in var)

    # -----------------------------------------------------------------------------

    def qa_filter(self, var, min_value, inplace=False):
        """
        Filter the variable array based on a quality assurance (QA) value threshold.

        Parameters
        ----------
        var : str or tuple
            Variable name or tuple of variable names to be filtered
        min_value : float
            The minimum QA value to retain data.
        inplace : bool, optional
            If True, the function applies the masking operation on the data array in-place.
            If False, the function returns a new masked array (default).

        Returns
        -------
        None or np.ma.MaskedArray or tuple of np.ma.MaskedArray
            If inplace is False and var is a single variable, a masked array with values below the minimum QA value
            masked is returned. If var is a tuple, a tuple of masked arrays is returned.
            If inplace is True, None is returned.
        """
        if 'qa_value' not in self.variables:
            raise AttributeError("Could not find 'qa_value' in between variables, please redefine your quality flag "
                                 "variable or use the minmax_filter method with the name of your variable.")

        if (self.variables['qa_value']['values'] == 0.0).all() is True:
            logging.error('Product seems to not have any valid quality value')
            raise ValueError('All quality flag values are equal to 0')

        return self.minmax_filter(var=var, min_value=min_value, from_var='qa_value', inplace=inplace)

    # -----------------------------------------------------------------------------

    def coordinates_filter(self, var, *args, inplace=False, **kwargs):
        """
        Filters the variable array based on coordinate borders.

        This method applies a filter to the variable array based on the specified longitude and latitude borders.
        The filtering is inclusive, meaning that values falling within the specified range are retained.

        Parameters
        ----------
        var : str or list or tuple
            Variable or list of variable names to filter.
        args : iterable
            Coordinate borders to use as filter. The coordinates should be provided as follows:
            - If providing individually: (min_lon, max_lon, min_lat, max_lat).
            - If providing as tuples: ((min_lon, max_lon), (min_lat, max_lat)).
        inplace : bool, optional
            Indicates if the operation should be performed inplace (default: False).
        **kwargs : dict, optional
            Additional keyword arguments. Can be used to provide single filter options.

        Returns
        -------
        np.ma.MaskedArray or tuple of np.ma.MaskedArray
            Masked array(s) resulting from the filtering operation. If inplace is True, returns None.

        Raises
        ------
        ValueError
            If the number of arguments is invalid or coordinate limits are out of range.

        """
        if len(args) in [1, 4]:
            lon_min, lon_max, lat_min, lat_max = args
        elif len(args) == 2:
            (lon_min, lon_max), (lat_min, lat_max) = args
        elif len(args) == 0 and kwargs:
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
            raise AssertionError(f'Longitude {lon_min} has to be smaller than {lon_max} and both in range [-180, 180]')
        if lat_min >= lat_max or lat_min > 90 or lat_max < -90:
            raise AssertionError(f'Latitude {lat_min} has to be smaller than {lat_max} and both in range [-90, 90]')

        bounds = shapely.bounds(self.polygons)

        logging.info(type(bounds))
        logging.info(bounds.shape)
        logging.info(bounds.flatten()[0])
        logging.info([lon_min, lon_max, lat_min, lat_max])
        mask = (bounds[..., 0] > lon_min) | (bounds[..., 1] > lat_min) | \
               (bounds[..., 2] < lon_max) | (bounds[..., 3] < lat_max)

        if isinstance(var, str):
            return self.filter(var, mask, inplace)

        return tuple(self.filter(v, mask, inplace) for v in var)

# =================================================================================
