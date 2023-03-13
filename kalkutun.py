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
import numpy as np
import pandas as pd
import shapely
from netCDF4 import Dataset
from . import geodata
from . import grid
from . import utils
from . import polygons
from . import scrap

# =================================================================================

class Kalkutun:
    """
    A class to handle satellite datasets for the TROPOMI sensor.

    This class supports the following products:
        - S5P_OFFL_L2__NO2
        - S5P_RPRO_L2__NO2

    The class provides an interface to easily access and manipulate the product data. It also includes methods to convert
    units, retrieve polygon data, and create a dataframe of the polygon data.

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
        Initializes the Kalkuntun object.

    get_polygons(self, return_data=True)
        Returns a list of Polygon objects created from the product's longitude and latitude.

    get_polygon_dataframe(self)
        Returns a Pandas dataframe containing the polygon data.

    convert_units(self, to_unit)
        Converts the units of the product data.

    Raises
    ------
    NotImplementedError
        If the sensor is not supported.
    """
    __module__ = 'kintunwenu'

    def __init__(self, dataset):
        """
        Initializes the Kalkutun object.

        Parameters
        ----------
        dataset : netCDF4.Dataset
            The netCDF4 dataset containing the product.

        Raises
        ------
        NotImplementedError
            If the sensor is not supported.
        """
        # ToDo: properly close the file if an error arise
        if isinstance(dataset, str):
            # ToDo: improve handling of string
            try:
                dataset = Dataset(dataset)
            except OSError:
                dataset = scrap.download_ncfile(dataset)
        elif isinstance(dataset, Kalkutun):
            # ToDo: handle dataset to be Kalkutun
            raise NotImplementedError('Kalkutun must be created from a path to file, an url or an actual Dataset')

        # Attempt to retrieve sensor information from product
        try:
            self.sensor = dataset.sensor.lower()
        except AttributeError:
            raise NotImplementedError('Unable to retrieve sensor information from product.')

        # Initialize class attributes
        self.longitude = None
        self.latitude = None
        self.tracer = None
        self.product = None
        self.units = None
        self.data = None
        self.qa_value = None

        if self.sensor in ['tropomi']:

            var_names = [
                'longitude',
                'latitude',
                'time_utc',
                'qa_value'
            ]

            self.tracer = dataset.id.split('__')[1]
            variables = dataset.groups['PRODUCT'].variables

            # Retrieve attributes from variables in product
            attrs = {}
            for key, value in variables.items():
                if key.lower().endswith('_column'):
                    attrs.update({'data': value[:].squeeze(),
                                  'units': utils.standardise_unit_string(value.units),
                                  'product': key})
                elif key in var_names:
                    # Here we are assuming that the 'time' dimension is 1
                    attrs.update({key: value[0]})

            for key, value in attrs.items():
                setattr(self, key, value[:])

        # Sensor not recognized
        else:
            raise NotImplementedError('Sensor has not been implemented.')

    # -----------------------------------------------------------------------------
    def __eq__(self, other):
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
        Kalkuntun
            A new instance of the current object.
        """
        new_object = self.__class__.__new__(self.__class__)
        new_dict = {k: (v.copy() if callable(getattr(v, 'copy', None)) else v) for k, v in self.__dict__.items()}
        new_object.__dict__.update(new_dict)
        return new_object

    # -----------------------------------------------------------------------------
    def max(self, *args, **kwargs):
        """This property returns the maximum value of the data array as a float32."""
        return self.data.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        """This property returns the minimum value of the data array as a float32."""
        return self.data.min(*args, **kwargs)

    @property
    def shape(self):
        """This property returns the shape of the data array as a float32."""
        return self.data.shape

    # -----------------------------------------------------------------------------
    def get_polygons(self, return_data=True):
        """
        Returns a list of Polygon objects created from the product's longitude and latitude.

        Parameters
        ----------
        return_data : bool, optional
            Indicates whether to return the corresponding data. Defaults to True.

        Returns
        -------
        list[shapely.geometry.polygon.Polygon] or tuple[list, np.ndarray]
            A list of Polygon objects created from the product's longitude and latitude. If `return_data`
            is True, it also returns a tuple containing the Polygon objects and the corresponding data.
        """
        mode = 'centers' if self.data.shape == self.longitude.shape else 'corners'
        corners = polygons.get_corners_from_coordinates(self.longitude, self.latitude, mode=mode)
        polygons_list = shapely.polygons(corners)

        if return_data is True:
            data = self.data[1:-1, 1:-1].flatten() if mode == 'centers' else self.data.flatten()
            return polygons_list, data
        else:
            return polygons_list

    # -----------------------------------------------------------------------------
    def get_polygon_dataframe(self, drop_masked=True):
        """
        Generates a Geo-DataFrame from the object coordinates and data.

        Parameters
        ----------
        drop_masked : bool, optional
            If True (default), drops any rows with masked values in the returned DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the polygons of each cell and its actual value.
            The DataFrame has two columns:
                - 'value': The data values associated with each polygon.
                - 'polygon': The Shapely Polygon objects representing geographical polygons.
        """
        polygons_list, data = self.get_polygons(return_data=True)
        df = geodata.create_geo_dataset(polygons_list, data)
        if drop_masked:
            return df.loc[~data.mask]
        else:
            return df

    # -----------------------------------------------------------------------------
    def convert_units(self, to_unit):
        """

        Parameters
        ----------
        to_unit : str
            The desired new unit of the data

        Returns
        -------
        None
        """
        to_unit = utils.standardise_unit_string(to_unit)
        self.data = utils.convert_units(self.data, from_unit=self.units, to_unit=to_unit,
                                        species=self.tracer)
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
            If False, the function returns a new masked array.

        Returns
        -------
            If inplace is False, A masked array with values below the minimum QA value masked.
            None otherwise.
        """
        if inplace:
            self.data = np.ma.masked_where(self.qa_value < min_value, self.data)
        else:
            return np.ma.masked_where(self.qa_value < min_value, self.data)

# =================================================================================

class GridCrafter:
    """
    Adapting data to its new grid like a master of disguise.
    Basically, just creates a regular grid, further capabilities are desired.
    """
    __module__ = 'kintunwenu'

    def __init__(
            self,
            coordinates=(-180, 180, -90, 90),
            resolution=(6, 4),
            interpolation='weighted',
            min_fill=None,
            **kwargs
    ):

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

        if interpolation in ['weighted']:
            self.interpolation = interpolation
        else:
            raise AssertionError('Interpolation method must be "weighted".')

        self.min_fill = min_fill
        self.units = kwargs.pop('units', None)
        self.geod = kwargs.pop('geod', None)
        self.qa_filter = kwargs.pop('qa_filter', None)
        self.lat_filter = kwargs.pop('lat_filter', None)

        # Create coordinate grids
        self.lons, self.lats = grid.create_grid(resolution, coordinates[0:2], coordinates[2:4])

    # -----------------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.regrid(*args, **kwargs)

    # -----------------------------------------------------------------------------
    def regrid(self, product, qa_filter=None, drop_negatives=False):
        """

        Parameters
        ----------
        product
        qa_filter
        drop_negatives

        Returns
        -------

        """
        kprod = product.copy() if isinstance(product, Kalkutun) else Kalkutun(product)

        # ToDo: cut outside limits

        if self.units is not None:
            kprod.convert_units(self.units)

        if qa_filter is not None:
            kprod.qa_filter(qa_filter, inplace=True)
        elif self.qa_filter is not None:
            kprod.qa_filter(self.qa_filter, inplace=True)

        df_obs = kprod.get_polygon_dataframe(drop_masked=True)

        if drop_negatives is True:
            df_obs = df_obs[df_obs['value'] > 0.0]

        if self.lat_filter is not None:
            df_obs = geodata.filter_by_latitude(df_obs, lat_thresh=self.lat_filter)

        df_obs = pd.DataFrame({
                k: v for k, v in zip(
                    ['polygon', 'value'],
                    polygons.split_anomaly_polygons(df_obs['polygon'], df_obs['value']))
        })

        if self.interpolation == 'weighted':
            regrid = grid.weighted_regrid(
                df_obs['polygon'], df_obs['value'], self.lons, self.lats,
                min_fill=self.min_fill, geod=self.geod
            )
        else:
            raise NotImplementedError('Interpolation type not implemented')

        return regrid

# =================================================================================
