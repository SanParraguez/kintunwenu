# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> CRAFTER

Provides class to grid Kalkutun product
"""

__all__ = [
    'GridCrafter'
]

# ============= IMPORTS ===============================

import logging
import numpy as np

from netCDF4 import Dataset

from .grid import create_grid, weighted_regrid
from .kalkutun import Kalkutun

# =================================================================================

class GridCrafter:
    """
    Adapting data to its new grid like a master of disguise.
    Basically, just creates a regular grid, further capabilities are desired.
    """
    __module__ = 'kintunwenu'

    def __init__(
            self,
            grid_lons, grid_lats,
            interpolation='weighted',
            min_fill=None, qa_filter=None,
            units=None, geod=None,
            lat_filter=None,
            **kwargs
    ):
        """
        Initializes the GridCrafter object.

        Parameters
        ----------
        grid_lons : np.ndarray
            Longitudes of grid corners.
        grid_lats : np.ndarray
            Latitudes of grid corners.
        interpolation : str, optional
            Method of interpolation to be used (default: 'weighted').
        min_fill : float, optional
            Fraction of the grid cell that has to be filled to be valid (default: None).
        qa_filter : float, optional
            Minimum value of the quality flag necessary to consider a measurement (default: None).
        units : str, optional
            Desired output units of the main data of the product. It will try to convert both the data and its
            standard deviation (default: None).
        geod : pyproj.Geod, optional
            Geodetic object to be used for calculating areas of cells. Assumes Earth if none is provided.
        lat_filter : float, optional
            Latitude filter (default: None).
        """

        # Check grid dimensions and raise appropriate errors if invalid
        self.lon_lim = np.min(grid_lons), np.max(grid_lons)
        self.lat_lim = np.min(grid_lats), np.max(grid_lats)

        if interpolation in ['weighted']:
            self.interpolation = interpolation
        else:
            raise NotImplementedError("Interpolation method must be 'weighted'.")

        self.min_fill = min_fill
        self.units = units
        self.geod = geod
        self.qa_filter = qa_filter
        self.lat_filter = lat_filter

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
        Creates an instance of GridCrafter using grid corner coordinates. Useful for non-monotonous grids.

        Parameters
        ----------
        grid_lons : np.ndarray
            Longitudes of grid corners.
        grid_lats : np.ndarray
            Latitudes of grid corners.
        """
        return cls(grid_lons, grid_lats, **kwargs)

    # -----------------------------------------------------------------------------

    @classmethod
    def from_size(cls, grid_size, lon_lim=(-180, 180), lat_lim=(-90, 90), method='corners', **kwargs):
        """
        Creates an instance of GridCrafter using grid size and limits. Useful for monotonous grids

        Parameters
        ----------
        grid_size : float or tuple[float, float]
            Size of the grid cells. If a float is given, it assumes a regular grid.
        lon_lim : tuple[float, float], optional
            Longitude limits of the grid, included (default: (-180, 180)).
        lat_lim : tuple[float, float], optional
            Latitude limits of the grid, included (default: (-90, 90)).
        method : str, optional
            Indicates if the points are the corners or the centers of the grid (default: 'corners').
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
        drop_poles = kwargs.pop('drop_poles', False)
        var_list = kwargs.pop('var_list', None)

        kprod = product.copy() if isinstance(product, Kalkutun) else Kalkutun(product, **kwargs)

        if var_list is None:
            var_list = list(kprod.variables.keys())

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
            logging.warning(f"    No polygons left to regrid for {product}, returning None (might check masked data)")
            return None

        if self.interpolation == 'weighted':
            regrid = weighted_regrid(
                self.lons, self.lats, df_obs['geometry'], df_obs.drop('geometry', axis=1),
                min_fill=self.min_fill, geod=self.geod, **kwargs
            )
            if not regrid:
                logging.warning(f"    No cell filled for {product}, returning None (might check masked data)")
                return None
        else:
            raise NotImplementedError('Interpolation type not implemented')

        # change data to actual name
        regrid[kprod.product] = regrid.pop('data')

        logging.debug(f"  Finished gridding of product ({next(iter(regrid.values())).count()} valid values)")
        return regrid

# =================================================================================
