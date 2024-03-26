# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================

CRAFTER MODULE
------------------

This module provides the GridCrafter class for adapting data to new grids based on the Kalkutun data class.

The GridCrafter class enables the creation of regular grids and offers functionality for interpolation,
filtering, and data conversion.

Classes:
    GridCrafter: A class for adapting data to new grids.
"""

__all__ = [
    'GridCrafter'
]

# ============= IMPORTS ===============================

import logging
import sys

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

    def regrid(self, product, varnames=None,
               qa_filter=None, coord_filter=None,
               **kwargs):
        """
        Perform a regridding over a product

        Parameters
        ----------
        product : Dataset or Kalkutun
            Product to be regridded.
        varnames : str or list or tuple
            Variable(s) to be regridded in the product. If None (default), assumes every variable with the
            correspondant dimensions to the polygons.
        qa_filter : float, optional
            Quality assurance filter to be applied.
        coord_filter : tuple
            Constrain the domain by masking values outside the limits given.
        **kwargs : dict, optional
            Additional keyword arguments. Used for initialization of Kalkutun product.

        Returns
        -------
        np.ndarray
            2D-array with the weighted values.
        """
        kprod = product.copy() if isinstance(product, Kalkutun) else Kalkutun(product, **kwargs)

        # ToDo: remove extra logging
        logging.info(f'Kalkutun variables: {list(kprod.variables.keys())}')
        logging.info(f"  grid_dimensions: {kprod.grid_dimensions}")

        if varnames is None:
            dims = kprod.grid_dimensions
            varnames = tuple(k for k, v in kprod.variables.items() if v['dims'][:len(dims)] == dims)
        elif isinstance(varnames, str):
            varnames = tuple([varnames])

        # ToDo: remove extra logging
        logging.info(f'Planning to regrid variables: {varnames}')

        if coord_filter is not None:
            kprod.coordinates_filter(varnames, coord_filter, inplace=True)
        else:
            kprod.coordinates_filter(varnames, self.lon_lim, self.lat_lim, inplace=True)

        # Try to convert units if possible
        if self.units is not None:
            for var in varnames:
                try:
                    kprod.convert_units(var, self.units)
                except ValueError:
                    continue

        # Apply quality assurance flag
        if qa_filter is not None:
            kprod.qa_filter(varnames, qa_filter, inplace=True)
        elif self.qa_filter is not None:
            kprod.qa_filter(varnames, self.qa_filter, inplace=True)

        # ToDo: remove extra logging
        # --- logging ---
        logging.info(list(kprod.variables.keys()))
        logging.info(varnames)
        data = {k: v['values'] for k, v in kprod.variables.items() if k in varnames}
        logging.info(f"variables to regrid: {list(data.keys())}")
        # ---------------

        if not [v for k, v in data.items() if v.size > 0]:
            logging.warning(f"    No polygons left to regrid for {product}, returning None (might check masked data)")
            return None

        if self.interpolation == 'weighted':
            regrid = weighted_regrid(
                self.lons, self.lats, kprod.polygons, data,
                min_fill=self.min_fill, geod=self.geod, **kwargs
            )
            if not regrid:
                logging.warning(f"    No cell filled for {product}, returning None (might check masked data)")
                return None
        else:
            raise NotImplementedError('Interpolation type not implemented')

        logging.debug(f"  finished gridding of product ({next(iter(regrid.values())).count()} valid values)")
        return regrid

# =================================================================================
