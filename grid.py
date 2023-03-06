# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> REGRID

Submodule that contains the functions for regridding.
"""
__all__ = [
    'weighted_regrid',
    'create_grid'
]

# === IMPORTS =========================================================
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod
from .geodata import create_geo_grid

# =================================================================================

def weighted_regrid(polygons, values, grid_lon, grid_lat, min_fill=None, geod=None):
    """
    Performs a weighted regridding of polygons into a given regular grid.

    Parameters
    ----------
    polygons : list or pd.Series pr np.ndarray of Polygon, len (n)
        The n polygons to be regridded.
    values : list or pd.Series pr np.ndarray, shape (n,)
        The values of each polygon.
    grid_lon : np.ndarray, shape (j,)
        Gridded longitudes.
    grid_lat : np.ndarray, shape (i,)
        Gridded latitudes.
    min_fill : float
        Minimum fraction of cell area needed to consider the cell new value valid.
        If not achieved, it is keep as nan. Using this parameter could lead to a
        decrease in performance.
    geod : pyproj.Geod

    Returns
    -------
    np.ndarray
        Regridded values with shape (i-1, j-1).
    """
    if type(polygons) == list:
        polygons = np.array(polygons)
    elif type(polygons) == pd.Series:
        polygons = polygons.to_numpy()
    if type(values) == list:
        values = np.array(values)
    elif type(values) == pd.Series:
        values = values.to_numpy()
    if geod is None:
        # proj = '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        geod = Geod('+a=6378137 +f=0.0033528106647475126')
    if min_fill is not None:
        assert 0.0 < min_fill < 1.0, f"Minimum fill value has to be a fraction, {min_fill} not valid."

    df_grid = create_geo_grid(grid_lon, grid_lat, mode='corners')

    # Get areas for single column and fill through longitudes
    df_grid['area'] = df_grid[df_grid['xi'] == 0]['polygon'].map(
        lambda poly: geod.geometry_area_perimeter(poly)[0]
    )
    df_grid['area'].fillna(method='ffill', inplace=True)

    # Create and query STRtree
    tree = shapely.STRtree(df_grid['polygon'].to_numpy())
    inters = tree.query(polygons)

    # Create GeoDataFrame with intersections
    df_inter = df_grid.loc[inters[1], 'area'].to_frame()
    df_inter['polygon'] = shapely.intersection(
        df_grid.loc[inters[1], 'polygon'],
        polygons.take(inters[0])
    )
    df_inter['value'] = values.take(inters[0])

    # Calculate intersection areas
    df_inter['inter_area'] = df_inter['polygon'].map(
        lambda poly: - geod.geometry_area_perimeter(poly)[0]
    )
    # Drop negative areas, undesired behaviour you will have
    df_inter = df_inter[df_inter['inter_area'] > 0.0]

    # Calculate new values
    df_inter['frac_area'] = df_inter['inter_area'] / df_inter['area']
    df_inter['weight'] = df_inter['value'] * df_inter['frac_area']
    df_inter.reset_index(inplace=True)
    df_inter = df_inter.loc[:, ('index', 'weight', 'frac_area')].groupby('index').sum()

    # Filter if min_fill
    if min_fill is not None:
        df_inter = df_inter[df_inter['frac_area'] > min_fill]

    df_grid['value'] = df_inter['weight'] / df_inter['frac_area']

    # Reshape to grid and mask
    grid_value = df_grid['value'].to_numpy().reshape((len(grid_lat) - 1, len(grid_lon) - 1))
    grid_value = np.ma.masked_where(np.isnan(grid_value), grid_value)

    return grid_value

# =================================================================================

def create_grid(grid_size, lon_lim=(-180, 180), lat_lim=(-90, 90)):
    """
    Creates equally spaced grid points.

    Parameters
    ----------
    grid_size : float or tuple
        Size of the grid cells, if a float is given it will assume regular grid
    lon_lim : tuple
        Longitude limits of the grid, included
    lat_lim : tuple
        Latitude limits of the grid, included

    Returns
    -------
    Tuple of np.ndarray
    """
    if isinstance(grid_size, (float, int)):
        grid_size = (grid_size, grid_size)
    else:
        grid_size = tuple(grid_size)

    nlon = int((lon_lim[1] - lon_lim[0]) / grid_size[0])
    nlat = int((lat_lim[1] - lat_lim[0]) / grid_size[1])
    grid_lon = np.linspace(*lon_lim, num=nlon + 1, endpoint=True)
    grid_lat = np.linspace(*lat_lim, num=nlat + 1, endpoint=True)

    return grid_lon, grid_lat
