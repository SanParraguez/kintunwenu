# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> GRID

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
from datetime import datetime
from .geodata import get_intersections, get_areas
from .polygons import get_corners_from_grid

# =================================================================================

def weighted_regrid(grid_lon, grid_lat, polygons, data, min_fill=None, geod=None, **kwargs):
    """
    Performs a weighted regridding of polygons into a given regular grid.

    Parameters
    ----------
    grid_lon : np.ndarray, shape (j,)
        Gridded longitudes.
    grid_lat : np.ndarray, shape (i,)
        Gridded latitudes.
    polygons : list or pd.Series or np.ndarray of Polygon, len (n)
        The n polygons to be regridded.
    data : list or pd.Series or np.ndarray or dict or pd.DataFrame, shape (n,)
        The values of each polygon.
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
    threads = kwargs.pop('threads', None)
    workers = kwargs.pop('workers', None)

    if isinstance(polygons, list):
        polygons = np.array(polygons)
    elif isinstance(polygons, pd.Series):
        polygons = polygons.to_numpy()

    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame().to_dict(orient='series')
    elif isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='series')

    if min_fill is not None:
        assert 0.0 < min_fill < 1.0, f"Minimum fill value has to be a fraction, {min_fill} not valid."

    df_grid = create_geo_grid(grid_lon, grid_lat, mode='corners')

    # Get areas for single column and fill through longitudes
    df_grid['area'] = df_grid[df_grid['xi'] == 0]['polygon'].map(
        lambda poly: geod.geometry_area_perimeter(poly)[0]
    )
    df_grid['area'].fillna(method='ffill', inplace=True)

    # ToDo: Implement KDtree and Rtree, check speeds.
    # Create and query STRtree
    tree = shapely.STRtree(df_grid['polygon'].to_numpy())
    inters = tree.query(polygons)

    # Create GeoDataFrame with intersections
    df_inter = df_grid.loc[inters[1], 'area'].to_frame()

    # Get intersection polygons
    df_inter['polygon'] = get_intersections(
        df_grid.loc[inters[1], 'polygon'].to_numpy(),
        polygons[inters[0]],
        threads=threads
    )

    # Calculate intersection areas (intersections are 'inverted' so we multiply by -1)
    df_inter['inter_area'] = -1 * get_areas(df_inter['polygon'], geod=geod, workers=workers)

    # Calculate new values
    df_inter['inter_area'] = df_inter['inter_area'] / df_inter['area']

    to_datetime = []
    if isinstance(data, dict):
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.datetime64):
                # np.array(0, dtype='datetime64[s]')
                value = (value - datetime(1970, 1, 1)).dt.total_seconds()
                to_datetime.append(key)
            df_inter['data_'+key] = np.asarray(value)[inters[0]]
    else:
        df_inter['data'] = data[inters[0]]

    # Drop negative areas, undesired behaviour you will have
    df_inter = df_inter[df_inter['inter_area'] > 0.0]

    for col in [col for col in df_inter.columns.to_list() if col[:4] == 'data']:
        df_inter[col] = df_inter[col] * df_inter['inter_area']

    df_inter.reset_index(inplace=True)
    df_inter = df_inter.drop(['area', 'polygon'], axis=1).groupby('index').sum()

    # Filter if min_fill
    if min_fill is not None:
        df_inter = df_inter[df_inter['inter_area'] > min_fill]

    for col in [col for col in df_inter.drop('inter_area', axis=1)]:
        df_grid[col] = df_inter[col] / df_inter['inter_area']

    for key in to_datetime:
        df_grid['data_'+key] = np.array(df_grid['data_'+key], dtype='datetime64[s]')

    # Reshape to grid
    grid_values = {}
    for col in df_inter.drop('inter_area', axis=1):
        grid_values[col.split('_')[-1]] = df_grid[col].to_numpy().reshape((len(grid_lat) - 1, len(grid_lon) - 1))

    return grid_values

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

# =================================================================================

def create_geo_grid(lons, lats, mode='corners'):
    """
    Generates a Geo-DataFrame containing a grid of polygons defined by the input latitude and longitude coordinates.

    Parameters
    ----------
    lons : array-like
        An array of longitude coordinates, in degrees.
    lats : array-like
        An array of latitude coordinates, in degrees.
    mode : str, optional
        Determines how the grid cells are defined. Defaults to 'corners', which creates cells with corners defined
        by the input coordinates. Alternatively, 'centers' can be used to create cells with centers defined
        by the input coordinates.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the polygons of each cell and its index (xi, yi). The DataFrame has three columns:
            - 'xi': The x index of the cell.
            - 'yi': The y index of the cell.
            - 'polygon': The Shapely Polygon object representing the cell.
    """
    lons = np.array(lons)
    lats = np.array(lats)
    if lons.ndim == 1 and lats.ndim == 1:
        lons, lats = np.meshgrid(lons, lats, indexing='xy')
    elif lons.ndim == 2 and lons.ndim == 2:
        assert lons.shape == lats.shape
    else:
        raise ValueError(f"Arrays must have same dimensions {lons.shape} and {lats.shape}"
                         f"not compatible.")

    polys_grid = shapely.polygons(
        get_corners_from_grid(lons, lats, mode=mode)
    )

    grid_shape = lons.shape
    grid_xi = np.tile(np.arange(grid_shape[0] - 1), grid_shape[1] - 1)
    grid_yi = np.arange(grid_shape[1] - 1).repeat(grid_shape[0] - 1)

    df_grid = pd.DataFrame({
        'xi': grid_xi,
        'yi': grid_yi,
        'polygon': polys_grid
    })

    return df_grid

# =================================================================================
