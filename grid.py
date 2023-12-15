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
import logging
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
        Gridded longitudes corners.
    grid_lat : np.ndarray, shape (i,)
        Gridded latitudes corners.
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
    dict
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
    if grid_lon.ndim > 1:
        grid_shape = tuple(dim-1 for dim in grid_lon.shape)
    else:
        grid_shape = (grid_lat.shape[0]-1, grid_lon.shape[0]-1)

    # Get areas for single column and fill through longitudes
    df_grid['area'] = df_grid[df_grid['xi'] == 0]['polygon'].map(
        lambda poly: geod.geometry_area_perimeter(poly)[0]
    )
    df_grid['area'].ffill(inplace=True)

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

    # Calculate fraction of the cell covered by the intersected polygon
    df_inter['inter_area'] = df_inter['inter_area'] / df_inter['area']

    # ToDo: change to avoid datetime calculations and just use timestamp in seconds since 1970
    #   this should increase performance
    to_datetime = []
    if isinstance(data, dict):
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.datetime64):
                value = (value - datetime(1970, 1, 1)).dt.total_seconds()
                to_datetime.append(key)
            df_inter['var_'+key] = np.asarray(value)[inters[0]]
    else:
        df_inter['data'] = data[inters[0]]

    # Drop negative areas, undesired behaviour you will have
    df_inter = df_inter[df_inter['inter_area'] > 0.0]

    # Calculate the weighted contribution (value * area_fraction)
    for col in [col for col in df_inter if col.split('_')[0] == 'var']:
        var_col = np.array(df_inter[col].to_list())
        if var_col.ndim > 1:
            df_inter[col] = [*(var_col * np.expand_dims(df_inter['inter_area'].to_numpy(),
                                                        axis=tuple(range(1, var_col.ndim))))]
        else:
            df_inter[col] = var_col * df_inter['inter_area']

    # Add up all the contributions per cell (now inter_area will be the total fraction of the cell covered)
    #    groupby seems to work from pandas v2.0
    df_inter.reset_index(inplace=True)
    df_inter = df_inter.drop(['area', 'polygon'], axis=1).groupby('index').sum()

    # Filter if min_fill is higher than the total area covered
    if min_fill is not None:
        df_inter = df_inter[df_inter['inter_area'] > min_fill]

    # In case of empty DataFrame, just return None
    if len(df_inter) == 0:
        logging.warning(f"    Regridding ended up empty, returning 'None' (might check masked data)")
        return None

    # Divide by the area covered, since it could be a value different from 1 for not completely covered cells
    for col in [col for col in df_inter.drop('inter_area', axis=1)]:
        df_grid[col] = df_inter[col] / df_inter['inter_area']

    grid_values = {}
    for col in df_inter.drop('inter_area', axis=1):
        # Try to get a numeric numpy array, if it fails because of combination of arrays and NaN values
        # creates empty arrays to fill the gaps
        try:
            col_array = df_grid[col].to_numpy(float)
        except ValueError:
            emp_array = np.full(df_inter[col].iloc[0].shape, np.nan)
            is_null = df_grid[col].isnull()
            df_grid.loc[is_null, col] = pd.Series([emp_array] * is_null.sum()).to_numpy()
            col_array = np.array(df_grid[col].to_list())

        # Reshape to grid
        grid_values['_'.join(col.split('_')[1:])] = np.ma.masked_invalid(
            col_array.reshape(grid_shape + col_array.shape[1:])
        )

    # As datetime variables
    for key in to_datetime:
        grid_values[key] = grid_values[key].astype('datetime64[s]')

    return grid_values

# =================================================================================

def create_grid(grid_size, lon_lim=(-180, 180), lat_lim=(-90, 90), method='corners'):
    """
    Creates equally spaced grid cells.

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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple with (lons, lats) 1D arrays.
    """
    if isinstance(grid_size, (float, int)):
        grid_size = (grid_size, grid_size)
    else:
        grid_size = tuple(grid_size)

    if len(lon_lim) != 2 or len(lat_lim) != 2:
        raise AssertionError('Both lon and lat limits have to be tuples with two elements.')

    if method == 'corners':
        nlon = int((lon_lim[1] - lon_lim[0]) / grid_size[0])
        nlat = int((lat_lim[1] - lat_lim[0]) / grid_size[1])
        grid_lon = np.linspace(*lon_lim, num=nlon + 1, endpoint=True)
        grid_lat = np.linspace(*lat_lim, num=nlat + 1, endpoint=True)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented, desirable")

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
    grid_xi = np.tile(np.arange(grid_shape[1] - 1), grid_shape[0] - 1)
    grid_yi = np.arange(grid_shape[0] - 1).repeat(grid_shape[1] - 1)

    df_grid = pd.DataFrame({
        'xi': grid_xi,
        'yi': grid_yi,
        'polygon': polys_grid
    })

    return df_grid

# =================================================================================
