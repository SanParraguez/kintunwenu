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
import pyproj
import shapely
import geopandas as gpd
from datetime import datetime
from .geodata import get_intersections, get_areas
from .polygons import get_corners_from_grid

import time


# =================================================================================

def weighted_regrid(grid_lon, grid_lat, grid_dims, polygons, data, min_fill=None, crs=None, **kwargs):
    """
    Performs a weighted regridding of polygons into a given regular grid.

    Parameters
    ----------
    grid_lon : np.ndarray, shape (j, )
        Gridded longitudes corners.
    grid_lat : np.ndarray, shape (i, )
        Gridded latitudes corners.
    grid_dims : tuple
        A tuple containing names of grid dimensions.
    polygons : list or pd.Series or np.ndarray of Polygon, len (n)
        The n polygons to be regridded.
    data : dict
        The variables with values and dimensions assigned to each polygon.
    min_fill : float
        Minimum fraction of cell area needed to consider the cell new value valid.
        If not achieved, it is kept as NaN. Using this parameter could lead to a
        decrease in performance.
    crs : str
        A crs for calculating area and perimeter (default: WGS84).

    Returns
    -------
    dict
        A dictionary with regridded values, where each key corresponds to a variable
        and contains a dict with updated 'values' and 'dims'.
    """
    # Validate input grid
    regular_grid = is_regular_grid(grid_lat, grid_lon)
    if not regular_grid:
        raise NotImplementedError("Irregular grids are not supported yet.")

    # Validate input polygons
    if isinstance(polygons, list):
        polygons = np.array(polygons)
    elif isinstance(polygons, pd.Series):
        polygons = polygons.to_numpy()

    # Validate input data
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary with 'values' and 'dims'")

    # Validate dictionary keys and dimensions
    for key, value in data.items():
        if 'values' not in value or 'dims' not in value:
            raise ValueError(f"Each variable in data must contain 'values' and 'dims'. Missing in {key}")
        if not set(value['dims']).issuperset(grid_dims):
            raise ValueError(f"dimensions of variables should be a superset of grid_dims for the regridding.")

    # Validate min_fill condition
    if min_fill is not None:
        if not (0.0 < min_fill < 1.0):
            raise ValueError(f"min_fill must be a fraction between 0 and 1. Got {min_fill}.")

    # Validate polygons dimensions
    # ToDo: handle when polygons come with more dimensions, not just the grid_dimensions
    if polygons.ndim != len(grid_dims):
        raise NotImplementedError(f"Multiple retrievals not implemented. polygons.shape and grid_dims "
                                  f"should have same length")

    # Assign default crs
    if crs is None:
        crs = 'WGS84'

    # ***

    # Initialize geoseries
    gdf = gpd.GeoSeries(polygons.flatten(), crs=crs)

    # Filter invalid polygons and store indexes for data
    valid_polygons = gdf.is_valid
    gdf = gdf[valid_polygons]

    # Generate grid dataframe
    gdf_grid = create_geo_grid(grid_lon, grid_lat, mode='corners', crs=crs)

    # Create grid tree and query using STRtree
    intersections = gdf_grid['geometry'].sindex.query(gdf)
    if intersections.size == 0:
        return None

    logging.info(f"len(inter[0]): {len(intersections[0])}, len(inter[1]): {len(intersections[1])}")
    logging.info(intersections)

    # Calculate grid areas in square meters
    if regular_grid:
        gdf_grid['cell_area'] = gdf_grid[gdf_grid['xi'] == 0].to_crs(epsg=6933).area
        gdf_grid['cell_area'] = gdf_grid['cell_area'].ffill()
    else:
        gdf_grid['cell_area'] = gdf_grid.to_crs(epsg=6933).area

    logging.info(gdf_grid)

    # Filter by intersections
    gdf_inter = gdf_grid.loc[intersections[1], ('xi', 'yi', 'geometry')]
    gdf_inter['poly_idx'] = intersections[0]

    # Calculate intersection geometry with pixels and get new areas
    gdf_inter['geometry'] = gdf_inter['geometry'].intersection(gdf.loc[intersections[0]], align=False)
    # gdf_inter['geometry'] = gdf_inter['geometry'].intersection(gdf.loc[gdf_inter['poly_idx']].reset_index(drop=True))
    gdf_inter['inter_area'] = gdf_inter['geometry'].to_crs(epsg=6933).area
    # # ToDo: add count without breaking groupby
    # gdf_inter['count'] = 1

    logging.info(gdf_inter)

    # Calculate fraction of the cell covered by the intersected polygon
    gdf_grid = gdf_grid.set_index(['xi', 'yi'])
    gdf_grid['inter_area'] = gdf_inter.drop(['geometry', 'poly_idx'], axis=1).groupby(['xi', 'yi']).sum()
    gdf_grid['coverage'] = gdf_grid['inter_area'] / gdf_grid['cell_area']

    logging.info(gdf_grid)

    # ***

    # Iterate over dictionary
    for key, value in data.items():

        # Get data and dimensions
        values = np.asarray(value['values'])
        dims = value['dims']

        # Get dimensions to regrid
        grid_dim_indices = [dims.index(dim) for dim in grid_dims]
        other_dim_indices = [i for i in range(len(dims)) if i not in grid_dim_indices]

        logging.info(f"grid_dim_indices: {grid_dim_indices}, other_dim_indices: {other_dim_indices}")

        # Reorder the axes to bring grid dimensions to the front
        reordering = grid_dim_indices + other_dim_indices
        reordered_values = np.transpose(values, axes=reordering)

        # Flatten the reordered array along the grid dimensions
        grid_shape = reordered_values.shape[:len(grid_dims)]
        other_shape = reordered_values.shape[len(grid_dims):]

        logging.info(f"grid_shape: {grid_shape}, other_shape: {other_shape}")

        # Flatten along the grid dimensions while keeping extra dimensions intact
        flat_values = reordered_values.reshape((-1,) + other_shape)[valid_polygons]

        logging.info(f"flat_values shape: {flat_values.shape}")
        logging.info(f"poly_idx shape: {gdf_inter['poly_idx'].shape}")

        # Initialize an array for the output
        output_shape = (grid_lon.size - 1, grid_lat.size - 1) + other_shape
        new_values = np.zeros(output_shape)

        # Iterate over the additional dimensions
        for idx in np.ndindex(*other_shape):
            # Slice along the extra dimensions
            slice_values = flat_values[:, idx].squeeze()

            logging.info(f"slice_idx: {idx}")
            logging.info(f"slice_values.shape: {slice_values.shape}")

            # Map values to polygons
            gdf_data = gdf_inter[['xi', 'yi']].copy()
            gdf_data['weighted_value'] = slice_values[gdf_inter['poly_idx']] * gdf_inter['inter_area']

            # Aggregate weighted values by grid cell and normalize
            weighted_sum = gdf_data.groupby(['xi', 'yi'])['weighted_value'].sum()
            normalized_values = (weighted_sum / gdf_grid['coverage']).fillna(0)

            # Reshape normalized values back into the grid
            reshaped_values = normalized_values.unstack(fill_value=0).values
            new_values[..., idx] = reshaped_values[..., None]

        # # Flatten the reordered array along the grid dimensions
        # # grid_shape = reordered_values.shape[:len(grid_dims)]
        # other_shape = reordered_values.shape[len(grid_dims):]
        # flat_values = reordered_values.reshape((-1,) + other_shape)[valid_polygons]
        #
        # logging.info(f"flat_values shape: {flat_values.shape}")
        # logging.info(f"poly_idx shape: {gdf_inter['poly_idx'].shape}")
        #
        # # Map values to corresponding polygons
        # gdf_data = gdf_inter[['xi', 'yi']].copy()
        # gdf_data['weighted_value'] = flat_values[gdf_inter['poly_idx']] * gdf_inter['inter_area']
        #
        # # Aggregate weighted values by grid cell and normalize
        # new_values = gdf_data.groupby(['xi', 'yi'])['weighted_value'].sum()
        # new_values /= gdf_grid['coverage']
        #
        # logging.info(new_values)


    # Obtain grid shape
    if grid_lon.ndim > 1:
        grid_shape = tuple(dim-1 for dim in grid_lon.shape)
    else:
        grid_shape = (grid_lat.shape[0]-1, grid_lon.shape[0]-1)

    # ToDo: change to avoid datetime calculations and just use timestamp in seconds since 1970
    #   this should increase performance
    to_datetime = []
    if isinstance(data, dict):
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.datetime64):
                value = (value - datetime(1970, 1, 1)).dt.total_seconds()
                to_datetime.append(key)
            df_inter['var_'+key] = np.asarray(value).flatten()[inters[0]]
    else:
        df_inter['data'] = data[inters[0]]

    # Drop negative areas, undesired behaviour you will have
    df_inter = df_inter[df_inter['inter_area'] > 0.0]
    logging.info(list(df_inter.columns))

    # Calculate the weighted contribution (value * area_fraction)
    for col in [col for col in df_inter if col.split('_')[0] == 'var']:
        var_col = np.array(df_inter[col].to_list())
        if var_col.ndim > 1:
            df_inter[col] = [*(var_col * np.expand_dims(df_inter['coverage'].to_numpy(),
                                                        axis=tuple(range(1, var_col.ndim))))]
        else:
            df_inter[col] = var_col * df_inter['coverage']
    # # Normalize by total coverage if greater than 100%
    # df_inter['normalized_weight'] = df_inter['coverage'] / df_inter['coverage'].sum()
    # for col in [col for col in df_inter if col.startswith('var_')]:
    #     df_inter[col] *= df_inter['normalized_weight']

    # Add up all the contributions per cell (now 'coverage' will be the total fraction of the cell covered)
    #    groupby seems to work from pandas v2.0
    df_inter.reset_index(inplace=True)
    df_inter['count'] = 1
    df_inter = df_inter.drop(['area', 'polygon', 'inter_area'], axis=1).groupby('index').sum()

    # Filter if min_fill is higher than the total area covered
    if min_fill is not None:
        df_inter = df_inter[df_inter['coverage'] > min_fill]

    # In case of empty DataFrame, just return None
    if len(df_inter) == 0:
        logging.warning(f"    Regridding ended up empty, returning 'None' (might check masked data)")
        return None

    # Divide by the area covered, since it could be a value different from 1 for not completely covered cells
    for col in [col for col in df_inter.drop(['coverage', 'count'], axis=1)]:
        df_grid[col] = df_inter[col] / df_inter['coverage']

    # Include covered fraction into data for output
    df_inter.rename(columns={'coverage': 'var_coverage', 'count': 'var_count'}, inplace=True)
    df_grid['var_coverage'] = df_inter['var_coverage']
    df_grid['var_count'] = df_inter['var_count'].astype(int)

    grid_values = {}
    for col in df_inter:
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
        A tuple with (lons, lats) 1D arrays, trimmed if needed to fit exact grid_size.
    """
    if isinstance(grid_size, (float, int)):
        grid_size = (grid_size, grid_size)
    else:
        grid_size = tuple(grid_size)

    if len(lon_lim) != 2 or len(lat_lim) != 2:
        raise AssertionError('Both lon and lat limits have to be tuples with two elements.')

    if method == 'corners':
        # Compute the exact number of grid cells in both directions
        nlon = int((lon_lim[1] - lon_lim[0]) / grid_size[0])
        nlat = int((lat_lim[1] - lat_lim[0]) / grid_size[1])

        # Adjust limits to match exactly the grid cell size
        adj_lon_lim = (lon_lim[0], lon_lim[0] + nlon * grid_size[0])
        adj_lat_lim = (lat_lim[0], lat_lim[0] + nlat * grid_size[1])

        # Generate the grid points
        grid_lon = np.linspace(*adj_lon_lim, num=nlon + 1, endpoint=True)
        grid_lat = np.linspace(*adj_lat_lim, num=nlat + 1, endpoint=True)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented, desirable")

    return grid_lon, grid_lat


# =================================================================================

def create_geo_grid(lons, lats, mode='corners', crs=None):
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
    if crs is None:
        crs = 'WGS84'

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

    df_grid = gpd.GeoDataFrame({
        'xi': grid_xi,
        'yi': grid_yi,
        'geometry': polys_grid
    }, crs=crs)

    return df_grid


# =================================================================================

def is_regular_grid(grid_lat, grid_lon):
    """
    Checks whether the given longitude and latitude arrays define a regular grid.

    Parameters
    ----------
    grid_lon : np.ndarray
        Array of grid longitudes (can be 1D or 2D).
    grid_lat : np.ndarray
        Array of grid latitudes (can be 1D or 2D).

    Returns
    -------
    bool
        True if the grid is regular, False otherwise.
    """
    if grid_lon.ndim == 2 and grid_lat.ndim == 2:
        # Check for consistent spacing along both dimensions
        lon_diff_row = np.diff(grid_lon, axis=1)
        lat_diff_col = np.diff(grid_lat, axis=0)
        if not (np.allclose(lon_diff_row, lon_diff_row[0, :]) and
                np.allclose(lat_diff_col, lat_diff_col[:, 0])):
            return False
    elif grid_lon.ndim == 1 and grid_lat.ndim == 1:
        # 1D arrays are regular by definition
        pass
    else:
        raise ValueError("grid_lon and grid_lat must be either both 1D or both 2D.")
    return True

# =================================================================================