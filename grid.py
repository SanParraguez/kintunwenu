"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> GRID

Submodule that contains the functions for regridding.
"""

__all__ = [
    'weighted_regrid',
]

# === IMPORTS =========================================================

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from datetime import datetime
from .geom_utils import create_geo_grid
from .geodata import is_regular_grid


# =============================================================================
# Weighted regrid
# =============================================================================

def weighted_regrid(grid_lat, grid_lon, grid_dims, polygons, data, min_fill=None, crs='WGS84', **kwargs):
    """
    Performs a weighted regridding of polygons into a given regular grid.

    Parameters
    ----------
    grid_lat : np.ndarray
        Gridded latitude corners (1D or 2D).
    grid_lon : np.ndarray
        Gridded longitude corners (1D or 2D).
    grid_dims : tuple[str]
        Names of the dimensions to grid.
    polygons : array-like of shapely.Polygon
        Polygons corresponding to be gridded.
    data : dict
        Dictionary of data variables. Each key maps to a dict with keys:
            'values': array-like, containing polygon data, and
            'dims': tuple of dimension names.
    min_fill : float, optional
        Minimum area fraction required to assign value to a grid cell.
    crs : str, optional
        Coordinate reference system (default: 'WGS84').

    Returns
    -------
    dict
        Dictionary with regridded variables per key, each as a dict with 'dims' and 'values'.
    """
    # Validate inputs and standardize polygons
    polygons = validate_inputs(grid_lat, grid_lon, grid_dims, polygons, data, min_fill)

    # Prepare geodata: polygons GeoSeries and grid GeoDataFrame
    gdf_polys, gdf_grid, intersections = prepare_geodata(polygons, crs, grid_lat, grid_lon)
    if intersections is None:
        logging.warning("No intersections were found between the grid and polygons.")
        return None

    # Compute grid cell coverage
    regular_grid = is_regular_grid(grid_lat, grid_lon)
    gdf_grid, gdf_inter = compute_grid_coverage(gdf_grid, gdf_polys, intersections, regular_grid=regular_grid)

    # valid_mask selects only the polygons kept after filtering invalid ones
    # This seems redundant, already done in prepare_geodata()
    valid_mask = gdf_polys.is_valid.to_numpy()

    # Process each variable and regrid
    out_data = {}
    for key, var in data.items():
        regridded = process_variable(var, grid_dims, grid_lat, grid_lon, valid_mask, gdf_inter, gdf_grid)
        out_data[key] = {
            'dims': ('lat', 'lon') + tuple(d for d in var['dims'] if d not in grid_dims),
            'values': regridded
        }
        logging.info(f"Variable '{key}' regridded. Output shape: {regridded.shape}")

    # Apply minimum fill: set cells with too low coverage to masked
    if min_fill is not None:
        # Convert grid_geo_df index back to 2D array shape
        coverage = gdf_grid['coverage'].unstack(fill_value=0).values.T
        for key in out_data:
            vals = out_data[key]['values']
            np.ma.masked_where(coverage < min_fill, vals, copy=False)
            out_data[key]['values'] = vals

    return out_data


# =============================================================================
# Helper functions
# =============================================================================

def validate_inputs(grid_lat, grid_lon, grid_dims, polygons, data, min_fill):
    if not is_regular_grid(grid_lat, grid_lon):
        raise NotImplementedError("Irregular grids are not supported yet.")

    # Standardize polygons: from list or pd.Series to np.array
    if isinstance(polygons, list):
        polygons = np.array(polygons)
    elif isinstance(polygons, pd.Series):
        polygons = polygons.to_numpy()

    # For now we require that polygon dimensions match the number of grid dimensions
    if polygons.ndim != len(grid_dims):
        raise NotImplementedError("Multiple retrievals not implemented. polygons.ndim must match len(grid_dims).")

    # Validate data structure
    if not isinstance(data, dict):
        raise TypeError("Data must be a dict of {'var_name': {'values': ..., 'dims': ...}}.")

    for key, var in data.items():
        if 'values' not in var or 'dims' not in var:
            raise ValueError(f"Variable '{key}' missing 'values' or 'dims'.")
        if not set(var['dims']).issuperset(grid_dims):
            raise ValueError(f"Variable '{key}' must contain grid dimensions {grid_dims}.")

    # Validate min_fill if provided
    if min_fill is not None and not (0.0 < min_fill < 1.0):
        raise ValueError("min_fill must be a float between 0 and 1.")

    return polygons


def prepare_geodata(polygons, crs, grid_lat, grid_lon):
    """Prepare the GeoSeries for the polygons and the GeoDataFrame for the grid."""
    # Build GeoSeries and filter invalid polygons
    gdf_polys = gpd.GeoSeries(polygons.flatten(), crs=crs)
    valid_mask = gdf_polys.is_valid
    gdf_polys = gdf_polys[valid_mask]

    # Create grid GeoDataFrame
    gdf_grid = create_geo_grid(grid_lat, grid_lon, mode='corners', crs=crs)

    # Build spatial index and query intersections
    intersections = gdf_grid['geometry'].sindex.query(gdf_polys)

    if intersections.size == 0:
        return None, None, None

    logging.info(f"Intersections: {len(intersections[0])} polygon indices and {len(intersections[1])} grid cell indices.")
    return gdf_polys, gdf_grid, intersections


def compute_grid_coverage(gdf_grid, gdf_polys, intersections, regular_grid=True):
    """Compute cell areas and coverage fraction."""
    # Compute grid cell area in square meters
    if regular_grid:
        gdf_grid['cell_area'] = gdf_grid[gdf_grid['xi'] == 0].to_crs(epsg=6933).area
        gdf_grid['cell_area'] = gdf_grid['cell_area'].ffill()
    else:
        gdf_grid['cell_area'] = gdf_grid.to_crs(epsg=6933).area

    # Prepare a GeoDataFrame with intersecting grid cells
    gdf_inter = gdf_grid.loc[intersections[1], ['xi', 'yi', 'geometry']].copy()
    # Associate each intersection with the corresponding polygon index
    gdf_inter['poly_idx'] = intersections[0]

    # Compute intersections between the grid cells and the polygons
    # Note: The order of gdf_polys is assumed to be preserved
    gdf_inter['geometry'] = gdf_inter['geometry'].intersection(
        gdf_polys.loc[intersections[0]].reset_index(drop=True), align=False
    )

    # Compute area of intersection
    gdf_inter['inter_area'] = gdf_inter['geometry'].to_crs(epsg=6933).area

    # Set grid index to (xi, yi) and compute total intersection per cell
    gdf_grid = gdf_grid.set_index(['xi', 'yi'])
    inter_summary = gdf_inter.drop(['geometry', 'poly_idx'], axis=1).groupby(['xi', 'yi']).sum()
    gdf_grid['inter_area'] = inter_summary['inter_area']
    gdf_grid['coverage'] = gdf_grid['inter_area'] / gdf_grid['cell_area']

    return gdf_grid, gdf_inter


def process_variable(var, grid_dims, grid_lat, grid_lon, valid_mask, gdf_inter, gdf_grid):
    """
    Process a single variable from data.

    The function will:
      - Reorder the axes so that grid dimensions come first.
      - Flatten the grid-dimension data (filtering by valid polygons).
      - For each extra dimension (or a singleton axis if 1D), compute the weighted sum using intersection areas.
    Returns the regridded array.
    """
    # If needed, convert datetime values into numeric seconds
    arr = np.asarray(var['values'])
    if np.issubdtype(arr.dtype, np.datetime64):
        arr = (arr - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    dims = var['dims']

    # Identify grid-dimension indices and extra (other) dimensions
    grid_dim_indices = [dims.index(dim) for dim in grid_dims]
    other_dim_indices = [i for i in range(len(dims)) if i not in grid_dim_indices]
    logging.info(f"Processing variable with grid indices {grid_dim_indices} and extra indices {other_dim_indices}")

    # Reorder axes: put grid dims first, then other dims
    reordering = grid_dim_indices + other_dim_indices
    reordered = np.transpose(arr, axes=reordering)
    grid_shape = reordered.shape[:len(grid_dims)]
    extra_shape = reordered.shape[len(grid_dims):]

    logging.info(f"Grid shape: {grid_shape}, extra shape: {extra_shape}")

    # Flatten across grid dimensions while preserving extra dims
    flat_values = reordered.reshape((-1,) + extra_shape)
    # Filter to the valid polygons only
    flat_values = flat_values[valid_mask]

    # Ensure there is always at least one extra axis: if extra_shape is empty, add a singleton dimension
    if flat_values.ndim == 1:
        flat_values = flat_values[:, np.newaxis]
        extra_shape = (1,)

    out_shape = (grid_lat.size - 1, grid_lon.size - 1) + extra_shape
    regridded = np.ma.masked_all(out_shape)

    # Process each extra-dim slice uniformly
    for idx in np.ndindex(extra_shape):
        slice_vals = flat_values[:, idx].squeeze()
        logging.info(f"Processing slice {idx} with shape {slice_vals.shape}")

        # Create temporary DataFrame for aggregation
        gdf_data = gdf_inter[['xi', 'yi']].copy()
        # Multiply each polygon's value with the intersect area
        # Note: It is assumed that gdf_inter['poly_idx'] indexes into slice_vals
        gdf_data['weighted_value'] = slice_vals[gdf_inter['poly_idx']] * gdf_inter['inter_area']

        # Group by grid cell (xi, yi) and compute weighted sum
        weighted_sum = gdf_data.groupby(['xi', 'yi'])['weighted_value'].sum()
        # Normalize by coverage fraction; fill missing values with 0
        norm_vals = (weighted_sum / (gdf_grid['coverage'] * gdf_grid['cell_area']))

        # Unstack to create a masked 2D grid array
        reshaped = np.ma.masked_invalid(norm_vals.unstack().values.T)

        # Assign results to the appropriate slice
        regridded[..., idx] = reshaped[..., None]

    # If extra_shape was a singleton dimension, remove that axis
    if extra_shape == (1,):
        regridded = regridded[..., 0]
    return regridded
