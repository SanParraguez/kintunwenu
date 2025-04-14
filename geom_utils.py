"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> GEOM UTILS

Contains lower-level functions for handling geometrical data.
"""

__all__ = [
    'create_geo_dataset'
    'create_geo_grid',
    'get_corners_from_grid',
]

# === IMPORTS =========================================================

import numpy as np
import pandas as pd
import shapely
import geopandas as gpd

# =================================================================================

def get_corners_from_grid(lats, lons, mode='center'):
    """
    Returns an array with dimensions (k, 4, 2), where k is the number of rectangles
    in a grid defined by the input coordinates. Each rectangle is represented by
    its four corners in (x, y) coordinates.

    Parameters
    ----------
    lats : np.ndarray
        2D array (n,m) with either center or corner latitudes of a grid
    lons : np.ndarray
        2D array (n,m) with either center or corner longitudes of a grid
    mode : str
        Indicates if the points given are 'center' or 'corner' of grid cell.

    Returns
    -------
    np.ndarray
        Array of shape (k, 4, 2) with the corner coordinates.
    """
    mode = mode.lower()
    if mode not in ['center', 'centers', 'corner', 'corners']:
        raise ValueError(f"Mode '{mode}' not supported.")

    if lats.shape != lons.shape:
        raise ValueError(f"Latitude and longitude must have the same shape. Got {lat.shape}, {lon.shape}")

    if mode in ['center', 'centers']:
        lon_corners = np.stack([lons[:-1, :-1], lons[1:, 1:], lons[1:, :-1], lons[:-1, 1:]], axis=0)
        anomaly = (np.max(lon_corners, axis=0) - np.min(lon_corners, axis=0)) > 180
        lon_corners[np.tile(anomaly, (4, 1, 1)) * (lon_corners < 0)] += 360
        lon_corners = lon_corners.sum(axis=0) / 4
        lat_corners = (lats[:-1, :-1] + lats[1:, 1:] + lats[1:, :-1] + lats[:-1, 1:]) / 4
    elif mode in ['corner', 'corners']:
        lon_corners = lons
        lat_corners = lats
    else:
        raise Exception("Something went very wrong here")

    # Create arrays of the corner coordinates for each rectangle
    # by stacking the midpoints of adjacent grid cells
    nw_corner = np.stack((lon_corners[:-1, :-1], lat_corners[:-1, :-1]))
    ne_corner = np.stack((lon_corners[:-1, 1:], lat_corners[:-1, 1:]))
    se_corner = np.stack((lon_corners[1:, 1:], lat_corners[1:, 1:]))
    sw_corner = np.stack((lon_corners[1:, :-1], lat_corners[1:, :-1]))

    # Combine the corner arrays into a single array
    corners = np.stack((nw_corner, ne_corner, se_corner, sw_corner))
    corners = np.moveaxis(corners.reshape((4, 2, -1)), -1, 0)
    corners[corners > 180] -= 360  # Get coordinates back to where they should be

    return corners


# =================================================================================

def create_geo_dataset(geometries, **kwargs):
    """
    Creates a pandas DataFrame that combines Shapely Polygon objects and data values.

    Parameters
    ----------
    geometries : List[shapely.geometry.BaseGeometry]
        A list of Shapely Geometries objects representing geographical polygons.
    kwargs :
        Data to include in the DataFrame in list or np.ndarray or pd.Series.

    Returns
    -------
    A pandas DataFrame with two columns:
        - 'value': The data values associated with each polygon.
        - 'geometry': The Shapely Polygon objects representing geographical polygons.
    """
    kwargs.update({'geometry': geometries})
    if np.asarray(geometries).ndim > 1:
        df = [
            pd.DataFrame({key: val.tolist() for key, val in zip(kwargs.keys(), value)})
            for value in zip(*kwargs.values())
        ]
        return df

    # Create DataFrame with every variable assigned to its geometry
    #   v.tolist() trick avoids error when n-dimensional arrays stored in pandas.
    df = pd.DataFrame({k: v.tolist() for k, v in kwargs.items()})
    return df


# =================================================================================

def create_geo_grid(grid_lat, grid_lon, mode='corners', crs='WGS84'):
    """
    Generates a GeoDataFrame with grid cell polygons defined by latitude and longitude coordinates.

    Parameters
    ----------
    grid_lat : array-like
        Latitudes (in degrees).
    grid_lon : array-like
        Longitudes (in degrees).
    mode : str, optional
        'corners' (default) defines cells by corner coordinates.
    crs : str, optional
        The coordinate reference system (default: 'WGS84')

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with columns 'xi', 'yi', and 'geometry' for each grid cell.
    """
    lats = np.array(grid_lat)
    lons = np.array(grid_lon)

    if lats.ndim == 1 and lons.ndim == 1:
        # When building the grid, meshgrid is performed with lons and lats in standard order
        lons, lats = np.meshgrid(lons, lats, indexing='xy')
    elif lons.ndim == 2 and lats.ndim == 2:
        if lons.shape != lats.shape:
            raise ValueError(f"Shapes of grid_lat {lats.shape} and grid_lon {lons.shape} do not match.")
    else:
        raise ValueError(f"Arrays must be either both 1D or both 2D. Got {lats.shape} and {lats.shape}")

    cell_corners = get_corners_from_grid(lats, lons, mode=mode)
    polys_grid = shapely.polygons(cell_corners)

    grid_shape = lons.shape
    grid_xi = np.tile(np.arange(grid_shape[1] - 1), grid_shape[0] - 1)
    grid_yi = np.arange(grid_shape[0] - 1).repeat(grid_shape[1] - 1)

    df_grid = gpd.GeoDataFrame({
        'xi': grid_xi,
        'yi': grid_yi,
        'geometry': polys_grid
    }, crs=crs)
    return df_grid
