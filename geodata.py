# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> GEODATA

Submodule that contains functions to deal with pd.DataFrame and pd.Series
"""
__all__ = [
    'create_geo_dataset',
    'create_geo_grid',
    'filter_over_pole',
    'filter_by_latitude',
    'split_polygons_at_meridian'
]

# === IMPORTS =========================================================
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely.geometry import Polygon
from .polygons import is_over_pole, create_meridian, get_corners_from_coordinates

# =================================================================================

def create_geo_dataset(polygons, data):
    """
    Creates a pandas DataFrame that combines Shapely Polygon objects and data values.

    Parameters
    ------------
    polygons : List[shapely.geometry.Polygon]
        A list of Shapely Polygon objects representing geographical polygons.
    data : List[float]
        A list of float values representing data associated with each polygon.

    Returns
    --------
    A pandas DataFrame with two columns:
        - 'value': The data values associated with each polygon.
        - 'polygon': The Shapely Polygon objects representing geographical polygons.
    """
    df = pd.DataFrame({'value': data, 'polygon': polygons})
    return df

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
        get_corners_from_coordinates(lons, lats, mode=mode)
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

def filter_over_pole(df, geod=None):
    """
    Filters out polygons that cross over the pole.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing a 'polygon' column with shapely Polygon objects.
    geod : pyproj.Geod, optional
        A Geod object that defines the ellipsoid to use for geodetic calculations. If not provided, an Equirectangular
        projection centered on the Prime Meridian will be used.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with the polygons that cross over the pole removed.
    """
    # ToDo: implementation for pd.Series.

    if not geod:
        proj = CRS.from_string('+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
        geod = proj.get_geod()

    over_pole = df['polygon'].map(lambda poly: is_over_pole(poly, geod))

    return df[~over_pole]

# =================================================================================

def filter_by_latitude(df, lat_thresh):
    """
    Filter a dataframe of polygons by latitude. Notice that this method is around one
    order of magnitude faster that 'filter_over_pole', but this method is less safe, use it if you know the
    size of your polygons.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of polygons with a 'polygon' column of
        shapely.geometry.Polygon objects.
    lat_thresh : float
        A latitude threshold to filter by. Polygons that intersect
        with a latitude band between `lat_thresh` and the corresponding pole are removed.

    Returns
    -------
    pandas.DataFrame
        A filtered dataframe with polygons that do not intersect
        with the latitude bands.

    """
    # ToDo: implementation for pd.Series.

    # Create latitude bands using shapely Polygon objects
    north_pole_band = Polygon([(-180, lat_thresh), (-180, 90), (180, 90), (180, lat_thresh)])
    south_pole_band = Polygon([(-180, -lat_thresh), (-180, -90), (180, -90), (180, -lat_thresh)])

    # Filter polygons that intersect with the latitude bands
    df = df[~shapely.intersects(df['polygon'], north_pole_band)]
    df = df[~shapely.intersects(df['polygon'], south_pole_band)]

    return df

# =================================================================================

def split_polygons_at_meridian(df, meridian):
    """
    Split polygons in a pandas DataFrame at a given meridian.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing a "polygon" column of Shapely polygons.
    meridian : float

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing the original and the split polygons.
    """
    # ToDo: implementation for pd.Series.

    meridian = create_meridian(meridian)

    # Get which polygons intersect meridian
    intersects = df['polygon']

    # Split each polygon in the DataFrame
    # splitted_polygons = df['polygon'].apply(lambda poly: split(poly, meridian))

    return
