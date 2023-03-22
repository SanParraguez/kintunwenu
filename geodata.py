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
    'get_intersections',
    'get_areas',
    'get_area'
]

# === IMPORTS =========================================================
import numpy as np
import pandas as pd
import shapely
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from pyproj import CRS, Geod
from shapely.geometry import Polygon
from .polygons import is_over_pole, get_corners_from_coordinates

# =================================================================================

def create_geo_dataset(polygons, data):
    """
    Creates a pandas DataFrame that combines Shapely Polygon objects and data values.

    Parameters
    ----------
    polygons : List[shapely.geometry.Polygon]
        A list of Shapely Polygon objects representing geographical polygons.
    data : List[float]
        A list of float values representing data associated with each polygon.

    Returns
    -------
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
    df : pandas.DataFrame or pandas.Series
        If a DataFrame, it must contain a 'polygon' column with shapely Polygon objects. If a Series, it must contain
        shapely Polygon objects.
    geod : pyproj.Geod, optional
        A Geod object that defines the ellipsoid to use for geodetic calculations. If not provided, an Equirectangular
        projection centered on the Prime Meridian will be used.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        If a DataFrame, it is the original DataFrame with the polygons that cross over the pole removed. If a Series,
        it is the original Series with the polygons that cross over the pole removed.
    """
    if not geod:
        proj = CRS.from_string('+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
        geod = proj.get_geod()

    series = df['polygon'] if isinstance(df, pd.DataFrame) else df
    over_pole = series.map(lambda poly: is_over_pole(poly, geod))

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

def get_intersections(a, b, threads=None):
    """
    Compute the intersections between geometries contained in two arrays.

    Parameters
    ----------
    a : np.ndarray or pd.Series or list or tuple or shapely.geometry.Geometry
        First array of geometries
    b : np.ndarray or pd.Series or list or tuple or shapely.geometry.Geometry
        Second array of geometries
    threads : int, optional
        The number of threads to use for parallel processing.
        If not provided or set to None, the computation will be performed sequentially.

    Returns
    -------
    np.ndarray or pd.Series
        The intersections between the two arrays or LineStrings.
    """
    # Convert to arrays
    a = np.array([a]) if isinstance(a, shapely.Geometry) else a
    b = np.array([b]) if isinstance(b, shapely.Geometry) else b
    a = np.array(a) if isinstance(a, (list, tuple)) else a
    b = np.array(b) if isinstance(b, (list, tuple)) else b

    if type(a) != type(b):
        # Raise warning if different types of objects are passed
        raise TypeError(f'Unexpected behavior could arise when indexing different type of objects '
                        f'[{type(a)}, {type(b)}]')

    if threads is None:
        intersections = shapely.intersection(a, b)

    else:
        chunksize = 1 + len(a) // threads
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            chunks = [(a.iloc[i * chunksize:(i + 1) * chunksize], b.iloc[i * chunksize:(i + 1) * chunksize]) for i in
                      range(threads)]
        else:
            chunks = [(a[i * chunksize:(i + 1) * chunksize], b[i * chunksize:(i + 1) * chunksize]) for i in
                      range(threads)]

        with ThreadPool(processes=threads) as pool:
            intersections = pool.starmap(shapely.intersection, chunks)

        intersections = pd.concat(intersections) if isinstance(a, pd.Series) else np.concatenate(intersections)

    return intersections

# =================================================================================

def get_areas(polys, geod=None, workers=None):
    """
    Computes the area of each polygon in a Pandas Series of Shapely polygons using either a single process or multiple
    processes.

    Parameters
    ----------
    polys : pd.Series of shapely.geometry.Polygon
        Series of input polygons.
    geod : pyproj.Geod, optional
        Geodetic calculator object, defaults to None (i.e., use the WGS84 ellipsoid).
    workers : int, optional
        Number of worker processes to use for parallel processing, defaults to None (i.e., single-process mode).

    Returns
    -------
    pd.Series of float
        Series of polygon areas.
    """
    if geod is None:
        # Default to WGS84 ellipsoid: proj = '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        geod = Geod('+a=6378137 +f=0.0033528106647475126')

    if workers is None:
        areas = polys.map(partial(get_area, geod=geod))
    else:
        chunksize = 1 + len(polys)//workers
        with Pool(processes=workers) as pool:
            areas = pool.map(partial(get_area, geod=geod),
                             polys, chunksize=chunksize)

    return pd.Series(areas, index=polys.index)

# =================================================================================

def get_area(polygon, geod):
    """
    Computes the area of a single Shapely polygon using the given geodetic calculator object.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input polygon.
    geod : pyproj.Geod
        Geodetic calculator object.

    Returns
    -------
    float
        Area of the polygon.
    """
    return geod.geometry_area_perimeter(polygon)[0]
