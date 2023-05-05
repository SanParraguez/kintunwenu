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
    'filter_over_pole',
    'filter_by_latitude',
    'get_intersections',
    'get_area',
    'get_areas',
    'is_over_pole',
    'are_over_pole',
]

# === IMPORTS =========================================================
import numpy as np
import pandas as pd
import shapely
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from pyproj import Geod
from shapely.geometry import Polygon

# =================================================================================

def create_geo_dataset(geometries, **kwargs):
    """
    Creates a pandas DataFrame that combines Shapely Polygon objects and data values.

    Parameters
    ----------
    geometries : List[shapely.geometry.BaseGeometry]
        A list of Shapely Geometries objects representing geographical polygons.
    kwargs : list or np.ndarray or pd.Series
        Data to include in the DataFrame

    Returns
    -------
    A pandas DataFrame with two columns:
        - 'value': The data values associated with each polygon.
        - 'geometry': The Shapely Polygon objects representing geographical polygons.
    """
    kwargs.update({'geometry': geometries})
    df = pd.DataFrame(kwargs)
    return df

# =================================================================================

def filter_over_pole(df, geod=None, workers=None):
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
    workers : int, optional
        Number of worker processes to use for parallel processing, defaults to None (i.e., single-process mode).

    Returns
    -------
    pandas.DataFrame or pandas.Series
        If a DataFrame, it is the original DataFrame with the polygons that cross over the pole removed. If a Series,
        it is the original Series with the polygons that cross over the pole removed.
    """
    series = df['geometry'] if isinstance(df, pd.DataFrame) else df
    over_pole = are_over_pole(series, geod=geod, workers=workers)

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
    df = df[~shapely.intersects(df['geometry'], north_pole_band)]
    df = df[~shapely.intersects(df['geometry'], south_pole_band)]

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

    # Raise warning if different types of objects are passed
    if type(a) != type(b):
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

# =================================================================================

def get_areas(polygons, geod=None, workers=None):
    """
    Computes the area of each polygon in a Pandas Series of Shapely polygons using either a single process or multiple
    processes.

    Parameters
    ----------
    polygons : pd.Series of shapely.geometry.Polygon
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
        areas = polygons.map(partial(get_area, geod=geod))
    else:
        chunksize = 1 + len(polygons)//workers
        with Pool(processes=workers) as pool:
            areas = pool.map(partial(get_area, geod=geod), polygons, chunksize=chunksize)

    return pd.Series(areas, index=polygons.index)

# =================================================================================

def is_over_pole(polygon, geod):
    """
    Determines if a shapely Polygon object crosses over one of the poles.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to test.
    geod : pyproj.Geod
        A Geod object that defines the ellipsoid to use for geodetic calculations.

    Returns
    -------
    bool
        True if the polygon crosses over the pole, False otherwise.
    """
    lons, lats = polygon.boundary.xy

    # Determine the azimuths between consecutive vertices of the polygon
    azimuths = np.array(geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:]))[:2]

    # Adjust the azimuths to the desired direction
    azimuths[1] += 180
    azimuths[azimuths > 180] -= 360

    # Flatten the array of azimuths and close the polygon by appending the azimuth between the last and first vertices
    azimuths = azimuths.T.flatten()
    azimuths = np.concatenate([azimuths, azimuths[:1]])

    # Determine the difference in azimuth between consecutive edges of the polygon
    diff = azimuths[1:] - azimuths[:-1]
    diff[diff > 180] -= 360
    diff[diff < -180] += 360

    # If the sum of the azimuth differences is zero, the polygon crosses over the pole
    return np.isclose(diff.sum(), 0.0)

# =================================================================================

def are_over_pole(polygons, geod=None, workers=None):
    """
    Takes a list of polygons and checks if they cross over the North or South Pole.
    It returns a boolean array indicating which polygons cross over the pole.

    Parameters
    ----------
    polygons : pd.Series
        Shapely Polygon objects to be checked if they cross over the pole.
    geod : pyproj.Geod
        Geographic projection to be used. If not provided, the function uses the WGS84 ellipsoid.
    workers : int
        Number of worker processes to use when checking the polygons.
        If not provided, the function runs in a single process.

    Returns
    -------
    np.ndarray
        Boolean array indicating which polygons cover any of the poles.
    """
    if geod is None:
        # Default to WGS84 ellipsoid: proj = '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        geod = Geod('+a=6378137 +f=0.0033528106647475126')

    if workers is None:
        over_pole = polygons.map(partial(is_over_pole, geod=geod))
    else:
        chunksize = 1 + len(polygons) // workers
        with Pool(processes=workers) as pool:
            over_pole = pool.map(partial(is_over_pole, geod=geod), polygons, chunksize=chunksize)
        over_pole = np.asarray(over_pole)

    return over_pole

# =================================================================================
