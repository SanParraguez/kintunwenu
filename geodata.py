"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> GEODATA

Submodule that contains functions to deal with geospatial datasets and geometries.
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
    'create_grid',
    'create_geo_grid',
    'is_regular_grid'
]

# === IMPORTS =========================================================

import numpy as np
import pandas as pd
import shapely
import pyproj
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
from shapely.geometry import Polygon
from .geom_utils import get_corners_from_grid


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
    north_band = Polygon([(-180, lat_thresh), (-180, 90), (180, 90), (180, lat_thresh)])
    south_band = Polygon([(-180, -lat_thresh), (-180, -90), (180, -90), (180, -lat_thresh)])
    df = df[~shapely.intersects(df['geometry'], north_band)]
    df = df[~shapely.intersects(df['geometry'], south_band)]
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
    a = np.array(a) if isinstance(a, (list, tuple)) else a
    b = np.array(b) if isinstance(b, (list, tuple)) else b

    if type(a) != type(b):
        raise TypeError(f"Geometry types don't match: {type(a)} vs {type(b)}")

    if threads is None or isinstance(a, shapely.Geometry):
        return shapely.intersection(a, b)

    chunksize = 1 + len(a) // threads
    if isinstance(a, pd.Series):
        chunks = [(a.iloc[i * chunksize:(i + 1) * chunksize], b.iloc[i * chunksize:(i + 1) * chunksize])
                  for i in range(threads)]
    else:
        chunks = [(a[i * chunksize:(i + 1) * chunksize], b[i * chunksize:(i + 1) * chunksize])
                  for i in range(threads)]

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
        # geod = Geod('+a=6378137 +f=0.0033528106647475126')
        geod = pyproj.CRS.from_epsg(4326).get_geod()

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
    # ToDo: change to accept any iterable and work with arrays

    if geod is None:
        # Default to WGS84 ellipsoid: proj = '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        # geod = Geod('+a=6378137 +f=0.0033528106647475126')
        geod = pyproj.CRS.from_epsg(4326).get_geod()

    if workers is None:
        over_pole = polygons.map(partial(is_over_pole, geod=geod))
    else:
        chunksize = 1 + len(polygons) // workers
        with Pool(processes=workers) as pool:
            over_pole = pool.map(partial(is_over_pole, geod=geod), polygons, chunksize=chunksize)
        over_pole = np.asarray(over_pole)

    return over_pole

# =================================================================================

def create_grid(grid_size, lon_lim=(-180, 180), lat_lim=(-90, 90), method='corners'):
    """
    Creates equally spaced grid cells.

    Parameters
    ----------
    grid_size : float or tuple[float, float]
        Size of the grid cells. If a float, a regular grid is assumed.
    lon_lim : tuple[float, float]
        Longitude limits of the grid.
    lat_lim : tuple[float, float]
        Latitude limits of the grid.
    method : str, optional
        Indicates if the points represent cell 'corners' (default) or centers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple with (lons, lats) as 1D arrays.
    """
    if isinstance(grid_size, (float, int)):
        grid_size = (grid_size, grid_size)
    else:
        grid_size = tuple(grid_size)

    if len(lat_lim) != 2 or len(lon_lim) != 2:
        raise AssertionError("lat_lim and lon_lim must be 2-tuples.")

    if method == 'corners':
        # Compute the number of grid cells in each direction
        nlon = int((lon_lim[1] - lon_lim[0]) / grid_size[0])
        nlat = int((lat_lim[1] - lat_lim[0]) / grid_size[1])
        # Adjust limits to match the exact cell size
        adj_lon_lim = (lon_lim[0], lon_lim[0] + nlon * grid_size[0])
        adj_lat_lim = (lat_lim[0], lat_lim[0] + nlat * grid_size[1])

        grid_lon = np.linspace(*adj_lon_lim, num=nlon + 1, endpoint=True)
        grid_lat = np.linspace(*adj_lat_lim, num=nlat + 1, endpoint=True)
    else:
        raise NotImplementedError(f"Method '{method}' is not supported.")

    return grid_lon, grid_lat


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
    grid_lat = np.asarray(grid_lat)
    grid_lon = np.asarray(grid_lon)

    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        lat_mesh, lon_mesh = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    elif grid_lat.shape == grid_lon.shape:
        lat_mesh = grid_lat
        lon_mesh = grid_lon
    else:
        raise ValueError(f"Latitude and longitude arrays must be same shape or 1D. Got {grid_lat.shape}, {grid_lon.shape}.")

    corners = get_corners_from_grid(lat_mesh, lon_mesh, mode=mode) # (n, 2) in lon, lat
    if corners.shape[-1] != 2:
        raise ValueError("Expected corner coordinates in (lon, lat) order with shape (..., 2)")

    # Validate coordinate order
    if np.abs(corners[..., 1]).max() > 90 and np.abs(corners[..., 0]).max() <= 90:
        raise ValueError("Coorners appear to be in (lat, lon) order, expected (lon, lat).")

    polys_grid = shapely.polygons(corners)

    shape = lat_mesh.shape
    grid_xi = np.tile(np.arange(shape[1] - 1), shape[0] - 1)
    grid_yi = np.arange(shape[0] - 1).repeat(shape[1] - 1)

    df_grid = gpd.GeoDataFrame({
        'xi': grid_xi,
        'yi': grid_yi,
        'geometry': polys_grid
    }, crs=crs)

    return df_grid

# =================================================================================

def is_regular_grid(grid_lat, grid_lon):
    """
    Checks if the provided latitude and longitude arrays define a regular grid.

    Parameters
    ----------
    grid_lon : np.ndarray
        Array of longitudes (1D or 2D).
    grid_lat : np.ndarray
        Array of latitudes (1D or 2D).

    Returns
    -------
    bool
        True if the grid is regular; otherwise, False.
    """
    if grid_lat.ndim == 2 and grid_lon.ndim == 2:
        lat_diff = np.diff(grid_lat, axis=0)
        lon_diff = np.diff(grid_lon, axis=1)
        return np.allclose(lat_diff, lat_diff[:, 0]) and np.allclose(lon_diff, lon_diff[0, :])
    elif grid_lat.ndim == 1 and grid_lon.ndim == 1:
        return True

    raise ValueError("grid_lat and grid_lon must both be 1D or both be 2D.")
