# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> POLYGON

Submodule that contains the functions for regridding.
"""
__all__ = [
    'get_corners_from_coordinates',
    'get_coords_from_polygons',
    # 'shift_polygon_lon',
    'intersects_meridian',
    'create_meridian',
    'split_polygon_at_line',
    'is_over_pole',
]

# === IMPORTS =========================================================
import numpy as np
import shapely
from shapely.ops import split
from multiprocessing.pool import ThreadPool, Pool

# =================================================================================

def get_corners_from_coordinates(longitude, latitude, mode='center'):
    """
    Returns an array with dimensions (k, 4, 2), where k is the number of rectangles
    in a grid defined by the input coordinates. Each rectangle is represented by
    its four corners in (lon, lat) coordinates.

    Parameters
    ----------
    latitude : np.ndarray
        2D array (n,m) with center latitudes of a grid
    longitude : np.ndarray
        2D array (n,m) with center longitudes of a grid
    mode : str
        Indicates if the points given are 'center' or 'corner' of grid cell.

    Returns
    -------
    np.ndarray
        3D array (k, 4, 2) with the (lon, lat) coordinates of each rectangle's corners
    """
    mode = mode.lower()
    if mode in ['center', 'centers']:
        # Calculate the midpoints of the grid cells
        lon_mid = (longitude[:-1, :-1] + longitude[1:, 1:]) / 2
        lat_mid = (latitude[:-1, :-1] + latitude[1:, 1:]) / 2
    elif mode in ['corner', 'corners']:
        lon_mid = longitude
        lat_mid = latitude
    else:
        raise ValueError(f'Mode {mode} not recognized')

    # Create arrays of the corner coordinates for each rectangle
    # by stacking the midpoints of adjacent grid cells
    nw_corner = np.stack((lon_mid[:-1, :-1], lat_mid[:-1, :-1]), axis=-1)
    ne_corner = np.stack((lon_mid[:-1, 1:], lat_mid[:-1, 1:]), axis=-1)
    se_corner = np.stack((lon_mid[1:, 1:], lat_mid[1:, 1:]), axis=-1)
    sw_corner = np.stack((lon_mid[1:, :-1], lat_mid[1:, :-1]), axis=-1)

    # Combine the corner arrays into a single array
    corners = np.stack((nw_corner, ne_corner, se_corner, sw_corner), axis=-1)

    return np.moveaxis(corners.reshape((-1, 2, 4)), 1, -1)

# =================================================================================

def get_coords_from_polygons(polygons):
    """
    Returns a list of arrays with (n, 2) coordinates from a list of polygons.

    Parameters
    ----------
    polygons : iterable or shapely.geometry.Polygon
        The input polygons. Can be a single Polygon object or an iterable of Polygon objects.

    Returns
    -------
    list of numpy.ndarray
        A list of numpy arrays with shape (n, 2), where n is the number of points in the polygon.
        Each row of the array contains the x and y coordinates of a point in the polygon's boundary.
    """
    if isinstance(polygons, shapely.Polygon):
        polygons = np.array([polygons])

    coords, indices = shapely.get_coordinates(polygons, return_index=True)

    return np.split(coords, np.where(indices[1:] != indices[:-1])[0] + 1)

# =================================================================================

# def shift_polygon_lon(polygons, degrees=180):
#     """
#     Shifts all longitude negative points in a Polygon object by 180.
#     Just meant to deal with the antimeridian problem.
#
#     Parameters
#     ----------
#     polygons : shapely.geometry.Polygon or np.ndarray or pd.Series
#         The Polygon object to be shifted.
#     degrees :
#
#     Returns
#     -------
#     shapely.geometry.Polygon
#         The shifted Polygon object.
#     """
#     return shapely.transform(polygons, lambda x: x + (degrees, 0))

# =================================================================================

def intersects_meridian(polygons, meridian):
    """
    Check if a polygon intersects a given meridian.

    Parameters
    ----------
    polygons : shapely.geometry.Polygon
        The polygon to check.
    meridian : float
        Meridian to be checked.
    Returns
    -------
    bool
        True if the polygon intersects given meridian,
        False otherwise.
    """
    return shapely.intersects(polygons, create_meridian(meridian))

# =================================================================================

def create_meridian(longitude):
    """
    Creates a LineString representing a meridian at the given longitude.

    Parameters
    ----------
    longitude : float
        The longitude of the meridian.

    Returns
    -------
    shapely.geometry.LineString
        A LineString representing the meridian at the given longitude.

    """
    return shapely.LineString([(longitude, -90), (longitude, 90)])

# =================================================================================

def split_polygon_at_line(polygon, line):
    """
    Split a polygon at a given line.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to be cut.
    line : shapely.geometry.LineString
        The line to cut the polygon at.

    Returns
    -------
    list of shapely.geometry.Polygon:
        A list of polygons that are split at the given line.
    """
    return list(split(polygon, line).geoms)

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

    # Adjust the azimuths to be relative to North
    azimuths[1] += 180
    azimuths[azimuths > 180] -= 360

    # Flatten the array of azimuths and close the polygon by appending the azimuth between the last and first vertices
    azimuths = azimuths.T.flatten()
    azimuths = np.concatenate([azimuths, azimuths[:1]])

    # Determine the difference in azimuth between consecutive edges of the polygon
    diff = azimuths[1:] - azimuths[:-1]
    diff[diff > 180] -= 360

    # If the sum of the azimuth differences is zero, the polygon crosses over the pole
    return diff.sum() == 0.0

# =================================================================================

def split_anomaly_polygons(polygons, data=None, verbose=1):
    """Splits polygons that cross the anomaly of 180 == -180.

    Parameters
    ----------
    polygons : np.ndarray
    data : np.ndarray, optional
        Data to carry with the polygons.
    verbose : int, optional
        Verbosity level (default is 1).

    Returns
    -------
    list of shapely.geometry.Polygon or tuple of list and numpy.ma.array
        The list of the new polygons (and the masked data array, if input data was not None).
    """
    # ToDo: finish implementation
    antimeridian = create_meridian(180.)
    coords = get_coords_from_polygons(polygons)

    try:
        coords = np.array(coords, dtype=np.float64)
    except ValueError:
        raise ValueError('Polygons must have the same amount of corners.')

    # Get weirdly long polygons
    anomaly = (coords[:, :, 0].max(axis=1) - coords[:, :, 0].min(axis=1)) > 180

    new_coords = coords[anomaly]
    new_coords[:, :, 0][new_coords[:, :, 0] < 0] += 360
    new_polygons = shapely.polygons(new_coords)
    new_polygons = new_polygons[shapely.intersects(new_polygons, antimeridian)]
    new_polygons = np.array([split_polygon_at_line(poly, antimeridian) for poly in new_polygons])

    # new_polygons = polygons[anomaly]
    # new_coords = coords[anomaly]
    # polygons = polygons[~anomaly]
    # new_coords[:, :, 0][new_coords[:, :, 0] < 0] += 360
    # new_polygons_2 = shapely.get_rings(shapely.polygons(new_coords))

    if verbose > 0:
        print(f"Splitted into {len(coords)} polygons")

    return polygons
