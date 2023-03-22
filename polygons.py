# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> POLYGON

Submodule that contains the functions for regridding.
"""
__all__ = [
    'get_coordinates_from_polygons',
    'get_corners_from_coordinates',
]

# === IMPORTS =========================================================
import numpy as np
import pandas as pd
import shapely
from shapely.ops import split

# =================================================================================

def get_coordinates_from_polygons(polygons):
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

def get_corners_from_coordinates(longitude, latitude, mode='center'):
    """
    Returns an array with dimensions (k, 4, 2), where k is the number of rectangles
    in a grid defined by the input coordinates. Each rectangle is represented by
    its four corners in (lon, lat) coordinates.

    Parameters
    ----------
    latitude : np.ndarray
        2D array (n,m) with either center or corner latitudes of a grid
    longitude : np.ndarray
        2D array (n,m) with either center or corner longitudes of a grid
    mode : str
        Indicates if the points given are 'center' or 'corner' of grid cell.

    Returns
    -------
    np.ndarray
        3D array (k, 4, 2) with the (lon, lat) coordinates of each rectangle's corners
    """
    mode = mode.lower()
    if mode in ['center', 'centers']:
        # Calculate the corners of the grid cells
        lon_corners = np.stack([longitude[:-1, :-1], longitude[1:, 1:], longitude[1:, :-1], longitude[:-1, 1:]], axis=0)
        anomaly = (np.max(lon_corners, axis=0) - np.min(lon_corners, axis=0)) > 180
        lon_corners[np.tile(anomaly, (4, 1, 1)) * (lon_corners < 0)] += 360
        lon_corners = lon_corners.sum(axis=0) / 4
        lat_corners = (latitude[:-1, :-1] + latitude[1:, 1:] + latitude[1:, :-1] + latitude[:-1, 1:]) / 4
    elif mode in ['corner', 'corners']:
        lon_corners = longitude
        lat_corners = latitude
    else:
        raise ValueError(f'Mode {mode} not recognized')

    # Create arrays of the corner coordinates for each rectangle
    # by stacking the midpoints of adjacent grid cells
    nw_corner = np.stack((lon_corners[:-1, :-1], lat_corners[:-1, :-1]))
    ne_corner = np.stack((lon_corners[:-1, 1:], lat_corners[:-1, 1:]))
    se_corner = np.stack((lon_corners[1:, 1:], lat_corners[1:, 1:]))
    sw_corner = np.stack((lon_corners[1:, :-1], lat_corners[1:, :-1]))

    # Combine the corner arrays into a single array
    corners = np.stack((nw_corner, ne_corner, se_corner, sw_corner))
    corners = np.moveaxis(corners.reshape((4, 2, -1)), -1, 0)

    corners[corners > 180] -= 360       # Get coordinates back to where they should be

    return corners

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

def cross_antimeridian(geometry):
    """
    Detects if a given geometry crosses the antimeridian (the 180-degree meridian).

    Parameters
    ----------
    geometry : shapely.Geometry or np.ndarray or pd.Series or list
        A geometry object or an array-like object of geometries to check.

    Returns
    -------
    list or np.ndarray
        An iterable indicating whether each geometry in the input list crosses the antimeridian.
        The output is a boolean array with the same length as the input geometry.

    Notes
    -----
    The function checks whether the geometry crosses the antimeridian by measuring the distance between the maximum
    and minimum longitude coordinates of the geometry. If the difference is greater than 180 degrees, the geometry is
    considered to cross the antimeridian. This method breaks if a polygon is big enough to cover half the Earth.

    This function supports different input types, including shapely.Geometry objects, numpy arrays, pandas Series, and
    lists.
    """
    if isinstance(geometry, np.ndarray):            # Case 0: Array [Polygon or list or ndarray]
        pass
    elif isinstance(geometry, shapely.Geometry):    # Case 1: Single Polygon
        geometry = np.array([geometry])
    elif isinstance(geometry, pd.Series):           # Case 2: Series [Polygon or list]
        geometry = geometry.to_numpy()
    elif isinstance(geometry, list):
        try:
            geometry = np.array(geometry, dtype=np.float64)     # Case 3: list with same dimensions
        except ValueError:
            pass                                                # Case 4: list with different dimensions

    if isinstance(geometry[0], shapely.Geometry):
        geometry = get_coordinates_from_polygons(geometry)

    try:
        geometry = np.array(geometry, dtype=np.float64)
    except ValueError:
        pass

    if isinstance(geometry, np.ndarray):
        anomaly = (geometry[:, :, 0].max(axis=1) - geometry[:, :, 0].min(axis=1)) > 180
    else:
        anomaly = [(geom[:, 0].max() - geom[:, 0].min()) > 180 for geom in geometry]

    return anomaly

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
    return diff.sum() == 0.0

# =================================================================================

def split_anomaly_polygons(polygons, data=None, verbose=1):
    """Splits polygons that cross the antimeridian (180 degrees longitude).

    Parameters
    ----------
    polygons : np.ndarray or pd.Series
        Array of shapely.geometry.Polygon objects to split.
    data : np.ndarray or pd.Series, optional
        Array of data to carry with the polygons. Default is None.
    verbose : int, optional
        Verbosity level. If greater than 0, print the number of polygons split. Default is 1.

    Returns
    -------
    list of shapely.geometry.Polygon or tuple of list and numpy.ma.array
        List of the new polygons if input data was None.
        Tuple of the new polygons and the masked data array if input data was not None.
    """
    if isinstance(polygons, pd.Series):
        polygons = polygons.to_numpy()
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    antimeridian = create_meridian(180.)
    coords = get_coordinates_from_polygons(polygons)
    new_data = None

    try:
        coords = np.array(coords, dtype=np.float64)
    except ValueError:
        pass

    if isinstance(coords, np.ndarray):
        # Get weirdly long polygons
        anomaly = (coords[:, :, 0].max(axis=-1) - coords[:, :, 0].min(axis=-1)) > 180

        if not anomaly.any():
            return (polygons, data) if data is not None else polygons

        if data is not None:
            old_polygons, old_data = polygons[~anomaly], data[~anomaly]
            coords, data = coords[anomaly], data[anomaly]
        else:
            old_polygons, coords = polygons[~anomaly], coords[anomaly]
            old_data = None

        # Create new polygons shifted by 360
        coords[:, :, 0][coords[:, :, 0] < 0] += 360.
        polygons = shapely.polygons(coords)

        polygons = polygons[shapely.intersects(polygons, antimeridian)]     # Just make sure that intersects
        polygons = [split(poly, antimeridian) for poly in polygons]     # Split new polygons
        new_coords = get_coordinates_from_polygons(polygons)  # Get coordinates to shift again

        # Shift coordinates backwards again
        try:
            new_coords = np.array(new_coords, dtype=np.float64)
            new_coords[(new_coords[:, :, 0] > 180.).any(axis=1), :, 0] -= 360.
        except ValueError:
            for new_coord in new_coords:
                if (new_coord[:, 0] > 180.).any():
                    new_coord[:, 0] -= 360.

        if data is not None:
            repeat_index = list(map(len, polygons))
            new_data = np.concatenate([np.tile(d, (n, 1)) for d, n in zip(data, repeat_index)], axis=0)

        polygons = shapely.polygons(new_coords)

    else:
        raise NotImplementedError('Support for lists will be implemented')

    if verbose > 0:
        print(f"Splitted into {len(old_polygons)+len(polygons)} polygons")

    if data is not None:
        return np.concatenate([old_polygons, polygons]), np.concatenate([old_data, new_data])
    else:
        return np.concatenate([old_polygons, polygons])

# =================================================================================

def shift_polygons(polygons, longitude):
    """
    Shifts the longitude of a list of polygons by a specified amount.

    Parameters
    ----------
    polygons : list or np.ndarray
        An iterable of `shapely.geometry.Polygon` objects to be shifted.
    longitude : float
        The amount by which the longitude of the polygons should be shifted.
        Positive values shift to the east, negative values shift to the west.

    Returns
    -------
    np.ndarray
        An array of `shapely.geometry.Polygon` objects with the longitude shifted.
    """
    coords = get_coordinates_from_polygons(polygons)

    try:
        coords = np.array(coords, dtype=np.float64)
        coords[:, :, 0] += longitude
    except ValueError:
        for coord in coords:
            coord[:, 0] += longitude

    return shapely.polygons(coords)
