# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> POLYGON

Submodule that contains the functions to handle polygons.
"""
__all__ = [
    'create_meridian',
    'intersects_meridian',
    'get_coordinates_from_polygons',
    'get_corners_from_grid',
    'split_anomaly_polygons',
    'shift_polygons',
]

# === IMPORTS =========================================================
import numpy as np
import pandas as pd
import shapely
from shapely.ops import split
from .geodata import create_geo_dataset

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

def get_corners_from_grid(longitude, latitude, mode='center'):
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

def split_anomaly_polygons(polygons, data=None, to_dataframe=True):
    """Splits polygons that cross the antimeridian (180 degrees longitude).

    Parameters
    ----------
    polygons : np.ndarray or pd.Series
        Array of shapely.geometry.Polygon objects to split.
    data : np.ndarray or pd.Series or list or dict, optional
        Data to carry with the polygons. Default is None.
    to_dataframe : bool, optional
        Indicates if a dataframe is returned when possible.

    Returns
    -------
    list of shapely.geometry.Polygon or tuple of list and numpy.ma.array
        List of the new polygons if input data was None.
        Tuple of the new polygons and the masked data array if input data was not None.
    """
    if isinstance(polygons, pd.Series):
        polygons = polygons.to_numpy()
    elif isinstance(polygons, pd.DataFrame) and data is None:
        data = polygons.copy()
        polygons = data.pop('geometry')

    antimeridian = create_meridian(180.)

    coords = get_coordinates_from_polygons(polygons)
    # Try to create array, not possible if geometries have different number of points
    try:
        coords = np.array(coords, dtype=np.float64)
    except ValueError:
        pass

    # ---------- Get new polygons ----------
    if isinstance(coords, np.ndarray):

        # Get weirdly long polygons, assuming there should not be huge polygons
        anomaly = (coords[..., 0].max(axis=-1) - coords[..., 0].min(axis=-1)) > 180

        if not anomaly.any():
            if to_dataframe:
                return create_geo_dataset(polygons, **data)
            else:
                return polygons, data if data is not None else polygons

        # Get only coords from anomaly polygons
        coords, new_coords = coords[~anomaly], coords[anomaly]
        # Shift negative coordinates by 360
        new_coords[..., 0][new_coords[..., 0] < 0] += 360
        # Create new polygons
        polygons, new_polygons = shapely.polygons(coords), shapely.polygons(new_coords)

        # Split polygons and get the repeat index for further data return
        new_polygons = [split(poly, antimeridian) for poly in new_polygons]
        repeat_index = [len(poly.geoms) for poly in new_polygons] if data is not None else None

        # Shift coordinates back and create new polygons
        new_coords = get_coordinates_from_polygons(np.concatenate([list(poly.geoms) for poly in new_polygons]))

        try:
            new_coords = np.array(new_coords, dtype=np.float64)
            new_coords[(new_coords[:, :, 0] > 180).any(axis=1), :, 0] -= 360.
            new_polygons = shapely.polygons(new_coords)
        except ValueError:
            # new_coords = np.array(new_coords, dtype=object)
            for new_coord in new_coords:
                if (new_coord[:, 0] > 180).any():
                    new_coord[:, 0] -= 360

            new_polygons = np.array([shapely.polygons(new_coord) for new_coord in new_coords])

    else:
        raise NotImplementedError('Support for lists will be implemented')
    # -------------------------------

    if data is not None:

        if isinstance(data, pd.Series):
            data = data.to_numpy()
        elif isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='series')
        elif isinstance(data, list):
            data = np.asarray(data)

        if isinstance(data, np.ndarray):
            data, new_data = data[~anomaly], data[anomaly]
            new_data = np.ma.concatenate([np.tile(d, (n, 1)) for d, n in zip(new_data, repeat_index)]).squeeze()
            data = np.concatenate([data, new_data])
        elif isinstance(data, dict):
            for key, value in data.items():
                value = np.asarray(value)
                value, new_value = value[~anomaly], value[anomaly]
                new_value = np.ma.concatenate([np.tile(d, (n, 1)) for d, n in zip(new_value, repeat_index)]).squeeze()
                data[key] = np.concatenate([value, new_value])
        else:
            raise TypeError(f'Data type {type(data)} not supported')

    polygons = np.concatenate([polygons, new_polygons])

    if to_dataframe:
        return create_geo_dataset(polygons, **data)
    else:
        return polygons, data if data is not None else polygons

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
