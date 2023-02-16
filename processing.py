# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> PROCESSING

Provides tools to manage GesDiscProduct and generate datasets.
"""
__all__ = [
    'regular_grid_data',
    '_generate_regular_grid_points']

# ============= IMPORTS ===============

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# =================================================================================

def regular_grid_data(x, y, z, xlim, ylim, grid_space, method='nearest'):
    """
    Interpolate into a regular grid, based on specified limits.
    Three methods are accepted: "nearest", "linear" and "cubic".

    Parameters
    ----------
    x : np.ma.MaskedArray
        Coordinates along the first axis.
    y : np.ma.MaskedArray
        Coordinates along the second axis.
    z : np.ma.MaskedArray
        Data to be gridded.
    xlim : tuple, list
        Iterable with x limits of the grid with shape (2,)
    ylim : tuple, list
        Iterable with y limits of the grid with shape (2,)
    grid_space : float, tuple, list
        A single value or an iterable with grid steps along each axis.
        If a single value is given, it assumes same step for x and y.
    method : str
        Interpolation method, can be 'nearest', 'linear' or 'cubic'.
    """
    if isinstance(grid_space, (float, int)):
        grid_space = (grid_space, grid_space)

    x, y, z = _cut_product_by_coordinates(x, y, z, xlim, ylim, margin=2)

    points = np.array((x[~z.mask], y[~z.mask])).T
    coord_points = np.array((x.compressed(), y.compressed())).T
    mask_map = z.mask[~x.mask] * 1 if np.ma.is_masked(x) else z.mask.flatten() * 1

    grid_points = _generate_regular_grid_points(xlim, ylim, grid_space)
    x, y = grid_points
    grid_points_flat = np.array((x.ravel(), y.ravel())).T

    if z.count() < 4:
        return x, y, np.ma.masked_where(True, x)

    z_grid = griddata(points, z.compressed(), grid_points, method=method)
    mask_grid = griddata(coord_points, mask_map, grid_points, method='linear')
    mask_grid[mask_grid > .5] = 1
    mask_grid[mask_grid <= .5] = 0

    threshold = np.sqrt(grid_space[0]**2 + grid_space[1]**2)/2
    distance, _ = KDTree(coord_points).query(grid_points_flat, workers=-1)
    outside = distance.reshape(z_grid.shape) > threshold

    mask = np.isnan(mask_grid) | (mask_grid == 1) | np.isnan(z_grid) | outside
    z = np.ma.masked_array(z_grid, mask=mask)

    return x, y, z

def _generate_regular_grid_points(xlim, ylim, grid_space) -> (np.ndarray, np.ndarray):
    """
    Creates a rectangular regular grid and returns both x and y matrices.

    Parameters
    ----------
    xlim : tuple, list
        First dimension minimum and maximum values.
    ylim : tuple, list
        Second dimension minimum and maximum values.
    grid_space : float, tuple, list
        Grid space, if a number it
    """
    if isinstance(grid_space, (float, int)):
        grid_space = (grid_space, grid_space)

    xi_grid = np.arange(xlim[0]+grid_space[0]/2, xlim[1], grid_space[0])
    yi_grid = np.arange(ylim[0]+grid_space[1]/2, ylim[1], grid_space[1])
    xi_grid, yi_grid = np.meshgrid(xi_grid, yi_grid)

    return xi_grid, yi_grid

def _cut_product_by_coordinates(x, y, z, xlim, ylim, margin=0):
    """
    Cut a given matrix to xlim and ylim by its coordinates x and y.
    A margin can be used, given in number of rows/columns.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates, used to cut the matrices.
    y : np.ndarray
        y-coordinates, used to cut the matrices.
    z : np.ndarray
        2D-array to be cut.
    xlim : list, tuple
        [x_min, x_max], indicating horizontal limits.
    ylim : list, tuple
        [y_min, y_max], indicating vertical limits.
    margin : int
        Positive integer indicating number of added rows/columns as margin.
    """
    pts_inside = np.argwhere((x >= xlim[0]) * (x <= xlim[-1]) * (y >= ylim[0]) * (y <= ylim[-1]))
    if pts_inside.size == 0:
        return np.ma.empty(0), np.ma.empty(0), np.ma.empty(0)
    else:
        mins, maxs = pts_inside.min(axis=0), pts_inside.max(axis=0)
        row_min, col_min, row_max, col_max = mins[0], mins[1], maxs[0], maxs[1]

        row_min = np.maximum(0, row_min-margin)
        col_min = np.maximum(0, col_min-margin)
        row_max += margin
        col_max += margin

        new_x = x[row_min:row_max+1, col_min:col_max+1]
        new_y = y[row_min:row_max+1, col_min:col_max+1]
        new_z = z[row_min:row_max+1, col_min:col_max+1]

        return new_x, new_y, new_z
