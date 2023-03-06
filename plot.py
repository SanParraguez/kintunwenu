# -*- coding: utf-8 -*-
"""
=======================================================================
          KINTUN-WENU IUP
=======================================================================

The PLOT submodule contains some functions to easily plot some data
"""
__all__ = [
    'polycolor',
]
# === IMPORTS =========================================================
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection

from .polygons import get_coords_from_polygons

# === FUNCTIONS ===================================================================

def polycolor(polygons, values, ax=None, **kwargs):
    """
    Similar to Matplotlib pcolor, but receives a 1D-array of polygons and values to assign their colors.

    Warning: this is slow as hell.

    Parameters
    ----------
    polygons : iterable
        List of shapely.Polygon to be plotted
    values : iterable
        List or array with values to plot
    ax : plt.Axes
        Axes to plot polygons, if None a new fig and ax is created
    kwargs

    Returns
    -------
    plt.Axes
    """
    cmap = kwargs.pop('cmap', 'inferno')
    figsize = kwargs.pop('figsize', (10, 4.5))
    norm = kwargs.pop('norm', Normalize())
    projection = kwargs.pop('projection', ccrs.Robinson())
    title = kwargs.pop('title', None)
    label = kwargs.pop('label', None)

    smap = ScalarMappable(norm, cmap)
    collection = PolyCollection(get_coords_from_polygons(polygons), facecolor=smap.to_rgba(values),
                                transform=ccrs.PlateCarree())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, subplot_kw={"projection": projection})
        ax.coastlines(color='gray', alpha=0.8)
        ax.set_global()
        ax.set_facecolor('white')
        ax.add_collection(collection, autolim=False)
        fig.colorbar(smap, orientation='vertical', label=label, ax=ax)
        ax.set_title(title)
    else:
        ax.add_collection(collection, autolim=False)

    return ax

# =================================================================================
