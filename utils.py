# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> UTILS

Provides classes to handle different type of datasets retrieved from https://disc.gsfc.nasa.gov
"""
__all__ = [
    '_standardise_unit_string',
    '_smoothn_fill'
]

# ============= IMPORTS ===============

import numpy as np
from scipy.ndimage import label, sum_labels
from .smoothn import smoothn

# =================================================================================

def _standardise_unit_string(units):
    """
    Standardise units strings.
    """
    if units == 'mol m-2':
        units = 'mol/m2'
    elif units == 'cm^-2':
        units = 'molec/cm2'

    return units

# =================================================================================

def _smoothn_fill(data, tolerance=5, s=None, robust=False, grid=None):
    """
    Fill missing values with a smoothn method, based on the cosine discrete transform.
    Returns the filled array.

    Parameters
    ----------
    data : np.ma.MaskedArray
        n-dimensional data to be filled.
    tolerance : int
        Maximum number of pixels in gap to be filled.
    s : float
        Smoothing parameter, calculated automatically if not provided to minimize GCV score.
    robust : bool
        Indicates if a robust iteration is executed to avoid outliers.
    grid : tuple, list
        Grid dimensions, assumed regular if not given.
    """
    labeled, n_labels = label(data.mask, np.ones((3, 3), dtype=np.int16))
    sizes = sum_labels(np.ones_like(data), labeled, index=range(n_labels + 1))
    mask_size = sizes <= tolerance
    mask_clean = mask_size[labeled]

    z = data.copy()
    z[mask_clean] = smoothn(data, s, robust=robust, di=grid)[mask_clean]

    return z
