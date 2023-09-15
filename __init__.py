# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
---  Santiago Parraguez Cerda                       ---
---  University of Bremen, Germany                  ---
---  mail: sanparra@uni-bremen.de                   ---
---  2023                                           ---
=======================================================

Provides classes to handle atmospheric trace datasets.
Suitable for download, process and visualize satellite data.
"""
__name__ = 'kintunwenu'
__version__ = '1.2.1'
__release__ = '2023-09'

__all__ = [
    'Kalkutun',
    'GridCrafter',
]

# ===== IMPORTS =======================================
from . import geodata
from . import grid
from . import netcdf as nc
from . import plot
from . import polygons
from . import scrap
from . import utils
from .kalkutun import Kalkutun, GridCrafter

# print("KintunWenu {version} ({release})".format(version=__version__, release=__release__))
# KINTUN WENU: from Mapuzungún, means "Search in the sky"
