# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN WENU                   ===
=======================================================
---  Santiago Parraguez Cerda                       ---
---  University of Bremen, Germany                  ---
---  mail: sanparra@uni-bremen.de                   ---
---  2024                                           ---
=======================================================

Provides classes to handle atmospheric trace datasets.
Suitable for process, grid and visualize satellite data.
"""
__name__ = 'kintunwenu'
__version__ = '2.0'
__release__ = '2024-03'
__author__ = 'Santiago Parraguez Cerda'
__email__ = 'sanparra@uni-bremen.de'

__all__ = [
    'Kalkutun',
    'GridCrafter',
]

# ===== IMPORTS =======================================
from . import geodata
from . import grid
from . import nc_tools
from . import plot
from . import polygons
from . import scrap
from . import utils
from .kalkutun import Kalkutun
from .crafter import GridCrafter

# print("KintunWenu {version} ({release})".format(version=__version__, release=__release__))
# KINTUN WENU: from Mapuzungún, means "Search in the sky"
