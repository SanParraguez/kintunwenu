# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
---  Santiago Parraguez Cerda                       ---
---  University of Bremen, Germany                  ---
---  mail: sanparra@uni-bremen.de                   ---
---  2021                                           ---
=======================================================

Provides classes to handle atmospheric trace datasets.
Suitable for download, process and visualize satellite data.
"""
__name__ = 'kintunwenu'
__version__ = 'v0.1'
__release__ = '2021-12'

# ===== IMPORTS =======================================

from . import scrap
from . import processing
from .ges_disc import GesDiscProduct, GesDiscDataset

print("Kintun Wenu {version} ({release})".format(version=__version__, release=__release__))
# KINTUN WENU: from Mapudungún, means "Search in the sky"
