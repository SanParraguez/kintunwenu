<p align="center" width="100%"><img src="/bin/banner.svg/" height="100" align="center"></p>

---
<div align="center">
<img src="https://img.shields.io/github/v/release/SanParraguez/kintunwenu?include_prereleases" alt="GitHub release (latest by date including pre-releases)">
<img src="https://img.shields.io/github/last-commit/SanParraguez/kintunwenu" alt="GitHub last commit">
<img src="https://img.shields.io/github/license/SanParraguez/kintunwenu" alt="GitHub license">
<img src="https://img.shields.io/github/downloads/SanParraguez/kintunwenu/total" alt="GitHub downloads">
</div>

Provides classes to handle satellite datasets. Allows process, regrid and visualize satellite data.

## Features

- Easy to add new product routines
- Vectorized gridding routine

### Supported products

- TROPOMI WFMD CH<sub>4</sub> v1.8 
- TROPOMI L2 NO<sub>2</sub>

## Basic usage

In your local directory just clone this repo

```shell
git clone git@github.com:SanParraguez/kintunwenu.git
```

If needed, you can also install the provided environment

```shell
conda env create --name ktw --file kintunwenu/bin/environment.yml
conda activate ktw
```

### Gridding class

```python
import numpy as np
import kintunwenu as ktw
from pyproj import CRS

# create grid
lons = np.linspace(-20,  15, 61)
lats = np.linspace(-50, -25, 61)

# create geodetic object for distance measurements
geod = CRS.from_string('+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs').get_geod(),

# path to my product file
path_to_file = "my/path/to/netcdf/file.nc"

# initialize gridder
gridder = ktw.GridCrafter(
    grid_lons=lons,
    grid_lats=lats,
    min_fill=0.5,
    qa_filter=0.75,
    geod=geod,
)

# grid
product = gridder(path_to_file)
```

## How to contribute

I'm still not sure, so any suggestion will be welcomed!

## ToDo

- [ ] Write proper README

## Dependencies

- Python $\geq$ 3.10
- [Numpy](https://github.com/numpy/numpy)
- [Pandas](https://github.com/pandas-dev/pandas)
- [Scipy](https://github.com/scipy/scipy)
- [NetCDF4](https://github.com/Unidata/netcdf4-python)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [Cartopy](https://github.com/SciTools/cartopy)
- [Shapely](https://github.com/shapely/shapely) $\geq$ 2.0
- [Requests](https://github.com/psf/requests)

## Contributors

- Santiago Parraguez Cerda
