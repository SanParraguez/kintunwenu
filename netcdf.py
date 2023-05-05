# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> NCDF

Contains functions for handling NetCDF files.
"""
__all__ = [
    'create_nc_file',
]

# === IMPORTS =========================================================
import os
from netCDF4 import Dataset

# === FUNCTIONS =======================================================

def create_nc_file(path, filename, dimensions=None, variables=None, attributes=None, clobber=True, verbose=1):
    """
    Easily creates a simple netCDF4 file. Can handle dimensions and variables from dictionaries.

    Parameters
    ----------
    path : str
        Directory in which the new file will be saved.
    filename : str
        Name of the file, should be supported by netCDF4.Dataset.
    dimensions : dict
        Dimensions of the nc file in format {name: value}.
    variables : dict[str, dict]
        Variables of the nc file in format {name: {attribute: value}}.
    attributes : dict
        Attributes of the nc file in format {name: value}.
    clobber : bool
        If False raise an Error when the file already exists, otherwise will be rewritten. Default: True
    verbose : int
        Activate verbose mode. Default: 1
    """
    dimensions = {} if not dimensions else dimensions
    variables = {} if not variables else variables
    attributes = {} if not attributes else attributes

    os.makedirs(path, exist_ok=True)
    filepath = path + filename

    with Dataset(filepath, 'w', format='NETCDF4', clobber=clobber) as ds:

        # Add dimensions to file
        for name, value in dimensions.items():
            ds.createDimension(name, value)

        # Add variables to file
        for name, value in variables.items():
            try:
                var_datatype = value.pop('datatype')
                var_dimensions = value.pop('dimensions')
            except KeyError:
                raise AttributeError('Both datatype and dimensions should be added to each variable')

            var = ds.createVariable(name, var_datatype, var_dimensions)
            var_value = value.pop('values', None)
            # TODO: Improve assigning method
            try:
                var[:] = var_value
            except IndexError:
                print('Avoided IndexError when assigning variable')
                var = var_value

            for val_name, val_value in value.items():
                setattr(var, val_name, val_value)

        # Add attributes to file
        for name, value in attributes.items():
            setattr(ds, name, value)

    if verbose > 0:
        print(f'Created file: {filepath}')

    return None

# =====================================================================
