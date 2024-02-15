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
    'WriteGridFile'
]

# === IMPORTS =========================================================

import os
import logging
import numpy as np
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
def WriteGridFile(filename, data, gridder, indent='', ncattrs=None, sup_vars=None, sup_dims=None):
    """
    Write gridded file
    """
    # create directory
    dirname = os.path.dirname(filename)
    if len(dirname) > 0 and not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)

    # check if ncattrs provided
    if not ncattrs:
        ncattrs = {}

    # write file
    with Dataset(filename, 'w', format="NETCDF4") as ds:

        # assign attributes
        for k, v in ncattrs.items():
            setattr(ds,  k, v)

        # ToDo: change hardcoded number of layers to a general value
        longitude = ds.createDimension('longitude', gridder.lons.shape[0] - 1)
        latitude = ds.createDimension('latitude', gridder.lats.shape[0] - 1)
        layer = ds.createDimension('layer', 34)
        level = ds.createDimension('level', 35)
        timed = ds.createDimension('time', 1)
        nv = ds.createDimension('nv', 2)

        lons = ds.createVariable('longitude', 'float32', ('longitude',))
        lons.standard_name = 'longitude'
        lons.units = 'degrees_east'
        lons.bounds = 'longitude_bounds'
        lons[:] = (gridder.lons[:-1] + gridder.lons[1:]) / 2

        lon_bounds = ds.createVariable('longitude_bounds', 'float32', ('nv', ))
        lon_bounds.standard_name = 'longitude'
        lon_bounds.units = 'degrees_east'
        lon_bounds[:] = [gridder.lons[0], gridder.lons[-1]]

        lats = ds.createVariable('latitude', 'float32', ('latitude',))
        lats.standard_name = 'latitude'
        lats.units = 'degrees_north'
        lats.bounds = 'latitude_bounds'
        lats[:] = (gridder.lats[:-1]+gridder.lats[1:])/2

        lat_bounds = ds.createVariable('latitude_bounds', 'float32', ('nv', ))
        lat_bounds.standard_name = 'latitude'
        lat_bounds.units = 'degrees_north'
        lat_bounds[:] = [gridder.lats[0], gridder.lats[-1]]

        for key, value in data.items():

            if value.squeeze().ndim == 2:
                ncvar = ds.createVariable(key, 'float32', ('time', 'latitude', 'longitude',))
            elif value.squeeze().ndim == 3 and value.shape[-1] == 34:
                ncvar = ds.createVariable(key, 'float32', ('time', 'latitude', 'longitude', 'layer'))
            elif value.squeeze().ndim == 3 and value.shape[-1] == 35:
                ncvar = ds.createVariable(key, 'float32', ('time', 'latitude', 'longitude', 'level'))
            else:
                print(f'Data shape not recognized for {key}: {value.shape}')
                raise ValueError(f'Data shape not recognized for {key}: {value.shape}')

            ncvar.standard_name = key

            if key.endswith('_column_precision') or key.endswith('_column'):
                ncvar.units = gridder.units
            elif 'pressure' in key:
                ncvar.units = 'Pa'

            if key == 'timestamp':
                ncvar[:] = value.astype('datetime64[s]').astype(np.float64)
            else:
                ncvar[:] = value

        if sup_vars:
            raw = ds.createGroup('RAW')

            for key, value in sup_dims.items():
                dim = raw.createDimension(key, value)

            for key, value in sup_vars.items():

                if value.shape == (sup_dims['time'], sup_dims['latitude'], sup_dims['longitude']):
                    var_shape = ('time', 'latitude', 'longitude',)
                elif value.shape == (sup_dims['time'], sup_dims['latitude'], sup_dims['longitude'], sup_dims['layer']):
                    var_shape = ('time', 'latitude', 'longitude', 'layer',)
                elif value.shape == (sup_dims['time'], sup_dims['latitude'], sup_dims['longitude'], sup_dims['level']):
                    var_shape = ('time', 'latitude', 'longitude', 'level',)
                elif value.shape == (sup_dims['level'],):
                    var_shape = ('level',)
                else:
                    print(f'Data shape not recognized for {key}: {value.shape}')
                    raise ValueError(f'Data shape not recognized for {key}: {value.shape}')

                ncvar = raw.createVariable(key, 'float32', var_shape)
                ncvar.standard_name = key

                if 'pressure' in key:
                    ncvar.units = 'Pa'

        logging.info(indent + f"written file {filename}")

        return

# =====================================================================
