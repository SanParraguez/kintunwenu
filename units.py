# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> UNITS

Module to deal with unit transformations and get some constants.
"""
__all__ = [
    'standardise_unit_string',
    'convert_units'
]

# ============= IMPORTS ===============================
import numpy as np

# =================================================================================
# ToDo: Define if these constants must be capitalized
# Define molar masses (in g/mol) for commonly measured gas species
molar_masses = {'o3': 47.9982, 'no2': 46.0055, 'so2': 64.064,
                'co2': 44.0095, 'ch4': 16.0425}

# Avogadro constant
Na = 6.02214076 * 1e23

# Unit convert dictionary
convert_dict = {
    'mol m-2': 'mol/m2',
    'cm^-2': 'molec/cm2',
    '1e-9': 'ppb',
}
# =================================================================================

def standardise_unit_string(units):
    """
    Standardise units strings.

    Parameters
    ----------
    units : str
        Unit to be standardised to a more normal notation.

    Returns
    -------
    str
        String with the same unit
    """
    return convert_dict.get(units, units)

# =================================================================================

def convert_units(data, from_unit, to_unit, species=None):
    """
    Converts data from one concentration unit to another. Units should be in
    ['mol/m2', 'umol/m2', 'molec/cm2', 'ppb', 'ppm', g/cm3]. To be honest, you
    could try a couple more. Also, I was tired, so you should check any output from
    this function.

    Parameters
    ----------
    data : np.ndarray
        The data to be converted.
    from_unit : str
        The current unit of the data.
    to_unit : str
        The desired unit of the data.
    species : str or None
        The species for which the molar mass should be used to convert the units.
        Should be specified if converting between mol/m2 and ppb. Defaults to None.

    Returns
    -------
    np.ndarray
        The converted data.
    """
    # ToDo: change to a two-step process, first to a standard unit and then to the desired.
    #   This should avoid the huge amount of if-else conditions.
    data = np.ma.array(data)

    from_unit = standardise_unit_string(from_unit.lower())
    to_unit = standardise_unit_string(to_unit.lower())

    if from_unit == to_unit:
        return data

    if from_unit in ['ppb', 'ppm']:
        if from_unit == 'ppb' and to_unit == 'ppm':
            return data * 1000
        elif from_unit == 'ppm' and to_unit == 'ppb':
            return data / 1000
        else:
            raise ValueError(f'Unit transformation {from_unit} -> {to_unit} not recognized')

    conv_factor = 1.0
    from_num, from_den = from_unit.split('/')
    to_num, to_den = to_unit.split('/')

    if from_den != to_den:
        # Area units
        if from_den == 'm2':
            if to_den == 'cm2':
                conv_factor *= 1e4
            elif to_den == 'km2':
                conv_factor *= 1e-6
            else:
                raise ValueError(f"Area unit '{to_den}' not recognized.")

        elif from_den == 'cm2':
            if to_den == 'm2':
                conv_factor *= 1e-4
            elif to_den == 'km2':
                conv_factor *= 1e-10
            else:
                raise ValueError(f"Area unit '{to_den}' not recognized.")

        elif from_den == 'km2':
            if to_den == 'm2':
                conv_factor *= 1e6
            elif to_den == 'cm2':
                conv_factor *= 1e10
            else:
                raise ValueError(f"Area unit '{to_den}' not recognized.")

        # Volume units
        elif from_den == 'm3':
            if to_den == 'cm3':
                conv_factor *= 1e6
            elif to_den == 'km3':
                conv_factor *= 1e-9
            else:
                raise ValueError(f"Volume unit '{to_den}' not recognized.")

        elif from_den == 'cm3':
            if to_den == 'm3':
                conv_factor *= 1e-6
            elif to_den == 'km3':
                conv_factor *= 1e-15
            else:
                raise ValueError(f"Volume unit '{to_den}' not recognized")

        elif from_den == 'km3':
            if to_den == 'm3':
                conv_factor *= 1e9
            elif to_den == 'cm3':
                conv_factor *= 1e15
            else:
                raise ValueError(f"Volume unit '{to_den}' not recognized")

        else:
            raise ValueError(f"Unit '{from_den}' not recognized")

    if from_num != to_num:
        # Number of molecules
        if from_num == 'molec':
            if to_num == 'mol':
                conv_factor *= 1 / Na
            elif to_num == 'umol':
                conv_factor *= 1e6 / Na
            elif to_num in ['g', 'kg']:
                if species is None:
                    raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
                molar_mass = molar_masses.get(species.lower())
                if molar_mass is None:
                    raise ValueError(f"Molar mass for species '{species}' is not available.")
                conv_factor *= molar_mass / Na
                if to_num == 'kg':
                    conv_factor *= 1e-3
            else:
                raise ValueError(f"Unit '{to_num}' not recognized")

        elif from_num == 'mol':
            if to_num == 'molec':
                conv_factor *= Na
            elif to_num == 'umol':
                conv_factor *= 1e6
            elif to_num in ['g', 'kg']:
                if species is None:
                    raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
                molar_mass = molar_masses.get(species.lower())
                if molar_mass is None:
                    raise ValueError(f"Molar mass for species '{species}' is not available.")
                conv_factor *= molar_mass
                if to_num == 'kg':
                    conv_factor *= 1e-3
            else:
                raise ValueError(f"Unit '{to_num}' not recognized")

        elif from_num == 'umol':
            if to_num == 'molec':
                conv_factor *= Na * 1e-6
            elif to_num == 'mol':
                conv_factor *= 1e-6
            elif to_num in ['g', 'kg']:
                if species is None:
                    raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
                molar_mass = molar_masses.get(species.lower())
                if molar_mass is None:
                    raise ValueError(f"Molar mass for species '{species}' is not available.")
                conv_factor *= 1e-6 * molar_mass
                if to_num == 'kg':
                    conv_factor *= 1e-3
            else:
                raise ValueError(f"Unit '{to_num}' not recognized")

        # Mass units
        elif from_num == 'g':
            if to_num == 'kg':
                conv_factor *= 1e-3
            elif to_num in ['molec', 'mol', 'umol']:
                if species is None:
                    raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
                molar_mass = molar_masses.get(species.lower())
                if molar_mass is None:
                    raise ValueError(f"Molar mass for species '{species}' is not available.")
                conv_factor *= 1 / molar_mass
                if to_num == 'umol':
                    conv_factor *= 1e6
                elif to_num == 'molec':
                    conv_factor *= Na
            else:
                raise ValueError(f"Unit '{to_num}' not recognized")

        elif from_num == 'kg':
            if to_num == 'g':
                conv_factor *= 1e3
            elif to_num in ['molec', 'mol', 'umol']:
                if species is None:
                    raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
                molar_mass = molar_masses.get(species.lower())
                if molar_mass is None:
                    raise ValueError(f"Molar mass for species '{species}' is not available.")
                conv_factor *= 1e3 / molar_mass
                if to_num == 'umol':
                    conv_factor *= 1e6
                elif to_num == 'molec':
                    conv_factor *= Na
            else:
                raise ValueError(f"Unit '{to_num}' not recognized")

        else:
            raise ValueError(f"Unit {from_num} not recognized")

    data[~data.mask] *= conv_factor
    return data

# =================================================================================
