# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN WENU                   ===
=======================================================

UNITS
-----

Module to deal with unit transformations and get some constants.

"""
__all__ = [
    'standardise_unit_string',
    'convert_units'
]

# ============= IMPORTS ===============================
import logging
import numpy as np

# =================================================================================

# Define molar masses (in g/mol) for commonly measured gas species
import kintunwenu.utils

MOLAR_MASS = {
    'o3' : 47.9982,
    'no2': 46.0055,
    'so2': 64.0640,
    'co2': 44.0095,
    'ch4': 16.0425,
}

# Avogadro constant
Na = 6.02214076 * 1e23

# Unit name dictionary
STANDARD_DICT = {
    'mol m-2'   : 'mol/m2',
    'cm^-2'     : 'molec/cm2',
    'ppb'       : '1e-9',
    'ppm'       : '1e-6',
    'tons'      : 'ton',
    'tonnes'    : 'ton',
}

# Dictionary to convert units to SI
CONVERT_TO_SI = {

    # Distance
    'm'     : 1,
    'cm'    : 1e-2,
    'km'    : 1e3,

    # Area
    'm2'    : 1,
    'cm2'   : 1e-4,
    'km2'   : 1e6,

    # Volume
    'm3'    : 1,
    'cm3'   : 1e-6,
    'km3'   : 1e9,

    # Mass
    'kg'    : 1,
    'g'     : 1e-3,
    'mg'    : 1e-6,
    'ug'    : 1e-9,
    'Pg'    : 1e12,
    'Tg'    : 1e9,
    'Gg'    : 1e6,
    'Mg'    : 1e3,
    'ton'   : 1e3,

    # Pressure
    'Pa'    : 1,
    'hPa'   : 1e3,

    # Amount of substance
    '1'     : 1,
    'molec' : 1,
    'mol'   : 1 / Na,
    'umol'  : 1e-6 / Na,

    # Time
    's'     : 1,
    'min'   : 60,
    'hr'    : 3600,
    'y'     : 365 * 24 * 3600,

}

# Dictionary to check consistency of unit transformation
UNIT_TYPE = {

    # Distance
    'm'  : 'length',
    'cm' : 'length',
    'km' : 'length',

    # Area
    'm2'  : 'area',
    'cm2' : 'area',
    'km2' : 'area',

    # Volume
    'm3'  : 'volume',
    'cm3' : 'volume',
    'km3' : 'volume',

    # Mass
    'kg' : 'mass',
    'g'  : 'mass',
    'mg' : 'mass',
    'ug' : 'mass',
    'Pg' : 'mass',
    'Tg' : 'mass',
    'Gg' : 'mass',
    'Mg' : 'mass',
    'ton': 'mass',

    # Pressure
    'Pa' : 'pressure',
    'hPa': 'pressure',

    # Amount of substance
    '1'    : 'amount',
    'molec': 'amount',
    'mol'  : 'amount',
    'umol' : 'amount',

    # Time
    's'    : 'time',
    'min'  : 'time',
    'hr'   : 'time',
    'y'    : 'time',

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
    return STANDARD_DICT.get(units, units)

# =================================================================================

def check_unit_types(*units):
    """
    Parameters
    ----------
    units : str
    """
    return all(UNIT_TYPE[i] == UNIT_TYPE[units[0]] for i in units[1:])

# =================================================================================

def convert_units(data, from_unit, to_unit, species=None):
    """
    Converts data from one unit to another. Units should be in
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
    # ToDo: implement handling of species

    data = np.ma.array(data, dtype=float)

    from_unit = standardise_unit_string(from_unit)
    to_unit = standardise_unit_string(to_unit)

    if from_unit == to_unit:
        return data

    conv_factor = 1.0

    try:
        from_num, from_den = from_unit.split('/')
    except ValueError:
        from_num, from_den = from_unit, '1'
    try:
        to_num, to_den = to_unit.split('/')
    except ValueError:
        to_num, to_den = to_unit, '1'

    if not check_unit_types(from_num, to_num) or not check_unit_types(from_den, to_den):
        raise ValueError(f"Units should be consistent, invalid unit conversion '{from_unit}' to '{to_unit}'")

    for x in [from_num, to_den]:
        try:
            conv_factor *= CONVERT_TO_SI[x]
        except KeyError as err:
            logging.error(err)
            raise NotImplementedError(f'Unit {x} not recognized')

    for x in [from_den, to_num]:
        try:
            conv_factor /= CONVERT_TO_SI[x]
        except KeyError as err:
            logging.error(err)
            raise NotImplementedError(f'Unit {x} not recognized')

    # -------------------------------------------
    #  LEGACY - Kept for reference in the future
    # -------------------------------------------
    # if from_num != to_num:
    #     # Number of molecules
    #     if from_num == 'molec':
    #         if to_num == 'mol':
    #             conv_factor *= 1 / Na
    #         elif to_num == 'umol':
    #             conv_factor *= 1e6 / Na
    #         elif to_num in ['g', 'kg']:
    #             if species is None:
    #                 raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
    #             molar_mass = MOLAR_MASS.get(species.lower())
    #             if molar_mass is None:
    #                 raise ValueError(f"Molar mass for species '{species}' is not available.")
    #             conv_factor *= molar_mass / Na
    #             if to_num == 'kg':
    #                 conv_factor *= 1e-3
    #         else:
    #             raise ValueError(f"Unit '{to_num}' not recognized")
    #
    #     elif from_num == 'mol':
    #         if to_num == 'molec':
    #             conv_factor *= Na
    #         elif to_num == 'umol':
    #             conv_factor *= 1e6
    #         elif to_num in ['g', 'kg']:
    #             if species is None:
    #                 raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
    #             molar_mass = MOLAR_MASS.get(species.lower())
    #             if molar_mass is None:
    #                 raise ValueError(f"Molar mass for species '{species}' is not available.")
    #             conv_factor *= molar_mass
    #             if to_num == 'kg':
    #                 conv_factor *= 1e-3
    #         else:
    #             raise ValueError(f"Unit '{to_num}' not recognized")
    #
    #     elif from_num == 'umol':
    #         if to_num == 'molec':
    #             conv_factor *= Na * 1e-6
    #         elif to_num == 'mol':
    #             conv_factor *= 1e-6
    #         elif to_num in ['g', 'kg']:
    #             if species is None:
    #                 raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
    #             molar_mass = MOLAR_MASS.get(species.lower())
    #             if molar_mass is None:
    #                 raise ValueError(f"Molar mass for species '{species}' is not available.")
    #             conv_factor *= 1e-6 * molar_mass
    #             if to_num == 'kg':
    #                 conv_factor *= 1e-3
    #         else:
    #             raise ValueError(f"Unit '{to_num}' not recognized")
    #
    #     # Mass units
    #     elif from_num == 'g':
    #         if to_num == 'kg':
    #             conv_factor *= 1e-3
    #         elif to_num in ['molec', 'mol', 'umol']:
    #             if species is None:
    #                 raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
    #             molar_mass = MOLAR_MASS.get(species.lower())
    #             if molar_mass is None:
    #                 raise ValueError(f"Molar mass for species '{species}' is not available.")
    #             conv_factor *= 1 / molar_mass
    #             if to_num == 'umol':
    #                 conv_factor *= 1e6
    #             elif to_num == 'molec':
    #                 conv_factor *= Na
    #         else:
    #             raise ValueError(f"Unit '{to_num}' not recognized")
    #
    #     elif from_num == 'kg':
    #         if to_num == 'g':
    #             conv_factor *= 1e3
    #         elif to_num in ['molec', 'mol', 'umol']:
    #             if species is None:
    #                 raise ValueError(f"Species must be specified when converting from {from_unit} to {to_unit}.")
    #             molar_mass = MOLAR_MASS.get(species.lower())
    #             if molar_mass is None:
    #                 raise ValueError(f"Molar mass for species '{species}' is not available.")
    #             conv_factor *= 1e3 / molar_mass
    #             if to_num == 'umol':
    #                 conv_factor *= 1e6
    #             elif to_num == 'molec':
    #                 conv_factor *= Na
    #         else:
    #             raise ValueError(f"Unit '{to_num}' not recognized")
    #
    #     else:
    #         raise ValueError(f"Unit {from_num} not recognized")
    # -------------------------------------------------------------------------

    logging.info(f"{from_unit} -> {to_unit}")
    logging.info(f"applying conversion factor: {conv_factor:.4g} for {from_unit} -> {to_unit}")
    data *= conv_factor
    return data

# =================================================================================
