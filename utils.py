# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================

UTILS MODULE
------------

Provides useful functions for the package.
"""
__all__ = [
    'timeit',
    'pick_random_value'
]

# ============= IMPORTS ===============================
import time
import numpy as np

# =================================================================================

def timeit(func, name=None):
    """
    Decorator that measures the execution time of a function and prints it to the console.

    Parameters
    ----------
    func : callable
        The function to be timed.
    name : str
        Name of the function to be displayed

    Returns
    -------
    callable
        A new function that wraps the original function with timing functionality.
    """
    name = func.__name__ if not name else name

    def wrapper(*args, **kwargs):
        """
        Wrapper function that calculates the execution time of the wrapped function.
        """

        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f'{name}: {toc-tic:.4g} s')
        return result

    return wrapper

# =================================================================================

def pick_random_value(*args, seed=None):
    """
    Randomly selects an index in the range of the size of the arguments,
    and returns that index and a tuple of values from the input arrays at that index.

    Parameters
    ----------
    *args : array-like
        One or more input arrays.
    seed : int
        Optional seed value for numpy.random.seed().

    Returns
    -------
        Tuple containing an integer index and a tuple of values from the input arrays at that index.
    """
    args = np.array(args)
    np.random.seed(seed)
    index = np.random.randint(args[0].size)
    values = tuple(arg.flatten()[index] for arg in args)
    return index, values

# =================================================================================
