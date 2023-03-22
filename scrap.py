# -*- coding: utf-8 -*-
"""
=======================================================
===                   KINTUN-WENU                   ===
=======================================================
-> SCRAP

Provides functions to download and process datasets from https://disc.gsfc.nasa.gov.
"""
__all__ = [
    'read_links_file',
    'download_ncfile'
]

# ============= IMPORTS ===============================

import requests
from netCDF4 import Dataset
from pathlib import Path
from urllib.parse import urlparse

# =================================================================================

def read_links_file(path, dtype=None):
    """
    Returns a list of links contained in one or more text files.

    Parameters
    ----------
    path : str, Path or list
        Path(s) to one or more text files.
        If a directory path is provided, all text files in the directory will be read.
    dtype : str, Path or list, optional
        Types of files to accept, specified by their extensions (e.g., 'txt', 'csv').
        If specified, only links from files of the specified types will be returned.
        All files are accepted by default (None).

    Returns
    -------
    list of str
        List of links contained in the text files.
    """
    # Ensure path is iterable
    if isinstance(path, (str, Path)):
        path = [path]

    links = []
    for single_path in path:
        single_path = Path(single_path)

        # If directory path is provided, read all text files in directory
        if single_path.is_dir():
            files = [file_path for file_path in single_path.iterdir()
                     if file_path.suffix == '.txt']
            for single_file in files:
                with open(single_file, 'r') as txt:
                    links += txt.read().splitlines()

        # If single text file is provided, read links from it
        else:
            with open(single_path, 'r') as txt:
                links += txt.read().splitlines()

    # Filter by file type if specified
    if dtype:
        if isinstance(dtype, str):
            dtype = [dtype]
        links = [link for link in links if link.split('.')[-1] in dtype]

    return links


# =================================================================================

def download_ncfile(url, save_path=None):
    """
    Downloads a netCDF4 dataset from the provided URL, and returns the dataset as an object.
    If a save_path is provided, the downloaded file will be saved to that location and not
    returned.

    Parameters
    ----------
    url : str
        URL to a .nc file.
    save_path : str, optional
        Path to save the downloaded file (default is None).

    Returns
    -------
    Dataset or None
        Returns a netCDF4 dataset object if no save_path is provided, otherwise returns None.
    """
    parse = urlparse(url)
    filename = parse.path.split('/')[-1]

    # If save_path is provided, download and save file to the given location
    if save_path:
        filename = Path(save_path) / filename
        print(f'Downloading {filename}...\r')
        with requests.get(url) as response:
            response.raise_for_status()
            with open(filename, 'wb') as file:
                file.write(response.content)
        return None

    # If no save_path is provided, load the dataset in memory and return it
    else:
        with requests.get(url) as response:
            response.raise_for_status()
            dataset = Dataset(filename, memory=response.content)
        return dataset

# =================================================================================
