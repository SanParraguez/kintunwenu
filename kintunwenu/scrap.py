# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
Universidad de Chile - 2021
mail: santiago.parraguez@ug.uchile.cl

=============
 KINTUN WENU
=============
  - SCRAP

Provides functions to download and process datasets from https://disc.gsfc.nasa.gov.
"""
__all__ = ['read_links_file', 'download_files', 'download_file', 'read_files']
# ============= IMPORTS ===============
import os
import requests
from netCDF4 import Dataset
from pathlib import Path
from urllib.parse import urlparse
# =================================================================================
def read_links_file(file_path) -> list:
    """
    Returns a list of strings with the links contained in a text file.

    Parameters
    ----------
    file_path : str
        Path of the text file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f'{file_path} should be a text file.')

    with open(file_path, 'r') as txt:
        links = txt.read().splitlines()[3:]

    return links

# =================================================================================
def download_files(urls, n=None, save_files=False, path='data', verbose=1) -> list:
    """
    Returns a list of netCDF4.Dataset files downloaded from the provided links, or download the files to disk.

    Parameters
    ----------
    urls : iterable
        Iterable of strings with urls to .nc files.
    n : int
        Number of files to be downloaded, None to get all the files.
    save_files: bool
        Whether the files are downloaded and saved to disk or just read and returned.
    path: str
        Path where the files will be stored if save_files is True.
    verbose: int
        To control amount of printed information.
    """
    if save_files:
        if path is not None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

    products = []

    links_to_download = urls[:n] if n is not None else urls
    n = len(links_to_download)
    n_succ = 0
    n_fail = 0

    if verbose > 0:
        print('Starting download of netCDF files')
    if verbose > 0:
        print(f'Downloaded {n_succ}/{n} files ({n_fail} omitted) from urls...', end='\r')

    for url in links_to_download:

        try:
            response = requests.get(url)
        except (ConnectionAbortedError or ConnectionError or ConnectionRefusedError or ConnectionResetError):
            n_fail += 1
            if verbose > 0:
                print(f'Downloaded {n_succ}/{n} files ({n_fail} omitted) from urls...', end='\r')
            continue

        filename = get_filename(url)
        nc = Dataset(filename + '.nc', memory=response.content)

        if hasattr(nc, 'errors'):
            n_fail += 1
            continue

        if save_files:
            filename = path / (filename + '.nc')
            with open(filename, 'wb') as file:
                file.write(response.content)
        else:
            products.append(nc)

        response.close()
        n_succ += 1
        if verbose > 0:
            print(f'Downloaded {n_succ}/{n} files ({n_fail} omitted) from urls...', end='\r')

    if verbose > 0:
        print(f'Downloaded {n_succ}/{n} files ({n_fail} omitted) from urls')
    return products if not save_files else None

# =================================================================================
def download_file(url) -> Dataset:
    """
    Returns a netCDF4.Dataset downloaded from the provided link.

    Parameters
    ----------
    url : str
        Url to a .nc file.
    """
    query = urlparse(url).query.split('&')
    filename = [subquery.split('=')[-1].split('.')[0] for subquery in query if 'LABEL' in subquery][0]
    response = requests.get(url)
    product = Dataset(filename, memory=response.content)
    response.close()
    return product

# =================================================================================
def read_files(path=None) -> list:

    list_dir = [f for f in os.listdir(path) if f.split('.')[-1] in ['nc', 'nc4', 'h5', 'he5']]
    if path is None:
        path = ''

    data_list = []
    n_succ, n_fail = 0, 0
    for f in list_dir:
        try:
            data_list.append(Dataset(path + '/' + f))
            n_succ += 1
        except ValueError:
            n_fail += 1

    print(f'Loaded {n_succ}/{len(list_dir)} successfully. Failed to load {n_fail} files.')
    return data_list

# =================================================================================
def get_filename(url):
    """
    Get the filename from a given url.

    Parameters
    ----------
    url : str
        Url to be parsed.
    """
    parse = urlparse(url)
    if parse.query:
        query = parse.query.split('&')
        filename = [q.split('=')[-1] for q in query if 'LABEL' in q][0]
    else:
        filename = parse.path.split('/')[-1]

    return filename.split('.')[0]
