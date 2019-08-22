# -*- coding: utf-8 -*-
"""
Common functions

Created on Thu Mar  7 10:37:27 2019
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import resample as sk_resample
from sklearn.utils import safe_indexing
from sklearn.utils.validation import check_consistent_length
from copy import deepcopy


def center_scale_array(x, scale='Z'):
    """Center and/or scale an array

    Parameters
    ----------
    x : array to center/scale

    scale : {None, 'Z', 'SS1'}, default is 'z'
        Parameter to specifcy scaling method. Use None if you only want to remove the column means
        from the data. Use 'Z' if you want to scale your columns by their standard deviations. Use
        'SS1' if you want to scale your columns so that their sum of squares will equal 1 (See
        Abdi & Williams, 2010).

    Returns
    -------
    cleaned : cleaned array
    """
    # Centering data
    cleaned = deepcopy(x).astype(float)
    cleaned_mean = cleaned.mean(axis=0)
    cleaned -= cleaned_mean

    if scale is 'Z':
        cleaned_std = cleaned.std(axis=0, ddof=1)
        cleaned_std[cleaned_std == 0.0] = 1.0
        cleaned /= cleaned_std

    elif scale is 'SS1':
        ss = np.sum(cleaned ** 2, axis=0)
        sqrt_ss = np.sqrt(ss)
        cleaned /= sqrt_ss

    return cleaned


def resample_array(*arrays_to_shuffle, **options):
    """Resample an array or arrays

    Parameters
    ----------
    *arrays_to_shuffle : array or sequence of arrays
        If multiple arrays are provided, they must have the same number of rows

    Other parameters
    ----------------
    bootstrap : bool, default is False
        Parameter to specify resampling method. If False (default), Resample the array
        without replacement (for permutation-based testing). If True, resample with
        replacement (for bootstrap testing). This parameter changes behavior of the
        shuffler parameter (see below).

    shuffler : {'indep', 'together'}, default is 'indep'
        Parameter to specify shuffling method. Depending on whether or not bootstrap is set
        to True, this function will behave differently, especially if more than one array
        is provided.

        When bootstrap == False:
            - 'indep':  For each column, shuffle the rows and add to the resampled array.
            Depending on the array size, this method may be extremely memory-intensive,
            though the correlational structure of the array is more likely to be destroyed.

            - 'together': The rows of the array are shuffled. This is the fastest and most
            memory-friendly method. However, correlations between columns may be preserved.

        When bootstrap == True and number of arrays > 1:
            - 'indep': For each array, resample the data using the bootstrap procedure
            independently. The subjects chosen for one resampled array will not necessarily
            be the same set of subjects for subsequent resampled arrays.

            - 'together': Apply the bootstrap resampling procedure to all arrays. A set of
            subjects will be chosen for the resampling procedure, and then their data will
            be sampled from all arrays. The bootstrap estimates may be slightly different
            than if the arrays are resampled independently.

            If a single arrays is provided, shuffler will be ignored.

    seed : {int or None}
        Parameter to set the RNG seed. If None, seed is automatically chosen.

    Returns
    -------
    resamp_array (numpy array): an (N x M) resampled array
    """
    seed = options.pop('seed', None)
    bootstrap = options.pop('bootstrap', False)
    shuffler = options.pop('shuffler', 'indep')

    rand_state = np.random.RandomState(seed)
    check_consistent_length(*arrays_to_shuffle)
    n_subjects = arrays_to_shuffle[0].shape[0]

    def _independent_shuffling(array_to_shuffle):
        n_rows, n_cols = array_to_shuffle.shape[0], array_to_shuffle.shape[1]
        shuffled_array = deepcopy(array_to_shuffle)
        for c in range(n_cols):
            perm_indices = np.arange(n_rows)
            rand_state.shuffle(perm_indices)
            shuffled_array[:, c] = safe_indexing(array_to_shuffle[:, c], perm_indices)
        return shuffled_array

    if bootstrap is False:
        if shuffler is 'indep':
            arrays_to_shuffle = [a.tolil() if issparse(a) else a for a in arrays_to_shuffle]
            resamp_arrays = [_independent_shuffling(a) for a in arrays_to_shuffle]

        elif shuffler is 'together':
            resamp_arrays = [sk_resample(a, replace=False, random_state=rand_state) for a in arrays_to_shuffle]

    else:
        if len(arrays_to_shuffle) == 1 or shuffler is 'indep':
            resamp_arrays = [sk_resample(a, replace=True, random_state=rand_state) for a in arrays_to_shuffle]
        elif shuffler is 'together':
            arrays_to_shuffle = [a.tolil() if issparse(a) else a for a in arrays_to_shuffle]
            boot_indices = rand_state.randint(0, n_subjects, size=n_subjects)
            resamp_arrays = [safe_indexing(a, boot_indices) for a in arrays_to_shuffle]

    if len(resamp_arrays) == 1:
        return resamp_arrays[0]
    else:
        return resamp_arrays


def permutation_p(observed, perm_array):
    """Non-parametric null hypothesis testing

    see Phipson & Smyth 2010 for more information
    """
    n_iters = len(perm_array)
    n_hits = np.where(np.abs(perm_array) >= np.abs(observed))
    return (len(n_hits[0]) + 1) / (n_iters + 1)


if __name__ == "__main__":
    # Performance testing
    
    seed = 2
    test_data = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13], [13, 14, 15, 16]])
    print(test_data)

    cleaned_test_data = center_scale_array(test_data, scale='SS1')
    print(cleaned_test_data)

