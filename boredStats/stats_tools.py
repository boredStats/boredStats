# -*- coding: UTF-8 -*-
"""Extra stats tools that don't have a home."""

import numpy as np
import utils


def cron_alpha(array, standardized=False):
    """Calculate Cronbach's alpha.

    Parameters
    ----------
    array : numpy array
        An (NxM) array where N is assumed to be the observations and M are
        the variables.
    standardized : bool (optional)
        Whether or not to calculate the standardized Cronbach's alpha.
        Default is False.

    Returns
    -------
    alpha : float
        Cronbach's alpha value.

    """
    k = array.shape[1]
    variance = np.var(array, axis=0, ddof=1)
    variance_avg = np.mean(variance)
    if standardized is False:
        # Uses covariance matrix
        covar = np.cov(array.astype(float), rowvar=False, ddof=1)
        top_triangle = utils.upper_tri_indexing(covar, diagonal=False)
        cov_avg = np.mean(top_triangle)

        alpha = (k*cov_avg) / (variance_avg + (k-1)*cov_avg)
    else:
        # Uses correlation matrix
        r = np.corrcoef(array.astype(float), rowvar=False)
        top_triangle = utils.upper_tri_indexing(r, diagonal=False)
        r_avg = np.mean(top_triangle)

        alpha = (k*r_avg) / (1 + (k-1)*r_avg)

    return alpha
