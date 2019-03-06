# -*- coding: utf-8 -*-
"""
Tools for correlation matrices    

Created on Wed Mar  6 14:18:19 2019
"""

import numpy as np

def center_matrix(a):
    """
    Remove the means from each column in a matrix
    """
    col_means = a.mean(0)
    n_rows = a.shape[0]
    rep_mean = np.reshape(np.repeat(col_means, n_rows), a.shape, order="F")
    
    return np.subtract(a, rep_mean)

def cross_corr(x, y):
    """
    Calculate Pearson's R the columns of two matrices
    """
    s = x.shape[0]
    if s != y.shape[0]:
        raise ValueError ("x and y must have the same number of subjects")
    
    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)
    
    cov = np.dot(center_matrix(x).T, center_matrix(y))
    
    return cov/np.dot(std_x[:, np.newaxis], std_y[np.newaxis, :])

def r_to_p(rmat, n):
    """
    Get the associated p-values for a correlation matrix
    
    Parametric method for calculating p-values:
        Input is assumed to be Pearson's R
        Hence p-values are calculated from the t-distribution
    """
    from scipy.stats import t as tfunc
    denmat = (1 - rmat**2) / (n - 2)
    tmat = rmat / np.sqrt(denmat)
    
    tvect = np.ndarray.flatten(tmat)
    pvect = np.ndarray(shape=tvect.shape)
    for ti, tval in enumerate(tvect):
        pvect[ti] = tfunc.sf(np.abs(tval), n-1) * 2
    
    return np.reshape(pvect, rmat.shape)
    
def r_to_se(rmat, n):
    """
    Get the associated standard errors for a correlation matrix
    """
    a = (1 - rmat**2) / (n - 2)
    return np.sqrt(a)

def fdr_p(pmat):
    """
    Apply an FDR correction to a matrix of p-values
    """
    from statsmodels.stats import multitest as mt
    pvect = np.ndarray.flatten(pmat)
    _, fdr_p = mt.fdrcorrection(pvect)
    return np.reshape(fdr_p, pmat.shape)

