# -*- coding: utf-8 -*-
"""
Common functions

Created on Thu Mar  7 10:37:27 2019
"""

import numpy as np
from numpy.random import permutation as pf
#from statsmodels.stats import multitest as mt

def center_matrix(a):
    """
    Remove the means from each column in a matrix
    """
#    col_means = a.mean(0)
#    rep_mean = np.reshape(np.repeat(col_means, a.shape[0]), a.shape, order="F")
#    
#    return np.subtract(a, rep_mean)
    return a - np.mean(a, axis=0)

def scale_matrix(a):
    """
    Convert centered data to z-scores
    """
    std = np.std(a, axis=0, ddof=1)
    b = np.ndarray(shape=a.shape)
    for c in range(a.shape[1]):
        b[:, c] = np.divide(a[:, c], std[c])
         
    return b

def perm_matrix(matrix):
    """
    Permute the columns of a matrix using as little memory as possible
    """
    return np.asarray([pf(matrix[:, col]) for col in range(matrix.shape[1])]).T

#def fdr_pmatrix(p_matrix):
#    """
#    Apply a FDR correction to a matrix of p-values
#    """
#    pvect = np.ndarray.flatten(p_matrix)
#    _, fdr_p = mt.fdrcorrection(pvect)
#    return np.reshape(fdr_p, p_matrix.shape)

def permutation_p(observed, perm_array):
    #see Phipson & Smyth 2010 for more information
    n_iters = len(perm_array)
    n_hits = np.where(np.abs(perm_array) >= np.abs(observed))
    return (len(n_hits[0]) + 1) / (n_iters + 1)

def resample_matrix(matrix):
    """
    Columnwise resampling with replacement
    """        
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    
    resamp_mat = np.ndarray(shape=matrix.shape)
    for col in range(n_cols):
        for row in range(n_rows):
            idx = np.random.randint(0, n_rows)
            resamp_mat[row, col] = matrix[idx, col]
    
    return resamp_mat