# -*- coding: utf-8 -*-
"""
Common functions

Created on Thu Mar  7 10:37:27 2019
"""

import numpy as np
from numpy.random import permutation as pf
from statsmodels.stats import multitest as mt

def center_matrix(a):
    """
    Remove the means from each column in a matrix
    """
    col_means = a.mean(0)
    rep_mean = np.reshape(np.repeat(col_means, a.shape[0]), a.shape, order="F")
    
    return np.subtract(a, rep_mean)

def perm_matrix(matrix):
    """
    Permute the columns of a matrix using as little memory as possible
    """
    return np.asarray([pf(matrix[:, col]) for col in range(matrix.shape[1])]).T

def fdr_pmatrix(p_matrix):
    """
    Apply a FDR correction to a matrix of p-values
    """
    pvect = np.ndarray.flatten(p_matrix)
    _, fdr_p = mt.fdrcorrection(pvect)
    return np.reshape(fdr_p, p_matrix.shape)

def permutation_p(observed, perm_array, n_iters):
    #see Phipson & Smyth 2010 for more information
    n_hits = np.where(np.abs(perm_array) >= np.abs(observed))
    return (len(n_hits) + 1)/ (n_iters + 1)