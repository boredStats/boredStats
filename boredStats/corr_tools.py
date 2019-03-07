# -*- coding: utf-8 -*-
"""
Tools for correlation matrices    

Created on Wed Mar  6 14:18:19 2019
"""

from boredStats import utils

import numpy as np

def cross_corr(x, y):
    """
    Calculate Pearson's R the columns of two matrices
    """
    s = x.shape[0]
    if s != y.shape[0]:
        raise ValueError ("x and y must have the same number of subjects")
    
    std_x = x.std(0, ddof=s - 1)
    std_y = y.std(0, ddof=s - 1)
    
    cov = np.dot(utils.center_matrix(x).T, utils.center_matrix(y))
    
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

class PermutationCorrelation(object):
    """
    Run permutation based inferential testing
    """
    def __init_(self, n_iters=1000, return_cube=False):
        self.n_iters = n_iters
        self.cube = return_cube
    
    @staticmethod
    def permutation_p(observed, perm_array, n_iters):
        #see Phipson & Smyth 2010 for more information
        n_hits = np.where(perm_array >= observed)
        return (len(n_hits) + 1)/ (n_iters + 1)
        
    def perm_corr(self, x, y):
        perm_3dmat = np.ndarray(shape=[x.shape[1], y.shape[1], self.n_iters])
        n = 0
        while n != self.n_iters:
            perm_x = utils.perm_matrix(x)
            perm_y = utils.perm_matrix(y)
            perm_3dmat[:, :, n] = cross_corr(perm_x, perm_y)
            n += 1
            
        corr_matrix = cross_corr(x, y)
        p_matrix = np.ndarray(shape=corr_matrix.shape)
        for r in range(corr_matrix.shape[0]):
            for c in range(corr_matrix.shape[1]):
                p_matrix[r, c] = self.permutation_p(
                        corr_matrix[r, c],
                        perm_3dmat[r, c, :],
                        self.n_iters)
        if self.cube:
            return corr_matrix, p_matrix, perm_3dmat
        else:
            return corr_matrix, p_matrix