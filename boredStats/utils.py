# -*- coding: utf-8 -*-
"""
Common functions

Created on Thu Mar  7 10:37:27 2019
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

def plotScree(eigenvalues, eigenPvals=None, kaiser=False, fname=None):
    """
    Create a scree plot for factor analysis using matplotlib
    
    Parameters
    ----------
    eigenvalues : numpy array
        A vector of eigenvalues
    
    eigenPvals : numpy array
        A vector of p-values corresponding to a permutation test
    
    kaiser : bool
        Plot the Kaiser criterion on the scree
        Note: Kaiser test only suitable for standarized data
        
    Optional
    --------
    fname : filepath
        filepath for saving the image
    Returns
    -------
    fig, ax1, ax2 : matplotlib figure handles
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    percentVar = (np.multiply(100, eigenvalues)) / np.sum(eigenvalues)
    cumulativeVar = np.zeros(shape=[len(percentVar)])
    c = 0
    for i,p in enumerate(percentVar):
        c = c+p
        cumulativeVar[i] = c
    
    fig,ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Scree plot", fontsize='xx-large')
    ax.plot(np.arange(1,len(percentVar)+1), eigenvalues, '-k')
    ax.set_ylim([0,(max(eigenvalues)*1.2)])
    ax.set_ylabel('Eigenvalues', fontsize='xx-large')
    ax.set_xlabel('Factors', fontsize='xx-large')
#    ax.set_xticklabels(fontsize='xx-large') #TO-DO: make tick labels bigger
    
    ax2 = ax.twinx()
    ax2.plot(np.arange(1,len(percentVar)+1), percentVar,'ok')
    ax2.set_ylim(0,max(percentVar)*1.2)
    ax2.set_ylabel('Percentage of variance explained', fontsize='xx-large')

    if eigenPvals is not None and len(eigenPvals) == len(eigenvalues):
        #TO-DO: add p<.05 legend?
        pvalueCheck = [i for i,t in enumerate(eigenPvals) if t<.05]
        eigenCheck = [e for i,e in enumerate(eigenvalues) for j in pvalueCheck if i==j]
        ax.plot(np.add(pvalueCheck,1), eigenCheck, 'ob', markersize=10)
    
    if kaiser:
        ax.axhline(1, color='k', linestyle=':', linewidth=2)
    
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig, ax, ax2