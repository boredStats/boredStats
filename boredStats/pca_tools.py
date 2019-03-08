# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:03:50 2019
"""

from . import utils

import numpy as np

class PermutationPCA(object):
    def __init__(self, eigen_perm=True, return_perms=False):
        self.eigtest = eigen_perm
        self.return_perms = return_perms
    
    @staticmethod
    def eigen_perm(obs_eigs, perm_eigs):
        n_iters = perm_eigs.shape[0]
        p_values = np.ndarray(shape=obs_eigs.shape)
        for e, eig in enumerate(obs_eigs):
            perm_data = perm_eigs[e, :]
            p_values[e] = utils.permutation_p(eig, perm_data, n_iters)
        return p_values
    
    def perm_pca(self, data, n_iters=1000, centered=False):
        if not centered:
            data_in = utils.center_matrix(data) 
        else:
            data_in = data
        
        left, obs_eigs, right = np.linalg.svd(data_in)
        
        output = {'eigenvalues' : obs_eigs,
                  'left_singular_values' : left,
                  'right_singular_values' : right}
        if not self.eigen_perm:
            return output
        
        perm_mat = np.ndarray(shape=[self.n_iters, obs_eigs.shape])
        n = 0
        while n != self.n_iters:
            perm_data_in = utils.perm_matrix(data_in)
            _, perm_eigs, _ = np.linalg.svd(perm_data_in)
            perm_mat[n, :] = perm_eigs
        
        p_values = self.eigen_perm(obs_eigs, perm_mat)
        output['eigen_p_values'] = p_values
        
        if self.return_perms:
            output['eigen_permutations'] = perm_mat
        return output
    
    