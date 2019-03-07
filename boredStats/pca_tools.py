# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:03:50 2019
"""

from boredStats import utils

import numpy as np

class PermutationPCA(object):
    def __init__(self, n_iters, return_cubes=False):
        self.n = n_iters
        self.cubes = return_cubes
        
    def perm_pca(self, data):
        
    