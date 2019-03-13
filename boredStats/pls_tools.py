"""Tools for running Partial-Least Squares Correlation

Complete:
Multi-table PLSC
Inference testing for # of latent variables to use

To-do:
Bootstrap testing for stabililty of saliences
Test suite
"""

import utils
from corr_tools import cross_corr

import numpy as np

class MultitablePLSC(object):
    def __init__(self, n_iters=None, return_perm=False):
        self.n_iters = n_iters
        self.return_perm = return_perm
    
    @staticmethod
    def build_corr_xy(y, x_list):
        """Build input for multitable PLSC
        Formula is:
            Y = X1 + X2 + ... + Xn
            
            Where Y is a table of outcome variables and
            Xn are N tables of vars to correlate with Y
        
        Parameters:
        -----------
        y : numpy array
        
        x_list : list
            A list of numpy arrays
            
        Notes:
        ------
        The arrays in y and x_list must have the same
        number of rows
        """
        
        num_vars_in_y = y.shape[1]
        num_vars_in_x = [x.shape[1] for x in x_list]
        
        cross_xy = np.ndarray(shape=[sum(num_vars_in_x), num_vars_in_y])
        
        for x_index, x_table in enumerate(x_list):
            cross_corrmat = cross_corr(x_table, y)
            start = num_vars_in_x[x_index] * x_index
            end = num_vars_in_x[x_index] * (x_index + 1)
            cross_xy[start:end, :] = cross_corrmat
            
        return cross_xy
    
    @staticmethod
    def procrustes_rotation(orig_svd, resamp_svd):
        """Apply a Procrustes rotation to resampled SVD results
        
        This rotation is to correct for:
            - axis rotation (change in order of components)
            - axis reflection (change in sign of loadings)
            
        See McIntosh & Lobaugh, 2004 for more
        
        Parameters:
        -----------
        orig_svd : tuple
        Tuple of SVD results corresponding to original data
        
        resamp_svd : tuple
        Tuple of SVD results corresponding to resampled data
        
        Returns:
        --------
        rot_u : rotated left singular values
        rot_s : rotated diagonal matrix
        rot_v : rotated right singular values
        """
        
        ov = orig_svd(0)
        rv = resamp_svd(0)
        rs = resamp_svd(1)
        ru = resamm_svd(2)

        n, _, p = np.svd(np.matmul(ov.T, rv))
        
        q = np.matmul(n, p.T)
        
        rot_u = ru * rs * q
        rot_v = rv * rs * q
        
        ss_rot_v = np.sum(rot_v**2, 0) #sum of squares of rotated v
        rot_s = np.sqrt(ss_rot_v)
        
        return rot_u, rot_s, rot_v
    
    @staticmethod
    def p_from_perm_mat(obs_vect, perm_array):
    """Calculate p-values columnwise
    
    Parameters:
    -----------
    obs_vect : numpy array
    Vector of true observations
    
    perm_array : numpy array
    N x M array of observations obtained through permutation
        N is the number of permutations used
        M is the number of variables
    
    Returns:
    --------
    p_values : numpy array
    Vector of p-values corresponding to obs_vect
    """

    n_iters = perm_array.shape[0]
    p_values = np.ndarray(shape=obs_vect.shape)
    for t, true in enumerate(obs_vect):
        perm_data = perm_array[t, :]
        p_values[t] = utils.permutation_p(true, perm_data, n_iters)
    return p_values
   
    @staticmethod
    def mult_plsc(y_table, x_tables):
        """Calculate multitable PLS-C, fixed effect model
        See Krishnan et al., 2010 for more
        """
        corr_xy = self.build_corr_xy(y_table, x_tables)
        centered_corr_xy = utils.center_matrix(corr_xy)
        
        u, delta, v = np.linalg.svd(centered_corr_xy)
        return u, delta, v.T

    def perm_mult_plsc(self, y_table, x_tables):
        """Run permutation based testing to determine
        best number of latent variables to use
        
        Parameters:
        -----------
        y_table : numpy array
        Array of variables for Y
        
        x_tables : list
        List of numpy arrays corresponding to X
        
        Returns:
        --------
        p_values : numpy array
        Array of p_values corresponding to the singular values
        
        perm_singular_array : numpy array (optional)
        N x M array of singular values obtained by permuting the
        rows in each table; self.return_perm must be True
        """
        if not self.n_iters:
            return AttributeError("Number of permutations cannot be None")
        
        orig_svd = self.mult_plsc(y_table, x_tables)
        orig_sing = orig_svd(1)
        
        n = 0
        perm_singular_values = np.ndarray(shape=[self.n_iters, len(orig_sing)])
        while n != self.n_iters:
            perm_y = utils.perm_matrix(y_table)
            perm_x_tables = [perm_matrix(x) for x in x_tables]
            
            perm_xy = self.build_corr_xy(perm_y, perm_x_tables)
            centered_perm_xy = utils.center_matrix(perm_xy)
            perm_svd = np.linalg.svd(centered_perm_xy)
            
            rot_perm = self.procrustes_rotation(orig_svd, perm_svd)
            perm_singular_values[n, :] = rot_perm
        
        p_values = utils.p_from_perm_mat(orig_sing, perm_singular_values)
        
        if self.return_perm:
            return p_values, perm_singular_values
        else:
            return p_values
    
    def bootstrap_saliences(self, y_table, x_tables, z_tester=2):
        """Run bootstrap testing on saliences
        
        Parameters:
        -----------
        y_taable, x_tables: inputs for multitable PLSC
        
        z_tester: int
        "Z-score" to test bootstrap samples with
        Default is 2 (or approximately 1.96)
        
        Returns:
        --------
        filt_y_sals, filt_x_sals: numpy arrays
        Arrays of saliences filtered by bootstrap testing
        
        See Krishnan et al. 2011, 'Deciding which latent variables to keep' 
        """
        if not self.n_iters:
            raise AttributeError("Number of iterations cannot be None")
            
        orig_svd = self.mult_plsc(y_table, x_tables)
        orig_y_sals = orig_svd(0)
        orig_x_sals = orig_svd(2)
        
        n = 0
        perm_y_sals = np.ndarray(shape=[orig_y_sals.shape, self.n_iters])
        perm_x_sals = np.ndarray(shape=[orig_x_sals.shape, self.n_iters])
        while n != self.n_iters:
            resamp_y = utils.resample_matrix(y_table)
            resamp_x_list = [utils.resample_matrix(x) for x in x_tables]
            
            resamp_svd = self.mult_plsc(resamp_y, resamp_x_list)
            rot_svd = self.procrustes_rotation(orig_svd, resamp_svd)
            perm_y_sals[:, :, n] = rot_svd(0)
            perm_x_sals[:, :, n] = rot_svd(2)
        
        perm_y_sals_std = np.std(perm_y_sals, axis=-1)
        perm_x_sals_std = np.std(perm_x_sals, axis=-1)
        
        perm_y_zscores = np.divide(orig_y_sals, perm_y_sals_std)
        perm_x_zscores = np.divide(orig_x_sals, perm_x_sals_std)
        
        filt_y_sals = orig_y_sals[perm_y_zscores < 2] = 0
        filt_x_sals = orig_x_sals[perm_x_zscores < 2] = 0
        
        return filt_y_sals, filt_x_sals
        
if __name__ == "__main__":
    n = 100
    y = np.random.rand(n, 15)
    
    x_list = []
    for numx in range(6):
        x_list.append(np.random.rand(n, 10))
    
    p = MultitablePLSC()
    res = p.mult_plsc(y, x_list)
    print(res)