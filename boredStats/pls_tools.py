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
        
        cross_xy = np.ndarray(shape=[num_vars_in_y, sum(num_vars_in_x)])
        
        start_index = 0
        for x_index, x_table in enumerate(x_list):
            cross_corrmat = cross_corr(y, x_table)
            end_index = start_index + num_vars_in_x[x_index]
            cross_xy[:, start_index:end_index] = cross_corrmat
            
        return cross_xy
    
    @staticmethod
    def procrustes_rotation(orig_svd, resamp_svd):
        """Apply a Procrustes rotation to resampled SVD results
        
        This rotation is to correct for:
            - axis rotation (change in order of components)
            - axis reflection (change in sign of loadings)
            
        See McIntosh & Lobaugh, 2004 'Assessment of significance'
        
        Parameters:
        -----------
        orig_svd : tuple
        Tuple of SVD results corresponding to original data
        
        resamp_svd : tuple
        Tuple of SVD results corresponding to resampled data
        
        Returns:
        --------
        rotated_u : rotated left singular values
        rotated_diag : rotated diagonal matrix values
        rotated_v : rotated right singular values
        """
        
        original_v = orig_svd[2]
        perm_u = resamp_svd[0]
        perm_diag = resamp_svd[1]
        perm_v = resamp_svd[2]

        n, _, p = np.linalg.svd(np.dot(original_v.T, perm_v), full_matrices=False)
        rotation_matrix = n.dot(p.T)
        
        rotated_u = np.dot(np.dot(perm_u, np.diagflat(perm_diag)), rotation_matrix)
        rotated_v = np.dot(rotation_matrix, np.dot(np.diagflat(perm_diag), perm_v))
        
        sum_of_squares_rotated_u = np.sum(rotated_v[:, :]**2, 0)
        rotated_diag = np.sqrt(sum_of_squares_rotated_u)
        
        return rotated_u, rotated_diag, rotated_v
    
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
        p_values = np.ndarray(shape=obs_vect.shape)
        for t, true in enumerate(obs_vect):
            perm_data = perm_array[:, t]
            p_values[t] = utils.permutation_p(true, perm_data)
        return p_values
    
    def mult_plsc(self, y_table, x_tables):
        """Calculate multitable PLS-C, fixed effect model
        See Krishnan et al., 2011 for more
        """
        corr_xy = self.build_corr_xy(y_table, x_tables)
        centered_corr_xy = utils.center_matrix(corr_xy)
        
        u, delta, v = np.linalg.svd(centered_corr_xy, full_matrices=False)
        return u, delta, v

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
        output : dict
        Dictionary of outputs
            Defaults to original eigenvalues and associated p-values
            return_perm=True includes matrix of permutation eigenvalues
        
        See Krishnan et al., 2011, 'Deciding which latent variables to keep' 
        """
        if not self.n_iters:
            return AttributeError("Number of permutations cannot be None")
        
        orig_svd = self.mult_plsc(y_table, x_tables)
        orig_sing = orig_svd[1]
        
        n = 0
        perm_singular_values = np.ndarray(shape=[self.n_iters, len(orig_sing)])
        while n != self.n_iters:
            #print('Working on iteration %d out of %d' % (int(n+1), self.n_iters))
            perm_y = utils.perm_matrix(y_table)
            perm_x_tables = [utils.perm_matrix(x) for x in x_tables]
            
            perm_svd = self.mult_plsc(perm_y, perm_x_tables)
            rot_perm = self.procrustes_rotation(orig_svd, perm_svd)

            perm_singular_values[n, :] = rot_perm[1]
            n += 1
        
        p_values = self.p_from_perm_mat(orig_sing, perm_singular_values)
        
        output = {'true_eigenvalues' : orig_sing,
                 'p_values' : p_values}
        if self.return_perm:
            output['permutation_eigs'] = perm_singular_values
        
        return output
    
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
        
        See Krishnan et al., 2011, 'Deciding which latent variables to keep' 
        """
        if not self.n_iters:
            raise AttributeError("Number of iterations cannot be None")
            
        orig_svd = self.mult_plsc(y_table, x_tables)
        orig_y_sals = orig_svd[0]
        orig_x_sals = orig_svd[2]
        
        n = 0
        perm_y_sals = np.ndarray(shape=[orig_y_sals.shape, self.n_iters])
        perm_x_sals = np.ndarray(shape=[orig_x_sals.shape, self.n_iters])
        while n != self.n_iters:
            resamp_y = utils.resample_matrix(y_table)
            resamp_x_list = [utils.resample_matrix(x) for x in x_tables]
            
            resamp_svd = self.mult_plsc(resamp_y, resamp_x_list)
            rot_svd = self.procrustes_rotation(orig_svd, resamp_svd)
            perm_y_sals[:, :, n] = rot_svd[0]
            perm_x_sals[:, :, n] = rot_svd[2]
        
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