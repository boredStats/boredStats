"""Tools for running Partial-Least Squares Analyses

Finished:
Multitable PLS-C

TODO:
Projecting observations into subspace
Multtable PLS-R
Support for pandas
Test suite

Note: These should be able to handle basic PLS-C/ PLS-R analyses since it only
requires a list of arrays
"""

from . import utils
from .corr_tools import cross_corr

import numpy as np
import pandas as pd

class MultitablePLSC(object):
    def __init__(self, center=True, scale=False,
                 n_iters=None, return_perm=False):
        self.center = center
        self.scale = scale
        self.n_iters = n_iters
        self.return_perm = return_perm

    @staticmethod
    def _clean_tables(table):
        #Check input type, return as list of numpy arrays
        outlist = []
        if isinstance(table, list):
            for t in table:
                if isinstance(t, np.ndarray):
                    outlist.append(t)
                elif isinstance(t, pd.DataFrame):
                    outlist.append(t.values)
                else:
                    return ValueError("Input was not an array or dataframe")
                
        if isinstance(table, np.ndarray):
            outlist.append(table)
        if isinstance(table, pd.DataFrame):
            outlist.append(table.values)
        
        return outlist
    
    @staticmethod
    def _procrustes_rotation(orig_svd, resamp_svd):
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

#        original_v = orig_svd[2]
#        perm_u = resamp_svd[0]
#        perm_diag = resamp_svd[1]
#        perm_v = resamp_svd[2]
#
#        n, _, p = np.linalg.svd(np.dot(original_v.T, perm_v), full_matrices=False)
#        rotation_matrix = np.dot(n, p.T)
#
#        rotated_u = np.dot(np.dot(perm_u, np.diagflat(perm_diag)), rotation_matrix)
#        rotated_v = np.dot(rotation_matrix, np.dot(np.diagflat(perm_diag), perm_v))
#
#        sum_of_squares_rotated_u = np.sum(rotated_v[:, :]**2, 0)
#        rotated_diag = np.sqrt(sum_of_squares_rotated_u)

        original_u = orig_svd[0]
        perm_u = resamp_svd[0]
        perm_v = resamp_svd[2]
        perm_diag = resamp_svd[1]

        n, _, p = np.linalg.svd(np.dot(original_u.T, perm_u),
                                full_matrices=False)

        rotation_matrix = np.dot(n, p.T)

        rotated_u = np.dot(np.dot(perm_u, np.diagflat(perm_diag)),
                           rotation_matrix)

        rotated_v = np.dot(rotation_matrix,
                           np.dot(np.diagflat(perm_diag), perm_v))

        try:
            sum_of_squares_rotated_u = np.sum(rotated_v[:, :]**2, 1)
            rotated_diag = np.sqrt(sum_of_squares_rotated_u)
        except RuntimeWarning as err:
            if 'overflow' in err:
                raise OverflowError #catch overflow to rerun permutation

        return rotated_u, rotated_diag, rotated_v

    @staticmethod
    def _p_from_perm_mat(obs_vect, perm_array):
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

    @staticmethod
    def _bootstrap_z(true_observations, permutation_cube, z_test):
        """Calculate "z-scores" from a cube of permutation data

        Works in tandem with mult_plsc_boostrap_saliences
        """
        standard_dev = np.std(permutation_cube, axis=-1)
        standard_err = standard_dev / np.sqrt(permutation_cube.shape[2])
        bootz = np.divide(true_observations, standard_err)

        zfilt = np.where(np.abs(bootz) < z_test)

        #create a copy of data safely using numpy only
        filtered_observations = np.ndarray(shape=true_observations.shape)
        filtered_observations[:, :] = true_observations
        for i in range(len(zfilt[0])):
            row = zfilt[0][i]
            col = zfilt[1][i]
            filtered_observations[row, col] = 0

        return filtered_observations, bootz

    def mult_plsc(self, y_tables=None, x_tables=None, corr_xy=None):
        """Calculate multitable PLS-C, fixed effect model

        Parameters
        ----------
        y_tables: numpy array, pandas dataframe, or list either

        x_tables:  numpy array, pandas dataframe, or list either

        corr_xy: numpy array
            Pre-calculated cross-correlation matrix

        See Krishnan et al., 2011 for more
        """
        
        if corr_xy is None:
            x = self._clean_tables(x_tables)
            y = self._clean_tables(y_tables)
            for yt in y:
                for xt in x:
                    if yt.shape[0] != xt.shape[0]:
                        raise RuntimeError("Tables need same number of subjects")
            
            corr_xy = cross_corr(np.hstack(x), np.hstack(y))
        
        if self.center:
            corr_xy = utils.center_matrix(corr_xy)
        if self.scale:
            corr_xy = utils.scale_matrix(corr_xy)
        
        u, delta, v = np.linalg.svd(corr_xy, full_matrices=False)
        return u, delta, v

    def mult_plsc_eigenperm(self, y_tables, x_tables):
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

        orig_svd = self.mult_plsc(y_tables, x_tables)
        orig_sing = orig_svd[1]

        n = 0
        perm_singular_values = np.ndarray(shape=[self.n_iters, len(orig_sing)])
        while n != self.n_iters:
            try:
                y_cleaned = self._clean_tables(y_tables)
                perm_y = [utils.perm_matrix(yt) for yt in y_cleaned]
                x_cleaned = self._clean_tables(x_tables)
                perm_x = [utils.perm_matrix(xt) for xt in x_cleaned]

                perm_svd = self.mult_plsc(perm_y, perm_x)
            except np.linalg.LinAlgError:
                continue #Rerun permutation if SVD doesn't converge

            try:
                rot_perm = self._procrustes_rotation(orig_svd, perm_svd)
            except OverflowError:
                continue #Rerun permutation if overflow

            perm_singular_values[n, :] = rot_perm[1]
            n += 1

        p_values = self._p_from_perm_mat(orig_sing, perm_singular_values)

        output = {'svd_res' : orig_svd,
                  'true_eigenvalues' : orig_sing,
                  'p_values' : p_values}
        if self.return_perm:
            output['permutation_eigs'] = perm_singular_values

        return output

    def mult_plsc_bootstrap_saliences(self, y_tables, x_tables, z_tester=2):
        """Run bootstrap testing on saliences

        Parameters:
        -----------
        y_taable, x_tables: inputs for multitable PLSC

        z_tester: int
            "Z-score" to test bootstrap samples with
            Default is 2 (or approximately 1.96)

        Returns:
        --------
        output: dict
            Dictionary of filtered saliences
            If return_perm is True, output will include the "z-scores" and
            permutation cube data for the salience matrices

        See Krishnan et al., 2011, 'Deciding which latent variables to keep'
        """
        if not self.n_iters:
            raise AttributeError("Number of iterations cannot be None")

        true_svd = self.mult_plsc(y_tables, x_tables)
        true_y_saliences = true_svd[2] #saliences for y-table
        true_x_saliences = true_svd[0] #saliences for x-tables

        perm_y_saliences = np.ndarray(shape=[true_y_saliences.shape[0],
                                             true_y_saliences.shape[1],
                                             self.n_iters])
        perm_x_saliences = np.ndarray(shape=[true_x_saliences.shape[0],
                                             true_x_saliences.shape[1],
                                             self.n_iters])

        n = 0
        while n != self.n_iters:
            try:
                y_cleaned = self._clean_tables(y_tables)
                resampled_y = [utils.resample_matrix(y) for y in y_cleaned]
                
                x_cleaned = self._clean_tables(x_tables)
                resampled_x = [utils.resample_matrix(x) for x in x_cleaned]

                resampled_svd = self.mult_plsc(resampled_y, resampled_x)
            except np.linalg.LinAlgError:
                continue #Rerun permutation if SVD doesn't converge

            try:
                rotated_svd = self._procrustes_rotation(true_svd,
                                                        resampled_svd)
            except OverflowError:
                continue #Rerun permutation if overflow

            perm_y_saliences[:, :, n] = rotated_svd[2]
            perm_x_saliences[:, :, n] = rotated_svd[0]
            n += 1

        threshold_y_saliences, y_zscores = self._bootstrap_z(true_y_saliences,
                                                             perm_y_saliences,
                                                             z_tester)
        threshold_x_saliences, x_zscores = self._bootstrap_z(true_x_saliences,
                                                             perm_x_saliences,
                                                             z_tester)

        output = {'y_saliences' : threshold_y_saliences,
                  'x_saliences' : threshold_x_saliences}
        
        if self.return_perm:
            output['zscores_y_saliences'] = y_zscores
            output['zscores_x_saliences'] = x_zscores
            output['permcube_y_saliences'] = perm_y_saliences
            output['permcube_x_saliences'] = perm_x_saliences

        return output

if __name__ == "__main__":
    pass