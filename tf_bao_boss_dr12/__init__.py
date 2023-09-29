import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
import tensorflow as tf

class tf_bao_boss_dr12(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # are there conflicting experiments?
        conflicting_experiments = [
            'bao', 'bao_boss', 'bao_known_rs'
            'bao_boss_aniso', 'bao_boss_aniso_gauss_approx']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'conflicting BAO measurments')

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.DM_rdfid_by_rd_in_Mpc = np.array([], 'float64')
        self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.array([], 'float64')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dM(rsfid/rs)':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.DM_rdfid_by_rd_in_Mpc = np.append(
                            self.DM_rdfid_by_rd_in_Mpc, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.append(
                            self.H_rd_by_rdfid_in_km_per_s_per_Mpc, float(this_line[2]))

        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))

        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.cov_data)[0]

        # end of initialization

    # compute likelihood
    def loglkl(self, cosmo, data):
        # Compute comoving angular diameter distance D_M = (1 + z) * D_A
        # and Hubble values for each z, using list comprehensions
        DM_at_z_values = [cosmo.angular_distance(z_val) * (1. + z_val) for z_val in self.z]
        H_at_z_values = [cosmo.Hubble(z_val) * conts.c / 1000.0 for z_val in self.z]

        # Convert these lists to TensorFlow tensors
        DM_at_z = tf.convert_to_tensor(DM_at_z_values, dtype=tf.float64)
        H_at_z = tf.convert_to_tensor(H_at_z_values, dtype=tf.float64)

        # Compute sound horizon at baryon drag rs_d
        rd = cosmo.rs_drag() * self.rs_rescale

        # Compute theoretical predictions
        theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.rd_fid_in_Mpc
        theo_H_rd_by_rdfid = H_at_z * rd / self.rd_fid_in_Mpc

        # Compute differences with observations
        DM_diff = theo_DM_rdfid_by_rd_in_Mpc - self.DM_rdfid_by_rd_in_Mpc
        H_diff = theo_H_rd_by_rdfid - self.H_rd_by_rdfid_in_km_per_s_per_Mpc

        # Stack DM_diff and H_diff into a single array
        data_array = tf.stack([DM_diff, H_diff], axis=-1)
        data_array = tf.reshape(data_array, [-1])

        # Compute chi squared
        inv_cov_data = tf.linalg.inv(self.cov_data)
        chi2 = tf.tensordot(tf.tensordot(data_array, inv_cov_data, 1), data_array, 1)

        # Return ln(L)
        loglkl = -0.5 * chi2
        
        return loglkl.numpy()
    










