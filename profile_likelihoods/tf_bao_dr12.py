import os
import numpy as np
import scipy.constants as conts
import tensorflow as tf
import pickle as pkl
from new_Spline import Spline_tri

class TF_bao_boss_dr12():

    # Initialization routine
    def __init__(self, model_name):

        # Load CONNECT model
        self.model = tf.keras.models.load_model(model_name, compile=False)
        self.output_info = eval(self.model.get_raw_info().numpy().decode('utf-8'))
        self.rs_rescale, self.rd_fid_in_Mpc = 1. , 147.78

        # Define emulated data
        self.all_z = np.array(self.output_info['z_bg'])
        self.ang_idx = self.output_info['interval']['bg']['ang.diam.dist.'][:2]
        self.hubble_idx = self.output_info['interval']['bg']['H [1/Mpc]'][:2]
        self.rs_drag_idx = self.output_info['interval']['extra']['rs_drag'][0]

        # Set path to datafile (Hardcoded paths would be avoided for general implementation)
        self.data_directory = 'data/COMBINEDDR12_BAO_consensus_dM_Hz/'
        self.data_file = 'BAO_consensus_results_dM_Hz.txt'
        self.cov_file = 'BAO_consensus_covtot_dM_Hz.txt'

        data = np.loadtxt(os.path.join(self.data_directory, self.data_file), skiprows=1, usecols=(0,2))
        self.z = np.array(list(set(data[:,0])))
        self.DM_rdfid_by_rd_in_Mpc, self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.reshape(data[:,1], [3,2]).T

        # Read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))
        self.inv_cov_data = tf.cast(tf.linalg.inv(self.cov_data), tf.float32)

        # Spline
        if set(self.all_z).intersection(self.z) != set(self.z):
            S = Spline_tri(tf.constant(self.all_z, dtype=tf.float32), tf.constant(self.z, dtype=tf.float32))
            self.spline = lambda x: S.do_spline(x)
        else:
            self.indices = np.searchsorted(self.all_z, self.z)
            self.spline = lambda x: tf.gather(x, self.indices, axis=1)

    # compute likelihood
    @tf.function
    def loglkl(self, x):
        # Define output
        output = self.model(x[:,:-1])
        
        # Compute comoving angular diameter distance D_M = (1 + z) * D_A and Hubble values for each z
        hubble = self.spline(output[:, self.hubble_idx[0]:self.hubble_idx[1]])
        da = self.spline(output[:, self.ang_idx[0]:self.ang_idx[1]])
        DM_at_z_values = da * (1. + self.z)
        H_at_z_values = hubble * conts.c / 1000.0
        rs_drag = output[:,self.rs_drag_idx:self.rs_drag_idx+1]

        # Compute sound horizon at baryon drag rs_d
        rd = rs_drag * self.rs_rescale

        # Compute theoretical predictions
        theo_DM_rdfid_by_rd_in_Mpc = DM_at_z_values / rd * self.rd_fid_in_Mpc
        theo_H_rd_by_rdfid = H_at_z_values * rd / self.rd_fid_in_Mpc

        # Compute differences with observations
        DM_diff = tf.transpose(theo_DM_rdfid_by_rd_in_Mpc - self.DM_rdfid_by_rd_in_Mpc)
        H_diff = tf.transpose(theo_H_rd_by_rdfid - self.H_rd_by_rdfid_in_km_per_s_per_Mpc)

        # Stack DM_diff and H_diff into a single array
        data_array = tf.transpose(tf.dynamic_stitch([[0,2,4],[1,3,5]], [DM_diff, H_diff]))

        # Compute chi squared
        chi2 = tf.reshape(tf.reduce_sum(tf.multiply(tf.tensordot(data_array, self.inv_cov_data,1), data_array), 1), [-1,1])
        
        # Return ln(L)
        loglkl = -0.5 * chi2
        
        return loglkl
