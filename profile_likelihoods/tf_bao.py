import os
import numpy as np
import tensorflow as tf
import pickle as pkl
from new_Spline import Spline_tri

class TF_bao():

    # Initialization routine
    def __init__(self, model_name):

        # Load CONNECT model
        self.model = tf.keras.models.load_model(model_name, compile=False)
        self.output_info = eval(self.model.get_raw_info().numpy().decode('utf-8'))

        # Set path to datafile (Hardcoded paths would be avoided for general implementation)
        self.data_directory = 'data/'
        self.file = 'bao_2012.txt'
        self.rs_rescale = 153.017 / 149.0808 # From tf_bao.data file

        # Define emulated data
        self.all_z = np.array(self.output_info['z_bg'])
        self.ang_idx = self.output_info['interval']['bg']['ang.diam.dist.'][:2]
        self.hubble_idx = self.output_info['interval']['bg']['H [1/Mpc]'][:2]
        self.rs_drag_idx = self.output_info['interval']['extra']['rs_drag'][0]

        # Define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # Read redshifts and data points
        for line in open(os.path.join(self.data_directory, self.file), 'r'):
            if (line.find('#') == -1):
                self.z = np.append(self.z, float(line.split()[0]))
                self.data = np.append(self.data, float(line.split()[1]))
                self.error = np.append(self.error, float(line.split()[2]))
                self.type = np.append(self.type, int(line.split()[3]))

        # Spline
        if set(self.all_z).intersection(self.z) != set(self.z):
            S = Spline_tri(tf.constant(self.all_z, dtype=tf.float32), tf.constant(self.z, dtype=tf.float32))
            self.spline = lambda x: S.do_spline(x)
        else:
            self.indices = np.searchsorted(self.all_z, self.z)
            self.spline = lambda x: tf.gather(x, self.indices, axis=1)

    @tf.function
    def loglkl(self, x):
        
        # Define output
        output = self.model(x[:,:-1])

        # Calculations
        hubble = self.spline(output[:, self.hubble_idx[0]:self.hubble_idx[1]])
        da = self.spline(output[:, self.ang_idx[0]:self.ang_idx[1]])
        dr = tf.divide(self.z, hubble)
        dv = tf.pow(da * da * (1 + self.z) * (1 + self.z) * dr, 1. / 3.)
        rs_drag = output[:,self.rs_drag_idx:self.rs_drag_idx+1]

        # Initialize a tensor for the 'theo' values
        theo = tf.zeros_like(self.z, dtype=tf.float32)
        
        # Handle the different types using boolean masks
        mask_type3 = tf.math.equal(self.type, 3)
        rs = rs_drag * self.rs_rescale
        theo = tf.where(mask_type3, dv / rs, theo)
        
        mask_type4 = tf.math.equal(self.type, 4)
        theo = tf.where(mask_type4, dv, theo)

        # Compute chi2
        chi2_values = ((theo - self.data) / self.error) ** 2
        chi2 = tf.reduce_sum(chi2_values, 1, keepdims=True)

        # Compute ln(L)
        lkl = -0.5 * chi2
        
        return lkl
