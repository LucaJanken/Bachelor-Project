import os
import numpy as np
from montepython.likelihood_class import Likelihood
import tensorflow as tf


class tf_bao(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        for line in open(os.path.join(
                self.data_directory, self.file), 'r'):
            if (line.find('#') == -1):
                self.z = np.append(self.z, float(line.split()[0]))
                self.data = np.append(self.data, float(line.split()[1]))
                self.error = np.append(self.error, float(line.split()[2]))
                self.type = np.append(self.type, int(line.split()[3]))

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # calculations
        da = [cosmo.angular_distance(z_val) for z_val in self.z]
        da = tf.convert_to_tensor(da, dtype=tf.float64)
        dr = [z_val / cosmo.Hubble(z_val) for z_val in self.z]
        dr = tf.convert_to_tensor(dr, dtype=tf.float64)
        dv = tf.math.pow(da * da * (1 + self.z) * (1 + self.z) * dr, 1. / 3.)

        # Initialize a tensor for the 'theo' values
        theo = tf.zeros_like(self.z, dtype=tf.float64)

        # Handle the different types using boolean masks
        mask_type3 = tf.math.equal(self.type, 3)
        rs = cosmo.rs_drag() * self.rs_rescale
        theo = tf.where(mask_type3, dv / rs, theo)

        mask_type4 = tf.math.equal(self.type, 4)
        theo = tf.where(mask_type4, dv, theo)

        # Check if there are any unhandled types
        if tf.reduce_any(~(mask_type3 | mask_type4)):
           raise ValueError("Unrecognized BAO data type.")

        # Compute chi2
        chi2_values = ((theo - self.data) / self.error) ** 2
        chi2 = tf.reduce_sum(chi2_values)

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl.numpy()
