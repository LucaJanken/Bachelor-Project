import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import tensorflow as tf


class tf_bao_boss(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # exclude the isotropic CMASS experiment when the anisotrpic
        # measurement is also used
        exclude_isotropic_CMASS = False

        conflicting_experiments = [
            'bao_boss_aniso', 'bao_boss_aniso_gauss_approx']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                exclude_isotropic_CMASS = True

        if exclude_isotropic_CMASS:
            warnings.warn("excluding isotropic CMASS measurement")
            if not hasattr(self, 'exclude') or self.exclude == None:
                self.exclude = ['CMASS']
            else:
                self.exclude.append('CMASS')

        # define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.file), 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()
                    # insert into array if this id is not manually excluded
                    if not this_line[0] in self.exclude:
                        self.z = np.append(self.z, float(this_line[1]))
                        self.data = np.append(self.data, float(this_line[2]))
                        self.error = np.append(self.error, float(this_line[3]))
                        self.type = np.append(self.type, int(this_line[4]))

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # Convert the z values to hubble values and angular distances
        hubble_values = [cosmo.Hubble(z_val) for z_val in self.z]
        hubble_tensor = tf.convert_to_tensor(hubble_values, dtype=tf.float64)
    
        da_values = [cosmo.angular_distance(z_val) for z_val in self.z]
        da = tf.convert_to_tensor(da_values, dtype=tf.float64)

        dr_values = [z_val / cosmo.Hubble(z_val) for z_val in self.z]
        dr = tf.convert_to_tensor(dr_values, dtype=tf.float64)

        dv = tf.math.pow(da * da * (1 + self.z) * (1 + self.z) * dr, 1. / 3.)

        # Initialize a tensor for the 'theo' values
        theo = tf.zeros_like(self.z, dtype=tf.float64)

        # Handle the different types using boolean masks and compute theo accordingly
        rs = cosmo.rs_drag()

        mask_type3 = tf.math.equal(self.type, 3)
        theo = tf.where(mask_type3, dv / rs, theo)

        mask_type4 = tf.math.equal(self.type, 4)
        theo = tf.where(mask_type4, dv, theo)

        mask_type5 = tf.math.equal(self.type, 5)
        theo = tf.where(mask_type5, da / rs, theo)

        mask_type6 = tf.math.equal(self.type, 6)
        theo = tf.where(mask_type6, 1. / hubble_tensor / rs, theo)

        mask_type7 = tf.math.equal(self.type, 7)
        theo = tf.where(mask_type7, rs / dv, theo)

        # Check if there are any unhandled types
        masks_combined = mask_type3 | mask_type4 | mask_type5 | mask_type6 | mask_type7
        if tf.reduce_any(~masks_combined):
            raise ValueError("Unrecognized BAO data type.")

        # Compute chi2
        chi2_values = ((theo - self.data) / self.error) ** 2
        chi2 = tf.reduce_sum(chi2_values)

        # Return ln(L)
        lkl = -0.5 * chi2

        return lkl.numpy()

