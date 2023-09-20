import os
import numpy as np
from montepython.likelihood_class import Likelihood
import tensorflow as tf


# Updated tf_bao class with filter_lines() function integrated

class tf_bao(Likelihood):

    # initialization routine
    def __init__(self, path, data, command_line):
        super().__init__(path, data, command_line)

        # read redshifts and data points into a list of lines
        with open(tf.io.gfile.join(self.data_directory, self.file), 'r') as f:
            lines = f.readlines()

        # convert the list of lines into a tensor of strings
        lines_array = tf.convert_to_tensor(lines, dtype=tf.string)

        # filter out lines containing the '#' character
        filtered_lines = self.filter_lines(lines_array)

        # split each line and extract the data columns
        data_columns = tf.strings.split(filtered_lines).to_list()

        # convert the list of lists into a 2D tensor
        data_matrix = tf.stack(data_columns)

        # assign the columns to the respective arrays of z and data points
        self.z = tf.strings.to_number(data_matrix[:, 0], tf.float64)
        self.data = tf.strings.to_number(data_matrix[:, 1], tf.float64)
        self.error = tf.strings.to_number(data_matrix[:, 2], tf.float64)
        self.type = tf.strings.to_number(data_matrix[:, 3], tf.int32)

        # number of data points
        self.num_points = tf.shape(self.z)[0]

    def filter_lines(self, lines_tensor):
        # Convert tensor to list
        lines_list = lines_tensor.numpy().tolist()

        # Decode byte-like strings to regular strings and filter out lines containing the '#' character
        filtered_lines_list = [line.decode('utf-8') for line in lines_list if b'#' not in line]

        # Convert the list back to a tensor
        filtered_lines_tensor = tf.convert_to_tensor(filtered_lines_list, dtype=tf.string)

        return filtered_lines_tensor

    # compute likelihood
    def loglkl(self, cosmo, data):

        # calculations
        da = [cosmo.angular_distance(z_val.numpy()) for z_val in self.z]
        da = tf.convert_to_tensor(da, dtype=tf.float64)
        dr = [z_val.numpy() / cosmo.Hubble(z_val.numpy()) for z_val in self.z]
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

