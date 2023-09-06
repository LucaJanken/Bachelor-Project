import os
import numpy as np
from montepython.likelihood_class import Likelihood


class tf_bao(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # read redshifts and data points into a list of lines
        with open(os.path.join(self.data_directory, self.file), 'r') as f:
            lines = f.readlines()

        # convert the list of lines into a numpy array of strings
        lines_array = np.array(lines)

        # filter out lines containing the '#' character
        filtered_lines = lines_array[np.char.find(lines_array, '#') == -1]

        # split each line and extract the data columns
        data_columns = np.char.split(filtered_lines).tolist()

        # convert the list of lists into a 2D numpy array
        data_matrix = np.array(data_columns, dtype='object')

        # assign the columns to the respective arrays of z and data points
        self.z = np.array(data_matrix[:, 0], dtype='float64')
        self.data = np.array(data_matrix[:, 1], dtype='float64')
        self.error = np.array(data_matrix[:, 2], dtype='float64')
        self.type = np.array(data_matrix[:, 3], dtype='int')

        # number of data points
        self.num_points = np.shape(self.z)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.

        # for each point, compute angular distance da, radial distance dr,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # theoretical prediction and chi2 contribution
        
        # Vectorized calculations
        da = cosmo.angular_distance(self.z)
        dr = self.z / cosmo.Hubble(self.z)
        dv = np.power(da * da * (1 + self.z) * (1 + self.z) * dr, 1. / 3.)

        # Initialize an array for the 'theo' values
        theo = np.zeros_like(self.z)

        # Handle the different types using boolean masks
        mask_type3 = self.type == 3
        rs = cosmo.rs_drag() * self.rs_rescale
        theo[mask_type3] = dv[mask_type3] / rs

        mask_type4 = self.type == 4
        theo[mask_type4] = dv[mask_type4]

        # Check if there are any unhandled types
        if np.any(~(mask_type3 | mask_type4)):
            raise io_mp.LikelihoodError("Unrecognized BAO data type.")

        # Compute chi2 in a vectorized manner
        chi2_values = ((theo - self.data) / self.error) ** 2
        chi2 = np.sum(chi2_values)

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl