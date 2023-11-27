import os
import numpy as np
import tensorflow as tf
import pickle as pkl

class TF_bao_boss():

    # Initialization routine
    def __init__(self, model_name):

        # Load CONNECT model
        self.model = tf.keras.models.load_model(model_name, compile=False)
        self.output_info = eval(self.model.get_raw_info().numpy().decode('utf-8'))
        
        # Set path to datafile (Hardcoded paths would be avoided for general implementation)
        self.data_directory = 'data/'
        self.file = 'bao_2014.txt'
        
        # Define emulated data
        self.ang_idx = self.output_info['interval']['bg']['ang.diam.dist.'][:2]
        self.hubble_idx = self.output_info['interval']['bg']['H [1/Mpc]'][:2]
        self.rs_drag_idx = self.output_info['interval']['extra']['rs_drag'][0]
        
        # Define array for values of z and data points
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'int')

        # Read redshifts and data points
        with open(os.path.join(self.data_directory, self.file), 'r') as filein:
            for line in filein:
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()
                    # insert into array if this id is not manually excluded
                    if not hasattr(self, 'exclude') or self.exclude == None or not this_line[0] in self.exclude:
                        self.z = np.append(self.z, float(this_line[1]))
                        self.data = np.append(self.data, float(this_line[2]))
                        self.error = np.append(self.error, float(this_line[3]))
                        self.type = np.append(self.type, int(this_line[4]))

        # Number of data points
        self.num_points = np.shape(self.z)[0]

    @tf.function
    def loglkl(self, x):

        # Define output
        output = self.model(x[:,:-1])
        
        # Calculations
        hubble = output[:, self.hubble_idx[0]:self.hubble_idx[1]]
        da = output[:, self.ang_idx[0]:self.ang_idx[1]]
        dr = tf.divide(self.z, hubble)
        dv = tf.pow(da * da * (1 + self.z) * (1 + self.z) * dr, 1. / 3.)
        rs_drag = output[:,self.rs_drag_idx:self.rs_drag_idx+1]

        # Initialize a tensor for the 'theo' values
        theo = tf.zeros_like(self.z, dtype=tf.float32)

        # Handle the different types using boolean masks and compute theo accordingly

        mask_type3 = tf.math.equal(self.type, 3)
        rs = rs_drag 
        theo = tf.where(mask_type3, dv / rs, theo)

        mask_type4 = tf.math.equal(self.type, 4)
        theo = tf.where(mask_type4, dv, theo)

        mask_type5 = tf.math.equal(self.type, 5)
        theo = tf.where(mask_type5, da / rs, theo)

        mask_type6 = tf.math.equal(self.type, 6)
        theo = tf.where(mask_type6, 1. / hubble / rs, theo)

        mask_type7 = tf.math.equal(self.type, 7)
        theo = tf.where(mask_type7, rs / dv, theo)

        # Compute chi2
        chi2_values = ((theo - self.data) / self.error) ** 2
        chi2 = tf.reduce_sum(chi2_values)

        # Compute ln(L)
        lkl = -0.5 * chi2

        return lkl.numpy()
