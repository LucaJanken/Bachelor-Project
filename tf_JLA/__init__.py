"""
.. module:: JLA
    :synopsis: JLA likelihood from Betoule et al. 2014

.. moduleauthor:: Benjamin Audren <benjamin.audren@gmail.com>

Copied from the original c++ code available at
`this address <http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html>`_.
The only modification was to keep the names of the covariance matrices to their
original names instead of going for C00, etc... Here is the conversion from
their original file names to theirs. Note that the convention is used for the
computations afterwards.

.. code::

    C00 = mag_covmat_file
    C11 = stretch_covmat_file
    C22 = colour_covmat_file
    C01 = mag_stretch_covmat_file
    C02 = mag_colour_covmat_file
    C12 = stretch_colour_covmat_file

.. note::

    Since there are a lot of file manipulation involved, the "pandas" library
    has to be installed -- it is an 8-fold improvement in speed over numpy, and
    a 2-fold improvement over a fast Python implementation. The "numexpr"
    library is also needed for doing the fast array manipulations, done with
    blas daxpy function in the original c++ code. Both can be installed with
    pip (Python package manager) easily.

"""
import numpy as np
import tensorflow as tf
import scipy.linalg as la
import montepython.io_mp as io_mp
try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "(sudo) pip install numexpr --user")
from montepython.likelihood_class import Likelihood_sn


class tf_JLA(Likelihood_sn):

    def __init__(self, path, data, command_line):

        # Unusual construction, since the data files are not distributed
        # alongside JLA (size problems)
        try:
            Likelihood_sn.__init__(self, path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The JLA data files were not found. Please download the "
                "following link "
                "http://supernovae.in2p3.fr/sdss_snls_jla/jla_likelihood_v6.tgz"
                ", extract it, and copy all files present in "
                "`jla_likelihood_v6/data` to `your_montepython/data/JLA`")

        # Load matrices from text files, whose names were read in the
        # configuration file
        self.C00 = self.read_matrix(self.mag_covmat_file)
        self.C11 = self.read_matrix(self.stretch_covmat_file)
        self.C22 = self.read_matrix(self.colour_covmat_file)
        self.C01 = self.read_matrix(self.mag_stretch_covmat_file)
        self.C02 = self.read_matrix(self.mag_colour_covmat_file)
        self.C12 = self.read_matrix(self.stretch_colour_covmat_file)

        # Reading light-curve parameters from self.data_file (jla_lcparams.txt)
        self.light_curve_params = self.read_light_curve_parameters()

    def loglkl(self, cosmo, data):
        """
        Compute negative log-likelihood (eq.15 Betoule et al. 2014)

        This version is rewritten for TensorFlow to allow automatic differentiation,
        with a temporary numpy workaround for the luminosity distance calculation.
        """
        # Ensure that input parameters are in TensorFlow format
        redshifts = tf.convert_to_tensor(self.light_curve_params.zcmb.values, dtype=tf.float64)

        # Numpy workaround for luminosity distance
        # Convert redshifts tensor to numpy array, compute luminosity distance, and convert back to tensor
        redshifts_np = redshifts.numpy()
        luminosity_distances_np = np.array([cosmo.luminosity_distance(z) for z in redshifts_np], dtype=np.float64)
        luminosity_distances_tf = tf.convert_to_tensor(luminosity_distances_np, dtype=tf.float64)

        # Distance modulus calculation in TensorFlow, ensuring all constants are float64
        moduli = 5 * tf.math.log1p(luminosity_distances_tf - tf.constant(1.0, dtype=tf.float64)) \
            / tf.math.log(tf.constant(10.0, dtype=tf.float64)) \
            + tf.constant(25.0, dtype=tf.float64)

        # Ensure nuisance parameters are TensorFlow variables or tensors
        alpha = tf.convert_to_tensor(data.mcmc_parameters['alpha']['current'] *
                                    data.mcmc_parameters['alpha']['scale'], dtype=tf.float64)
        beta = tf.convert_to_tensor(data.mcmc_parameters['beta']['current'] *
                                    data.mcmc_parameters['beta']['scale'], dtype=tf.float64)
        M = tf.convert_to_tensor(data.mcmc_parameters['M']['current'] *
                                data.mcmc_parameters['M']['scale'], dtype=tf.float64)
        Delta_M = tf.convert_to_tensor(data.mcmc_parameters['Delta_M']['current'] *
                                    data.mcmc_parameters['Delta_M']['scale'], dtype=tf.float64)

        # Convert light curve parameters and covariance matrix computation to TensorFlow
        mb_tf = tf.convert_to_tensor(self.light_curve_params.mb.values, dtype=tf.float64)
        x1_tf = tf.convert_to_tensor(self.light_curve_params.x1.values, dtype=tf.float64)
        color_tf = tf.convert_to_tensor(self.light_curve_params.color.values, dtype=tf.float64)
        thirdvar_tf = tf.convert_to_tensor(self.light_curve_params.thirdvar.values, dtype=tf.float64)

        C00, C11, C22 = tf.convert_to_tensor(self.C00, dtype=tf.float64), tf.convert_to_tensor(self.C11, dtype=tf.float64), tf.convert_to_tensor(self.C22, dtype=tf.float64)
        C01, C02, C12 = tf.convert_to_tensor(self.C01, dtype=tf.float64), tf.convert_to_tensor(self.C02, dtype=tf.float64), tf.convert_to_tensor(self.C12, dtype=tf.float64)
        
        cov = (C00 + alpha**2 * C11 + beta**2 * C22 +
            2. * alpha * C01 - 2. * beta * C02 - 2. * alpha * beta * C12)

        # Compute residuals in TensorFlow
        residuals = mb_tf - (M - alpha * x1_tf + beta * color_tf + Delta_M * tf.cast(thirdvar_tf > self.scriptmcut, tf.float64))
        residuals -= moduli

        # Add statistical errors to diagonal of covariance matrix
        dmb_tf = tf.convert_to_tensor(self.light_curve_params.dmb.values, dtype=tf.float64)
        dx1_tf = tf.convert_to_tensor(self.light_curve_params.dx1.values, dtype=tf.float64)
        dcolor_tf = tf.convert_to_tensor(self.light_curve_params.dcolor.values, dtype=tf.float64)
        cov_m_s_tf = tf.convert_to_tensor(self.light_curve_params.cov_m_s.values, dtype=tf.float64)
        cov_m_c_tf = tf.convert_to_tensor(self.light_curve_params.cov_m_c.values, dtype=tf.float64)
        cov_s_c_tf = tf.convert_to_tensor(self.light_curve_params.cov_s_c.values, dtype=tf.float64)

        cov += tf.linalg.diag(dmb_tf**2 + (alpha * dx1_tf)**2 + (beta * dcolor_tf)**2 +
                            2. * alpha * cov_m_s_tf - 2. * beta * cov_m_c_tf - 2. * alpha * beta * cov_s_c_tf)

        # Perform Cholesky decomposition and solve
        chol = tf.linalg.cholesky(cov)
        solved_residuals = tf.linalg.cholesky_solve(chol, residuals[..., tf.newaxis])

        # Compute chi^2 as a dot product
        chi2 = tf.reduce_sum(residuals * tf.squeeze(solved_residuals))

        return -0.5 * chi2.numpy()



