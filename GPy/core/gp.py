# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from gp_base import GPBase
from ..util.linalg import dtrtrs
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from .. import likelihoods

class GP(GPBase):
    """
    Gaussian Process model for regression and EP

    :param X: input observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :param likelihood: a GPy likelihood
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """
    def __init__(self, X, Y, kernel, likelihood, inference_method=None, name='gp'):

        #find a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
        else:
            inference_method = expectation_propagation
            print "defaulting to ", inference_method, "for latent function inference"

        super(GP, self).__init__(X, Y, kernel, likelihood, inference_method, name)
        self.parameters_changed()

    def parameters_changed(self):
        super(GP, self).parameters_changed()
        self.K = self.kern.K(self.X)
        self.posterior = self.inference_method.inference(self.K, self.likelihood, self.Y)

    def dL_dtheta_K(self):
        return self.kern.dK_dtheta(self.posterior.dL_dK, self.X)

    def log_likelihood(self):
        return self.posterior.log_marginal

    def _raw_predict(self, _Xnew, which_parts='all', full_cov=False, stop=False):
        """
        Internal helper function for making predictions, does not account
        for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        """
        Kx = self.kern.K(_Xnew, self.X, which_parts=which_parts).T
        LiKx, _ = dtrtrs(self.posterior._woodbury_chol, np.asfortranarray(Kx), lower=1)
        mu = np.dot(Kx.T, self.posterior._woodbury_vector)
        if full_cov:
            Kxx = self.kern.K(_Xnew, which_parts=which_parts)
            var = Kxx - tdot(LiKx.T)
        else:
            Kxx = self.kern.Kdiag(_Xnew, which_parts=which_parts)
            var = Kxx - np.sum(LiKx*LiKx, 0)
            var = var.reshape(-1, 1)
        return mu, var

    def predict(self, Xnew, which_parts='all', full_cov=False, **likelihood_args):
        """
        Predict the function(s) at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: bool
        :returns: mean: posterior mean,  a Numpy array, Nnew x self.input_dim
        :returns: var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :returns: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.input_dim


           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        # normalize X values
        Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, which_parts=which_parts)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov, **likelihood_args)
        return mean, var, _025pm, _975pm


