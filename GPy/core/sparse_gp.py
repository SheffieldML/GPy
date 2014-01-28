# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.linalg import mdot, tdot, symmetrify, backsub_both_sides, chol_inv, dtrtrs, dpotrs, dpotri
from gp import GP
from parameterization.param import Param
from ..inference.latent_function_inference import varDTC

class SparseGP(GP):
    """
    A general purpose Sparse GP model

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param Z: inducing inputs
    :type Z: np.ndarray (num_inducing x input_dim)
    :param num_inducing: Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type num_inducing: int

    """

    def __init__(self, X, Y, Z, kernel, likelihood, inference_method=None, X_variance=None, name='sparse gp'):

        #pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = varDTC.Gaussian_inference()
        else:
            #inference_method = ??
            raise NotImplementedError, "what to do what to do?"
            print "defaulting to ", inference_method, "for latent function inference"

        GP.__init__(self, X, Y, likelihood, inference_method, kernel, name)

        self.Z = Z
        self.num_inducing = Z.shape[0]

        if X_variance is None:
            self.has_uncertain_inputs = False
            self.X_variance = None
        else:
            assert X_variance.shape == X.shape
            self.has_uncertain_inputs = True
            self.X_variance = X_variance

        self.Z = Param('inducing inputs', self.Z)
        self.add_parameter(self.Z, gradient=self.dL_dZ, index=0)
        self.add_parameter(self.kern, gradient=self.dL_dtheta)
        self.add_parameter(self.likelihood, gradient=lambda:self.likelihood._gradients(partial=self.partial_for_likelihood))

    def parameters_changed(self):
        self.posterior = self.inference_method.inference(self.kern, self.X, self.X_variance, self.Z, self.likelihood, self.Y)

                #The derivative of the bound wrt the inducing inputs Z
        self.Z.gradient = self.kern.dK_dX(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            self.Z.gradient += self.kern.dpsi1_dZ(self.dL_dpsi1, self.Z, self.X, self.X_variance)
            self.Z.gradient += self.kern.dpsi2_dZ(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            self.Z.gradient += self.kern.dK_dX(self.dL_dpsi1.T, self.Z, self.X)

    def _raw_predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False):
        """
        Make a prediction for the latent function values
        """
        #TODO!!!


    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return GP._getstate(self) + [self.Z,
                self.num_inducing,
                self.has_uncertain_inputs,
                self.X_variance]

    def _setstate(self, state):
        self.X_variance = state.pop()
        self.has_uncertain_inputs = state.pop()
        self.num_inducing = state.pop()
        self.Z = state.pop()
        GP._setstate(self, state)
