# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.linalg import mdot, tdot, symmetrify, backsub_both_sides, dtrtrs, dpotrs, dpotri
from gp import GP
from parameterization.param import Param
from ..inference.latent_function_inference import varDTC
from .. import likelihoods
from GPy.util.misc import param_to_array

class SparseGP(GP):
    """
    A general purpose Sparse GP model

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

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
                inference_method = varDTC.VarDTC()
            else:
                #inference_method = ??
                raise NotImplementedError, "what to do what to do?"
            print "defaulting to ", inference_method, "for latent function inference"

        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        if not (X_variance is None):
            assert X_variance.shape == X.shape
        self.X_variance = X_variance

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name)
        self.add_parameter(self.Z, index=0)
        self.parameters_changed()

    def _update_gradients_Z(self, add=False):
    #The derivative of the bound wrt the inducing inputs Z ( unless they're all fixed)
        if not self.Z.is_fixed:
            if add: self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
            else: self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
            if self.X_variance is None:
                self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
            else:
                self.Z.gradient += self.kern.dpsi1_dZ(self.grad_dict['dL_dpsi1'], self.Z, self.X, self.X_variance)
                self.Z.gradient += self.kern.dpsi2_dZ(self.grad_dict['dL_dpsi2'], self.Z, self.X, self.X_variance)

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.X_variance, self.Z, self.likelihood, self.Y)
        self._update_gradients_Z(add=False)

    def _raw_predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False):
        """
        Make a prediction for the latent function values
        """
        if X_variance_new is None:
            Kx = self.kern.K(self.Z, Xnew, which_parts=which_parts)
            mu = np.dot(Kx.T, self.posterior.woodbury_vector)
            if full_cov:
                Kxx = self.kern.K(Xnew, which_parts=which_parts)
                var = Kxx - mdot(Kx.T, self.posterior.woodbury_inv, Kx) # NOTE this won't work for plotting
            else:
                Kxx = self.kern.Kdiag(Xnew, which_parts=which_parts)
                var = Kxx - np.sum(Kx * np.dot(self.posterior.woodbury_inv, Kx), 0)
        else:
            # assert which_parts=='all', "swithching out parts of variational kernels is not implemented"
            Kx = self.kern.psi1(self.Z, Xnew, X_variance_new) # , which_parts=which_parts) TODO: which_parts
            mu = np.dot(Kx, self.Cpsi1V)
            if full_cov:
                raise NotImplementedError, "TODO"
            else:
                Kxx = self.kern.psi0(self.Z, Xnew, X_variance_new)
                psi2 = self.kern.psi2(self.Z, Xnew, X_variance_new)
                var = Kxx - np.sum(np.sum(psi2 * Kmmi_LmiBLmi[None, :, :], 1), 1)
        return mu, var[:,None]


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
