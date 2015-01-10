# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from gp import GP
from parameterization.param import Param
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from parameterization.variational import VariationalPosterior

import logging
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.inference.optimization.stochastics import SparseGPStochastics,\
    SparseGPMissing
#no stochastics.py file added! from GPy.inference.optimization.stochastics import SparseGPStochastics,\
    #SparseGPMissing
logger = logging.getLogger("sparse gp")

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

    def __init__(self, X, Y, Z, kernel, likelihood, inference_method=None,
                 name='sparse gp', Y_metadata=None, normalizer=False):
        #pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = var_dtc.VarDTC(limit=1 if not self.missing_data else Y.shape[1])
            else:
                #inference_method = ??
                raise NotImplementedError, "what to do what to do?"
            print "defaulting to ", inference_method, "for latent function inference"

        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)

        logger.info("Adding Z as parameter")
        self.link_parameter(self.Z, index=0)
        self.posterior = None

    def has_uncertain_inputs(self):
        return isinstance(self.X, VariationalPosterior)

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata)

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])

        if isinstance(self.X, VariationalPosterior):
            #gradients wrt kernel
            dL_dKmm = self.grad_dict['dL_dKmm']
            self.kern.update_gradients_full(dL_dKmm, self.Z, None)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_expectations(variational_posterior=self.X,
                                                    Z=self.Z,
                                                    dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                                    dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                                    dL_dpsi2=self.grad_dict['dL_dpsi2'])
            self.kern.gradient += kerngrad

            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(dL_dKmm, self.Z)
            self.Z.gradient += self.kern.gradients_Z_expectations(
                               self.grad_dict['dL_dpsi0'],
                               self.grad_dict['dL_dpsi1'],
                               self.grad_dict['dL_dpsi2'],
                               Z=self.Z,
                               variational_posterior=self.X)
        else:
            #gradients wrt kernel
            self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
            kerngrad = self.kern.gradient.copy()
            self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
            kerngrad += self.kern.gradient
            self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z, None)
            self.kern.gradient += kerngrad
            #gradients wrt Z
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
            self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)


    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values
        """

        if kern is None: kern = self.kern

        if not isinstance(Xnew, VariationalPosterior):
            Kx = kern.K(self.Z, Xnew)
            mu = np.dot(Kx.T, self.posterior.woodbury_vector)
            if full_cov:
                Kxx = kern.K(Xnew)
                if self.posterior.woodbury_inv.ndim == 2:
                    var = Kxx - np.dot(Kx.T, np.dot(self.posterior.woodbury_inv, Kx))
                elif self.posterior.woodbury_inv.ndim == 3:
                    var = Kxx[:,:,None] - np.tensordot(np.dot(np.atleast_3d(self.posterior.woodbury_inv).T, Kx).T, Kx, [1,0]).swapaxes(1,2)
                var = var
            else:
                Kxx = kern.Kdiag(Xnew)
                var = (Kxx - np.sum(np.dot(np.atleast_3d(self.posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T
        else:
            Kx = kern.psi1(self.Z, Xnew).T
            mu = np.dot(Kx.T, self.posterior.woodbury_vector)
            if full_cov:
                Kxx = kern.K(Xnew.mean)
                if self.posterior.woodbury_inv.ndim == 2:
                    var = Kxx - np.dot(Kx.T, np.dot(self.posterior.woodbury_inv, Kx))
                elif self.posterior.woodbury_inv.ndim == 3:
                    var = Kxx[:,:,None] - np.tensordot(np.dot(np.atleast_3d(self.posterior.woodbury_inv).T, Kx).T, Kx, [1,0]).swapaxes(1,2)
            else:
                Kxx = kern.psi0(self.Z, Xnew)
                var = (Kxx - np.sum(np.dot(np.atleast_3d(self.posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T
        return mu, var
