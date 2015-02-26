# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .gp import GP
from .parameterization.param import Param
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from .parameterization.variational import VariationalPosterior, NormalPosterior
from ..util.linalg import mdot

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
                raise NotImplementedError("what to do what to do?")
            print("defaulting to ", inference_method, "for latent function inference")

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
        Make a prediction for the latent function values. 
    
        For certain inputs we give back a full_cov of shape NxN,
        if there is missing data, each dimension has its own full_cov of shape NxNxD, and if full_cov is of, 
        we take only the diagonal elements across N.
        
        For uncertain inputs, the SparseGP bound produces a full covariance structure across D, so for full_cov we 
        return a NxDxD matrix and in the not full_cov case, we return the diagonal elements across D (NxD).
        This is for both with and without missing data.
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
            psi0_star = self.kern.psi0(self.Z, Xnew)
            psi1_star = self.kern.psi1(self.Z, Xnew)
            #psi2_star = self.kern.psi2(self.Z, Xnew) # Only possible if we get NxMxM psi2 out of the code.
            la = self.posterior.woodbury_vector
            mu = np.dot(psi1_star, la) # TODO: dimensions?
            
            if full_cov: 
                var = np.empty((Xnew.shape[0], la.shape[1], la.shape[1]))
                di = np.diag_indices(la.shape[1])
            else: 
                var = np.empty((Xnew.shape[0], la.shape[1]))
                
            for i in range(Xnew.shape[0]):
                _mu, _var = Xnew.mean.values[[i]], Xnew.variance.values[[i]]
                psi2_star = self.kern.psi2(self.Z, NormalPosterior(_mu, _var))
                tmp = (psi2_star[:, :] - psi1_star[[i]].T.dot(psi1_star[[i]]))

                var_ = mdot(la.T, tmp, la)
                p0 = psi0_star[i]
                t = np.atleast_3d(self.posterior.woodbury_inv)
                t2 = np.trace(t.T.dot(psi2_star), axis1=1, axis2=2)
                
                if full_cov:
                    var_[di] += p0
                    var_[di] += -t2
                    var[i] = var_
                else:
                    var[i] = np.diag(var_)+p0-t2
        return mu, var
