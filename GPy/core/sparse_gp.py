# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .gp import GP
from .parameterization.param import Param
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from GPy.core.parameterization.variational import VariationalPosterior

import logging
logger = logging.getLogger("sparse gp")

class SparseGP(GP):
    """
    A general purpose Sparse GP model

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

    This is not for missing data, as the implementation for missing data involves
    some inefficient optimization routine decisions.
    See missing data SparseGP implementation in py:class:'~GPy.models.sparse_gp_minibatch.SparseGPMiniBatch'.

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

    def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, X_variance=None, inference_method=None,
                 name='sparse gp', Y_metadata=None, normalizer=False):

        #pick a sensible inference method
        if inference_method is None:
            if isinstance(likelihood, likelihoods.Gaussian):
                inference_method = var_dtc.VarDTC(limit=3)
            else:
                #inference_method = ??
                raise NotImplementedError("what to do what to do?")
            print(("defaulting to ", inference_method, "for latent function inference"))

        self.Z = Param('inducing inputs', Z)
        self.num_inducing = Z.shape[0]

        GP.__init__(self, X, Y, kernel, likelihood, mean_function, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)

        logger.info("Adding Z as parameter")
        self.link_parameter(self.Z, index=0)
        self.posterior = None

    @property
    def _predictive_variable(self):
        return self.Z

    def has_uncertain_inputs(self):
        return isinstance(self.X, VariationalPosterior)

    def set_Z(self, Z, trigger_update=True):
        if trigger_update: self.update_model(False)
        self.unlink_parameter(self.Z)
        self.Z = Param('inducing inputs',Z)
        self.link_parameter(self.Z, index=0)
        if trigger_update: self.update_model(True)

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata)
        self._update_gradients()

    def _update_gradients(self):
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
        self._Zgrad = self.Z.gradient.copy()


    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values.

        For certain inputs we give back a full_cov of shape NxN,
        if there is missing data, each dimension has its own full_cov of shape NxNxD, and if full_cov is of,
        we take only the diagonal elements across N.

        For uncertain inputs, the SparseGP bound produces cannot predict the full covariance matrix full_cov for now.
        The implementation of that will follow. However, for each dimension the
        covariance changes, so if full_cov is False (standard), we return the variance
        for each dimension [NxD].
        """        
        if hasattr(self.posterior, '_raw_predict'):
            mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
            if self.mean_function is not None:
                mu += self.mean_function.f(Xnew)
            return mu, var

        if kern is None: kern = self.kern

        if not isinstance(Xnew, VariationalPosterior):
            # Kx = kern.K(self._predictive_variable, Xnew)
            # mu = np.dot(Kx.T, self.posterior.woodbury_vector)
            # if full_cov:
            #     Kxx = kern.K(Xnew)
            #     if self.posterior.woodbury_inv.ndim == 2:
            #         var = Kxx - np.dot(Kx.T, np.dot(self.posterior.woodbury_inv, Kx))
            #     elif self.posterior.woodbury_inv.ndim == 3:
            #         var = np.empty((Kxx.shape[0],Kxx.shape[1],self.posterior.woodbury_inv.shape[2]))
            #         for i in range(var.shape[2]):
            #             var[:, :, i] = (Kxx - mdot(Kx.T, self.posterior.woodbury_inv[:, :, i], Kx))
            #     var = var
            # else:
            #     Kxx = kern.Kdiag(Xnew)
            #     if self.posterior.woodbury_inv.ndim == 2:
            #         var = (Kxx - np.sum(np.dot(self.posterior.woodbury_inv.T, Kx) * Kx, 0))[:,None]
            #     elif self.posterior.woodbury_inv.ndim == 3:
            #         var = np.empty((Kxx.shape[0],self.posterior.woodbury_inv.shape[2]))
            #         for i in range(var.shape[1]):
            #             var[:, i] = (Kxx - (np.sum(np.dot(self.posterior.woodbury_inv[:, :, i].T, Kx) * Kx, 0)))
            #     var = var
            # #add in the mean function
            # if self.mean_function is not None:
            #     mu += self.mean_function.f(Xnew)
            mu, var = super(SparseGP, self)._raw_predict(Xnew, full_cov, kern)
        else:
            psi0_star = kern.psi0(self._predictive_variable, Xnew)
            psi1_star = kern.psi1(self._predictive_variable, Xnew)
            psi2_star = kern.psi2n(self._predictive_variable, Xnew) 
            la = self.posterior.woodbury_vector
            mu = np.dot(psi1_star, la) # TODO: dimensions?
            N,M,D = psi0_star.shape[0],psi1_star.shape[1], la.shape[1]

            if full_cov:
                raise NotImplementedError("Full covariance for Sparse GP predicted with uncertain inputs not implemented yet.")
                var = np.zeros((Xnew.shape[0], la.shape[1], la.shape[1]))
                di = np.diag_indices(la.shape[1])
            else:
                tmp = psi2_star - psi1_star[:,:,None]*psi1_star[:,None,:]
                var = (tmp.reshape(-1,M).dot(la).reshape(N,M,D)*la[None,:,:]).sum(1) + psi0_star[:,None] 
                if self.posterior.woodbury_inv.ndim==2:
                    var += -psi2_star.reshape(N,-1).dot(self.posterior.woodbury_inv.flat)[:,None]
                else:
                    var += -psi2_star.reshape(N,-1).dot(self.posterior.woodbury_inv.reshape(-1,D))
            assert np.all(var>=-1e-5), "The predicted variance goes negative!: "+str(var)
            var = np.clip(var,1e-15,np.inf)

#             for i in range(Xnew.shape[0]):
#                 _mu, _var = Xnew.mean.values[[i]], Xnew.variance.values[[i]]
#                 psi2_star = kern.psi2(self._predictive_variable, NormalPosterior(_mu, _var))
#                 tmp = (psi2_star[:, :] - psi1_star[[i]].T.dot(psi1_star[[i]]))
# 
#                 var_ = mdot(la.T, tmp, la)
#                 p0 = psi0_star[i]
#                 t = np.atleast_3d(self.posterior.woodbury_inv)
#                 t2 = np.trace(t.T.dot(psi2_star), axis1=1, axis2=2)
# 
#                 if full_cov:
#                     var_[di] += p0
#                     var_[di] += -t2
#                     var[i] = var_
#                 else:
#                     var[i] = np.diag(var_)+p0-t2

        return mu, var
