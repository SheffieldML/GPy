# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import logging
from .. import kern
from ..likelihoods import Gaussian
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior
from .sparse_gp_minibatch import SparseGPMiniBatch
from ..core.parameterization.param import Param

class BayesianGPLVMMiniBatch(SparseGPMiniBatch):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None,
                 name='bayesian gplvm', normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1):
        self.logger = logging.getLogger(self.__class__.__name__)
        if X is None:
            from ..util.initialization import initialize_latent
            self.logger.info("initializing latent space X with method {}".format(init))
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if Z is None:
            self.logger.info("initializing inducing inputs")
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if X_variance is False:
            self.logger.info('no variance on X, activating sparse GPLVM')
            X = Param("latent space", X)
        else:
            if X_variance is None:
                self.logger.info("initializing latent space variance ~ uniform(0,.1)")
                X_variance = np.random.uniform(0,.1,X.shape)
            self.variational_prior = NormalPrior()
            X = NormalPosterior(X, X_variance)

        if kernel is None:
            self.logger.info("initializing kernel RBF")
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) #+ kern.Bias(input_dim) + kern.White(input_dim)

        if likelihood is None:
            likelihood = Gaussian()

        self.kl_factr = 1.

        if inference_method is None:
            from ..inference.latent_function_inference.var_dtc import VarDTC
            self.logger.debug("creating inference_method var_dtc")
            inference_method = VarDTC(limit=3 if not missing_data else Y.shape[1])

        super(BayesianGPLVMMiniBatch,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method,
                                           normalizer=normalizer,
                                           missing_data=missing_data, stochastic=stochastic,
                                           batchsize=batchsize)
        self.X = X
        self.link_parameter(self.X, 0)

    #def set_X_gradients(self, X, X_grad):
    #    """Set the gradients of the posterior distribution of X in its specific form."""
    #    X.mean.gradient, X.variance.gradient = X_grad

    #def get_X_gradients(self, X):
    #    """Get the gradients of the posterior distribution of X in its specific form."""
    #    return X.mean.gradient, X.variance.gradient

    def _outer_values_update(self, full_values):
        """
        Here you put the values, which were collected before in the right places.
        E.g. set the gradients of parameters, etc.
        """
        super(BayesianGPLVMMiniBatch, self)._outer_values_update(full_values)
        if self.has_uncertain_inputs():
            meangrad_tmp, vargrad_tmp = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z, dL_dpsi0=full_values['dL_dpsi0'],
                                            dL_dpsi1=full_values['dL_dpsi1'],
                                            dL_dpsi2=full_values['dL_dpsi2'],
                                            psi0=self.psi0, psi1=self.psi1, psi2=self.psi2)

            self.X.mean.gradient = meangrad_tmp
            self.X.variance.gradient = vargrad_tmp
        else:
            self.X.gradient = self.kern.gradients_X(full_values['dL_dKnm'], self.X, self.Z)
            self.X.gradient += self.kern.gradients_X_diag(full_values['dL_dKdiag'], self.X)

    def _outer_init_full_values(self):
        return super(BayesianGPLVMMiniBatch, self)._outer_init_full_values()

    def parameters_changed(self):
        super(BayesianGPLVMMiniBatch,self).parameters_changed()

        kl_fctr = self.kl_factr
        if kl_fctr > 0 and self.has_uncertain_inputs():
            Xgrad = self.X.gradient.copy()
            self.X.gradient[:] = 0
            self.variational_prior.update_gradients_KL(self.X)

            if self.missing_data or not self.stochastics:
                self.X.mean.gradient = kl_fctr*self.X.mean.gradient
                self.X.variance.gradient = kl_fctr*self.X.variance.gradient
            else:
                d = self.output_dim
                self.X.mean.gradient = kl_fctr*self.X.mean.gradient*self.stochastics.batchsize/d
                self.X.variance.gradient = kl_fctr*self.X.variance.gradient*self.stochastics.batchsize/d
            self.X.gradient += Xgrad

            if self.missing_data or not self.stochastics:
                self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)
            else: #self.stochastics is given:
                d = self.output_dim
                self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)*self.stochastics.batchsize/d

        self._Xgrad = self.X.gradient.copy()
