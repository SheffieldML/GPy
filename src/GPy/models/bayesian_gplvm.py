# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .. import kern
from ..core.sparse_gp_mpi import SparseGP_MPI
from ..likelihoods import Gaussian
from GPy.core.parameterization.variational import NormalPosterior, NormalPrior
from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
import logging

class BayesianGPLVM(SparseGP_MPI):
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
                 name='bayesian gplvm', mpi_comm=None, normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1, Y_metadata=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        if X is None:
            from ..util.initialization import initialize_latent
            self.logger.info("initializing latent space X with method {}".format(init))
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if X_variance is None:
            self.logger.info("initializing latent space variance ~ uniform(0,.1)")
            X_variance = np.random.uniform(0,.1,X.shape)

        if Z is None:
            self.logger.info("initializing inducing inputs")
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            self.logger.info("initializing kernel RBF")
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) #+ kern.Bias(input_dim) + kern.White(input_dim)

        if likelihood is None:
            likelihood = Gaussian()

        self.variational_prior = NormalPrior()
        X = NormalPosterior(X, X_variance)

        if inference_method is None:
            if mpi_comm is not None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                from ..inference.latent_function_inference.var_dtc import VarDTC
                self.logger.debug("creating inference_method var_dtc")
                inference_method = VarDTC(limit=3 if not missing_data else Y.shape[1])
        if isinstance(inference_method,VarDTC_minibatch):
            inference_method.mpi_comm = mpi_comm

        super(BayesianGPLVM,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method,
                                           normalizer=normalizer, mpi_comm=mpi_comm,
                                           variational_prior=self.variational_prior,
                                           Y_metadata=Y_metadata
                                           )
        self.link_parameter(self.X, index=0)

    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient = X_grad

    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient

    def parameters_changed(self):
        super(BayesianGPLVM,self).parameters_changed()
        if isinstance(self.inference_method, VarDTC_minibatch):
            return

        kl_fctr = 1.
        self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])

        self.variational_prior.update_gradients_KL(self.X)
        self._Xgrad = self.X.gradient.copy()

