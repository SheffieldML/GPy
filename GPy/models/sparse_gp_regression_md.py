# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference.vardtc_md import VarDTC_MD
from GPy.core.parameterization.variational import NormalPosterior

class SparseGPRegressionMD(SparseGP_MPI):
    """
    Sparse Gaussian Process Regression with Missing Data
    """

    def __init__(self, X, Y, indexD, kernel=None, Z=None, num_inducing=10, X_variance=None, normalizer=None, mpi_comm=None, individual_Y_noise=False, name='sparse_gp'):

        assert len(Y.shape)==1 or Y.shape[1]==1
        self.individual_Y_noise = individual_Y_noise
        self.indexD = indexD
        output_dim = int(np.max(indexD))+1

        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.RBF(input_dim)#  + kern.white(input_dim, variance=1e-3)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[:min(num_inducing, num_data)]
            Z = X.view(np.ndarray)[i].copy()
        else:
            assert Z.shape[1] == input_dim

        if individual_Y_noise:
            likelihood = likelihoods.Gaussian(variance=np.array([np.var(Y[indexD==d]) for d in range(output_dim)])*0.01)
        else:
            likelihood = likelihoods.Gaussian(variance=np.var(Y)*0.01)

        if not (X_variance is None):
            X = NormalPosterior(X,X_variance)

        infr = VarDTC_MD()

        SparseGP_MPI.__init__(self, X, Y, Z, kernel, likelihood, inference_method=infr, normalizer=normalizer, mpi_comm=mpi_comm, name=name)
        self.output_dim = output_dim

    def parameters_changed(self):

        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.indexD, self.output_dim, self.Y_metadata)

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'] if self.individual_Y_noise else self.grad_dict['dL_dthetaL'].sum())

        self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
        kerngrad = self.kern.gradient.copy()
        self.kern.update_gradients_full(self.grad_dict['dL_dKnm'], self.X, self.Z)
        kerngrad += self.kern.gradient
        self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z, None)
        self.kern.gradient += kerngrad
        #gradients wrt Z
        self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
        self.Z.gradient += self.kern.gradients_X(self.grad_dict['dL_dKnm'].T, self.Z, self.X)
