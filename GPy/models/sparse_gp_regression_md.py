# Copyright (c) 2017, Zhenwen Dai
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

    This model targets at the use case, in which there are multiple output dimensions (different dimensions are assumed to be independent following the same GP prior) and each output dimension is observed at a different set of inputs. The model takes a different data format: the inputs and outputs observations of all the output dimensions are stacked together correspondingly into two matrices. An extra array is used to indicate the index of output dimension for each data point. The output dimensions are indexed using integers from 0 to D-1 assuming there are D output dimensions.

    :param X: input observations.
    :type X: numpy.ndarray
    :param Y: output observations, each column corresponding to an output dimension.
    :type Y: numpy.ndarray
    :param indexD: the array containing the index of output dimension for each data point
    :type indexD: numpy.ndarray
    :param kernel: a GPy kernel for GP of individual output dimensions ** defaults to RBF **
    :type kernel: GPy.kern.Kern or None
    :param Z: inducing inputs
    :type Z: numpy.ndarray or None
    :param num_inducing: a tuple (M, Mr). M is the number of inducing points for GP of individual output dimensions. Mr is the number of inducing points for the latent space.
    :type num_inducing: (int, int)
    :param boolean individual_Y_noise: whether individual output dimensions have their own noise variance or not, boolean
    :param str name: the name of the model
    """

    def __init__(self, X, Y, indexD, kernel=None, Z=None, num_inducing=10,  normalizer=None, mpi_comm=None, individual_Y_noise=False, name='sparse_gp'):

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

        infr = VarDTC_MD()

        super(SparseGPRegressionMD, self).__init__(X, Y, Z, kernel, likelihood, inference_method=infr, normalizer=normalizer, mpi_comm=mpi_comm, name=name)
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
