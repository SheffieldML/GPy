# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import VarDTC
from GPy.core.parameterization.variational import NormalPosterior

class SparseGPRegression(SparseGP_MPI):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the SparseGP class, with a set of sensible defalts

    :param X: input observations
    :param X_variance: input uncertainties, one per input X
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf+white
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param num_inducing: number of inducing points (ignored if Z is passed, see note)
    :type num_inducing: int
    :rtype: model object

    .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored
    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Z=None, num_inducing=10, X_variance=None, normalizer=None, mpi_comm=None):
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

        likelihood = likelihoods.Gaussian()

        if not (X_variance is None):
            X = NormalPosterior(X,X_variance)

        if mpi_comm is not None:
            from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
            infr = VarDTC_minibatch(mpi_comm=mpi_comm)
        else:
            infr = VarDTC()

        SparseGP_MPI.__init__(self, X, Y, Z, kernel, likelihood, inference_method=infr, normalizer=normalizer, mpi_comm=mpi_comm)

    def parameters_changed(self):
        from ..inference.latent_function_inference.var_dtc_parallel import update_gradients_sparsegp,VarDTC_minibatch
        if isinstance(self.inference_method,VarDTC_minibatch):
            update_gradients_sparsegp(self, mpi_comm=self.mpi_comm)
        else:
            super(SparseGPRegression, self).parameters_changed()
