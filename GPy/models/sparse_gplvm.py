# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
import sys, pdb
from GPy.models.sparse_gp_regression import SparseGPRegression
from GPy.models.gplvm import GPLVM
# from .. import kern
# from ..core import model
# from ..util.linalg import pdinv, PCA

class SparseGPLVM(SparseGPRegression):
    """
    Sparse Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, kernel=None, init='PCA', num_inducing=10):
        if X is None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        SparseGPRegression.__init__(self, X, Y, kernel=kernel, num_inducing=num_inducing)

    def parameters_changed(self):
        super(SparseGPLVM, self).parameters_changed()
        self.X.gradient = self.kern.gradients_X(self.grad_dict['dL_dKnm'], self.X, self.Z)
