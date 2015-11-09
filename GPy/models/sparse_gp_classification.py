# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import EPDTC

class SparseGPClassification(SparseGP):
    """
    Sparse Gaussian Process model for classification

    This is a thin wrapper around the sparse_GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param likelihood: a GPy likelihood, defaults to Binomial with probit link_function
    :param kernel: a GPy kernel, defaults to rbf+white
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :rtype: model object

    """

    def __init__(self, X, Y=None, likelihood=None, kernel=None, Z=None, num_inducing=10, Y_metadata=None):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method=EPDTC(), name='SparseGPClassification',Y_metadata=Y_metadata)

class SparseGPClassificationUncertainInput(SparseGP):
    """
    Sparse Gaussian Process model for classification with uncertain inputs.

    This is a thin wrapper around the sparse_GP class, with a set of sensible defaults

    :param X: input observations
    :type X: np.ndarray (num_data x input_dim)
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance, optional)
    :type X_variance: np.ndarray (num_data x input_dim)
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
    def __init__(self, X, X_variance, Y, kernel=None, Z=None, num_inducing=10, Y_metadata=None, normalizer=None):
        from GPy.core.parameterization.variational import NormalPosterior
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        X = NormalPosterior(X, X_variance)

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood,
                          inference_method=EPDTC(),
                          name='SparseGPClassification', Y_metadata=Y_metadata, normalizer=normalizer)

    def parameters_changed(self):
        #Compute the psi statistics for N once, but don't sum out N in psi2
        self.psi0 = self.kern.psi0(self.Z, self.X)
        self.psi1 = self.kern.psi1(self.Z, self.X)
        self.psi2 = self.kern.psi2n(self.Z, self.X)
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata, psi0=self.psi0, psi1=self.psi1, psi2=self.psi2)
        self._update_gradients()





