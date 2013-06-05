# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
from scipy import integrate

class exponential(Kernpart):
    """
    Exponential kernel (aka Ornstein-Uhlenbeck or Matern 1/2)

    .. math::

       k(r) = \sigma^2 \exp(- r) \ \ \ \ \  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param lengthscale: the vector of lengthscale :math:`\ell_i`
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one single lengthscale parameter \ell), otherwise there is one lengthscale parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False):
        self.input_dim = input_dim
        self.ARD = ARD
        if ARD == False:
            self.num_params = 2
            self.name = 'exp'
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                lengthscale = np.ones(1)
        else:
            self.num_params = self.input_dim + 1
            self.name = 'exp'
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == self.input_dim, "bad number of lengthscales"
            else:
                lengthscale = np.ones(self.input_dim)
        self._set_params(np.hstack((variance, lengthscale.flatten())))

    def _get_params(self):
        """return the value of the parameters."""
        return np.hstack((self.variance, self.lengthscale))

    def _set_params(self, x):
        """set the value of the parameters."""
        assert x.size == self.num_params
        self.variance = x[0]
        self.lengthscale = x[1:]

    def _get_param_names(self):
        """return parameter names."""
        if self.num_params == 2:
            return ['variance', 'lengthscale']
        else:
            return ['variance'] + ['lengthscale_%i' % i for i in range(self.lengthscale.size)]

    def K(self, X, X2, target):
        """Compute the covariance matrix between X and X2."""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:, None, :] - X2[None, :, :]) / self.lengthscale), -1))
        np.add(self.variance * np.exp(-dist), target, target)

    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix associated to X."""
        np.add(target, self.variance, target)

    def dK_dtheta(self, dL_dK, X, X2, target):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:, None, :] - X2[None, :, :]) / self.lengthscale), -1))
        invdist = 1. / np.where(dist != 0., dist, np.inf)
        dist2M = np.square(X[:, None, :] - X2[None, :, :]) / self.lengthscale ** 3
        dvar = np.exp(-dist)
        target[0] += np.sum(dvar * dL_dK)
        if self.ARD == True:
            dl = self.variance * dvar[:, :, None] * dist2M * invdist[:, :, None]
            target[1:] += (dl * dL_dK[:, :, None]).sum(0).sum(0)
        else:
            dl = self.variance * dvar * dist2M.sum(-1) * invdist
            target[1] += np.sum(dl * dL_dK)

    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        """derivative of the diagonal of the covariance matrix with respect to the parameters."""
        # NB: derivative of diagonal elements wrt lengthscale is 0
        target[0] += np.sum(dL_dKdiag)

    def dK_dX(self, dL_dK, X, X2, target):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:, None, :] - X2[None, :, :]) / self.lengthscale), -1))[:, :, None]
        ddist_dX = (X[:, None, :] - X2[None, :, :]) / self.lengthscale ** 2 / np.where(dist != 0., dist, np.inf)
        dK_dX = -np.transpose(self.variance * np.exp(-dist) * ddist_dX, (1, 0, 2))
        target += np.sum(dK_dX * dL_dK.T[:, :, None], 0)

    def dKdiag_dX(self, dL_dKdiag, X, target):
        pass

    def Gram_matrix(self, F, F1, lower, upper):
        """
        Return the Gram matrix of the vector of functions F with respect to the RKHS norm. The use of this function is limited to input_dim=1.

        :param F: vector of functions
        :type F: np.array
        :param F1: vector of derivatives of F
        :type F1: np.array
        :param lower,upper: boundaries of the input domain
        :type lower,upper: floats
        """
        assert self.input_dim == 1
        def L(x, i):
            return(1. / self.lengthscale * F[i](x) + F1[i](x))
        n = F.shape[0]
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                G[i, j] = G[j, i] = integrate.quad(lambda x : L(x, i) * L(x, j), lower, upper)[0]
        Flower = np.array([f(lower) for f in F])[:, None]
        return(self.lengthscale / 2. / self.variance * G + 1. / self.variance * np.dot(Flower, Flower.T))
