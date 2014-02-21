# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from ... import util
import numpy as np
from scipy import integrate

class Stationary(Kern):
    def __init__(self, input_dim, variance, lengthscale, ARD, name):
        super(Stationary, self).__init__(input_dim, name)
        self.ARD = ARD
        if not ARD:
            if lengthscale is None:
                lengthscale = np.ones(1)
            else:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1 "Only  lengthscale needed for non-ARD kernel"
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size in [1, input_dim], "Bad lengthscales"
                if lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim)*lengthscale
            else:
                lengthscale = np.ones(self.input_dim)
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size==1
        self.add_parameters(self.variance, self.lengthscale)

    def _dist(self, X, X2):
        if X2 is None:
            X2 = X
        return X[:, None, :] - X2[None, :, :]

    def _scaled_dist(self, X, X2=None):
        return np.sqrt(np.sum(np.square(self._dist(X, X2) / self.lengthscale), -1))

    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.

    def update_gradients_full(self, dL_dK, X, X2=None):
        K = self.K(X, X2)
        self.variance.gradient = np.sum(K * dL_dK)/self.variance

        rinv = self._inv_dist(X, X2)
        dL_dr = self.dK_dr(X, X2) * dL_dK
        x_xl3 = np.square(self._dist(X, X2)) / self.lengthscale**3

        if self.ARD:
            self.lengthscale.gradient = -((dL_dr*rinv)[:,:,None]*x_xl3).sum(0).sum(0)
        else:
            self.lengthscale.gradient = -((dL_dr*rinv)[:,:,None]*x_xl3).sum()

    def _inv_dist(self, X, X2=None):
        dist = self._scaled_dist(X, X2)
        if X2 is None:
            nondiag = util.diag.offdiag_view(dist)
            nondiag[:] = 1./nondiag
            return dist
        else:
            return 1./np.where(dist != 0., dist, np.inf)

    def gradients_X(self, dL_dK, X, X2=None):
        dL_dr = self.dK_dr(X, X2) * dL_dK
        invdist = self._inv_dist(X, X2)
        ret = np.sum((invdist*dL_dr)[:,:,None]*self._dist(X, X2),1)/self.lengthscale**2
        if X2 is None:
            ret *= 2.
        return ret

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)




class Exponential(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='Exponential'):
        super(Exponential, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K(self, X, X2=None):
        dist = self._scaled_dist(X, X2)
        return self.variance * np.exp(-0.5 * dist)

    def dK_dr(self, X, X2):
        return -0.5*self.K(X, X2)

class Matern32(Stationary):
    """
    Matern 3/2 kernel:

    .. math::

       k(r) = \\sigma^2 (1 + \\sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='Mat32'):
        super(Matern32, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K(self, X, X2=None):
        dist = self._scaled_dist(X, X2)
        return self.variance * (1. + np.sqrt(3.) * dist) * np.exp(-np.sqrt(3.) * dist)

    def dK_dr(self, X, X2):
        dist = self._scaled_dist(X, X2)
        return -3.*self.variance*dist*np.exp(-np.sqrt(3.)*dist)

    def Gram_matrix(self, F, F1, F2, lower, upper):
        """
        Return the Gram matrix of the vector of functions F with respect to the
        RKHS norm. The use of this function is limited to input_dim=1.

        :param F: vector of functions
        :type F: np.array
        :param F1: vector of derivatives of F
        :type F1: np.array
        :param F2: vector of second derivatives of F
        :type F2: np.array
        :param lower,upper: boundaries of the input domain
        :type lower,upper: floats
        """
        assert self.input_dim == 1
        def L(x, i):
            return(3. / self.lengthscale ** 2 * F[i](x) + 2 * np.sqrt(3) / self.lengthscale * F1[i](x) + F2[i](x))
        n = F.shape[0]
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                G[i, j] = G[j, i] = integrate.quad(lambda x : L(x, i) * L(x, j), lower, upper)[0]
        Flower = np.array([f(lower) for f in F])[:, None]
        F1lower = np.array([f(lower) for f in F1])[:, None]
        return(self.lengthscale ** 3 / (12.*np.sqrt(3) * self.variance) * G + 1. / self.variance * np.dot(Flower, Flower.T) + self.lengthscale ** 2 / (3.*self.variance) * np.dot(F1lower, F1lower.T))


class Matern52(Stationary):
    """
    Matern 5/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{5} r + \\frac53 r^2) \exp(- \sqrt{5} r) \ \ \ \ \  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }
       """

    def K(self, X, X2=None):
        r = self._scaled_dist(X, X2)
        return self.variance*(1+np.sqrt(5.)*r+5./3*r**2)*np.exp(-np.sqrt(5.)*r)

    def dK_dr(self, X, X2):
        r = self._scaled_dist(X, X2)
        return self.variance*(10./3*r -5.*r -5.*np.sqrt(5.)/3*r**2)*np.exp(-np.sqrt(5.)*r)

    def Gram_matrix(self,F,F1,F2,F3,lower,upper):
        """
        Return the Gram matrix of the vector of functions F with respect to the RKHS norm. The use of this function is limited to input_dim=1.

        :param F: vector of functions
        :type F: np.array
        :param F1: vector of derivatives of F
        :type F1: np.array
        :param F2: vector of second derivatives of F
        :type F2: np.array
        :param F3: vector of third derivatives of F
        :type F3: np.array
        :param lower,upper: boundaries of the input domain
        :type lower,upper: floats
        """
        assert self.input_dim == 1
        def L(x,i):
            return(5*np.sqrt(5)/self.lengthscale**3*F[i](x) + 15./self.lengthscale**2*F1[i](x)+ 3*np.sqrt(5)/self.lengthscale*F2[i](x) + F3[i](x))
        n = F.shape[0]
        G = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                G[i,j] = G[j,i] = integrate.quad(lambda x : L(x,i)*L(x,j),lower,upper)[0]
        G_coef = 3.*self.lengthscale**5/(400*np.sqrt(5))
        Flower = np.array([f(lower) for f in F])[:,None]
        F1lower = np.array([f(lower) for f in F1])[:,None]
        F2lower = np.array([f(lower) for f in F2])[:,None]
        orig = 9./8*np.dot(Flower,Flower.T) + 9.*self.lengthscale**4/200*np.dot(F2lower,F2lower.T)
        orig2 = 3./5*self.lengthscale**2 * ( np.dot(F1lower,F1lower.T) + 1./8*np.dot(Flower,F2lower.T) + 1./8*np.dot(F2lower,Flower.T))
        return(1./self.variance* (G_coef*G + orig + orig2))




class ExpQuad(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='ExpQuad'):
        super(ExpQuad, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K(self, X, X2=None):
        r = self._scaled_dist(X, X2)
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, X, X2):
        dist = self._scaled_dist(X, X2)
        return -dist*self.K(X, X2)



