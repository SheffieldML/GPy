# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from ...util.linalg import tdot
from ... import util
import numpy as np
from scipy import integrate
from ...util.caching import Cache_this

class Stationary(Kern):
    """
    Stationary kernels (covaraince functions).

    Stationary covariance fucntion depend only on r, where r is defined as 

      r = \sqrt{ \sum_{q=1}^Q (x_q - x'_q)^2 }

    The covaraince function k(x, x' can then be written k(r). 

    In this implementation, r is scaled by the lengthscales parameter(s):

      r = \sqrt{ \sum_{q=1}^Q \frac{(x_q - x'_q)^2}{\ell_q^2} }. 
    
    By default, there's only one lengthscale: seaprate lengthscales for each
    dimension can be enables by setting ARD=True. 

    To implement a stationary covaraince function using this class, one need
    only define the covariance function k(r), and it derivative. 

      ...
      def K_of_r(self, r):
          return foo
      def dK_dr(self, r):
          return bar

    The lengthscale(s) and variance parameters are added to the structure automatically. 
          
    """
    
    def __init__(self, input_dim, variance, lengthscale, ARD, name):
        super(Stationary, self).__init__(input_dim, name)
        self.ARD = ARD
        if not ARD:
            if lengthscale is None:
                lengthscale = np.ones(1)
            else:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size in [1, input_dim], "Bad number of lengthscales"
                if lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim)*lengthscale
            else:
                lengthscale = np.ones(self.input_dim)
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size==1
        self.add_parameters(self.variance, self.lengthscale)

    def K_of_r(self, r):
        raise NotImplementedError, "implement the covariance function as a fn of r to use this class"

    def dK_dr(self, r):
        raise NotImplementedError, "implement derivative of the covariance function wrt r to use this class"

    #@Cache_this(limit=5, ignore_args=())
    def K(self, X, X2=None):
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r)

    #@Cache_this(limit=5, ignore_args=(0,))
    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            return np.sqrt(-2.*tdot(X) + (Xsq[:,None] + Xsq[None,:]))
        else:
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            return np.sqrt(-2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:]))

    #@Cache_this(limit=5, ignore_args=())
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if self.ARD:
            if X2 is not None:
                X2 = X2 / self.lengthscale
            return self._unscaled_dist(X/self.lengthscale, X2)
        else:
            return self._unscaled_dist(X, X2)/self.lengthscale

    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.

    def update_gradients_full(self, dL_dK, X, X2=None):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)

        dL_dr = self.dK_dr(r) * dL_dK

        if self.ARD:
            #rinv = self._inv_dis# this is rather high memory? Should we loop instead?t(X, X2)
            #d =  X[:, None, :] - X2[None, :, :]
            #x_xl3 = np.square(d) 
            #self.lengthscale.gradient = -((dL_dr*rinv)[:,:,None]*x_xl3).sum(0).sum(0)/self.lengthscale**3
            tmp = dL_dr*self._inv_dist(X, X2)
            if X2 is None: X2 = X
            [np.copyto(self.lengthscale.gradient[q:q+1], -np.sum(tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T))/self.lengthscale[q]**3) for q in xrange(self.input_dim)]
        else:
            self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale

        self.variance.gradient = np.sum(K * dL_dK)/self.variance

    def _inv_dist(self, X, X2=None):
        """
        Compute the elementwise inverse of the distance matrix, expecpt on the
        diagonal, where we return zero (the distance on the diagonal is zero).
        This term appears in derviatives.
        """
        dist = self._scaled_dist(X, X2)
        if X2 is None:
            nondiag = util.diag.offdiag_view(dist)
            nondiag[:] = 1./nondiag
            return dist
        else:
            return 1./np.where(dist != 0., dist, np.inf)

    def gradients_X(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
        """
        r = self._scaled_dist(X, X2)
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr(r) * dL_dK
        #The high-memory numpy way:
        #d =  X[:, None, :] - X2[None, :, :]
        #ret = np.sum((invdist*dL_dr)[:,:,None]*d,1)/self.lengthscale**2
        #if X2 is None:
            #ret *= 2.

        #the lower memory way with a loop
        tmp = invdist*dL_dr
        if X2 is None:
            tmp *= 2.
            X2 = X
        ret = np.empty(X.shape, dtype=np.float64)
        [np.copyto(ret[:,q], np.sum(tmp*(X[:,q][:,None]-X2[:,q][None,:]), 1)) for q in xrange(self.input_dim)]
        ret /= self.lengthscale**2

        return ret

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)

    def input_sensitivity(self):
        return np.ones(self.input_dim)/self.lengthscale

class Exponential(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='Exponential'):
        super(Exponential, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r)

    def dK_dr(self, r):
        return -0.5*self.K_of_r(r)

class Matern32(Stationary):
    """
    Matern 3/2 kernel:

    .. math::

       k(r) = \\sigma^2 (1 + \\sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='Mat32'):
        super(Matern32, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K_of_r(self, r):
        return self.variance * (1. + np.sqrt(3.) * r) * np.exp(-np.sqrt(3.) * r)

    def dK_dr(self,r):
        return -3.*self.variance*r*np.exp(-np.sqrt(3.)*r)

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

       k(r) = \sigma^2 (1 + \sqrt{5} r + \\frac53 r^2) \exp(- \sqrt{5} r) 
       """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='Mat52'):
        super(Matern52, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K_of_r(self, r):
        return self.variance*(1+np.sqrt(5.)*r+5./3*r**2)*np.exp(-np.sqrt(5.)*r)

    def dK_dr(self, r):
        return self.variance*(10./3*r -5.*r -5.*np.sqrt(5.)/3*r**2)*np.exp(-np.sqrt(5.)*r)

    def Gram_matrix(self, F, F1, F2, F3, lower, upper):
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

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

class Cosine(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, name='Cosine'):
        super(Cosine, self).__init__(input_dim, variance, lengthscale, ARD, name)

    def K_of_r(self, r):
        return self.variance * np.cos(r)

    def dK_dr(self, r):
        return -self.variance * np.sin(r)


class RatQuad(Stationary):
    """
    Rational Quadratic Kernel

    .. math::

       k(r) = \sigma^2 \\bigg( 1 + \\frac{r^2}{2} \\bigg)^{- \\alpha}

    """


    def __init__(self, input_dim, variance=1., lengthscale=None, power=2., ARD=False, name='ExpQuad'):
        super(RatQuad, self).__init__(input_dim, variance, lengthscale, ARD, name)
        self.power = Param('power', power, Logexp())
        self.add_parameters(self.power)

    def K_of_r(self, r):
        r2 = np.power(r, 2.)
        return self.variance*np.power(1. + r2/2., -self.power)

    def dK_dr(self, r):
        r2 = np.power(r, 2.)
        return -self.variance*self.power*r*np.power(1. + r2/2., - self.power - 1.)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(RatQuad, self).update_gradients_full(dL_dK, X, X2)
        r = self._scaled_dist(X, X2)
        r2 = np.power(r, 2.)
        dK_dpow = -self.variance * np.power(2., self.power) * np.power(r2 + 2., -self.power) * np.log(0.5*(r2+2.))
        grad = np.sum(dL_dK*dK_dpow)
        self.power.gradient = grad

    def update_gradients_diag(self, dL_dKdiag, X):
        super(RatQuad, self).update_gradients_diag(dL_dKdiag, X)
        self.power.gradient = 0.


