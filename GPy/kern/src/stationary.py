# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import integrate
from .kern import Kern
from ...core.parameterization import Param
from ...util.linalg import tdot
from ... import util
from ...util.config import config # for assesing whether to use cython
from paramz.caching import Cache_this
from paramz.transformations import Logexp

try:
    from . import stationary_cython
except ImportError:
    print('warning in stationary: failed to import cython module: falling back to numpy')
    config.set('cython', 'working', 'false')


class Stationary(Kern):
    """
    Stationary kernels (covariance functions).

    Stationary covariance fucntion depend only on r, where r is defined as

    .. math::
        r(x, x') = \\sqrt{ \\sum_{q=1}^Q (x_q - x'_q)^2 }

    The covariance function k(x, x' can then be written k(r).

    In this implementation, r is scaled by the lengthscales parameter(s):

    .. math::

        r(x, x') = \\sqrt{ \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\ell_q^2} }.

    By default, there's only one lengthscale: seaprate lengthscales for each
    dimension can be enables by setting ARD=True.

    To implement a stationary covariance function using this class, one need
    only define the covariance function k(r), and it derivative.

    ```
    def K_of_r(self, r):
        return foo
    def dK_dr(self, r):
        return bar
    ```

    The lengthscale(s) and variance parameters are added to the structure automatically.

    """

    def __init__(self, input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=False):
        super(Stationary, self).__init__(input_dim, active_dims, name,useGPU=useGPU)
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
        self.link_parameters(self.variance, self.lengthscale)

    def K_of_r(self, r):
        raise NotImplementedError("implement the covariance function as a fn of r to use this class")

    def dK_dr(self, r):
        raise NotImplementedError("implement derivative of the covariance function wrt r to use this class")

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr(self, r):
        raise NotImplementedError("implement second derivative of covariance wrt r to use this method")

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.

        K(X, X2) = K_of_r((X-X2)**2)
        """
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r)

    @Cache_this(limit=3, ignore_args=())
    def dK_dr_via_X(self, X, X2):
        """
        compute the derivative of K wrt X going through X
        """
        #a convenience function, so we can cache dK_dr
        return self.dK_dr(self._scaled_dist(X, X2))

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr_via_X(self, X, X2):
        #a convenience function, so we can cache dK_dr
        return self.dK2_drdr(self._scaled_dist(X, X2))

    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + X1sq[:,None] + X2sq[None,:]
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)

    @Cache_this(limit=3, ignore_args=())
    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
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
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and stor in the <parameter>.gradient field.

        See also update_gradients_full
        """
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance

        #now the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        if self.ARD:

            tmp = dL_dr*self._inv_dist(X, X2)
            if X2 is None: X2 = X
            if config.getboolean('cython', 'working'):
                self.lengthscale.gradient = self._lengthscale_grads_cython(tmp, X, X2)
            else:
                self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
        else:
            r = self._scaled_dist(X, X2)
            self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale


    def _inv_dist(self, X, X2=None):
        """
        Compute the elementwise inverse of the distance matrix, expecpt on the
        diagonal, where we return zero (the distance on the diagonal is zero).
        This term appears in derviatives.
        """
        dist = self._scaled_dist(X, X2).copy()
        return 1./np.where(dist != 0., dist, np.inf)

    def _lengthscale_grads_pure(self, tmp, X, X2):
        return -np.array([np.sum(tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T)) for q in range(self.input_dim)])/self.lengthscale**3

    def _lengthscale_grads_cython(self, tmp, X, X2):
        N,M = tmp.shape
        Q = self.input_dim
        X, X2 = np.ascontiguousarray(X), np.ascontiguousarray(X2)
        grads = np.zeros(self.input_dim)
        stationary_cython.lengthscale_grads(N, M, Q, tmp, X, X2, grads)
        return -grads/self.lengthscale**3

    def gradients_X(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
        """
        if config.getboolean('cython', 'working'):
            return self._gradients_X_cython(dL_dK, X, X2)
        else:
            return self._gradients_X_pure(dL_dK, X, X2)

    def gradients_XX(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective K(dL_dK), compute the second derivative of K wrt X and X2:

        ..math:
          \frac{\partial^2 K}{\partial X\partial X2}

        ..returns:
            dL2_dXdX2: NxMxQ, for X [NxQ] and X2[MxQ] (X2 is X if, X2 is None)
            Thus, we return the second derivative in X2.
        """
        # The off diagonals in Q are always zero, this should also be true for the Linear kernel...
        # According to multivariable chain rule, we can chain the second derivative through r:
        # d2K_dXdX2 = dK_dr*d2r_dXdX2 + d2K_drdr * dr_dX * dr_dX2:
        invdist = self._inv_dist(X, X2)
        invdist2 = invdist**2

        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        tmp1 = dL_dr * invdist

        dL_drdr = self.dK2_drdr_via_X(X, X2) * dL_dK
        tmp2 = dL_drdr * invdist2

        l2 = np.ones(X.shape[1]) * self.lengthscale**2

        if X2 is None:
            X2 = X
            tmp1 -= np.eye(X.shape[0])*self.variance
        else:
            tmp1[X==X2.T] -= self.variance

        grad = np.empty((X.shape[0], X2.shape[0], X.shape[1]), dtype=np.float64)
        #grad = np.empty(X.shape, dtype=np.float64)
        for q in range(self.input_dim):
            tmpdist2 = (X[:,[q]]-X2[:,[q]].T) ** 2
            grad[:, :, q] = ((tmp1*invdist2 - tmp2)*tmpdist2/l2[q] - tmp1)/l2[q]
            #grad[:, :, q] = ((tmp1*(((tmpdist2)*invdist2/l2[q])-1)) - (tmp2*(tmpdist2))/l2[q])/l2[q]
            #np.sum(((tmp1*(((tmpdist2)*invdist2/l2[q])-1)) - (tmp2*(tmpdist2))/l2[q])/l2[q], axis=1, out=grad[:,q])
            #np.sum( - (tmp2*(tmpdist**2)), axis=1, out=grad[:,q])
        return grad

    def gradients_XX_diag(self, dL_dK, X):
        """
        Given the derivative of the objective K(dL_dK), compute the second derivative of K wrt X and X2:

        ..math:
          \frac{\partial^2 K}{\partial X\partial X2}

        ..returns:
            dL2_dXdX2: NxMxQ, for X [NxQ] and X2[MxQ]
        """
        return np.ones(X.shape) * self.variance/self.lengthscale**2

    def _gradients_X_pure(self, dL_dK, X, X2=None):
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        tmp = invdist*dL_dr
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X

        #The high-memory numpy way:
        #d =  X[:, None, :] - X2[None, :, :]
        #grad = np.sum(tmp[:,:,None]*d,1)/self.lengthscale**2

        #the lower memory way with a loop
        grad = np.empty(X.shape, dtype=np.float64)
        for q in range(self.input_dim):
            np.sum(tmp*(X[:,q][:,None]-X2[:,q][None,:]), axis=1, out=grad[:,q])
        return grad/self.lengthscale**2

    def _gradients_X_cython(self, dL_dK, X, X2=None):
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        tmp = invdist*dL_dr
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X
        X, X2 = np.ascontiguousarray(X), np.ascontiguousarray(X2)
        grad = np.zeros(X.shape)
        stationary_cython.grad_X(X.shape[0], X.shape[1], X2.shape[0], X, X2, tmp, grad)
        return grad/self.lengthscale**2

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)

    def input_sensitivity(self, summarize=True):
        return self.variance*np.ones(self.input_dim)/self.lengthscale**2




class Exponential(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Exponential'):
        super(Exponential, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r)

    def dK_dr(self, r):
        return -0.5*self.K_of_r(r)




class OU(Stationary):
    """
    OU kernel:

    .. math::

       k(r) = \\sigma^2 \exp(- r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^{\text{input_dim}} \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='OU'):
        super(OU, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.exp(-r)

    def dK_dr(self,r):
        return -1.*self.variance*np.exp(-r)


class Matern32(Stationary):
    """
    Matern 3/2 kernel:

    .. math::

       k(r) = \\sigma^2 (1 + \\sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^{\\text{input_dim}} \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat32'):
        super(Matern32, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

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
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat52'):
        super(Matern52, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

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
    """
    The Exponentiated quadratic covariance function.

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{5} r + \\frac53 r^2) \exp(- \sqrt{5} r)

    notes::
     - Yes, this is exactly the same as the RBF covariance function, but the
       RBF implementation also has some features for doing variational kernels
       (the psi-statistics).

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='ExpQuad'):
        super(ExpQuad, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

class Cosine(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Cosine'):
        super(Cosine, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

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


    def __init__(self, input_dim, variance=1., lengthscale=None, power=2., ARD=False, active_dims=None, name='RatQuad'):
        super(RatQuad, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
        self.power = Param('power', power, Logexp())
        self.link_parameters(self.power)

    def K_of_r(self, r):
        r2 = np.square(r)
#         return self.variance*np.power(1. + r2/2., -self.power)
        return self.variance*np.exp(-self.power*np.log1p(r2/2.))

    def dK_dr(self, r):
        r2 = np.square(r)
#         return -self.variance*self.power*r*np.power(1. + r2/2., - self.power - 1.)
        return-self.variance*self.power*r*np.exp(-(self.power+1)*np.log1p(r2/2.))

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(RatQuad, self).update_gradients_full(dL_dK, X, X2)
        r = self._scaled_dist(X, X2)
        r2 = np.square(r)
#        dK_dpow = -self.variance * np.power(2., self.power) * np.power(r2 + 2., -self.power) * np.log(0.5*(r2+2.))
        dK_dpow = -self.variance * np.exp(self.power*(np.log(2.)-np.log1p(r2+1)))*np.log1p(r2/2.)
        grad = np.sum(dL_dK*dK_dpow)
        self.power.gradient = grad

    def update_gradients_diag(self, dL_dKdiag, X):
        super(RatQuad, self).update_gradients_diag(dL_dKdiag, X)
        self.power.gradient = 0.


