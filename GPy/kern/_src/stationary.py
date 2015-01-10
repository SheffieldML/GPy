# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
from ...util.linalg import tdot
from ... import util
import numpy as np
from scipy import integrate, weave
from ...util.config import config # for assesing whether to use weave
from ...util.caching import Cache_this

class Stationary(Kern):
    """
    Stationary kernels (covariance functions).

    Stationary covariance fucntion depend only on r, where r is defined as

      r = \sqrt{ \sum_{q=1}^Q (x_q - x'_q)^2 }

    The covariance function k(x, x' can then be written k(r).

    In this implementation, r is scaled by the lengthscales parameter(s):

      r = \sqrt{ \sum_{q=1}^Q \frac{(x_q - x'_q)^2}{\ell_q^2} }.

    By default, there's only one lengthscale: seaprate lengthscales for each
    dimension can be enables by setting ARD=True.

    To implement a stationary covariance function using this class, one need
    only define the covariance function k(r), and it derivative.

      ...
      def K_of_r(self, r):
          return foo
      def dK_dr(self, r):
          return bar

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
        raise NotImplementedError, "implement the covariance function as a fn of r to use this class"

    def dK_dr(self, r):
        raise NotImplementedError, "implement derivative of the covariance function wrt r to use this class"

    @Cache_this(limit=5, ignore_args=())
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
        #a convenience function, so we can cache dK_dr
        return self.dK_dr(self._scaled_dist(X, X2))

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

    @Cache_this(limit=5, ignore_args=())
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
        self.variance.gradient = np.einsum('ij,ij,i', self.K(X, X2), dL_dK, 1./self.variance)

        #now the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        if self.ARD:
            #rinv = self._inv_dis# this is rather high memory? Should we loop instead?t(X, X2)
            #d =  X[:, None, :] - X2[None, :, :]
            #x_xl3 = np.square(d)
            #self.lengthscale.gradient = -((dL_dr*rinv)[:,:,None]*x_xl3).sum(0).sum(0)/self.lengthscale**3
            tmp = dL_dr*self._inv_dist(X, X2)
            if X2 is None: X2 = X


            if config.getboolean('weave', 'working'):
                try:
                    self.lengthscale.gradient = self.weave_lengthscale_grads(tmp, X, X2)
                except:
                    print "\n Weave compilation failed. Falling back to (slower) numpy implementation\n"
                    config.set('weave', 'working', 'False')
                    self.lengthscale.gradient = np.array([np.einsum('ij,ij,...', tmp, np.square(X[:,q:q+1] - X2[:,q:q+1].T), -1./self.lengthscale[q]**3) for q in xrange(self.input_dim)])
            else:
                self.lengthscale.gradient = np.array([np.einsum('ij,ij,...', tmp, np.square(X[:,q:q+1] - X2[:,q:q+1].T), -1./self.lengthscale[q]**3) for q in xrange(self.input_dim)])
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

    def weave_lengthscale_grads(self, tmp, X, X2):
        """Use scipy.weave to compute derivatives wrt the lengthscales"""
        N,M = tmp.shape
        Q = X.shape[1]
        if hasattr(X, 'values'):X = X.values
        if hasattr(X2, 'values'):X2 = X2.values
        grads = np.zeros(self.input_dim)
        code = """
        double gradq;
        for(int q=0; q<Q; q++){
          gradq = 0;
          for(int n=0; n<N; n++){
            for(int m=0; m<M; m++){
              gradq += tmp(n,m)*(X(n,q)-X2(m,q))*(X(n,q)-X2(m,q));
            }
          }
          grads(q) = gradq;
        }
        """
        weave.inline(code, ['tmp', 'X', 'X2', 'grads', 'N', 'M', 'Q'], type_converters=weave.converters.blitz, support_code="#include <math.h>")
        return -grads/self.lengthscale**3

    def gradients_X(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
        """
        if config.getboolean('weave', 'working'):
            try:
                return self.gradients_X_weave(dL_dK, X, X2)
            except:
                print "\n Weave compilation failed. Falling back to (slower) numpy implementation\n"
                config.set('weave', 'working', 'False')
                return self.gradients_X_(dL_dK, X, X2)
        else:
            return self.gradients_X_(dL_dK, X, X2)

    def gradients_X_(self, dL_dK, X, X2=None):
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        tmp = invdist*dL_dr
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X

        #The high-memory numpy way:
        #d =  X[:, None, :] - X2[None, :, :]
        #ret = np.sum(tmp[:,:,None]*d,1)/self.lengthscale**2

        #the lower memory way with a loop
        ret = np.empty(X.shape, dtype=np.float64)
        for q in xrange(self.input_dim):
            np.sum(tmp*(X[:,q][:,None]-X2[:,q][None,:]), axis=1, out=ret[:,q])
        ret /= self.lengthscale**2

        return ret

    def gradients_X_weave(self, dL_dK, X, X2=None):
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        tmp = invdist*dL_dr
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X

        code = """
        int n,m,d;
        double retnd;
        #pragma omp parallel for private(n,d, retnd, m)
        for(d=0;d<D;d++){
          for(n=0;n<N;n++){
            retnd = 0.0;
            for(m=0;m<M;m++){
              retnd += tmp(n,m)*(X(n,d)-X2(m,d));
            }
            ret(n,d) = retnd;
          }
        }

        """
        if hasattr(X, 'values'):X = X.values #remove the GPy wrapping to make passing into weave safe
        if hasattr(X2, 'values'):X2 = X2.values
        ret = np.zeros(X.shape)
        N,D = X.shape
        N,M = tmp.shape
        from scipy import weave
        support_code = """
        #include <omp.h>
        #include <stdio.h>
        """
        weave_options = {'headers'           : ['<omp.h>'],
                         'extra_compile_args': ['-fopenmp -O3'], # -march=native'],
                         'extra_link_args'   : ['-lgomp']}
        weave.inline(code, ['ret', 'N', 'D', 'M', 'tmp', 'X', 'X2'], type_converters=weave.converters.blitz, support_code=support_code, **weave_options)
        return ret/self.lengthscale**2

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

       k(r) = \\sigma^2 \exp(- r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

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

       k(r) = \\sigma^2 (1 + \\sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

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


