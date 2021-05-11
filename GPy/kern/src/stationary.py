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
    use_stationary_cython = config.getboolean('cython', 'working')
except ImportError:
    print('warning in stationary: failed to import cython module: falling back to numpy')
    use_stationary_cython = False


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

    Thanks to @strongh:
    In Stationary, a covariance function is defined in GPy as stationary when it depends only on the l2-norm |x_1 - x_2 |.
    However this is the typical definition of isotropy, while stationarity is usually a bit more relaxed.
    The more common version of stationarity is that the covariance is a function of x_1 - x_2 (See e.g. R&W first paragraph of section 4.1).
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

    def _save_to_input_dict(self):
        input_dict = super(Stationary, self)._save_to_input_dict()
        input_dict["variance"] =  self.variance.values.tolist()
        input_dict["lengthscale"] = self.lengthscale.values.tolist()
        input_dict["ARD"] = self.ARD
        return input_dict

    def K_of_r(self, r):
        raise NotImplementedError("implement the covariance function as a fn of r to use this class")

    def dK_dr(self, r):
        raise NotImplementedError("implement derivative of the covariance function wrt r to use this class")

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr(self, r):
        raise NotImplementedError("implement second derivative of covariance wrt r to use this method")

    @Cache_this(limit=3, ignore_args=())
    def dK2_drdr_diag(self):
        "Second order derivative of K in r_{i,i}. The diagonal entries are always zero, so we do not give it here."
        raise NotImplementedError("implement second derivative of covariance wrt r_diag to use this method")

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
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
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

    def reset_gradients(self):
        self.variance.gradient = 0.
        if not self.ARD:
            self.lengthscale.gradient = 0.
        else:
            self.lengthscale.gradient = np.zeros(self.input_dim)

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and stor in the <parameter>.gradient field.

        See also update_gradients_full
        """
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.

    def update_gradients_full(self, dL_dK, X, X2=None, reset=True):
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
            if use_stationary_cython:
                self.lengthscale.gradient = self._lengthscale_grads_cython(tmp, X, X2)
            else:
                self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
        else:
            r = self._scaled_dist(X, X2)
            self.lengthscale.gradient = -np.sum(dL_dr*r)/self.lengthscale

    def update_gradients_direct(self, dL_dVar, dL_dLen):
        """
        Specially intended for the Grid regression case.
        Given the computed log likelihood derivates, update the corresponding
        kernel and likelihood gradients.
        Useful for when gradients have been computed a priori.
        """
        self.variance.gradient = dL_dVar
        self.lengthscale.gradient = dL_dLen

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
        if use_stationary_cython:
            return self._gradients_X_cython(dL_dK, X, X2)
        else:
            return self._gradients_X_pure(dL_dK, X, X2)

    def gradients_XX(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective K(dL_dK), compute the second derivative of K wrt X and X2:

        returns the full covariance matrix [QxQ] of the input dimensionfor each pair or vectors, thus
        the returned array is of shape [NxNxQxQ].

        ..math:
            \frac{\partial^2 K}{\partial X2 ^2} = - \frac{\partial^2 K}{\partial X\partial X2}

        ..returns:
            dL2_dXdX2:  [NxMxQxQ] in the cov=True case, or [NxMxQ] in the cov=False case,
                        for X [NxQ] and X2[MxQ] (X2 is X if, X2 is None)
                        Thus, we return the second derivative in X2.
        """
        # According to multivariable chain rule, we can chain the second derivative through r:
        # d2K_dXdX2 = dK_dr*d2r_dXdX2 + d2K_drdr * dr_dX * dr_dX2:
        invdist = self._inv_dist(X, X2)
        invdist2 = invdist**2
        dL_dr = self.dK_dr_via_X(X, X2) #* dL_dK # we perform this product later
        tmp1 = dL_dr * invdist
        dL_drdr = self.dK2_drdr_via_X(X, X2) #* dL_dK # we perofrm this product later
        tmp2 = dL_drdr*invdist2
        l2 =  np.ones(X.shape[1])*self.lengthscale**2 #np.multiply(np.ones(X.shape[1]) ,self.lengthscale**2)

        if X2 is None:
            X2 = X
            tmp1 -= np.eye(X.shape[0])*self.variance
        else:
            tmp1[invdist2==0.] -= self.variance

        #grad = np.empty((X.shape[0], X2.shape[0], X2.shape[1], X.shape[1]), dtype=np.float64)
        dist = X[:,None,:] - X2[None,:,:]
        dist = (dist[:,:,:,None]*dist[:,:,None,:])
        I = np.ones((X.shape[0], X2.shape[0], X2.shape[1], X.shape[1]))*np.eye((X2.shape[1]))
        grad = (((dL_dK*(tmp1*invdist2 - tmp2))[:,:,None,None] * dist)/l2[None,None,:,None]
                - (dL_dK*tmp1)[:,:,None,None] * I)/l2[None,None,None,:]
        return grad

    def gradients_XX_diag(self, dL_dK_diag, X):
        """
        Given the derivative of the objective dL_dK, compute the second derivative of K wrt X:

        ..math:
          \frac{\partial^2 K}{\partial X\partial X}

        ..returns:
            dL2_dXdX: [NxQxQ]
        """
        dL_dK_diag = dL_dK_diag.copy().reshape(-1, 1, 1)
        assert (dL_dK_diag.size == X.shape[0]) or (dL_dK_diag.size == 1), "dL_dK_diag has to be given as row [N] or column vector [Nx1]"

        l4 =  np.ones(X.shape[1])*self.lengthscale**2
        return dL_dK_diag * (np.eye(X.shape[1]) * -self.dK2_drdr_diag()/(l4))[None, :,:]# np.zeros(X.shape+(X.shape[1],))
        #return np.ones(X.shape) * d2L_dK * self.variance/self.lengthscale**2 # np.zeros(X.shape)
    
    def dgradients_dX(self, X, X2, dimX):
        g1 = self.dK2_dvariancedX(X, X2, dimX)
        g2 = self.dK2_dlengthscaledX(X, X2, dimX)
        return [g1, g2]

    def dgradients_dX2(self, X, X2, dimX2):
        g1 = self.dK2_dvariancedX2(X, X2, dimX2)
        g2 = self.dK2_dlengthscaledX2(X, X2, dimX2)
        return [g1, g2]

    def dgradients2_dXdX2(self, X, X2, dimX, dimX2):
        g1 = self.dK3_dvariancedXdX2(X, X2, dimX, dimX2)
        g2 = self.dK3_dlengthscaledXdX2(X, X2, dimX, dimX2)
        return [g1, g2]

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

    def get_one_dimensional_kernel(self, dimensions):
        """
        Specially intended for the grid regression case
        For a given covariance kernel, this method returns the corresponding kernel for
        a single dimension. The resulting values can then be used in the algorithm for
        reconstructing the full covariance matrix.
        """
        raise NotImplementedError("implement one dimensional variation of kernel")




class Exponential(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Exponential'):
        super(Exponential, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.exp(-r)

    def dK_dr(self, r):
        return -self.K_of_r(r)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Exponential, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.Exponential"
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Exponential(**input_dict)

#    def sde(self):
#        """
#        Return the state space representation of the covariance.
#        """
#        F  = np.array([[-1/self.lengthscale]])
#        L  = np.array([[1]])
#        Qc = np.array([[2*self.variance/self.lengthscale]])
#        H = np.array([[1]])
#        Pinf = np.array([[self.variance]])
#        # TODO: return the derivatives as well
#
#        return (F, L, Qc, H, Pinf)


class OU(Stationary):
    """
    OU kernel:

    .. math::

       k(r) = \\sigma^2 \exp(- r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^{\text{input_dim}} \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='OU'):
        super(OU, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(OU, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.OU"
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return OU(**input_dict)

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

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Matern32, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.Matern32"
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Matern32(**input_dict)

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

    def sde(self):
        """
        Return the state space representation of the covariance.
        """
        variance = float(self.variance.values)
        lengthscale = float(self.lengthscale.values)
        foo  = np.sqrt(3.)/lengthscale
        F    = np.array([[0, 1], [-foo**2, -2*foo]])
        L    = np.array([[0], [1]])
        Qc   = np.array([[12.*np.sqrt(3) / lengthscale**3 * variance]])
        H    = np.array([[1, 0]])
        Pinf = np.array([[variance, 0],
        [0,              3.*variance/(lengthscale**2)]])
        # Allocate space for the derivatives
        dF    = np.empty([F.shape[0],F.shape[1],2])
        dQc   = np.empty([Qc.shape[0],Qc.shape[1],2])
        dPinf = np.empty([Pinf.shape[0],Pinf.shape[1],2])
        # The partial derivatives
        dFvariance       = np.zeros([2,2])
        dFlengthscale    = np.array([[0,0],
        [6./lengthscale**3,2*np.sqrt(3)/lengthscale**2]])
        dQcvariance      = np.array([12.*np.sqrt(3)/lengthscale**3])
        dQclengthscale   = np.array([-3*12*np.sqrt(3)/lengthscale**4*variance])
        dPinfvariance    = np.array([[1,0],[0,3./lengthscale**2]])
        dPinflengthscale = np.array([[0,0],
        [0,-6*variance/lengthscale**3]])
        # Combine the derivatives
        dF[:,:,0]    = dFvariance
        dF[:,:,1]    = dFlengthscale
        dQc[:,:,0]   = dQcvariance
        dQc[:,:,1]   = dQclengthscale
        dPinf[:,:,0] = dPinfvariance
        dPinf[:,:,1] = dPinflengthscale

        return (F, L, Qc, H, Pinf, dF, dQc, dPinf)

class Matern52(Stationary):
    """
    Matern 5/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{5} r + \\frac53 r^2) \exp(- \sqrt{5} r)
    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat52'):
        super(Matern52, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Matern52, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.Matern52"
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Matern52(**input_dict)

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

       k(r) = \sigma^2 \exp(- 0.5 r^2)

    notes::
     - This is exactly the same as the RBF covariance function, but the
       RBF implementation also has some features for doing variational kernels
       (the psi-statistics).

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='ExpQuad'):
        super(ExpQuad, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(ExpQuad, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.ExpQuad"
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return ExpQuad(**input_dict)

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

class Cosine(Stationary):
    """
    Cosine Covariance function
    
    .. math::

        k(r) = \sigma^2 \cos(r)

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Cosine'):
        super(Cosine, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.cos(r)

    def dK_dr(self, r):
        return -self.variance * np.sin(r)

class ExpQuadCosine(Stationary):
    """
    Exponentiated quadratic multiplied by cosine covariance function (spectral mixture kernel).
    
    .. math::

        k(r) = \sigma^2 \exp(-2\pi^2r^2)\cos(2\pi r/T)

    """
    
    def __init__(self, input_dim, variance=1., lengthscale=None, period=1., ARD=False, active_dims=None, name='ExpQuadCosine'):
        super(ExpQuadCosine, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
        self.period = Param('period', period, Logexp())
        self.link_parameters(self.period)
        
    def K_of_r(self, r):
        return self.variance * np.exp(-2*np.pi**2*r**2)*np.cos(2*np.pi*r/self.period)

    def dK_dr(self, r):
        return -4*np.pi**2*r*self.K_of_r(r) - self.variance * 2*np.pi/self.period*np.exp(-2*np.pi**2*r**2)*np.sin(2*np.pi*r/self.period)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(ExpQuadCosine, self).update_gradients_full(dL_dK, X, X2)
        r = self._scaled_dist(X, X2)
        r2 = np.square(r)
        dK_dperiod = self.variance * 2*np.pi*r/self.period**2*np.exp(-2*np.pi**2*r**2)*np.sin(2*np.pi*r/self.period)
        grad = np.sum(dL_dK*dK_dperiod)
        self.period.gradient = grad

    def update_gradients_diag(self, dL_dKdiag, X):
        super(ExpQuadCosine, self).update_gradients_diag(dL_dKdiag, X)
        self.period.gradient = 0.

    
    
class Sinc(Stationary):
    """
    Sinc Covariance function
    
    .. math::

        k(r) = \sigma^2 \sinc(\pi r)

    """
    
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Sinc'):
        super(Sinc, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.sinc(2*r)

    def dK_dr(self, r):
        # small angle approximation to avoid divide by zero errors.
        return np.where(r<1e-5, -self.variance*4/3*np.pi*np.pi*r, self.variance/r * (np.cos(2*np.pi*r)-np.sinc(2*r)))

    

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

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(RatQuad, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RatQuad"
        input_dict["power"] = self.power.values.tolist()
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return RatQuad(**input_dict)

    def K_of_r(self, r):
        r2 = np.square(r)
#         return self.variance*np.power(1. + r2/2., -self.power)
        return self.variance*np.exp(-self.power*np.log1p(r2/2.))

    def dK_dr(self, r):
        r2 = np.square(r)
#         return -self.variance*self.power*r*np.power(1. + r2/2., - self.power - 1.)
        return -self.variance*self.power*r*np.exp(-(self.power+1)*np.log1p(r2/2.))

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
