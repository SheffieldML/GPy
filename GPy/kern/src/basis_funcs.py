# #Copyright (c) 2012, Max Zwiessele (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from .kern import Kern
from ...core.parameterization.param import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this
from ...util.linalg import tdot, mdot

class BasisFuncKernel(Kern):
    def __init__(self, input_dim, variance=1., active_dims=None, ARD=False, name='basis func kernel'):
        """
        Abstract superclass for kernels with explicit basis functions for use in GPy.

        This class does NOT automatically add an offset to the design matrix phi!
        """
        super(BasisFuncKernel, self).__init__(input_dim, active_dims, name)
        assert self.input_dim==1, "Basis Function Kernel only implemented for one dimension. Use one kernel per dimension (and add them together) for more dimensions"
        self.ARD = ARD
        if self.ARD:
            phi_test = self._phi(np.random.normal(0, 1, (1, self.input_dim)))
            variance = variance * np.ones(phi_test.shape[1])
        else:
            variance = np.array(variance)
        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)

    def parameters_changed(self):
        self.alpha = np.sqrt(self.variance)
        self.beta = 1./self.variance

    @Cache_this(limit=3, ignore_args=())
    def phi(self, X):
        return self._phi(X)

    def _phi(self, X):
        raise NotImplementedError('Overwrite this _phi function, which maps the input X into the higher dimensional space and returns the design matrix Phi')

    def K(self, X, X2=None):
        return self._K(X, X2)

    def Kdiag(self, X, X2=None):
        return np.diag(self._K(X, X2))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if self.ARD:
            phi1 = self.phi(X)
            if X2 is None or X is X2:
                self.variance.gradient = np.einsum('ij,iq,jq->q', dL_dK, phi1, phi1)
            else:
                phi2 = self.phi(X2)
                self.variance.gradient = np.einsum('ij,iq,jq->q', dL_dK, phi1, phi2)
        else:
            self.variance.gradient = np.einsum('ij,ij', dL_dK, self._K(X, X2)) * self.beta

    def update_gradients_diag(self, dL_dKdiag, X):
        if self.ARD:
            phi1 = self.phi(X)
            self.variance.gradient = np.einsum('i,iq,iq->q', dL_dKdiag, phi1, phi1)
        else:
            self.variance.gradient = np.einsum('i,i', dL_dKdiag, self.Kdiag(X)) * self.beta

    def concatenate_offset(self, X):
        """
        Convenience function to add an offset column to phi.
        You can use this function to add an offset (bias on y axis)
        to phi in your custom self._phi(X).
        """
        return np.c_[np.ones((X.shape[0], 1)), X]

    def posterior_inf(self, X=None, posterior=None):
        """
        Do the posterior inference on the parameters given this kernels functions
        and the model posterior, which has to be a GPy posterior, usually found at m.posterior, if m is a GPy model.
        If not given we search for the the highest parent to be a model, containing the posterior, and for X accordingly.
        """
        if X is None:
            try:
                X = self._highest_parent_.X
            except NameError:
                raise RuntimeError("This kernel is not part of a model and cannot be used for posterior inference")
        if posterior is None:
            try:
                posterior = self._highest_parent_.posterior
            except NameError:
                raise RuntimeError("This kernel is not part of a model and cannot be used for posterior inference")
        phi_alpha = self.phi(X) * self.variance
        return (phi_alpha).T.dot(posterior.woodbury_vector), (np.eye(phi_alpha.shape[1])*self.variance - mdot(phi_alpha.T, posterior.woodbury_inv, phi_alpha))

    @Cache_this(limit=3, ignore_args=())
    def _K(self, X, X2):
        if X2 is None or X is X2:
            phi = self.phi(X) * self.alpha
            if phi.ndim != 2:
                phi = phi[:, None]
            return tdot(phi)
        else:
            phi1 = self.phi(X) * self.alpha
            phi2 = self.phi(X2) * self.alpha
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
                phi2 = phi2[:, None]
            return phi1.dot(phi2.T)


class LinearSlopeBasisFuncKernel(BasisFuncKernel):
    def __init__(self, input_dim, start, stop, variance=1., active_dims=None, ARD=False, name='linear_segment'):
        """
        A linear segment transformation. The segments start at start, \
        are then linear to stop and constant again. The segments are
        normalized, so that they have exactly as much mass above
        as below the origin.

        Start and stop can be tuples or lists of starts and stops.
        Behaviour of start stop is as np.where(X<start) would do.
        """

        self.start = np.array(start)
        self.stop = np.array(stop)
        super(LinearSlopeBasisFuncKernel, self).__init__(input_dim, variance, active_dims, ARD, name)

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        phi = np.where(X < self.start, self.start, X)
        phi = np.where(phi > self.stop, self.stop, phi)
        return ((phi-(self.stop+self.start)/2.))#/(.5*(self.stop-self.start)))-1.

class ChangePointBasisFuncKernel(BasisFuncKernel):
    """
    The basis function has a changepoint. That is, it is constant, jumps at a
    single point (given as changepoint) and is constant again. You can
    give multiple changepoints. The changepoints are calculated using
    np.where(self.X < self.changepoint), -1, 1)
    """
    def __init__(self, input_dim, changepoint, variance=1., active_dims=None, ARD=False, name='changepoint'):
        self.changepoint = np.array(changepoint)
        super(ChangePointBasisFuncKernel, self).__init__(input_dim, variance, active_dims, ARD, name)

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        return np.where((X < self.changepoint), -1, 1)

class DomainKernel(LinearSlopeBasisFuncKernel):
    """
    Create a constant plateou of correlation between start and stop and zero
    elsewhere. This is a constant shift of the outputs along the yaxis
    in the range from start to stop.
    """
    def __init__(self, input_dim, start, stop, variance=1., active_dims=None, ARD=False, name='constant_domain'):
        super(DomainKernel, self).__init__(input_dim, start, stop, variance, active_dims, ARD, name)

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        phi = np.where((X>self.start)*(X<self.stop), 1, 0)
        return phi#((phi-self.start)/(self.stop-self.start))-.5

class LogisticBasisFuncKernel(BasisFuncKernel):
    """
    Create a series of logistic basis functions with centers given. The
    slope gets computed by datafit. The number of centers determines the
    number of logistic functions.
    """
    def __init__(self, input_dim, centers, variance=1., slope=1., active_dims=None, ARD=False, ARD_slope=True, name='logistic'):
        self.centers = np.atleast_2d(centers)
        if ARD:
            assert ARD_slope, "If we have one variance per center, we want also one slope per center."
        self.ARD_slope = ARD_slope
        if self.ARD_slope:
            self.slope = Param('slope', slope * np.ones(self.centers.size))
        else:
            self.slope = Param('slope', slope)
        super(LogisticBasisFuncKernel, self).__init__(input_dim, variance, active_dims, ARD, name)
        self.link_parameter(self.slope)

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        phi = 1/(1+np.exp(-((X-self.centers)*self.slope)))
        return np.where(np.isnan(phi), 0, phi)#((phi-self.start)/(self.stop-self.start))-.5

    def parameters_changed(self):
        BasisFuncKernel.parameters_changed(self)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(LogisticBasisFuncKernel, self).update_gradients_full(dL_dK, X, X2)
        if X2 is None or X is X2:
            phi1 = self.phi(X)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
            dphi1_dl = (phi1**2) * (np.exp(-((X-self.centers)*self.slope)) * (X-self.centers))
            if self.ARD_slope:
                self.slope.gradient = self.variance * 2 * np.einsum('ij,iq,jq->q', dL_dK, phi1, dphi1_dl)
            else:
                self.slope.gradient = np.sum(self.variance * 2 * (dL_dK * phi1.dot(dphi1_dl.T)).sum())
        else:
            phi1 = self.phi(X)
            phi2 = self.phi(X2)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
                phi2 = phi2[:, None]
            dphi1_dl = (phi1**2) * (np.exp(-((X-self.centers)*self.slope)) * (X-self.centers))
            dphi2_dl = (phi2**2) * (np.exp(-((X2-self.centers)*self.slope)) * (X2-self.centers))
            if self.ARD_slope:
                self.slope.gradient = (self.variance * np.einsum('ij,iq,jq->q', dL_dK, phi1, dphi2_dl) + np.einsum('ij,iq,jq->q', dL_dK, phi2, dphi1_dl))
            else:
                self.slope.gradient = np.sum(self.variance * (dL_dK * phi1.dot(dphi2_dl.T)).sum() + (dL_dK * phi2.dot(dphi1_dl.T)).sum())
        self.slope.gradient = np.where(np.isnan(self.slope.gradient), 0, self.slope.gradient)
