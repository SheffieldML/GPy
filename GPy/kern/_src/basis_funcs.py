# #Copyright (c) 2012, Max Zwiessele (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from .kern import Kern
from ...core.parameterization.param import Param
from ...core.parameterization.transformations import Logexp
import numpy as np
from ...util.caching import Cache_this
from ...util.linalg import tdot

class BasisFuncKernel(Kern):
    def __init__(self, input_dim, variance=1., active_dims=None, name='basis func kernel'):
        """
        Abstract superclass for kernels with explicit basis functions for use in GPy.
        
        This class does NOT automatically add an offset to the design matrix phi!
        """
        super(BasisFuncKernel, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)
    
    def phi(self, X):
        raise NotImplementedError('Overwrite this phi function, which maps the input X into the higher dimensional space and forms the design matrix Phi')
        
    def K(self, X, X2=None):
        return self.variance * self._K(X, X2)

    def Kdiag(self, X, X2=None):
        return self.variance * np.diag(self._K(X, X2))
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.einsum('ij,ij', dL_dK, self._K(X, X2))
        
    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.einsum('i,i', dL_dKdiag, self._K(X))
        
    def concatenate_offset(self, X):
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
        phi = self.phi(X)
        return self.variance * phi.T.dot(posterior.woodbury_vector), self.variance * (1 - self.variance * phi.T.dot(posterior.woodbury_inv.dot(phi)))
    
    @Cache_this(limit=3, ignore_args=())
    def _K(self, X, X2):
        if X2 is None or X is X2:
            phi = self.phi(X)
            if phi.ndim != 2:
                phi = phi[:, None]
            return tdot(phi)
        else:
            phi1 = self.phi(X)
            phi2 = self.phi(X2)
            if phi1.ndim != 2:
                phi1 = phi1[:, None]
                phi2 = phi2[:, None]
            return phi1.dot(phi2.T)
        
        
class LinearSlopeBasisFuncKernel(BasisFuncKernel):
    def __init__(self, input_dim, start, stop, variance=1., active_dims=None, name='linear_segment'):
        super(LinearSlopeBasisFuncKernel, self).__init__(input_dim, variance, active_dims, name)
        self.start = np.array(start)
        self.stop = np.array(stop)
    
    @Cache_this(limit=3, ignore_args=())
    def phi(self, X):
        phi = np.where(X < self.start, self.start, X)
        phi = np.where(phi > self.stop, self.stop, phi)
        return ((phi-self.start)/(self.stop-self.start))-.5
        return self.concatenate_offset(phi)  # ((phi-self.start)/(self.stop-self.start))-.5
    
class ChangePointBasisFuncKernel(BasisFuncKernel):
    def __init__(self, input_dim, changepoint, variance=1., active_dims=None, name='changepoint'):
        super(ChangePointBasisFuncKernel, self).__init__(input_dim, variance, active_dims, name)
        self.changepoint = changepoint
    
    @Cache_this(limit=3, ignore_args=())
    def phi(self, X):
        return self.concatenate_offset(np.where((X < self.changepoint), -1, 1))

class DomainKernel(LinearSlopeBasisFuncKernel):
    @Cache_this(limit=3, ignore_args=())
    def phi(self, X):
        phi = np.where((X>self.start)*(X<self.stop), 1., 0.)
        return phi#((phi-self.start)/(self.stop-self.start))-.5
        return self.concatenate_offset(phi)  # ((phi-self.start)/(self.stop-self.start))-.5
