# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

class TruncLinear(Kern):
    """
    Truncated Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^input_dim \sigma^2_i \max(0, x_iy_i - \sigma_q)

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, input_dim, variances=None, delta=None, ARD=False, active_dims=None, name='linear'):
        super(TruncLinear, self).__init__(input_dim, active_dims, name)
        self.ARD = ARD
        if not ARD:
            if variances is not None:
                variances = np.asarray(variances)
                delta = np.asarray(delta)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
                delta = np.zeros(1)
        else:
            if variances is not None:
                variances = np.asarray(variances)
                delta = np.asarray(delta)
                assert variances.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances = np.ones(self.input_dim)
                delta = np.zeros(self.input_dim)

        self.variances = Param('variances', variances, Logexp())
        self.delta = Param('delta', delta)
        self.add_parameter(self.variances)
        self.add_parameter(self.delta)

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        XX = self.variances*self._product(X, X2)
        return XX.sum(axis=-1)

    @Cache_this(limit=3)
    def _product(self, X, X2=None):
        if X2 is None:
            X2 = X
        XX = np.einsum('nq,mq->nmq',X-self.delta,X2-self.delta)
        XX[XX<0] = 0
        return XX

    def Kdiag(self, X):
        return (self.variances*np.square(X-self.delta)).sum(axis=-1)

    def update_gradients_full(self, dL_dK, X, X2=None):
        dK_dvar = self._product(X, X2)
        if X2 is None:
            X2=X
        dK_ddelta = self.variances*(2*self.delta-X[:,None,:]-X2[None,:,:])*(dK_dvar>0)
        if self.ARD:
            self.variances.gradient[:] = np.einsum('nmq,nm->q',dK_dvar,dL_dK)
            self.delta.gradient[:] = np.einsum('nmq,nm->q',dK_ddelta,dL_dK)
        else:
            self.variances.gradient[:] = np.einsum('nmq,nm->',dK_dvar,dL_dK)
            self.delta.gradient[:] = np.einsum('nmq,nm->',dK_ddelta,dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        if self.ARD:
            self.variances.gradient[:] = np.einsum('nq,n->q',np.square(X-self.delta),dL_dKdiag)
            self.delta.gradient[:] = np.einsum('nq,n->q',2*self.variances*(self.delta-X),dL_dKdiag)
        else:
            self.variances.gradient[:] = np.einsum('nq,n->',np.square(X-self.delta),dL_dKdiag)
            self.delta.gradient[:] = np.einsum('nq,n->',2*self.variances*(self.delta-X),dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2=None):
        XX = self._product(X, X2)
        if X2 is None:
            Xp = (self.variances*(X-self.delta))*(XX>0)
        else:
            Xp = (self.variances*(X2-self.delta))*(XX>0)
        if X2 is None:
            return np.einsum('nmq,nm->nq',Xp,dL_dK)+np.einsum('mnq,nm->mq',Xp,dL_dK)
        else:
            return np.einsum('nmq,nm->nq',Xp,dL_dK)

    def gradients_X_diag(self, dL_dKdiag, X):
        return 2.*self.variances*dL_dKdiag[:,None]*(X-self.delta)

    def input_sensitivity(self):
        return np.ones(self.input_dim) * self.variances

class TruncLinear_inf(Kern):
    """
    Truncated Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^input_dim \sigma^2_i \max(0, x_iy_i - \sigma_q)

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: array or list of the appropriate size (or float if there
                     is only one variance parameter)
    :param ARD: Auto Relevance Determination. If False, the kernel has only one
                variance parameter \sigma^2, otherwise there is one variance
                parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    """

    def __init__(self, input_dim, interval, variances=None, ARD=False, active_dims=None, name='linear'):
        super(TruncLinear_inf, self).__init__(input_dim, active_dims, name)
        self.interval = interval
        self.ARD = ARD
        if not ARD:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
        else:
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.input_dim, "bad number of variances, need one ARD variance per input_dim"
            else:
                variances = np.ones(self.input_dim)

        self.variances = Param('variances', variances, Logexp())
        self.add_parameter(self.variances)


#     @Cache_this(limit=3)
    def K(self, X, X2=None):
        tmp = self._product(X, X2)
        return (self.variances*tmp).sum(axis=-1)

#     @Cache_this(limit=3)
    def _product(self, X, X2=None):
        if X2 is None:
            X2 = X
        X_X2 = X[:,None,:]-X2[None,:,:]
        tmp = np.abs(X_X2**3)/6+np.einsum('nq,mq->nmq',X,X2)*(self.interval[1]-self.interval[0]) \
              -(X[:,None,:]+X2[None,:,:])*(self.interval[1]*self.interval[1]-self.interval[0]*self.interval[0])/2+(self.interval[1]**3-self.interval[0]**3)/3.
        return tmp

    def Kdiag(self, X):
        tmp = np.square(X)*(self.interval[1]-self.interval[0]) \
              -X*(self.interval[1]*self.interval[1]-self.interval[0]*self.interval[0])+(self.interval[1]**3-self.interval[0]**3)/3
        return (self.variances*tmp).sum(axis=-1)

    def update_gradients_full(self, dL_dK, X, X2=None):
        dK_dvar = self._product(X, X2)
        if self.ARD:
            self.variances.gradient[:] = np.einsum('nmq,nm->q',dK_dvar,dL_dK)
        else:
            self.variances.gradient[:] = np.einsum('nmq,nm->',dK_dvar,dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        tmp = np.square(X)*(self.interval[1]-self.interval[0]) \
              -X*(self.interval[1]*self.interval[1]-self.interval[0]*self.interval[0])+(self.interval[1]**3-self.interval[0]**3)/3
        if self.ARD:
            self.variances.gradient[:] = np.einsum('nq,n->q',tmp,dL_dKdiag)
        else:
            self.variances.gradient[:] = np.einsum('nq,n->',tmp,dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2=None):
        XX = self._product(X, X2)
        if X2 is None:
            Xp = (self.variances*(X-self.delta))*(XX>0)
        else:
            Xp = (self.variances*(X2-self.delta))*(XX>0)
        if X2 is None:
            return np.einsum('nmq,nm->nq',Xp,dL_dK)+np.einsum('mnq,nm->mq',Xp,dL_dK)
        else:
            return np.einsum('nmq,nm->nq',Xp,dL_dK)

    def gradients_X_diag(self, dL_dKdiag, X):
        return 2.*self.variances*dL_dKdiag[:,None]*(X-self.delta)

    def input_sensitivity(self):
        return np.ones(self.input_dim) * self.variances


