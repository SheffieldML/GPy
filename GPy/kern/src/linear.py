# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .kern import Kern
from ...util.linalg import tdot
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this
from .psi_comp import PSICOMP_Linear

class Linear(Kern):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^{\\text{input_dim}} \sigma^2_i x_iy_i

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

    def __init__(self, input_dim, variances=None, ARD=False, active_dims=None, name='linear'):
        super(Linear, self).__init__(input_dim, active_dims, name)
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
        self.link_parameter(self.variances)
        self.psicomp = PSICOMP_Linear()

    def to_dict(self):
        input_dict = super(Linear, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.Linear"
        input_dict["variances"] = self.variances.values.tolist()
        input_dict["ARD"] = self.ARD
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Linear(**input_dict)

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        if self.ARD:
            if X2 is None:
                return tdot(X*np.sqrt(self.variances))
            else:
                rv = np.sqrt(self.variances)
                return np.dot(X*rv, (X2*rv).T)
        else:
            return self._dot_product(X, X2) * self.variances

    @Cache_this(limit=3, ignore_args=(0,))
    def _dot_product(self, X, X2=None):
        if X2 is None:
            return tdot(X)
        else:
            return np.dot(X, X2.T)

    def Kdiag(self, X):
        return np.sum(self.variances * np.square(X), -1)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: dL_dK = (dL_dK+dL_dK.T)/2
        if self.ARD:
            if X2 is None:
                #self.variances.gradient = np.array([np.sum(dL_dK * tdot(X[:, i:i + 1])) for i in range(self.input_dim)])
                self.variances.gradient = (dL_dK.dot(X)*X).sum(0) #np.einsum('ij,iq,jq->q', dL_dK, X, X)
            else:
                #product = X[:, None, :] * X2[None, :, :]
                #self.variances.gradient = (dL_dK[:, :, None] * product).sum(0).sum(0)
                self.variances.gradient = (dL_dK.dot(X2)*X).sum(0)  #np.einsum('ij,iq,jq->q', dL_dK, X, X2)
        else:
            self.variances.gradient = np.sum(self._dot_product(X, X2) * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        tmp = dL_dKdiag[:, None] * X ** 2
        if self.ARD:
            self.variances.gradient = tmp.sum(0)
        else:
            self.variances.gradient = np.atleast_1d(tmp.sum())


    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None: dL_dK = (dL_dK+dL_dK.T)/2
        if X2 is None:
            return dL_dK.dot(X)*(2*self.variances) #np.einsum('jq,q,ij->iq', X, 2*self.variances, dL_dK)
        else:
            #return (((X2[None,:, :] * self.variances)) * dL_dK[:, :, None]).sum(1)
            return dL_dK.dot(X2)*self.variances #np.einsum('jq,q,ij->iq', X2, self.variances, dL_dK)

    def gradients_XX(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective K(dL_dK), compute the second derivative of K wrt X and X2:

        returns the full covariance matrix [QxQ] of the input dimensionfor each pair or vectors, thus
        the returned array is of shape [NxNxQxQ].

        ..math:
            \frac{\partial^2 K}{\partial X2 ^2} = - \frac{\partial^2 K}{\partial X\partial X2}

        ..returns:
            dL2_dXdX2:  [NxMxQxQ] for X [NxQ] and X2[MxQ] (X2 is X if, X2 is None)
                        Thus, we return the second derivative in X2.
        """
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0], X2.shape[0], X.shape[1], X.shape[1]))
        #if X2 is None: dL_dK = (dL_dK+dL_dK.T)/2
        #if X2 is None:
        #    return np.ones(np.repeat(X.shape, 2)) * (self.variances[None,:] + self.variances[:, None])[None, None, :, :]
        #else:
        #    return np.ones((X.shape[0], X2.shape[0], X.shape[1], X.shape[1])) * (self.variances[None,:] + self.variances[:, None])[None, None, :, :]


    def gradients_X_diag(self, dL_dKdiag, X):
        return 2.*self.variances*dL_dKdiag[:,None]*X

    def gradients_XX_diag(self, dL_dKdiag, X):
        return np.zeros((X.shape[0], X.shape[1], X.shape[1]))

        #dims = X.shape
        #if cov:
        #    dims += (X.shape[1],)
        #return 2*np.ones(dims)*self.variances

    def input_sensitivity(self, summarize=True):
        return np.ones(self.input_dim) * self.variances

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[0]
        if self.ARD:
            self.variances.gradient = dL_dvar
        else:
            self.variances.gradient = dL_dvar.sum()

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[1]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2:]

class LinearFull(Kern):
    def __init__(self, input_dim, rank, W=None, kappa=None, active_dims=None, name='linear_full'):
        super(LinearFull, self).__init__(input_dim, active_dims, name)
        if W is None:
            W = np.ones((input_dim, rank))
        if kappa is None:
            kappa = np.ones(input_dim)
        assert W.shape == (input_dim, rank)
        assert kappa.shape == (input_dim,)

        self.W = Param('W', W)
        self.kappa = Param('kappa', kappa, Logexp())
        self.link_parameters(self.W, self.kappa)

    def K(self, X, X2=None):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return np.einsum('ij,jk,lk->il', X, P, X if X2 is None else X2)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: dL_dK = (dL_dK+dL_dK.T)/2
        self.kappa.gradient = np.einsum('ij,ik,kj->j', X, dL_dK, X if X2 is None else X2)
        self.W.gradient = np.einsum('ij,kl,ik,lm->jm', X, X if X2 is None else X2, dL_dK, self.W)
        self.W.gradient += np.einsum('ij,kl,ik,jm->lm', X, X if X2 is None else X2, dL_dK, self.W)

    def Kdiag(self, X):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return np.einsum('ij,jk,ik->i', X, P, X)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.kappa.gradient = np.einsum('ij,i->j', np.square(X), dL_dKdiag)
        self.W.gradient = 2.*np.einsum('ij,ik,jl,i->kl', X, X, self.W, dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None: dL_dK = (dL_dK+dL_dK.T)/2
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        if X2 is None:
            return 2.*np.einsum('ij,jk,kl->il', dL_dK, X, P)
        else:
            return np.einsum('ij,jk,kl->il', dL_dK, X2, P)

    def gradients_X_diag(self, dL_dKdiag, X):
        P = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        return 2.*np.einsum('jk,i,ij->ik', P, dL_dKdiag, X)
