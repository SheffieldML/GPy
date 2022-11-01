# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .stationary import Stationary
from .psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from ...core import Param
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from .grid_kerns import GridRBF

class RBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    """
    _support_GPU = True
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False, inv_l=False):
        super(RBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if inv_l:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param('inv_lengthscale',1./self.lengthscale**2, Logexp())
            self.link_parameter(self.inv_l)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(RBF, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RBF"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"] == True:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    @Cache_this(limit=3, ignore_args=())
    def dK_dX(self, X, X2, dimX):
        """
        Compute the derivative of K with respect to:
            dimension dimX of set X.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.lengthscale))[dimX]
        dist = X[:,None,dimX] - X2[None,:,dimX]
        return -dist*(lengthscaleinv**2)*self._clean_K(X, X2)

    @Cache_this(limit=3, ignore_args=())
    def dK_dXdiag(self, X, dimX):
        """
        Compute the derivative of K with respect to:
            dimension dimX of set X.

        Returns only diagonal elements.
        """
        return np.zeros(X.shape[0])

    @Cache_this(limit=3, ignore_args=())
    def dK_dX2(self, X, X2, dimX2):
        """
        Compute the derivative of K with respect to:
            dimension dimX2 of set X2.
        """
        return -self._clean_dK_dX(X, X2, dimX2)

    @Cache_this(limit=3, ignore_args=())
    def dK_dX2diag(self, X, dimX2):
        """
        Compute the derivative of K with respect to:
            dimension dimX2 of set X2.

        Returns only diagonal elements.
        """
        return np.zeros(X.shape[0])

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the second derivative of K with respect to:
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.lengthscale))
        dist = np.rollaxis(X[:,None,:] - X2[None,:,:], 2, 0)

        term = dist[dimX]*(lengthscaleinv[dimX]**2)
        term *= dist[dimX2]*(lengthscaleinv[dimX2]**2)
        if dimX == dimX2:
            term -= (lengthscaleinv[dimX]**2)
        return -term*self._clean_K(X, X2)

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdX2diag(self, X, dimX, dimX2):
        """
        Compute the second derivative of K with respect to:
            dimension dimX of set X, and
            dimension dimX2 of set X2.

        Returns only diagonal elements.
        """
        if dimX == dimX2:
            lengthscaleinv = np.ones((X.shape[1]))/(self.lengthscale)
            return np.ones(X.shape[0])*(lengthscaleinv[dimX]**2)*self.variance
        else:
            return np.zeros(X.shape[0])

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdX(self, X, X2, dimX_0, dimX_1):
        """
        Compute the second derivative of K with respect to:
            dimension dimX_0 of set X, and
            dimension dimX_1 of set X.
        """
        return -self._clean_dK2_dXdX2(X, X2, dimX_0, dimX_1)

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdXdiag(self, X, dimX_0, dimX_1):
        """
        Compute the second derivative of K with respect to:
            dimension dimX_0 of set X, and
            dimension dimX_1 of set X.

        Returns only diagonal elements.
        """
        return -self._clean_dK2_dXdX2diag(X, dimX_0, dimX_1)

    @Cache_this(limit=3, ignore_args=())
    def dK3_dXdXdX2(self, X, X2, dimX_0, dimX_1, dimX2):
        """
        Compute the third derivative of K with respect to:
            dimension dimX_0 of set X,
            dimension dimX_1 of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.lengthscale))
        dist = np.rollaxis(X[:,None,:] - X2[None,:,:], 2, 0)

        term = dist[dimX_0]*(lengthscaleinv[dimX_0]**2)
        term *= dist[dimX_1]*(lengthscaleinv[dimX_1]**2)
        term *= dist[dimX2]*(lengthscaleinv[dimX2]**2)
        if dimX_0 == dimX_1:
            term -= dist[dimX2]*(lengthscaleinv[dimX2]**2)*(lengthscaleinv[dimX_0]**2)
        if dimX_0 == dimX2:
            term -= dist[dimX_1]*(lengthscaleinv[dimX_1]**2)*(lengthscaleinv[dimX_0]**2)
        if dimX_1 == dimX2:
            term -= dist[dimX_0]*(lengthscaleinv[dimX_0]**2)*(lengthscaleinv[dimX_1]**2)
        return term*self._clean_K(X, X2)

    @Cache_this(limit=3, ignore_args=())
    def dK3_dXdXdX2diag(self, X, dimX_0, dimX_1):
        """
        Compute the third derivative of K with respect to:
            dimension dimX_0 of set X,
            dimension dimX_1 of set X, and
            dimension dimX_1 of set X2.

        Returns only diagonal elements of the covariance matrix.
        """
        return np.zeros(X.shape[0])

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r**2-1)*self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance # as the diagonal of r is always filled with zeros

    @Cache_this(limit=3, ignore_args=())
    def dK_dvariance(self, X, X2):
        """
        Compute the derivative of K with respect to variance.
        """
        return self._clean_K(X, X2)/self.variance

    @Cache_this(limit=3, ignore_args=())
    def dK_dlengthscale(self, X, X2):
        """
        Compute the derivative(s) of K with respect to lengthscale(s).
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.lengthscale))
        dist = np.rollaxis(X[:,None,:] - X2[None,:,:], 2, 0)

        K = self._clean_K(X, X2)

        if self.ARD:
            g = []
            for diml in range(self.input_dim):
                g += [(dist[diml]**2)*(lengthscaleinv[diml]**3)*K]
        else:
            g = (lengthscaleinv[0]**3)*np.sum(dist**2, axis=0)*K
        return g

    @Cache_this(limit=3, ignore_args=())
    def dK2_dvariancedX(self, X, X2, dimX):
        """
        Compute the second derivative of K with respect to:
            variance, and
            dimension dimX of set X.
        """
        return self._clean_dK_dX(X, X2, dimX)/self.variance

    @Cache_this(limit=3, ignore_args=())
    def dK2_dvariancedX2(self, X, X2, dimX2):
        """
        Compute the second derivative of K with respect to:
            variance, and
            dimension dimX2 of set X2.
        """
        return -self.dK2_dvariancedX(X, X2, dimX2)

    @Cache_this(limit=3, ignore_args=())
    def dK2_dlengthscaledX(self, X, X2, dimX):
        """
        Compute the second derivative(s) of K with respect to:
            lengthscale(s), and
            dimension dimX of set X.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.lengthscale))
        dist = np.rollaxis(X[:,None,:] - X2[None,:,:], 2, 0)

        dK_dX = self._clean_dK_dX(X, X2, dimX)
        dK_dl = self.dK_dlengthscale(X, X2)

        if self.ARD:
            g = []
            for diml in range(self.input_dim):
                term = -dist[dimX]*(lengthscaleinv[dimX]**2)*dK_dl[diml]
                if diml == dimX:
                    term -= 2*lengthscaleinv[dimX]*dK_dX
                g += [term]
        else:
            term = -dist[dimX]*(lengthscaleinv[0]**2)*dK_dl
            term -= 2*lengthscaleinv[0]*dK_dX
            g = term
        return g

    @Cache_this(limit=3, ignore_args=())
    def dK2_dlengthscaledX2(self, X, X2, dimX2):
        """
        Compute the second derivative(s) of K with respect to:
            lengthscale(s), and
            dimension dimX2 of set X2.
        """
        dK2_dlengthscaledX = self.dK2_dlengthscaledX(X, X2, dimX2)
        if self.ARD:
            return [-1.*g for g in dK2_dlengthscaledX]
        else:
            return -1*dK2_dlengthscaledX
    
    @Cache_this(limit=3, ignore_args=())
    def dK3_dvariancedXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the third derivative of K with respect to:
            variance,
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        return self._clean_dK2_dXdX2(X, X2, dimX, dimX2)/self.variance

    @Cache_this(limit=3, ignore_args=())
    def dK3_dlengthscaledXdX2(self, X, X2, dimX, dimX2):
        """
        Compute the third derivative(s) of K with respect to:
            lengthscale(s),
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        lengthscaleinv = (np.ones(X.shape[1])/(self.lengthscale))
        dist = np.rollaxis(X[:,None,:] - X2[None,:,:], 2, 0)

        K = self._clean_K(X, X2)
        dK_dX = self._clean_dK_dX(X, X2, dimX)
        dK_dX2 = self._clean_dK_dX(X, X2, dimX2)
        dK2_dXdX2 = self._clean_dK2_dXdX2(X, X2, dimX, dimX2)

        if self.ARD:
            g = []
            for diml in range(self.input_dim):
                term = (dist[diml]**2)*(lengthscaleinv[diml]**3)*dK2_dXdX2
                if diml == dimX:
                    term -= 2*dist[dimX]*(lengthscaleinv[dimX]**3)*dK_dX2
                if diml == dimX2:
                    term -= 2*dist[dimX2]*(lengthscaleinv[dimX2]**3)*dK_dX
                if diml == dimX == dimX2:
                    term -= 2*(lengthscaleinv[dimX]**3)*K
                g += [term]
        else:
            term = np.sum(dist**2, axis=0)*dK2_dXdX2
            term -= 4*dist[dimX2]*dK_dX
            if dimX == dimX2:
                term -= 2*K
            g = (lengthscaleinv[0]**3)*term
        return g

    def __getstate__(self):
        dc = super(RBF, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(RBF, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1 #TODO: higher dim spectra?
        return self.variance*np.sqrt(2*np.pi)*self.lengthscale*np.exp(-self.lengthscale*2*omega**2/2)

    def parameters_changed(self):
        if self.use_invLengthscale: self.lengthscale[:] = 1./np.sqrt(self.inv_l+1e-200)
        super(RBF,self).parameters_changed()


    def get_one_dimensional_kernel(self, dim):
        """
        Specially intended for Grid regression.
        """
        oneDkernel = GridRBF(input_dim=1, variance=self.variance.copy(), originalDimensions=dim)
        return oneDkernel

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale*(self.lengthscale**3/-2.)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]

    def update_gradients_diag(self, dL_dKdiag, X):
        super(RBF,self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale: self.inv_l.gradient =self.lengthscale.gradient*(self.lengthscale**3/-2.)

    def update_gradients_full(self, dL_dK, X, X2=None):
        super(RBF,self).update_gradients_full(dL_dK, X, X2)
        if self.use_invLengthscale: self.inv_l.gradient =self.lengthscale.gradient*(self.lengthscale**3/-2.)
