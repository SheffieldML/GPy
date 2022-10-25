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
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        dist = X[:,None,dimX]-X2[None,:,dimX]
        lengthscale2inv = (np.ones((X.shape[1]))/(self.lengthscale**2))[dimX]
        return -1.*K*dist*lengthscale2inv

    @Cache_this(limit=3, ignore_args=())
    def dK_dX2(self, X, X2, dimX2):
        return -self._clean_dK_dX(X, X2, dimX2)

    @Cache_this(limit=3, ignore_args=())
    def dK_dXdiag(self, X, dimX):
        return np.zeros(X.shape[0])

    @Cache_this(limit=3, ignore_args=())
    def dK_dX2diag(self, X, dimX2):
        return np.zeros(X.shape[0])
    
    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        if X2 is None:
            X2=X
        dist = X[:,None,:]-X2[None,:,:]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        return -1.*K*dist[:,:,dimX]*dist[:,:,dimX2]*lengthscale2inv[dimX]*lengthscale2inv[dimX2] + (dimX==dimX2)*K*lengthscale2inv[dimX]

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdX(self, X, X2, dim_pred_grads, dimX):
        return -self._clean_dK2_dXdX2(X, X2, dim_pred_grads, dimX)

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdX2diag(self, X, dimX, dimX2):
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        return np.ones(X.shape[0])*self.variance*lengthscale2inv[dimX]*(dimX==dimX2)

    @Cache_this(limit=3, ignore_args=())
    def dK2_dXdXdiag(self, X, dimX, dimX2):
        return -self._clean_dK2_dXdX2diag(X, dimX, dimX2)

    @Cache_this(limit=3, ignore_args=())
    def dK3_dXdXdX2(self, X, X2, dim_pred_grads, dimX, dimX2):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        if X2 is None:
            X2=X
        dist = X[:,None,:]-X2[None,:,:]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        return K*(dist[:,:,dim_pred_grads]*lengthscale2inv[dim_pred_grads]*\
                  dist[:,:,dimX]*lengthscale2inv[dimX]*\
                  dist[:,:,dimX2]*lengthscale2inv[dimX2]-\
                  (dim_pred_grads==dimX)*dist[:,:,dimX2]*lengthscale2inv[dimX2]*lengthscale2inv[dim_pred_grads]-\
                  (dimX==dimX2)*dist[:,:,dim_pred_grads]*lengthscale2inv[dim_pred_grads]*lengthscale2inv[dimX] -\
                  (dimX2==dim_pred_grads)*dist[:,:,dimX]*lengthscale2inv[dimX]*lengthscale2inv[dimX2] )

    @Cache_this(limit=3, ignore_args=())
    def dK3_dXdXdX2diag(self, X, dim_pred_grads, dimX):
        return np.zeros(X.shape[0])

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r**2-1)*self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance # as the diagonal of r is always filled with zeros
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dvariance(self,X,X2):
        return self._clean_K(X,X2)/self.variance
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dlengthscale(self,X,X2):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        if X2 is None:
            X2=X
        dist = X[:,None,:]-X2[None,:,:]
        lengthscaleinv = np.ones((X.shape[1]))/(self.lengthscale)
        if self.ARD:
            g = []
            for diml in range(X.shape[1]):
                g += [ (dist[:,:,diml]**2)*(lengthscaleinv[diml]**3)*K]
        else:
            g = np.sum(dist**2, axis=2)*(lengthscaleinv[0]**3)*K
        return g
    
    @Cache_this(limit=3, ignore_args=())
    def dK2_dvariancedX(self, X, X2, dim):
        return self._clean_dK_dX(X,X2, dim)/self.variance
    
    @Cache_this(limit=3, ignore_args=())
    def dK2_dvariancedX2(self, X, X2, dim):
        return self._clean_dK_dX2(X,X2, dim)/self.variance
    
    @Cache_this(limit=3, ignore_args=())
    def dK3_dvariancedXdX2(self, X, X2, dim, dimX2):
        return self._clean_dK2_dXdX2(X, X2, dim, dimX2)/self.variance

    @Cache_this(limit=3, ignore_args=())
    def dK2_dlengthscaledX(self, X, X2, dimX):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        if X2 is None:
            X2=X
        dist = X[:,None,:]-X2[None,:,:]
        lengthscaleinv = np.ones((X.shape[1]))/(self.lengthscale)
        if self.ARD:
            g = []
            for diml in range(X.shape[1]):
                g += [-1.*K*dist[:,:,dimX]*(dist[:,:,diml]**2)*(lengthscaleinv[dimX]**2)*(lengthscaleinv[diml]**3) + 2.*dist[:,:,dimX]*(lengthscaleinv[diml]**3)*K*(dimX == diml)]
        else:
            g = -1.*K*dist[:,:,dimX]*np.sum(dist**2, axis=2)*(lengthscaleinv[dimX]**5) + 2.*dist[:,:,dimX]*(lengthscaleinv[dimX]**3)*K
        return g
    
    @Cache_this(limit=3, ignore_args=())
    def dK2_dlengthscaledX2(self, X, X2, dimX2):
        tmp = self.dK2_dlengthscaledX(X, X2, dimX2)
        if self.ARD:
            return [-1.*g for g in tmp]
        else:
            return -1*tmp
    
    @Cache_this(limit=3, ignore_args=())
    def dK3_dlengthscaledXdX2(self, X, X2, dimX, dimX2):
        r = self._scaled_dist(X, X2)
        K = self.K_of_r(r)
        if X2 is None:
            X2=X
        dist = X[:,None,:]-X2[None,:,:]
        lengthscaleinv = np.ones((X.shape[1]))/(self.lengthscale)
        lengthscale2inv = lengthscaleinv**2
        if self.ARD:
            g = []
            for diml in range(X.shape[1]):
                tmp = -1.*K*dist[:,:,dimX]*dist[:,:,dimX2]*(dist[:,:,diml]**2)*lengthscale2inv[dimX]*lengthscale2inv[dimX2]*(lengthscaleinv[diml]**3)
                if dimX == dimX2:
                    tmp += K*lengthscale2inv[dimX]*(lengthscaleinv[diml]**3)*(dist[:,:,diml]**2)
                if diml == dimX:
                    tmp += 2.*K*dist[:,:,dimX]*dist[:,:,dimX2]*lengthscale2inv[dimX2]*(lengthscaleinv[dimX]**3)
                if diml == dimX2:
                    tmp += 2.*K*dist[:,:,dimX]*dist[:,:,dimX2]*lengthscale2inv[dimX]*(lengthscaleinv[dimX2]**3)
                    if dimX == dimX2:
                        tmp += -2.*K*(lengthscaleinv[dimX]**3)
                g += [tmp]
        else:
            g = -1.*K*dist[:,:,dimX]*dist[:,:,dimX2]*np.sum(dist**2, axis=2)*(lengthscaleinv[dimX]**7) +4*K*dist[:,:,dimX]*dist[:,:,dimX2]*(lengthscaleinv[dimX]**5)
            if dimX == dimX2:
                g += -2.*K*(lengthscaleinv[dimX]**3) + K*(lengthscaleinv[dimX]**5)*np.sum(dist**2, axis=2)
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
