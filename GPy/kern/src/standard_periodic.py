# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
The standard periodic kernel which mentioned in:

[1] Gaussian Processes for Machine Learning, C. E. Rasmussen, C. K. I. Williams.
The MIT Press, 2005.


[2] Introduction to Gaussian processes. D. J. C. MacKay. In C. M. Bishop, editor,
Neural Networks and Machine Learning, pages 133-165. Springer, 1998.
"""

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp

import numpy as np

class StdPeriodic(Kern):
    """
    Standart periodic kernel

    .. math::

       k(x,y) = \theta_1 \exp \left[  - \frac{1}{2} \sum_{i=1}^{input\_dim}
       \left( \frac{\sin(\frac{\pi}{T_i} (x_i - y_i) )}{l_i} \right)^2 \right] }

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\theta_1` in the formula above
    :type variance: float
    :param period: the vector of periods :math:`\T_i`. If None then 1.0 is assumed.
    :type period: array or list of the appropriate size (or float if there is only one period parameter)
    :param lengthscale: the vector of lengthscale :math:`\l_i`. If None then 1.0 is assumed.
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD1: Auto Relevance Determination with respect to period.
        If equal to "False" one single period parameter :math:`\T_i` for
        each dimension is assumed, otherwise there is one lengthscale
        parameter per dimension.
    :type ARD1: Boolean
    :param ARD2: Auto Relevance Determination with respect to lengthscale.
        If equal to "False" one single lengthscale parameter :math:`l_i` for
        each dimension is assumed, otherwise there is one lengthscale
        parameter per dimension.
    :type ARD2: Boolean
    :param active_dims: indices of dimensions which are used in the computation of the kernel
    :type active_dims: array or list of the appropriate size
    :param name: Name of the kernel for output
    :type String
    :param useGPU: whether of not use GPU
    :type Boolean
    """

    def __init__(self, input_dim, variance=1., period=None, lengthscale=None, ARD1=False, ARD2=False, active_dims=None, name='std_periodic',useGPU=False):
        super(StdPeriodic, self).__init__(input_dim, active_dims, name, useGPU=useGPU)
        self.ARD1 = ARD1 # correspond to periods
        self.ARD2 = ARD2 # correspond to lengthscales

        self.name = name

        if self.ARD1 == False:
            if period is not None:
                period = np.asarray(period)
                assert period.size == 1, "Only one period needed for non-ARD kernel"
            else:
                period = np.ones(1)
        else:
            if period is not None:
                period = np.asarray(period)
                assert period.size == input_dim, "bad number of periods"
            else:
                period = np.ones(input_dim)

        if self.ARD2 == False:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                lengthscale = np.ones(1)
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == input_dim, "bad number of lengthscales"
            else:
                lengthscale = np.ones(input_dim)

        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size==1, "Variance size must be one"
        self.period =  Param('period', period, Logexp())
        self.lengthscale =  Param('lengthscale', lengthscale, Logexp())

        self.link_parameters(self.variance,  self.period, self.lengthscale)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(StdPeriodic, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.StdPeriodic"
        input_dict["variance"] = self.variance.values.tolist()
        input_dict["period"] = self.period.values.tolist()
        input_dict["lengthscale"] = self.lengthscale.values.tolist()
        input_dict["ARD1"] = self.ARD1
        input_dict["ARD2"] = self.ARD2
        return input_dict


    def parameters_changed(self):
        """
        This functions deals as a callback for each optimization iteration.
        If one optimization step was successfull and the parameters
        this callback function will be called to be able to update any
        precomputations for the kernel.
        """

        pass


    def K(self, X, X2=None):
        """Compute the covariance matrix between X and X2."""
        if X2 is None:
            X2 = X

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.period
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale ), axis = -1 ) )

        return self.variance * exp_dist


    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix associated to X."""
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def dK_dX(self, X, X2, dimX):
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        periodinv = np.ones(X.shape[1])/(self.period)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist * periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        k = self.variance*exp_dist[None,:,:]
        full = -k*np.pi/2.*np.sin(2.*base)*lengthscale2inv[:,None,None]*periodinv[:,None,None]
        return full[dimX,:,:]

    def dK_dX2(self, X, X2, dimX2):
        return -self._clean_dK_dX(X, X2, dimX2)

    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        periodinv  = np.ones(X.shape[1])/(self.period)
        period2inv = np.ones(X.shape[1])/(self.period**2)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist * periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        k = self.variance*exp_dist
        dk_dx2 = self._clean_dK_dX2( X, X2, dimX2)
        ret = -dk_dx2*np.pi/2.*lengthscale2inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:])
        if dimX == dimX2:
            ret += k*(np.pi**2)*period2inv[dimX]*lengthscale2inv[dimX]*np.cos(2.*base[dimX,:,:])
        return ret

    def dK_dvariance(self, X, X2=None):
        if X2 is None:
            X2 = X
        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.period
        return np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale ), axis = -1 ) )

    def dK_dlengthscale(self, X, X2=None):
        if X2 is None:
            X2=X
        lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
        periodinv = np.ones(X.shape[1])/(self.period)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist *periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        return self.variance*np.sum((np.sin(base))**2, axis=0)*exp_dist/(self.lengthscale**3) if not self.ARD2 else self.variance*exp_dist[None,:,:]*(np.sin(base))**2*lengthscale3inv[:,None,None]

    def dK_dperiod(self, X, X2=None):
        if X2 is None:
            X2=X
        periodinv = 1/self.period
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist *periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        return self.variance*exp_dist*np.sum(np.sin(base)*np.cos(base)*base/self.period[:,None,None]/(self.lengthscale[:,None,None]**2), axis=0) if not self.ARD1 else self.variance*exp_dist[None,:,:]*np.sin(base)*np.cos(base)*base/self.period[:,None,None]/(self.lengthscale[:,None,None]**2);

    def dK2_dvariancedX(self, X, X2, dimX):
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        periodinv = np.ones(X.shape[1])/(self.period)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist * periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        ret = -exp_dist*np.pi/2.*np.sin(2.*base)*lengthscale2inv[:,None,None]*periodinv[:,None,None]
        return ret[dimX,:,:]

    def dK2_dlengthscaledX(self, X, X2, dimX):
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
        periodinv = np.ones(X.shape[1])/(self.period)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist *periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        if not self.ARD2:
            ret = np.pi*self.variance*lengthscale3inv[dimX]*periodinv[dimX]*np.sin(2.*base)*exp_dist*(1 - 0.5*self.lengthscale*np.sum(lengthscale3inv[dimX]*np.sin(base)**2, axis=0)) 
        else:
            tmp = np.pi*self.variance*lengthscale3inv[:,None,None]*periodinv[dimX]*np.sin(2.*base[dimX, :, :])*exp_dist
            ret = -0.5*(np.sin(base)**2)*lengthscale2inv[dimX]*tmp
            ret[dimX,:,:] += tmp[dimX,:,:]
        return ret

    def dK2_dperioddX(self, X, X2, dimX):
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        periodinv = np.ones(X.shape[1])/(self.period)
        period2inv = np.ones(X.shape[1])/(self.period**2)
        period3inv = np.ones(X.shape[1])/(self.period**3)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist *periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        k = self._clean_K(X,X2)
        dk_dperiod = self.dK_dperiod(X,X2)
        if self.ARD1:
            ret = -dk_dperiod*np.pi*np.sin(2.*base[dimX,:,:])*lengthscale2inv[dimX]*periodinv[dimX]/2.
            ret[dimX,:,:] += k*np.pi*lengthscale2inv[dimX]*period2inv[dimX]*(np.pi*np.cos(2.*base[dimX,:,:])*dist[dimX,:,:]*periodinv[dimX] + np.sin(2.*base[dimX,:,:])/2.)
            return ret
        else:
            ret = self.variance*exp_dist[None,:,:]*np.pi*lengthscale2inv[:,None,None]*(np.pi*period3inv[:,None,None]*np.cos(2.*base)*dist - 0.25*np.pi*period3inv[:,None,None]*np.sin(2.*base)*np.sum(dist*np.sin(2.*base)*lengthscale2inv[:,None,None], axis=0) +0.5*np.sin(2.*base)*period2inv[:,None,None])
            return ret[dimX,:,:]

    def dK2_dvariancedX2(self, X, X2, dimX2):
        return -self.dK2_dvariancedX(X, X2, dimX2)

    def dK2_dlengthscaledX2(self, X, X2, dimX2):
        return -self.dK2_dlengthscaledX(X, X2, dimX2)

    def dK2_dperioddX2(self, X, X2, dimX2):
        return -self.dK2_dperioddX(X, X2, dimX2)

    def dK3_dvariancedXdX2(self, X, X2, dimX, dimX2):
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        periodinv  = np.ones(X.shape[1])/(self.period)
        period2inv = np.ones(X.shape[1])/(self.period**2)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist * periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        I = np.eye((X.shape[1]))
        dk2_dvariancedx2 = self.dK2_dvariancedX2(X, X2, dimX2)
        dk_dvariance = self.dK_dvariance(X, X2)
        ret = -dk2_dvariancedx2*np.pi/2.*lengthscale2inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:])
        ret[dimX,dimX2] += (dk_dvariance*(np.pi**2)*period2inv[dimX]*lengthscale2inv[dimX]*np.cos(2.*base[dimX,:,:]))[dimX, dimX2]
        return ret

    def dK3_dlengthscaledXdX2(self, X, X2, dimX, dimX2):
        lengthscaleinv = np.ones(X.shape[1])/(self.lengthscale)
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
        lengthscale4inv = np.ones(X.shape[1])/(self.lengthscale**4)
        lengthscale5inv = np.ones(X.shape[1])/(self.lengthscale**5)
        periodinv = np.ones(X.shape[1])/(self.period)
        period2inv = np.ones(X.shape[1])/(self.period**2)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist * periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        k = self.variance*exp_dist;
        dk2_dlengthgthscaledx2 = self.dK2_dlengthscaledX2(X, X2, dimX2)
        dk_dx2 = self._clean_dK_dX2(X,X2,dimX2)
        dk_dlengthscale = self.dK_dlengthscale(X, X2)
        I = np.eye((X.shape[1]))
        if self.ARD2:
            tmp1 =-dk2_dlengthgthscaledx2*np.pi/2.*lengthscale2inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:])
            tmp2 = np.zeros_like(tmp1)
            tmp3 = np.zeros_like(tmp1)
            tmp4 = np.zeros_like(tmp1)
            # i = j
            tmp2 = np.zeros_like(tmp1)
            tmp2[dimX,:,:] = dk_dx2*np.pi*lengthscale3inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:])
            if dimX == dimX2: # j = k
                tmp3 = dk_dlengthscale*(np.pi**2)*lengthscale2inv[dimX]*period2inv[dimX]*np.cos(2.*base[dimX,:,:])
                # i = j = k
                tmp4 = np.zeros_like(tmp1)
                tmp4[dimX,:,:] = -2.*k[:,:]*(np.pi**2)*lengthscale3inv[dimX]*period2inv[dimX]*np.cos(2.*base[dimX,:,:])
            return tmp1+tmp2+tmp3+tmp4 
        else:
            tmp1 = -dk2_dlengthgthscaledx2[dimX2,:,:]*np.pi/2.*lengthscale2inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:]) + dk_dx2*np.pi*lengthscale3inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:])
            tmp2 = np.zeros_like(tmp1)
            if dimX == dimX2:
                tmp2[:,:] = (dk_dlengthscale*(np.pi**2)*lengthscale2inv[dimX]*period2inv[dimX]*np.cos(2.*base[dimX]) -2.*k*(np.pi**2)*lengthscale3inv[dimX]*period2inv[dimX]*np.cos(2.*base[dimX,:,:]))
            return tmp1+tmp2 

    def dK3_dperioddXdX2(self, X, X2, dimX, dimX2):
        lengthscaleinv = np.ones(X.shape[1])/(self.lengthscale)
        lengthscale2inv = np.ones(X.shape[1])/(self.lengthscale**2)
        lengthscale3inv = np.ones(X.shape[1])/(self.lengthscale**3)
        lengthscale4inv = np.ones(X.shape[1])/(self.lengthscale**4)
        lengthscale5inv = np.ones(X.shape[1])/(self.lengthscale**5)
        periodinv = np.ones(X.shape[1])/(self.period)
        period2inv = np.ones(X.shape[1])/(self.period**2)
        period3inv = np.ones(X.shape[1])/(self.period**3)
        period4inv = np.ones(X.shape[1])/(self.period**4)
        dist = np.rollaxis(X[:, None, :] - X2[None, :, :],2,0)
        base = np.pi * dist * periodinv[:,None,None]
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscale[:,None,None] ), axis = 0 ) )
        k = self.variance*exp_dist
        dk2_dperioddx2 = self.dK2_dperioddX2(X, X2, dimX2)
        dk_dx2 = self._clean_dK_dX2(X, X2, dimX2)
        dk_dperiod = self.dK_dperiod(X, X2)
        if self.ARD1:
            tmp1 = -dk2_dperioddx2*np.pi/2.*lengthscale2inv[dimX]*periodinv[dimX]*np.sin(2.*base[dimX,:,:])
            tmp2 = np.zeros_like(tmp1)
            tmp3 = np.zeros_like(tmp1)
            tmp4 = np.zeros_like(tmp1)
            # i = j
            tmp2[dimX,:,:] = dk_dx2[:,:]*(np.pi*lengthscale2inv[dimX]*period2inv[dimX]*np.sin(2.*base[dimX,:,:])/2. + (np.pi**2)*lengthscale2inv[dimX]*period3inv[dimX]*np.cos(2.*base[dimX,:,:])*dist[dimX,:,:])
            if dimX == dimX2: # j = k
                tmp3 = dk_dperiod[:,:,:]*(np.pi**2)*period2inv[dimX]*lengthscale2inv[dimX]*np.cos(2.*base[dimX,:,:])
                # i = j = k
                tmp4[dimX,:,:] = -2.*k*(np.pi**2)*lengthscale2inv[dimX]*(period3inv[dimX]*np.cos(2.*base[dimX,:,:])-np.pi*period4inv[dimX]*np.sin(2.*base[dimX,:,:])*dist[dimX,:,:])
            return tmp1+tmp2+tmp3+tmp4
        else:
            tmp1 = np.pi*lengthscale2inv[dimX]/2.*(-dk2_dperioddx2*periodinv[dimX]*np.sin(2.*base[dimX,:,:]) + dk_dx2*period2inv[dimX]*np.sin(2.*base[dimX,:,:]) + np.pi*dk_dx2*2.*period3inv[dimX]*np.cos(2.*base[dimX,:,:])*dist[dimX,:,:] )
            tmp2 = np.zeros_like(tmp1)
            if dimX == dimX2:            
                tmp2 = (np.pi**2)*lengthscale2inv[dimX]*period2inv[dimX]*(dk_dperiod*np.cos(2.*base[dimX,:,:]) -2.*k*periodinv[dimX]*np.cos(2.*base[dimX,:,:]) +2.*k*np.sin(2.*base[dimX,:,:])*np.pi*period2inv[dimX]*dist[dimX,:,:] )
            return tmp1+tmp2

    def update_gradients_full(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None:
            X2 = X

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.period

        sin_base = np.sin( base )
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / self.lengthscale ), axis = -1 ) )

        dwl = self.variance * (1.0/np.square(self.lengthscale)) * sin_base*np.cos(base) * (base / self.period)

        dl = self.variance * np.square( sin_base) / np.power( self.lengthscale, 3)

        self.variance.gradient = np.sum(exp_dist * dL_dK)
        #target[0] += np.sum( exp_dist * dL_dK)

        if self.ARD1: # different periods
            self.period.gradient = (dwl * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else:  # same period
            self.period.gradient = np.sum(dwl.sum(-1) * exp_dist * dL_dK)

        if self.ARD2: # different lengthscales
            self.lengthscale.gradient = (dl * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else: # same lengthscales
            self.lengthscale.gradient = np.sum(dl.sum(-1) * exp_dist * dL_dK)

    def update_gradients_direct(self, dL_dVar, dL_dPer, dL_dLen):
        self.variance.gradient = dL_dVar
        self.period.gradient = dL_dPer
        self.lengthscale.gradient = dL_dLen

    def reset_gradients(self):
        self.variance.gradient = 0.
        if not self.ARD1:
            self.period.gradient = 0.
        else:
            self.period.gradient = np.zeros(self.input_dim)
        if not self.ARD2:
            self.lengthscale.gradient = 0.
        else:
            self.lengthscale.gradient = np.zeros(self.input_dim)

    def update_gradients_diag(self, dL_dKdiag, X):
        """derivative of the diagonal of the covariance matrix with respect to the parameters."""
        self.variance.gradient = np.sum(dL_dKdiag)
        self.period.gradient = 0
        self.lengthscale.gradient = 0

    def dgradients(self, X, X2):
        g1 = self.dK_dvariance(X, X2)
        g2 = self.dK_dperiod(X, X2)
        g3 = self.dK_dlengthscale(X, X2)
        return [g1, g2, g3]

    def dgradients_dX(self, X, X2, dimX):
        g1 = self.dK2_dvariancedX(X, X2, dimX)
        g2 = self.dK2_dperioddX(X, X2, dimX)
        g3 = self.dK2_dlengthscaledX(X, X2, dimX)
        return [g1, g2, g3]

    def dgradients_dX2(self, X, X2, dimX2):
        g1 = self.dK2_dvariancedX2(X, X2, dimX2)
        g2 = self.dK2_dperioddX2(X, X2, dimX2)
        g3 = self.dK2_dlengthscaledX2(X, X2, dimX2)
        return [g1, g2, g3]

    def dgradients2_dXdX2(self, X, X2, dimX, dimX2):
        g1 = self.dK3_dvariancedXdX2(X, X2, dimX, dimX2)
        g2 = self.dK3_dperioddXdX2(X, X2, dimX, dimX2)
        g3 = self.dK3_dlengthscaledXdX2(X, X2, dimX, dimX2)
        return [g1, g2, g3]

    def gradients_X(self, dL_dK, X, X2=None):
        K = self.K(X, X2)
        if X2 is None:
            dL_dK = dL_dK+dL_dK.T
            X2 = X
        dX = -np.pi*((dL_dK*K)[:,:,None]*np.sin(2*np.pi/self.period*(X[:,None,:] - X2[None,:,:]))/(2.*np.square(self.lengthscale)*self.period)).sum(1)
        return dX
    
    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)
    
    def input_sensitivity(self, summarize=True):
        return self.variance*np.ones(self.input_dim)/self.lengthscale**2