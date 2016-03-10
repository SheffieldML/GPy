# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np
from paramz.caching import Cache_this
four_over_tau = 2./np.pi

class MLP(Kern):
    """

    Multi layer perceptron kernel (also known as arc sine kernel or neural network kernel)

    .. math::

          k(x,y) = \\sigma^{2}\\frac{2}{\\pi }  \\text{asin} \\left ( \\frac{ \\sigma_w^2 x^\\top y+\\sigma_b^2}{\\sqrt{\\sigma_w^2x^\\top x + \\sigma_b^2 + 1}\\sqrt{\\sigma_w^2 y^\\top y \\sigma_b^2 +1}} \\right )


    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param weight_variance: the vector of the variances of the prior over input weights in the neural network :math:`\sigma^2_w`
    :type weight_variance: array or list of the appropriate size (or float if there is only one weight variance parameter)
    :param bias_variance: the variance of the prior over bias parameters :math:`\sigma^2_b`
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one weight variance parameter \sigma^2_w), otherwise there is one weight variance parameter per dimension.
    :type ARD: Boolean
    :rtype: Kernpart object


    """

    def __init__(self, input_dim, variance=1., weight_variance=1., bias_variance=1., ARD=False, active_dims=None, name='mlp'):
        super(MLP, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.ARD= ARD
        if ARD:
            wv = np.empty((input_dim,))
            wv[:] = weight_variance
            weight_variance = wv
        self.weight_variance = Param('weight_variance', weight_variance, Logexp())
        self.bias_variance = Param('bias_variance', bias_variance, Logexp())
        self.link_parameters(self.variance, self.weight_variance, self.bias_variance)


    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        if X2 is None:
            X_denom = np.sqrt(self._comp_prod(X)+1.)
            X2_denom = X_denom
            X2 = X
        else:
            X_denom = np.sqrt(self._comp_prod(X)+1.)
            X2_denom = np.sqrt(self._comp_prod(X2)+1.)
        XTX = self._comp_prod(X,X2)/X_denom[:,None]/X2_denom[None,:]
        return self.variance*four_over_tau*np.arcsin(XTX)

    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix for X."""
        X_prod = self._comp_prod(X)
        return self.variance*four_over_tau*np.arcsin(X_prod/(X_prod+1.))

    def update_gradients_full(self, dL_dK, X, X2=None):
        """Derivative of the covariance with respect to the parameters."""
        dvar, dw, db = self._comp_grads(dL_dK, X, X2)[:3]
        self.variance.gradient = dvar
        self.weight_variance.gradient = dw
        self.bias_variance.gradient = db

    def update_gradients_diag(self, dL_dKdiag, X):
        dvar, dw, db = self._comp_grads_diag(dL_dKdiag, X)[:3]
        self.variance.gradient = dvar
        self.weight_variance.gradient = dw
        self.bias_variance.gradient = db
        
    def gradients_X(self, dL_dK, X, X2):
        """Derivative of the covariance matrix with respect to X"""
        return self._comp_grads(dL_dK, X, X2)[3]

    def gradients_X_X2(self, dL_dK, X, X2):
        """Derivative of the covariance matrix with respect to X"""
        return self._comp_grads(dL_dK, X, X2)[3:]

    def gradients_X_diag(self, dL_dKdiag, X):
        """Gradient of diagonal of covariance with respect to X"""
        return self._comp_grads_diag(dL_dKdiag, X)[3]

    @Cache_this(limit=3, ignore_args=())
    def _comp_prod(self, X, X2=None):
        if X2 is None:
            return (np.square(X)*self.weight_variance).sum(axis=1)+self.bias_variance
        else:
            return (X*self.weight_variance).dot(X2.T)+self.bias_variance
    
    @Cache_this(limit=3, ignore_args=(1,))
    def _comp_grads(self, dL_dK, X, X2=None):
        var,w,b = self.variance, self.weight_variance, self.bias_variance
        K = self.K(X, X2)
        dvar = (dL_dK*K).sum()/var
        X_prod = self._comp_prod(X)
        X2_prod = self._comp_prod(X2) if X2 is not None else X_prod
        XTX = self._comp_prod(X,X2) if X2 is not None else self._comp_prod(X, X)
        common = var*four_over_tau/np.sqrt((X_prod[:,None]+1.)*(X2_prod[None,:]+1.)-np.square(XTX))*dL_dK
        if self.ARD:
            if X2 is not None:
                XX2 = X[:,None,:]*X2[None,:,:] if X2 is not None else X[:,None,:]*X[None,:,:]
                XX = np.square(X)
                X2X2 = np.square(X2)
                Q = self.weight_variance.shape[0]
                common_XTX = common*XTX
                dw =  np.dot(common.flat,XX2.reshape(-1,Q)) -( (common_XTX.sum(1)/(X_prod+1.)).T.dot(XX)+(common_XTX.sum(0)/(X2_prod+1.)).dot(X2X2))/2
            else:
                XX2 = X[:,None,:]*X[None,:,:]
                XX = np.square(X)
                Q = self.weight_variance.shape[0]
                common_XTX = common*XTX
                dw =  np.dot(common.flat,XX2.reshape(-1,Q)) - ((common_XTX.sum(0)+common_XTX.sum(1))/(X_prod+1.)).dot(XX)/2
        else:
            dw = (common*((XTX-b)/w-XTX*(((X_prod-b)/(w*(X_prod+1.)))[:,None]+((X2_prod-b)/(w*(X2_prod+1.)))[None,:])/2.)).sum()
        db = (common*(1.-XTX*(1./(X_prod[:,None]+1.)+1./(X2_prod[None,:]+1.))/2.)).sum()
        if X2 is None:
            common = common+common.T
            dX = common.dot(X)*w-((common*XTX).sum(axis=1)/(X_prod+1.))[:,None]*X*w
            dX2 = dX
        else:
            dX = common.dot(X2)*w-((common*XTX).sum(axis=1)/(X_prod+1.))[:,None]*X*w
            dX2 = common.T.dot(X)*w-((common*XTX).sum(axis=0)/(X2_prod+1.))[:,None]*X2*w
        return dvar, dw, db, dX, dX2
    
    @Cache_this(limit=3, ignore_args=(1,))
    def _comp_grads_diag(self, dL_dKdiag, X):
        var,w,b = self.variance, self.weight_variance, self.bias_variance
        K = self.Kdiag(X)
        dvar = (dL_dKdiag*K).sum()/var
        X_prod = self._comp_prod(X)
        common = var*four_over_tau/(np.sqrt(1-np.square(X_prod/(X_prod+1)))*np.square(X_prod+1))*dL_dKdiag
        if self.ARD:
            XX = np.square(X)
            dw = np.dot(common,XX)
        else:
            dw = (common*(X_prod-b)).sum()/w
        db = common.sum()
        dX = common[:,None]*X*w*2
        return dvar, dw, db, dX
