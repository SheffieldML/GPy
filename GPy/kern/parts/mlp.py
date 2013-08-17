# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
four_over_tau = 2./np.pi

class MLP(Kernpart):
    """
    multi layer perceptron kernel (also known as arc sine kernel or neural network kernel)

    .. math::

       k(x,y) = \sigma^2 \frac{2}{\pi}  \text{asin} \left(\frac{\sigma_w^2 x^\top y+\sigma_b^2}{\sqrt{\sigma_w^2x^\top x + \sigma_b^2 + 1}\sqrt{\sigma_w^2 y^\top y \sigma_b^2 +1}} \right)

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

    def __init__(self, input_dim, variance=1., weight_variance=None, bias_variance=100., ARD=False):
        ARD = False
        self.input_dim = input_dim
        self.ARD = ARD
        if not ARD:
            self.num_params=3
            if weight_variance is not None:
                weight_variance = np.asarray(weight_variance)
                assert weight_variance.size == 1, "Only one weight variance needed for non-ARD kernel"
            else:
                weight_variance = 100.*np.ones(1)
        else:
            self.num_params = self.input_dim + 2
            if weight_variance is not None:
                weight_variance = np.asarray(weight_variance)
                assert weight_variance.size == self.input_dim, "bad number of weight variances"
            else:
                weight_variance = np.ones(self.input_dim)

        self.name='mlp'
        self._set_params(np.hstack((variance, weight_variance.flatten(), bias_variance)))

    def _get_params(self):
        return np.hstack((self.variance, self.weight_variance.flatten(), self.bias_variance))

    def _set_params(self, x):
        assert x.size == (self.num_params)
        self.variance = x[0]
        self.weight_variance = x[1:-1]
        self.weight_std = np.sqrt(self.weight_variance)
        self.bias_variance = x[-1]

    def _get_param_names(self):
        if self.num_params == 3:
            return ['variance', 'weight_variance', 'bias_variance']
        else:
            return ['variance'] + ['weight_variance_%i' % i for i in range(self.lengthscale.size)] + ['bias_variance']

    def K(self, X, X2, target):
        """Return covariance between X and X2."""
        self._K_computations(X, X2)
        target += self.variance*self._K_dvar

    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix for X."""
        self._K_diag_computations(X)
        target+= self.variance*self._K_diag_dvar

    def dK_dtheta(self, dL_dK, X, X2, target):
        """Derivative of the covariance with respect to the parameters."""
        self._K_computations(X, X2)
        denom3 = self._K_denom*self._K_denom*self._K_denom
        base = four_over_tau*self.variance/np.sqrt(1-self._K_asin_arg*self._K_asin_arg)
        base_cov_grad = base*dL_dK

        if X2 is None:
            vec = np.diag(self._K_inner_prod)
            target[1] += ((self._K_inner_prod/self._K_denom 
                           -.5*self._K_numer/denom3
                           *(np.outer((self.weight_std*vec+self.bias_variance+1.), self.weight_std*vec) 
                             +np.outer(self.weight_std*vec,(self.weight_std*vec+self.bias_variance+1.))))*base_cov_grad).sum()
            target[2] += ((1./self._K_denom 
                           -.5*self._K_numer/denom3 
                           *((vec[None, :]+vec[:, None])*self.weight_variance
                           +2.*self.bias_variance + 2.))*base_cov_grad).sum()
        else:
            vec1 = (X*X).sum(1)
            vec2 = (X2*X2).sum(1)
            target[1] += ((self._K_inner_prod/self._K_denom 
                           -.5*self._K_numer/denom3
                           *(np.outer((self.weight_variance*vec1+self.bias_variance+1.), vec2) + np.outer(vec1, self.weight_variance*vec2 + self.bias_variance+1.)))*base_cov_grad).sum()
            target[2] += ((1./self._K_denom 
                           -.5*self._K_numer/denom3 
                           *((vec1[:, None]+vec2[None, :])*self.weight_variance
                             + 2*self.bias_variance + 2.))*base_cov_grad).sum()
            
        target[0] += np.sum(self._K_dvar*dL_dK)


    def dK_dX(self, dL_dK, X, X2, target):
        """Derivative of the covariance matrix with respect to X"""
        self._K_computations(X, X2)
        gX = np.zeros((X2.shape[0], X.shape[1], X.shape[0]))
        
        for i in range(X.shape[0]):
            gX[:, :, i] = self._dK_dX_point(X, X2, i)
            
            
    def _dK_dX_point(self, X, X2, i):
        """Gradient with respect to one point of X"""
        
        inner_prod = self._K_inner_prod[i, :].T
        numer = self._K_numer[i, :].T
        denom = self._K_denom[i, :].T
        arg = self._K_asin_arg[i, :].T
        vec1 = (X[i, :]*X[i, :]).sum()*self.weight_variance + self.bias_variance + 1.
        vec2 = (X2*X2).sum(1)*self.weight_variance + self.bias_variance + 1.
        #denom = np.sqrt(np.outer(vec2,vec1))
        #arg = numer/denom
        gX = np.zeros(X2.shape)
        denom3 = denom*denom*denom
        for j in range(X2.shape[1]):
            gX[:, j]=X2[:, j]/denom - vec2*X[i, j]*numer/denom3
            gX[:, j] = four_over_tau*self.weight_variance*self.variance*gX[:, j]/np.sqrt(1-arg*arg)

    
    
    def _K_computations(self, X, X2):
        if self.ARD:
            pass
        else:
            if X2 is None:
                self._K_inner_prod = np.dot(X,X.T)
                self._K_numer = self._K_inner_prod*self.weight_variance+self.bias_variance
                vec = np.diag(self._K_numer) + 1.
                self._K_denom = np.sqrt(np.outer(vec,vec))
                self._K_asin_arg = self._K_numer/self._K_denom
                self._K_dvar = four_over_tau*np.arcsin(self._K_asin_arg)
            else:
                self._K_inner_prod = np.dot(X,X2.T)
                self._K_numer = self._K_inner_prod*self.weight_variance + self.bias_variance
                vec1 = (X*X).sum(1)*self.weight_variance + self.bias_variance + 1.
                vec2 = (X2*X2).sum(1)*self.weight_variance + self.bias_variance + 1.
                self._K_denom = np.sqrt(np.outer(vec1,vec2))
                self._K_asin_arg = self._K_numer/self._K_denom
                self._K_dvar = four_over_tau*np.arcsin(self._K_asin_arg)

    def _K_diag_computations(self, X):
        if self.ARD:
            pass
        else:
            self._K_diag_numer = (X*X).sum(1)*self.weight_variance + self.bias_variance
            self._K_diag_denom = self._K_diag_numer+1.
            self._K_diag_dvar = four_over_tau*np.arcsin(self._K_diag_numer/self._K_diag_denom)

  


