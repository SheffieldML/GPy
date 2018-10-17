# Copyright (c) 2015, Thomas Hornung
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp

class Spline(Kern):
    """
    Linear spline kernel. You need to specify 2 parameters: the variance and c.
    The variance is defined in powers of 10. Thus specifying -2 means 10^-2.
    The parameter c allows to define the stiffness of the spline fit. A very stiff
    spline equals linear regression.
    See https://www.youtube.com/watch?v=50Vgw11qn0o starting at minute 1:17:28
    Lit: Wahba, 1990
    """

    def __init__(self, input_dim, variance=1., c=1., active_dims=None, name='spline'):
        super(Spline, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.c = Param('c', c)
        self.link_parameters(self.variance,self.c)


    def K(self, X, X2=None):
        if X2 is None: X2=X
        term1 = (X+8.)*(X2.T+8.)/16.
        term2 = abs((X-X2.T)/16.)**3
        term3 = ((X+8.)/16.)**3 + ((X2.T+8.)/16.)**3
        return (self.variance**2 * (1. + (1.+self.c) * term1 + self.c/3. * (term2 - term3)))

    def Kdiag(self, X):
        term1 = np.square(X+8.,X+8.)/16.
        term3 = 2. * ((X+8.)/16.)**3
        return (self.variance**2 * (1. + (1.+self.c) * term1 - self.c/3. * term3))[:,0]

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2=X
        term1 = (X+8.)*(X2.T+8.)/16.
        term2 = abs((X-X2.T)/16.)**3
        term3 = ((X+8.)/16.)**3 + ((X2.T+8.)/16.)**3
        self.variance.gradient = np.sum(dL_dK * (2*self.variance * (1. + (1.+self.c) * term1 + self.c/3. * ( term2 - term3))))
        self.c.gradient = np.sum(dL_dK * (self.variance**2* (term1 + 1./3.*(term2 - term3))))

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError
