# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np

class rational_quadratic(Kernpart):
    """
    rational quadratic kernel

    .. math::

       k(r) = \sigma^2 \\bigg( 1 + \\frac{r^2}{2 \ell^2} \\bigg)^{- \\alpha} \ \ \ \ \  \\text{ where  } r^2 = (x-y)^2

    :param input_dim: the number of input dimensions
    :type input_dim: int (input_dim=1 is the only value currently supported)
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param lengthscale: the lengthscale :math:`\ell`
    :type lengthscale: float
    :param power: the power :math:`\\alpha`
    :type power: float
    :rtype: Kernpart object

    """
    def __init__(self,input_dim,variance=1.,lengthscale=1.,power=1.):
        assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.input_dim = input_dim
        self.num_params = 3
        self.name = 'rat_quad'
        self.variance = variance
        self.lengthscale = lengthscale
        self.power = power

    def _get_params(self):
        return np.hstack((self.variance,self.lengthscale,self.power))

    def _set_params(self,x):
        self.variance = x[0]
        self.lengthscale = x[1]
        self.power = x[2]

    def _get_param_names(self):
        return ['variance','lengthscale','power']

    def K(self,X,X2,target):
        if X2 is None: X2 = X
        dist2 = np.square((X-X2.T)/self.lengthscale)
        target += self.variance*(1 + dist2/2.)**(-self.power)

    def Kdiag(self,X,target):
        target += self.variance

    def dK_dtheta(self,dL_dK,X,X2,target):
        if X2 is None: X2 = X
        dist2 = np.square((X-X2.T)/self.lengthscale)

        dvar = (1 + dist2/2.)**(-self.power)
        dl = self.power * self.variance * dist2 * self.lengthscale**(-3) * (1 + dist2/2./self.power)**(-self.power-1)
        dp = - self.variance * np.log(1 + dist2/2.) * (1 + dist2/2.)**(-self.power)

        target[0] += np.sum(dvar*dL_dK)
        target[1] += np.sum(dl*dL_dK)
        target[2] += np.sum(dp*dL_dK)

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        target[0] += np.sum(dL_dKdiag)
        # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged

    def dK_dX(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        dist2 = np.square((X-X2.T)/self.lengthscale)

        dX = -self.variance*self.power * (X-X2.T)/self.lengthscale**2 *  (1 + dist2/2./self.lengthscale)**(-self.power-1)
        target += np.sum(dL_dK*dX,1)[:,np.newaxis]

    def dKdiag_dX(self,dL_dKdiag,X,target):
        pass
