
from stationary import Matern32  #from kern import Matern32
import numpy as np
from GPy.util.linalg import tdot
from GPy import util


class Matern32grad(Matern32):
    """
    Matern 3/2 kernel:

    .. math::

       k(r) = \\sigma^2 (1 + \\sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat32'):
        #super(Matern32, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
        super(Matern32grad, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def dK_dtheta(self, X):
        """
        Compute the Euclidean distance between each pair of rows of X.
        """
        Xsq = np.sum(np.square(X),1)
        r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
        util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
        r2 = np.clip(r2, 0, np.inf)
        dist = np.sqrt(r2)
        le2 = self.lengthscale**2
        dkdl = self.variance * ( np.sqrt(3)*dist*(-1)/le2*np.exp(-np.sqrt(3*r2/le2)) \
             +(1.+np.sqrt(3.*r2/le2)) *np.exp(-np.sqrt(3.*r2/le2) )*np.sqrt(3*r2)/le2 )

        return dkdl #self.variance * (1. + np.sqrt(3.) * r) * np.exp(-np.sqrt(3.) * r)

## self.kern = kern.Matern32(1,lengthscale=1)     self.kern.dkdl(X)

