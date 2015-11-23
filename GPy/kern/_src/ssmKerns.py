# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
# Kurt Cutajar

from kern import Kern
from ... import util
import numpy as np
from scipy import integrate, weave
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp

class SsmKerns(Kern):
    """
    Kernels suitable for GP SSM regression with one-dimensional inputs

    """

    def __init__(self, input_dim, variance, lengthscale, active_dims, name, useGPU=False):
    	
        super(SsmKerns, self).__init__(input_dim, active_dims, name,useGPU=useGPU)
        self.lengthscale = np.abs(lengthscale)
        self.variance = Param('variance', variance, Logexp())

    def K_of_r(self, r):
        raise NotImplementedError("implement the covariance function as a fn of r to use this class")

    def dK_dr(self, r):
        raise NotImplementedError("implement derivative of the covariance function wrt r to use this class")

    def Q_of_r(self, r):
        raise NotImplementedError("implement covariance function of SDE wrt r to use this class")

    def Phi_of_r(self, r):
        raise NotImplementedError("implement transition function of SDE wrt r to use this class")

    def noise(self):
    	raise NotImplementedError("implement noise function of SDE to use this class")

   	def dPhidLam(self, r):
		raise NotImplementedError("implement derivative of the transition function wrt to lambda to use this class")

    def dQ(self, r):
    	raise NotImplementedError("implement detivatives of Q wrt to lambda and variance to use this class")


class Matern32_SSM(SsmKerns):
    """
    Matern 3/2 kernel:

    .. math::

       k(r) = \\sigma^2 (1 + \\sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \\text{ where  } r = \sqrt{\sum_{i=1}^input_dim \\frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def __init__(self, input_dim, variance=1., lengthscale=1, active_dims=None, name='Mat32'):
        super(Matern32_SSM, self).__init__(input_dim, variance, lengthscale, active_dims, name)
        lambd = np.sqrt(2*1.5)/lengthscale # additional paramter of model (derived from lengthscale)
        self.lam = Param('lambda', lambd, Logexp())
        self.link_parameters(self.lam, self.variance)
        self.order = 2

    def noise(self):
        """
        Compute noise for the kernel
        """
        p = 1
        lp_fact = np.sum(np.log(range(1,p+1)))
        l2p_fact = np.sum(np.log(range(1,2*p +1)))
        logq = np.log(2*self.variance) + p*np.log(4) + 2*lp_fact + (2*p + 1)*np.log(self.lam) - l2p_fact
        q = np.exp(logq)
        return q

    def K_of_r(self, r):
        lengthscale = np.sqrt(3.) / self.lam
        r = r / lengthscale
        return self.variance * (1. + np.sqrt(3.) * r) * np.exp(-np.sqrt(3.) * r)

    def dK_dr(self,r):
        return -3.*self.variance*r*np.exp(-np.sqrt(3.)*r)

    def Q_of_r(self, r):
        """
        Compute process variance (Q)
        """
        q = self.noise()
        Q = np.zeros((2,2))
        Q[0][0] = 1/(4*self.lam**3) - (4*(r**2)*(self.lam**2) 
                            + 4*r*self.lam + 2)/(8*(self.lam**3)*np.exp(2*r*self.lam))
        Q[0][1] = (r**2)/(2*np.exp(2*r*self.lam))
        Q[1][0] = Q[0][1]
        Q[1][1] = 1/(4*self.lam) - (2*(r**2)*(self.lam**2) 
                            - 2*r*self.lam + 1)/(4*self.lam*np.exp(2*r*self.lam))
        return q*Q

    def Phi_of_r(self, r):
        """
        Compute transition function (Phi)
        """
        if r < 0:
            phi = np.zeros((2,2))
            phi[0][0] = self.variance
            phi[0][1] = 0
            phi[1][0] = 0
            phi[1][1] = np.power(self.lam,2)*self.variance
            return phi
        else:
            mult = np.exp(-self.lam*r)
            phi = np.zeros((2, 2))
            phi[0][0] = (1 + self.lam*r)*mult
            phi[0][1] = mult*r
            phi[1][0] = -r*mult*self.lam**2
            phi[1][1] = mult*(1 - self.lam*r)
            return phi

    def dPhidLam(self, r):
        """
        Compute derivative of transition function (Phi) wrt lambda
        """
    	mult = np.exp(self.lam*r)
    	dPhi = np.zeros((2, 2))
    	dPhi[0][0] = (r / mult) - (r * (self.lam*r + 1))/mult
    	dPhi[0][1] = (-1 * (r**2)) / mult
    	dPhi[1][0] = (self.lam**2 * r**2) / mult - (2*self.lam*r)/mult
    	dPhi[1][1] = (r * (self.lam*r - 1))/mult - r/mult
    	return dPhi

    def dQ(self, r):
        """
        Compute derivatives of Q with respect to lambda and variance
        """
        if (r == -1):
            dQdLam = np.array([[0, 0], [0, 2*self.lam*self.variance]])
            dQdVar = None
        else:
            q = self.noise()
            Q = self.Q_of_r(r)
            dQdLam = np.zeros((2, 2))
            mult = np.exp(2*self.lam*r)
            dQdLam[0][0] = (3*(4*(r**2)*(self.lam**2) + 4*r*self.lam + 2))/(8*(self.lam**4)*mult) - 3/(4*(self.lam**4)) - (8*self.lam*(r**2) + 4*r)/(8*(self.lam**3)*mult) + (r*(4*(r**2)*(self.lam**2) + 4*r*self.lam + 2))/(4*(self.lam**3)*mult)
            dQdLam[0][1] = -(r**3)/mult
            dQdLam[1][0] = dQdLam[0][1]
            dQdLam[1][1] = (2*(r**2)*(self.lam**2) - 2*r*self.lam + 1)/(4*(self.lam**2)*mult) - 1/(4*(self.lam**2)) + (2*r - 4*(r**2)*self.lam)/(4*self.lam*mult) + (r*(2*(r**2)*(self.lam**2) - 2*r*self.lam + 1))/(2*self.lam*mult)
            dq = (q*(2+1))/self.lam
            dQdLam = q*dQdLam + dq*(Q/q)
            dQdVar = Q / self.variance
    	return [dQdLam, dQdVar]