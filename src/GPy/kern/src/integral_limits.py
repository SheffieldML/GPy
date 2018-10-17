# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import math
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp


class Integral_Limits(Kern): 
    """
    Integral kernel. This kernel allows 1d histogram or binned data to be modelled.
    The outputs are the counts in each bin. The inputs (on two dimensions) are the start and end points of each bin.
    The kernel's predictions are the latent function which might have generated those binned results.
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='integral'):
        """
        """
        super(Integral_Limits, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.

    def h(self, z):
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def dk_dl(self, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        return l * ( self.h((t-sprime)/l) - self.h((t - tprime)/l) + self.h((tprime-s)/l) - self.h((s-sprime)/l))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:  #we're finding dK_xx/dTheta
            dK_dl = np.zeros([X.shape[0],X.shape[0]])
            dK_dv = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    dK_dl[i,j] = self.variances[0]*self.dk_dl(x[0],x2[0],x[1],x2[1],self.lengthscale[0])
                    dK_dv[i,j] = self.k_xx(x[0],x2[0],x[1],x2[1],self.lengthscale[0])  #the gradient wrt the variance is k_xx.
            self.lengthscale.gradient = np.sum(dK_dl * dL_dK)
            self.variances.gradient = np.sum(dK_dv * dL_dK)
        else:     #we're finding dK_xf/Dtheta
            raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")

    #useful little function to help calculate the covariances.
    def g(self,z):
        return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def k_xx(self,t,tprime,s,sprime,l):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
        return 0.5 * (l**2) * ( self.g((t-sprime)/l) + self.g((tprime-s)/l) - self.g((t - tprime)/l) - self.g((s-sprime)/l))

    def k_ff(self,t,tprime,l):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        return np.exp(-((t-tprime)**2)/(l**2)) #rbf

    def k_xf(self,t,tprime,s,l):
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
        return 0.5 * np.sqrt(math.pi) * l * (math.erf((t-tprime)/l) + math.erf((tprime-s)/l))

    def K(self, X, X2=None):
        """Note: We have a latent function and an output function. We want to be able to find:
          - the covariance between values of the output function
          - the covariance between values of the latent function
          - the "cross covariance" between values of the output function and the latent function
        This method is used by GPy to either get the covariance between the outputs (K_xx) or
        is used to get the cross covariance (between the latent function and the outputs (K_xf).
        We take advantage of the places where this function is used:
         - if X2 is none, then we know that the items being compared (to get the covariance for)
         are going to be both from the OUTPUT FUNCTION.
         - if X2 is not none, then we know that the items being compared are from two different
         sets (the OUTPUT FUNCTION and the LATENT FUNCTION).
        
        If we want the covariance between values of the LATENT FUNCTION, we take advantage of
        the fact that we only need that when we do prediction, and this only calls Kdiag (not K).
        So the covariance between LATENT FUNCTIONS is available from Kdiag.        
        """
        if X2 is None:
            K_xx = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    K_xx[i,j] = self.k_xx(x[0],x2[0],x[1],x2[1],self.lengthscale[0])
            return K_xx * self.variances[0]
        else:
            K_xf = np.zeros([X.shape[0],X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    K_xf[i,j] = self.k_xf(x[0],x2[0],x[1],self.lengthscale[0]) #x2[1] unused, see k_xf docstring for explanation.
            return K_xf * self.variances[0]

    def Kdiag(self, X):
        """I've used the fact that we call this method during prediction (instead of K). When we
        do prediction we want to know the covariance between LATENT FUNCTIONS (K_ff) (as that's probably
        what the user wants).
        $K_{ff}^{post} = K_{ff} - K_{fx} K_{xx}^{-1} K_{xf}$"""
        K_ff = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            K_ff[i] = self.k_ff(x[0],x[0],self.lengthscale[0])
        return K_ff * self.variances[0]
