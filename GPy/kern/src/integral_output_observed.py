# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import math

class Integral_Output_Observed(Kern): #todo do I need to inherit from Stationary
    """
    Unlike the other integral kernel, this one returns predictions of integrals.
    
    Integral kernel, can include limits on each integral value. This kernel allows an n-dimensional
    histogram or binned data to be modelled. The outputs are the counts in each bin. The inputs
    are the start and end points of each bin: Pairs of inputs act as the limits on each bin. So
    inputs 4 and 5 provide the start and end values of each bin in the 3rd dimension.
    
    Unlike the other classes, here the kernel's predictions are the observed function.
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='integral'):
        super(Integral_Output_Observed, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)
            
        assert len(lengthscale)==input_dim/2            

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.

    def h(self, z):
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def dk_dl(self, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        return l * ( self.h((t-sprime)/l) - self.h((t - tprime)/l) + self.h((tprime-s)/l) - self.h((s-sprime)/l))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:  #we're finding dK_xx/dTheta
            dK_dl_term = np.zeros([X.shape[0],X.shape[0],self.lengthscale.shape[0]])
            k_term = np.zeros([X.shape[0],X.shape[0],self.lengthscale.shape[0]])
            dK_dl = np.zeros([X.shape[0],X.shape[0],self.lengthscale.shape[0]])
            dK_dv = np.zeros([X.shape[0],X.shape[0]])
            for il,l in enumerate(self.lengthscale):
                idx = il*2
                for i,x in enumerate(X):
                    for j,x2 in enumerate(X):
                        dK_dl_term[i,j,il] = self.dk_dl(x[idx],x2[idx],x[idx+1],x2[idx+1],l)
                        k_term[i,j,il] = self.k_xx(x[idx],x2[idx],x[idx+1],x2[idx+1],l)
            for il,l in enumerate(self.lengthscale):
                dK_dl = self.variances[0] * dK_dl_term[:,:,il]
                for jl, l in enumerate(self.lengthscale):
                    if jl!=il:
                        dK_dl *= k_term[:,:,jl]
                self.lengthscale.gradient[il] = np.sum(dK_dl * dL_dK)
            dK_dv = self.calc_K_xx_wo_variance(X) #the gradient wrt the variance is k_xx.
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

    def calc_K_xx_wo_variance(self,X):
        """Calculates K_xx without the variance term"""
        K_xx = np.ones([X.shape[0],X.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X):
                for il,l in enumerate(self.lengthscale):
                    idx = il*2 #each pair of input dimensions describe the limits on one actual dimension in the data
                    K_xx[i,j] *= self.k_xx(x[idx],x2[idx],x[idx+1],x2[idx+1],l)
        return K_xx

#    def K(self, X, X2=None):
#        if X2 is None: #X vs X
#            print X
#            K_xx = self.calc_K_xx_wo_variance(X)
#            return K_xx * self.variances[0]
#        else: #X vs X2
#            K_xf = np.ones([X.shape[0],X2.shape[0]])
#            for i,x in enumerate(X):
#                for j,x2 in enumerate(X2):
#                    for il,l in enumerate(self.lengthscale):
#                        idx = il*2
#                        K_xf[i,j] *= self.k_xf(x[idx],x2[idx],x[idx+1],l)
#            return K_xf * self.variances[0]
            
    def K(self, X, X2=None):
        if X2 is None: #X vs X
            K_xx = self.calc_K_xx_wo_variance(X)
            return K_xx * self.variances[0]
        else: #X vs X2
            K_xx = np.ones([X.shape[0],X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    for il,l in enumerate(self.lengthscale):
                        idx = il*2
                        K_xx[i,j] *= self.k_xx(x[idx],x2[idx],x[idx+1],x2[idx+1],l)
            return K_xx * self.variances[0]

    def Kdiag(self, X):
        """I've used the fact that we call this method for K_ff when finding the covariance as a hack so
        I know if I should return K_ff or K_xx. In this case we're returning K_ff!!
        $K_{ff}^{post} = K_{ff} - K_{fx} K_{xx}^{-1} K_{xf}$"""
        return np.diag(self.K(X))
        
        #K_ff = np.ones(X.shape[0])
        #for i,x in enumerate(X):
        #    for il,l in enumerate(self.lengthscale):
        #        idx = il*2
        #        K_ff[i] *= self.k_ff(x[idx],x[idx],l)
        #return K_ff * self.variances[0]
