# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import math
from rbf import RBF

class ShapeIntegral(Kern):
    """
    
    """

    def __init__(self, input_dim, active_dims=None, kernel=None, name='shapeintegral',Nperunit=100):
        super(ShapeIntegral, self).__init__(input_dim, active_dims, name)
        
        if kernel is None:
            kernel = RBF(2)
        self.kernel = kernel
        self.Nperunit = Nperunit
        
    def getarea(self,a,b,c):
        B = b-a
        C = c-a
        area = np.abs(np.cross(B,C)/2.0)
        return area

    def placetrianglepoints(self,a,b,c,Nperunit=400):
        x = b-a
        y = c-a
        N = Nperunit * self.getarea(a,b,c)
        u = np.random.rand(N,1)**0.5
        v = np.random.rand(N,1)*u
        return a+(x*(1-u) + y*v)


    def placepoints(self,shape,Nperunit=100):
        a = np.mean(shape,0)
        allps = []
        for b,c in zip(np.r_[shape[0:-1,:],shape[-1:,:]],np.r_[shape[1:,:],shape[0:1,:]]):
            allps.extend(self.placetrianglepoints(a,b,c,Nperunit))
        return np.array(allps)
        
    def calc_K_xx_wo_variance(self,X):
        """Calculates K_xx without the variance term"""

        ps = []
        qs = []
        for i in range(0,X.shape[0],2):
            s = X[i:i+2,:]
            s = s[:,~np.isnan(s[0,:])]
            ps.append(self.placepoints(s.T,self.Nperunit))
            qs.append(self.placepoints(s.T,self.Nperunit))

        K_xx = np.ones([len(ps),len(qs)])
        
        for i,p in enumerate(ps):
            for j,q in enumerate(qs): 
                cov = self.kernel.K(p,q)
                v = np.sum(cov)/(self.Nperunit**2)
                K_xx[i,j] = v
        return K_xx, ps, qs 

    def K(self, X, X2=None):
        if X2 is None: #X vs X
            K_xx = self.calc_K_xx_wo_variance(X)
            return K_xx
        else: #X vs X2
            pass #TODO

    def Kdiag(self, X):
        """I've used the fact that we call this method for K_ff when finding the covariance as a hack so
        I know if I should return K_ff or K_xx. In this case we're returning K_ff!!
        $K_{ff}^{post} = K_{ff} - K_{fx} K_{xx}^{-1} K_{xf}$"""
        pass #TODO
