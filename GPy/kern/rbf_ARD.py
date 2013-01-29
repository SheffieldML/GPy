# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib

class rbf_ARD(kernpart):
    def __init__(self,D,variance=1.,lengthscales=None):
        """
        Arguments
        ----------
        D: int - the number of input dimensions
        variance: float
        lengthscales : np.ndarray of shape (D,)
        """
        self.D = D
        if lengthscales is not None:
            assert lengthscales.shape==(self.D,)
        else:
            lengthscales = np.ones(self.D)
        self.Nparam = self.D + 1
        self.name = 'rbf_ARD'
        self.set_param(np.hstack((variance,lengthscales)))

        #initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3,1))
        self._X, self._X2, self._params = np.empty(shape=(3,1))

    def get_param(self):
        return np.hstack((self.variance,self.lengthscales))

    def set_param(self,x):
        assert x.size==(self.D+1)
        self.variance = x[0]
        self.lengthscales = x[1:]
        self.lengthscales2 = np.square(self.lengthscales)
        #reset cached results
        self._Z, self._mu, self._S = np.empty(shape=(3,1)) # cached versions of Z,mu,S

    def get_param_names(self):
        if self.D==1:
            return ['variance','lengthscale']
        else:
            return ['variance']+['lengthscale_%i'%i for i in range(self.lengthscales.size)]

    def K(self,X,X2,target):
        self._K_computations(X,X2)
        np.add(self.variance*self._K_dvar, target,target)

    def Kdiag(self,X,target):
        np.add(target,self.variance,target)

    def dK_dtheta(self,partial,X,X2,target):
        self._K_computations(X,X2)
        dl = self._K_dvar[:,:,None]*self.variance*self._K_dist2/self.lengthscales
        target[0] += np.sum(self._K_dvar*partial)
        target[1:] += (dl*partial[:,:,None]).sum(0).sum(0)

    def dKdiag_dtheta(self,X,target):
        target[0] += np.sum(partial)

    def dK_dX(self,partial,X,X2,target):
        self._K_computations(X,X2)
        dZ = self.variance*self._K_dvar[:,:,None]*self._K_dist/self.lengthscales2
        dK_dX = -dZ.transpose(1,0,2)
        target += np.sum(dK_dX*partial.T[:,:,None],0)

    def dKdiag_dX(self,partial,X,target):
        pass

    def psi0(self,Z,mu,S,target):
        target += self.variance

    def dpsi0_dtheta(self,partial,Z,mu,S,target):
        target[0] += np.sum(partial)

    def dpsi0_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        pass

    def psi1(self,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        np.add(target, self._psi1,target)

    def dpsi1_dtheta(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        denom_deriv = S[:,None,:]/(self.lengthscales**3+self.lengthscales*S[:,None,:])
        d_length = self._psi1[:,:,None]*(self.lengthscales*np.square(self._psi1_dist/(self.lengthscales2+S[:,None,:])) + denom_deriv)
        target[0] += np.sum(partial*self._psi1/self.variance)
        target[1:] += (d_length*partial[:,:,None]).sum(0).sum(0)

    def dpsi1_dZ(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        # np.add(target,-self._psi1[:,:,None]*self._psi1_dist/self.lengthscales2/self._psi1_denom,target)
        denominator = (self.lengthscales2*(self._psi1_denom))
        dpsi1_dZ = - self._psi1[:,:,None] * ((self._psi1_dist/denominator))
        target += np.sum(partial.T[:,:,None] * dpsi1_dZ, 0)

    def dpsi1_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """return shapes are N,M,Q"""
        self._psi_computations(Z,mu,S)
        tmp = self._psi1[:,:,None]/self.lengthscales2/self._psi1_denom
        target_mu += np.sum(partial.T[:, :, None]*tmp*self._psi1_dist,1)
        target_S += np.sum(partial.T[:, :, None]*0.5*tmp*(self._psi1_dist_sq-1),1)

    def psi2(self,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        target += self._psi2.sum(0) #TODO: psi2 should be NxMxM (for het. noise)

    def dpsi2_dtheta(self,partial,Z,mu,S,target):
        """Shape N,M,M,Ntheta"""
        self._psi_computations(Z,mu,S)
        d_var = np.sum(2.*self._psi2/self.variance,0)
        d_length = self._psi2[:,:,:,None]*(0.5*self._psi2_Zdist_sq*self._psi2_denom + 2.*self._psi2_mudist_sq + 2.*S[:,None,None,:]/self.lengthscales2)/(self.lengthscales*self._psi2_denom)
        d_length = d_length.sum(0)
        target[0] += np.sum(partial*d_var)
        target[1:] += (d_length*partial[:,:,None]).sum(0).sum(0)

    def dpsi2_dZ(self,partial,Z,mu,S,target):
        """Returns shape N,M,M,Q"""
        self._psi_computations(Z,mu,S)
        term1 = 0.5*self._psi2_Zdist/self.lengthscales2 # M, M, Q
        term2 = self._psi2_mudist/self._psi2_denom/self.lengthscales2 # N, M, M, Q
        dZ = self._psi2[:,:,:,None] * (term1[None] + term2) 
        target += (partial[None,:,:,None]*dZ).sum(0).sum(0)

    def dpsi2_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """Think N,M,M,Q """
        self._psi_computations(Z,mu,S)
        tmp = self._psi2[:,:,:,None]/self.lengthscales2/self._psi2_denom
        target_mu += (partial[None,:,:,None]*-tmp*2.*self._psi2_mudist).sum(1).sum(1)
        target_S += (partial[None,:,:,None]*tmp*(2.*self._psi2_mudist_sq-1)).sum(1).sum(1)

    def _K_computations(self,X,X2):
        if not (np.all(X==self._X) and np.all(X2==self._X2)):
            self._X = X
            self._X2 = X2
            if X2 is None: X2 = X
            self._K_dist = X[:,None,:]-X2[None,:,:] # this can be computationally heavy
            self._params = np.empty(shape=(1,0))#ensure the next section gets called
        if not np.all(self._params == self.get_param()):
            self._params == self.get_param()
            self._K_dist2 = np.square(self._K_dist/self.lengthscales)
            self._K_exponent = -0.5*self._K_dist2.sum(-1)
            self._K_dvar = np.exp(-0.5*self._K_dist2.sum(-1))

    def _psi_computations(self,Z,mu,S):
        #here are the "statistics" for psi1 and psi2
        if not np.all(Z==self._Z):
            #Z has changed, compute Z specific stuff
            self._psi2_Zhat = 0.5*(Z[:,None,:] +Z[None,:,:]) # M,M,Q
            self._psi2_Zdist = Z[:,None,:]-Z[None,:,:] # M,M,Q
            self._psi2_Zdist_sq = np.square(self._psi2_Zdist)/self.lengthscales2 # M,M,Q
            self._Z = Z

        if not (np.all(Z==self._Z) and np.all(mu==self._mu) and np.all(S==self._S)):
            #something's changed. recompute EVERYTHING

            #psi1
            self._psi1_denom = S[:,None,:]/self.lengthscales2 + 1.
            self._psi1_dist = Z[None,:,:]-mu[:,None,:]
            self._psi1_dist_sq = np.square(self._psi1_dist)/self.lengthscales2/self._psi1_denom
            self._psi1_exponent = -0.5*np.sum(self._psi1_dist_sq+np.log(self._psi1_denom),-1)
            self._psi1 = self.variance*np.exp(self._psi1_exponent)

            #psi2
            self._psi2_denom = 2.*S[:,None,None,:]/self.lengthscales2+1. # N,M,M,Q
            self._psi2_mudist = mu[:,None,None,:]-self._psi2_Zhat #N,M,M,Q
            self._psi2_mudist_sq = np.square(self._psi2_mudist)/(self.lengthscales2*self._psi2_denom)
            self._psi2_exponent = np.sum(-self._psi2_Zdist_sq/4. -self._psi2_mudist_sq -0.5*np.log(self._psi2_denom),-1) #N,M,M
            self._psi2 = np.square(self.variance)*np.exp(self._psi2_exponent) # N,M,M

            self._Z, self._mu, self._S = Z, mu,S


if __name__=='__main__':
    #run some simple tests on the kernel (TODO:move these to unititest)
    #TODO: these are broken in this new structure!
    N = 10
    M = 5
    Q = 3

    Z = np.random.randn(M,Q)
    mu = np.random.randn(N,Q)
    S = np.random.rand(N,Q)

    var = 2.5
    lengthscales = np.ones(Q)*0.7

    k = rbf(Q,var,lengthscales)

    from checkgrad import checkgrad

    def k_theta_test(param,k):
        k.set_param(param)
        K = k.K(Z)
        dK_dtheta = k.dK_dtheta(Z)
        f = np.sum(K)
        df = dK_dtheta.sum(0).sum(0)
        return f,np.array(df)
    print "dk_dtheta_test"
    checkgrad(k_theta_test,np.random.randn(1+Q),args=(k,))


    def psi1_mu_test(mu,k):
        mu = mu.reshape(N,Q)
        f = np.sum(k.psi1(Z,mu,S))
        df = k.dpsi1_dmuS(Z,mu,S)[0].sum(1)
        return f,df.flatten()
    print "psi1_mu_test"
    checkgrad(psi1_mu_test,np.random.randn(N*Q),args=(k,))

    def psi1_S_test(S,k):
        S = S.reshape(N,Q)
        f = np.sum(k.psi1(Z,mu,S))
        df = k.dpsi1_dmuS(Z,mu,S)[1].sum(1)
        return f,df.flatten()
    print "psi1_S_test"
    checkgrad(psi1_S_test,np.random.rand(N*Q),args=(k,))

    def psi1_theta_test(theta,k):
        k.set_param(theta)
        f = np.sum(k.psi1(Z,mu,S))
        df = np.array([np.sum(grad) for grad in k.dpsi1_dtheta(Z,mu,S)])
        return f,df
    print "psi1_theta_test"
    checkgrad(psi1_theta_test,np.random.rand(1+Q),args=(k,))


    def psi2_mu_test(mu,k):
        mu = mu.reshape(N,Q)
        f = np.sum(k.psi2(Z,mu,S))
        df = k.dpsi2_dmuS(Z,mu,S)[0].sum(1).sum(1)
        return f,df.flatten()
    print "psi2_mu_test"
    checkgrad(psi2_mu_test,np.random.randn(N*Q),args=(k,))

    def psi2_S_test(S,k):
        S = S.reshape(N,Q)
        f = np.sum(k.psi2(Z,mu,S))
        df = k.dpsi2_dmuS(Z,mu,S)[1].sum(1).sum(1)
        return f,df.flatten()
    print "psi2_S_test"
    checkgrad(psi2_S_test,np.random.rand(N*Q),args=(k,))

    def psi2_theta_test(theta,k):
        k.set_param(theta)
        f = np.sum(k.psi2(Z,mu,S))
        df = np.array([np.sum(grad) for grad in k.dpsi2_dtheta(Z,mu,S)])
        return f,df
    print "psi2_theta_test"
    checkgrad(psi2_theta_test,np.random.rand(1+Q),args=(k,))
