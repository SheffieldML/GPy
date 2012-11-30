# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib

class rbf(kernpart):
    """
    Radial Basis Function kernel, aka squared-exponential or Gaussian kernel.

    :param D: the number of input dimensions
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float

    .. Note: for rbf with different lengthscales on each dimension, see rbf_ARD
    """

    def __init__(self,D,variance=1.,lengthscale=1.):
        self.D = D
        self.Nparam = 2
        self.name = 'rbf'
        self.set_param(np.hstack((variance,lengthscale)))

        #initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3,1))
        self._X, self._X2, self._params = np.empty(shape=(3,1))

    def get_param(self):
        return np.hstack((self.variance,self.lengthscale))

    def set_param(self,x):
        self.variance, self.lengthscale = x
        self.lengthscale2 = np.square(self.lengthscale)
        #reset cached results
        self._X, self._X2, self._params = np.empty(shape=(3,1))
        self._Z, self._mu, self._S = np.empty(shape=(3,1)) # cached versions of Z,mu,S

    def get_param_names(self):
        return ['variance','lengthscale']

    def K(self,X,X2,target):
        self._K_computations(X,X2)
        np.add(self.variance*self._K_dvar, target,target)

    def Kdiag(self,X,target):
        np.add(target,self.variance,target)

    def dK_dtheta(self,partial,X,X2,target):
        self._K_computations(X,X2)
        target[0] += np.sum(self._K_dvar*partial)
        target[1] += np.sum(self._K_dvar*self.variance*self._K_dist2/self.lengthscale*partial)

    def dKdiag_dtheta(self,partial,X,target):
        #NB: derivative of diagonal elements wrt lengthscale is 0
        target[0] += np.sum(partial)

    def dK_dX(self,partial,X,X2,target):
        self._K_computations(X,X2)
        _K_dist = X[:,None,:]-X2[None,:,:]
        dK_dX = np.transpose(-self.variance*self._K_dvar[:,:,np.newaxis]*_K_dist/self.lengthscale2,(1,0,2))
        target += np.sum(dK_dX*partial.T[:,:,None],0)

    def dKdiag_dX(self,X,target):
        pass

    def _K_computations(self,X,X2):
        if not (np.all(X==self._X) and np.all(X2==self._X2)):
            self._X = X
            self._X2 = X2
            if X2 is None: X2 = X
            XXT = np.dot(X,X2.T)
            if X is X2:
                self._K_dist2 = (-2.*XXT + np.diag(XXT)[:,np.newaxis] + np.diag(XXT)[np.newaxis,:])/self.lengthscale2
            else:
                self._K_dist2 = (-2.*XXT + np.sum(np.square(X),1)[:,None] + np.sum(np.square(X2),1)[None,:])/self.lengthscale2
            self._K_exponent = -0.5*self._K_dist2
            self._K_dvar = np.exp(-0.5*self._K_dist2)

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
