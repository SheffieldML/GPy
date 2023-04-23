# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .kern import Kern
from ...util.linalg import mdot
from ...util.decorators import silence_errors
from ...core.parameterization.param import Param
from paramz.transformations import Logexp

class Periodic(Kern):
    def __init__(self, input_dim, variance, lengthscale, period, n_freq, lower, upper, active_dims, name):
        """
        :type input_dim: int
        :param variance: the variance of the Matern kernel
        :type variance: float
        :param lengthscale: the lengthscale of the Matern kernel
        :type lengthscale: np.ndarray of size (input_dim,)
        :param period: the period
        :type period: float
        :param n_freq: the number of frequencies considered for the periodic subspace
        :type n_freq: int
        :rtype: kernel object
        """

        assert input_dim==1, "Periodic kernels are only defined for input_dim=1"
        super(Periodic, self).__init__(input_dim, active_dims, name)
        self.input_dim = input_dim
        self.lower,self.upper = lower, upper
        self.n_freq = n_freq
        self.n_basis = 2*n_freq
        self.variance = Param('variance', float(variance), Logexp())
        self.lengthscale = Param('lengthscale', float(lengthscale), Logexp())
        self.period = Param('period', float(period), Logexp())
        self.link_parameters(self.variance, self.lengthscale, self.period)

    def _cos(self, alpha, omega, phase):
        def f(x):
            return alpha*np.cos(omega*x + phase)
        return f

    @silence_errors
    def _cos_factorization(self, alpha, omega, phase):
        r1 = np.sum(alpha*np.cos(phase),axis=1)[:,None]
        r2 = np.sum(alpha*np.sin(phase),axis=1)[:,None]
        r =  np.sqrt(r1**2 + r2**2)
        psi = np.where(r1 != 0, (np.arctan(r2/r1) + (r1<0.)*np.pi),np.arcsin(r2))
        return r,omega[:,0:1], psi

    @silence_errors
    def _int_computation(self,r1,omega1,phi1,r2,omega2,phi2):
        Gint1 = 1./(omega1+omega2.T)*( np.sin((omega1+omega2.T)*self.upper+phi1+phi2.T) - np.sin((omega1+omega2.T)*self.lower+phi1+phi2.T)) + 1./(omega1-omega2.T)*( np.sin((omega1-omega2.T)*self.upper+phi1-phi2.T) - np.sin((omega1-omega2.T)*self.lower+phi1-phi2.T) )
        Gint2 = 1./(omega1+omega2.T)*( np.sin((omega1+omega2.T)*self.upper+phi1+phi2.T) - np.sin((omega1+omega2.T)*self.lower+phi1+phi2.T)) +  np.cos(phi1-phi2.T)*(self.upper-self.lower)
        Gint = np.dot(r1,r2.T)/2 * np.where(np.isnan(Gint1),Gint2,Gint1)
        return Gint

    def K(self, X, X2=None):
        FX = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)
        if X2 is None:
            FX2 = FX
        else:
            FX2 = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X2)
        return mdot(FX,self.Gi,FX2.T)

    def Kdiag(self,X):
        return np.diag(self.K(X))


class PeriodicExponential(Periodic):
    """
    Kernel of the periodic subspace (up to a given frequency) of a exponential
    (Matern 1/2) RKHS.

    Only defined for input_dim=1.
    """

    def __init__(self, input_dim=1, variance=1., lengthscale=1., period=2.*np.pi, n_freq=10, lower=0., upper=4*np.pi, active_dims=None, name='periodic_exponential'):
        super(PeriodicExponential, self).__init__(input_dim, variance, lengthscale, period, n_freq, lower, upper, active_dims, name)

    def parameters_changed(self):
        self.a = [1./self.lengthscale, 1.]
        self.b = [1]

        self.basis_alpha = np.ones((self.n_basis,))
        self.basis_omega = (2*np.pi*np.arange(1,self.n_freq+1)/self.period).repeat(2)
        self.basis_phi =   np.zeros(self.n_freq * 2)
        self.basis_phi[::2] = -np.pi/2

        self.G = self.Gram_matrix()
        self.Gi = np.linalg.inv(self.G)

    def Gram_matrix(self):
        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)),self.a[1]*self.basis_omega))
        Lo = np.column_stack((self.basis_omega,self.basis_omega))
        Lp = np.column_stack((self.basis_phi,self.basis_phi+np.pi/2))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)
        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]
        return(self.lengthscale/(2*self.variance) * Gint + 1./self.variance*np.dot(Flower,Flower.T))

    @silence_errors
    def update_gradients_full(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to the parameters (shape is N x num_inducing x num_params)"""
        if X2 is None: X2 = X
        FX  = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)
        FX2 = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X2)

        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)),self.a[1]*self.basis_omega))
        Lo = np.column_stack((self.basis_omega,self.basis_omega))
        Lp = np.column_stack((self.basis_phi,self.basis_phi+np.pi/2))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)

        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]

        #dK_dvar
        dK_dvar = 1./self.variance*mdot(FX,self.Gi,FX2.T)

        #dK_dlen
        da_dlen = [-1./self.lengthscale**2,0.]
        dLa_dlen =  np.column_stack((da_dlen[0]*np.ones((self.n_basis,1)),da_dlen[1]*self.basis_omega))
        r1,omega1,phi1 = self._cos_factorization(dLa_dlen,Lo,Lp)
        dGint_dlen = self._int_computation(r1,omega1,phi1, r,omega,phi)
        dGint_dlen = dGint_dlen + dGint_dlen.T
        dG_dlen = 1./2*Gint + self.lengthscale/2*dGint_dlen
        dK_dlen = -mdot(FX,self.Gi,dG_dlen/self.variance,self.Gi,FX2.T)

        #dK_dper
        dFX_dper  = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X ,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X)
        dFX2_dper = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X2,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X2)

        dLa_dper = np.column_stack((-self.a[0]*self.basis_omega/self.period, -self.a[1]*self.basis_omega**2/self.period))
        dLp_dper = np.column_stack((self.basis_phi+np.pi/2,self.basis_phi+np.pi))
        r1,omega1,phi1 =  self._cos_factorization(dLa_dper,Lo,dLp_dper)

        IPPprim1 =  self.upper*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi/2))
        IPPprim1 -= self.lower*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi/2))
        # SIMPLIFY!!!       IPPprim1 = (self.upper - self.lower)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi/2))
        IPPprim2 =  self.upper*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  + self.upper*np.cos(phi-phi1.T))
        IPPprim2 -= self.lower*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi/2)  + self.lower*np.cos(phi-phi1.T))
        IPPprim = np.where(np.logical_or(np.isnan(IPPprim1), np.isinf(IPPprim1)), IPPprim2, IPPprim1)


        IPPint1 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi)
        IPPint1 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi)
        IPPint2 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  + 1./2*self.upper**2*np.cos(phi-phi1.T)
        IPPint2 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  + 1./2*self.lower**2*np.cos(phi-phi1.T)
        #IPPint2[0,0] = (self.upper**2 - self.lower**2)*np.cos(phi[0,0])*np.cos(phi1[0,0])
        IPPint = np.where(np.isnan(IPPint1),IPPint2,IPPint1)

        dLa_dper2 = np.column_stack((-self.a[1]*self.basis_omega/self.period))
        dLp_dper2 = np.column_stack((self.basis_phi+np.pi/2))
        r2,omega2,phi2 = dLa_dper2.T,Lo[:,0:1],dLp_dper2.T

        dGint_dper = np.dot(r,r1.T)/2 * (IPPprim - IPPint) + self._int_computation(r2,omega2,phi2, r,omega,phi)
        dGint_dper = dGint_dper + dGint_dper.T

        dFlower_dper  = np.array(self._cos(-self.lower*self.basis_alpha*self.basis_omega/self.period,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]

        dG_dper = 1./self.variance*(self.lengthscale/2*dGint_dper + self.b[0]*(np.dot(dFlower_dper,Flower.T)+np.dot(Flower,dFlower_dper.T)))

        dK_dper = mdot(dFX_dper,self.Gi,FX2.T) - mdot(FX,self.Gi,dG_dper,self.Gi,FX2.T) + mdot(FX,self.Gi,dFX2_dper.T)

        self.variance.gradient = np.sum(dK_dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dK_dlen*dL_dK)
        self.period.gradient = np.sum(dK_dper*dL_dK)



class PeriodicMatern32(Periodic):
    """
    Kernel of the periodic subspace (up to a given frequency) of a Matern 3/2 RKHS. Only defined for input_dim=1.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance of the Matern kernel
    :type variance: float
    :param lengthscale: the lengthscale of the Matern kernel
    :type lengthscale: np.ndarray of size (input_dim,)
    :param period: the period
    :type period: float
    :param n_freq: the number of frequencies considered for the periodic subspace
    :type n_freq: int
    :rtype: kernel object

    """

    def __init__(self, input_dim=1, variance=1., lengthscale=1., period=2.*np.pi, n_freq=10, lower=0., upper=4*np.pi, active_dims=None, name='periodic_Matern32'):
        super(PeriodicMatern32, self).__init__(input_dim, variance, lengthscale, period, n_freq, lower, upper, active_dims, name)
    def parameters_changed(self):
        self.a = [3./self.lengthscale**2, 2*np.sqrt(3)/self.lengthscale, 1.]
        self.b = [1,self.lengthscale**2/3]

        self.basis_alpha = np.ones((self.n_basis,))
        self.basis_omega = (2*np.pi*np.arange(1,self.n_freq+1)/self.period).repeat(2)
        self.basis_phi =   np.zeros(self.n_freq * 2)
        self.basis_phi[::2] = -np.pi/2

        self.G = self.Gram_matrix()
        self.Gi = np.linalg.inv(self.G)

    def Gram_matrix(self):
        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)),self.a[1]*self.basis_omega,self.a[2]*self.basis_omega**2))
        Lo = np.column_stack((self.basis_omega,self.basis_omega,self.basis_omega))
        Lp = np.column_stack((self.basis_phi,self.basis_phi+np.pi/2,self.basis_phi+np.pi))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)

        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]
        F1lower = np.array(self._cos(self.basis_alpha*self.basis_omega,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        return(self.lengthscale**3/(12*np.sqrt(3)*self.variance) * Gint + 1./self.variance*np.dot(Flower,Flower.T) + self.lengthscale**2/(3.*self.variance)*np.dot(F1lower,F1lower.T))


    @silence_errors
    def update_gradients_full(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to the parameters (shape is num_data x num_inducing x num_params)"""
        if X2 is None: X2 = X
        FX  = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)
        FX2 = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X2)

        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)),self.a[1]*self.basis_omega,self.a[2]*self.basis_omega**2))
        Lo = np.column_stack((self.basis_omega,self.basis_omega,self.basis_omega))
        Lp = np.column_stack((self.basis_phi,self.basis_phi+np.pi/2,self.basis_phi+np.pi))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)

        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]
        F1lower = np.array(self._cos(self.basis_alpha*self.basis_omega,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]

        #dK_dvar
        dK_dvar = 1./self.variance*mdot(FX,self.Gi,FX2.T)

        #dK_dlen
        da_dlen = [-6/self.lengthscale**3,-2*np.sqrt(3)/self.lengthscale**2,0.]
        db_dlen = [0.,2*self.lengthscale/3.]
        dLa_dlen =  np.column_stack((da_dlen[0]*np.ones((self.n_basis,1)),da_dlen[1]*self.basis_omega,da_dlen[2]*self.basis_omega**2))
        r1,omega1,phi1 = self._cos_factorization(dLa_dlen,Lo,Lp)
        dGint_dlen = self._int_computation(r1,omega1,phi1, r,omega,phi)
        dGint_dlen = dGint_dlen + dGint_dlen.T
        dG_dlen = self.lengthscale**2/(4*np.sqrt(3))*Gint + self.lengthscale**3/(12*np.sqrt(3))*dGint_dlen + db_dlen[0]*np.dot(Flower,Flower.T) + db_dlen[1]*np.dot(F1lower,F1lower.T)
        dK_dlen = -mdot(FX,self.Gi,dG_dlen/self.variance,self.Gi,FX2.T)

        #dK_dper
        dFX_dper  = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X ,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X)
        dFX2_dper = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X2,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X2)

        dLa_dper = np.column_stack((-self.a[0]*self.basis_omega/self.period, -self.a[1]*self.basis_omega**2/self.period, -self.a[2]*self.basis_omega**3/self.period))
        dLp_dper = np.column_stack((self.basis_phi+np.pi/2,self.basis_phi+np.pi,self.basis_phi+np.pi*3/2))
        r1,omega1,phi1 =  self._cos_factorization(dLa_dper,Lo,dLp_dper)

        IPPprim1 =  self.upper*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi/2))
        IPPprim1 -= self.lower*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi/2))
        IPPprim2 =  self.upper*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  + self.upper*np.cos(phi-phi1.T))
        IPPprim2 -= self.lower*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi/2)  + self.lower*np.cos(phi-phi1.T))
        IPPprim = np.where(np.isnan(IPPprim1),IPPprim2,IPPprim1)

        IPPint1 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi)
        IPPint1 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi)
        IPPint2 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  + 1./2*self.upper**2*np.cos(phi-phi1.T)
        IPPint2 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  + 1./2*self.lower**2*np.cos(phi-phi1.T)
        IPPint = np.where(np.isnan(IPPint1),IPPint2,IPPint1)

        dLa_dper2 = np.column_stack((-self.a[1]*self.basis_omega/self.period, -2*self.a[2]*self.basis_omega**2/self.period))
        dLp_dper2 = np.column_stack((self.basis_phi+np.pi/2,self.basis_phi+np.pi))
        r2,omega2,phi2 =  self._cos_factorization(dLa_dper2,Lo[:,0:2],dLp_dper2)

        dGint_dper = np.dot(r,r1.T)/2 * (IPPprim - IPPint) +  self._int_computation(r2,omega2,phi2, r,omega,phi)
        dGint_dper = dGint_dper + dGint_dper.T

        dFlower_dper  = np.array(self._cos(-self.lower*self.basis_alpha*self.basis_omega/self.period,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        dF1lower_dper = np.array(self._cos(-self.lower*self.basis_alpha*self.basis_omega**2/self.period,self.basis_omega,self.basis_phi+np.pi)(self.lower)+self._cos(-self.basis_alpha*self.basis_omega/self.period,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]

        dG_dper = 1./self.variance*(self.lengthscale**3/(12*np.sqrt(3))*dGint_dper + self.b[0]*(np.dot(dFlower_dper,Flower.T)+np.dot(Flower,dFlower_dper.T)) + self.b[1]*(np.dot(dF1lower_dper,F1lower.T)+np.dot(F1lower,dF1lower_dper.T)))

        dK_dper = mdot(dFX_dper,self.Gi,FX2.T) - mdot(FX,self.Gi,dG_dper,self.Gi,FX2.T) + mdot(FX,self.Gi,dFX2_dper.T)

        self.variance.gradient = np.sum(dK_dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dK_dlen*dL_dK)
        self.period.gradient = np.sum(dK_dper*dL_dK)



class PeriodicMatern52(Periodic):
    """
    Kernel of the periodic subspace (up to a given frequency) of a Matern 5/2 RKHS. Only defined for input_dim=1.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance of the Matern kernel
    :type variance: float
    :param lengthscale: the lengthscale of the Matern kernel
    :type lengthscale: np.ndarray of size (input_dim,)
    :param period: the period
    :type period: float
    :param n_freq: the number of frequencies considered for the periodic subspace
    :type n_freq: int
    :rtype: kernel object

    """

    def __init__(self, input_dim=1, variance=1., lengthscale=1., period=2.*np.pi, n_freq=10, lower=0., upper=4*np.pi, active_dims=None, name='periodic_Matern52'):
        super(PeriodicMatern52, self).__init__(input_dim, variance, lengthscale, period, n_freq, lower, upper, active_dims, name)

    def parameters_changed(self):
        self.a = [5*np.sqrt(5)/self.lengthscale**3, 15./self.lengthscale**2,3*np.sqrt(5)/self.lengthscale, 1.]
        self.b  = [9./8, 9*self.lengthscale**4/200., 3*self.lengthscale**2/5., 3*self.lengthscale**2/(5*8.), 3*self.lengthscale**2/(5*8.)]

        self.basis_alpha = np.ones((2*self.n_freq,))
        self.basis_omega = (2*np.pi*np.arange(1,self.n_freq+1)/self.period).repeat(2)
        self.basis_phi =   np.zeros(self.n_freq * 2)
        self.basis_phi[::2] = -np.pi/2

        self.G = self.Gram_matrix()
        self.Gi = np.linalg.inv(self.G)

    def Gram_matrix(self):
        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)), self.a[1]*self.basis_omega, self.a[2]*self.basis_omega**2, self.a[3]*self.basis_omega**3))
        Lo = np.column_stack((self.basis_omega, self.basis_omega, self.basis_omega, self.basis_omega))
        Lp = np.column_stack((self.basis_phi, self.basis_phi+np.pi/2, self.basis_phi+np.pi, self.basis_phi+np.pi*3/2))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)

        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]
        F1lower = np.array(self._cos(self.basis_alpha*self.basis_omega,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        F2lower = np.array(self._cos(self.basis_alpha*self.basis_omega**2,self.basis_omega,self.basis_phi+np.pi)(self.lower))[:,None]
        lower_terms = self.b[0]*np.dot(Flower,Flower.T) + self.b[1]*np.dot(F2lower,F2lower.T) + self.b[2]*np.dot(F1lower,F1lower.T) + self.b[3]*np.dot(F2lower,Flower.T) + self.b[4]*np.dot(Flower,F2lower.T)
        return(3*self.lengthscale**5/(400*np.sqrt(5)*self.variance) * Gint + 1./self.variance*lower_terms)

    @silence_errors
    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2 = X
        FX  = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)
        FX2 = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X2)

        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)), self.a[1]*self.basis_omega, self.a[2]*self.basis_omega**2, self.a[3]*self.basis_omega**3))
        Lo = np.column_stack((self.basis_omega, self.basis_omega, self.basis_omega, self.basis_omega))
        Lp = np.column_stack((self.basis_phi, self.basis_phi+np.pi/2, self.basis_phi+np.pi, self.basis_phi+np.pi*3/2))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)

        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]
        F1lower = np.array(self._cos(self.basis_alpha*self.basis_omega,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        F2lower = np.array(self._cos(self.basis_alpha*self.basis_omega**2,self.basis_omega,self.basis_phi+np.pi)(self.lower))[:,None]

        #dK_dvar
        dK_dvar = 1./self.variance*mdot(FX,self.Gi,FX2.T)

        #dK_dlen
        da_dlen = [-3*self.a[0]/self.lengthscale, -2*self.a[1]/self.lengthscale, -self.a[2]/self.lengthscale, 0.]
        db_dlen = [0., 4*self.b[1]/self.lengthscale, 2*self.b[2]/self.lengthscale, 2*self.b[3]/self.lengthscale, 2*self.b[4]/self.lengthscale]
        dLa_dlen =  np.column_stack((da_dlen[0]*np.ones((self.n_basis,1)), da_dlen[1]*self.basis_omega, da_dlen[2]*self.basis_omega**2, da_dlen[3]*self.basis_omega**3))
        r1,omega1,phi1 = self._cos_factorization(dLa_dlen,Lo,Lp)
        dGint_dlen = self._int_computation(r1,omega1,phi1, r,omega,phi)
        dGint_dlen = dGint_dlen + dGint_dlen.T
        dlower_terms_dlen = db_dlen[0]*np.dot(Flower,Flower.T) + db_dlen[1]*np.dot(F2lower,F2lower.T) + db_dlen[2]*np.dot(F1lower,F1lower.T) + db_dlen[3]*np.dot(F2lower,Flower.T) + db_dlen[4]*np.dot(Flower,F2lower.T)
        dG_dlen = 15*self.lengthscale**4/(400*np.sqrt(5))*Gint + 3*self.lengthscale**5/(400*np.sqrt(5))*dGint_dlen + dlower_terms_dlen
        dK_dlen = -mdot(FX,self.Gi,dG_dlen/self.variance,self.Gi,FX2.T)

        #dK_dper
        dFX_dper  = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X ,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X)
        dFX2_dper = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X2,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X2)

        dLa_dper = np.column_stack((-self.a[0]*self.basis_omega/self.period, -self.a[1]*self.basis_omega**2/self.period, -self.a[2]*self.basis_omega**3/self.period, -self.a[3]*self.basis_omega**4/self.period))
        dLp_dper = np.column_stack((self.basis_phi+np.pi/2,self.basis_phi+np.pi,self.basis_phi+np.pi*3/2,self.basis_phi))
        r1,omega1,phi1 =  self._cos_factorization(dLa_dper,Lo,dLp_dper)

        IPPprim1 =  self.upper*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi/2))
        IPPprim1 -= self.lower*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi/2)  +  1./(omega-omega1.T)*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi/2))
        IPPprim2 =  self.upper*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi/2)  + self.upper*np.cos(phi-phi1.T))
        IPPprim2 -= self.lower*(1./(omega+omega1.T)*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi/2)  + self.lower*np.cos(phi-phi1.T))
        IPPprim = np.where(np.isnan(IPPprim1),IPPprim2,IPPprim1)

        IPPint1 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi)
        IPPint1 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi)
        IPPint2 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  + 1./2*self.upper**2*np.cos(phi-phi1.T)
        IPPint2 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  + 1./2*self.lower**2*np.cos(phi-phi1.T)
        IPPint = np.where(np.isnan(IPPint1),IPPint2,IPPint1)

        dLa_dper2 = np.column_stack((-self.a[1]*self.basis_omega/self.period, -2*self.a[2]*self.basis_omega**2/self.period, -3*self.a[3]*self.basis_omega**3/self.period))
        dLp_dper2 = np.column_stack((self.basis_phi+np.pi/2, self.basis_phi+np.pi, self.basis_phi+np.pi*3/2))
        r2,omega2,phi2 =  self._cos_factorization(dLa_dper2,Lo[:,0:2],dLp_dper2)

        dGint_dper = np.dot(r,r1.T)/2 * (IPPprim - IPPint) +  self._int_computation(r2,omega2,phi2, r,omega,phi)
        dGint_dper = dGint_dper + dGint_dper.T

        dFlower_dper  = np.array(self._cos(-self.lower*self.basis_alpha*self.basis_omega/self.period,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        dF1lower_dper = np.array(self._cos(-self.lower*self.basis_alpha*self.basis_omega**2/self.period,self.basis_omega,self.basis_phi+np.pi)(self.lower)+self._cos(-self.basis_alpha*self.basis_omega/self.period,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        dF2lower_dper = np.array(self._cos(-self.lower*self.basis_alpha*self.basis_omega**3/self.period,self.basis_omega,self.basis_phi+np.pi*3/2)(self.lower) + self._cos(-2*self.basis_alpha*self.basis_omega**2/self.period,self.basis_omega,self.basis_phi+np.pi)(self.lower))[:,None]

        dlower_terms_dper  = self.b[0] * (np.dot(dFlower_dper,Flower.T) + np.dot(Flower.T,dFlower_dper))
        dlower_terms_dper += self.b[1] * (np.dot(dF2lower_dper,F2lower.T) + np.dot(F2lower,dF2lower_dper.T)) - 4*self.b[1]/self.period*np.dot(F2lower,F2lower.T)
        dlower_terms_dper += self.b[2] * (np.dot(dF1lower_dper,F1lower.T) + np.dot(F1lower,dF1lower_dper.T)) - 2*self.b[2]/self.period*np.dot(F1lower,F1lower.T)
        dlower_terms_dper += self.b[3] * (np.dot(dF2lower_dper,Flower.T) + np.dot(F2lower,dFlower_dper.T)) - 2*self.b[3]/self.period*np.dot(F2lower,Flower.T)
        dlower_terms_dper += self.b[4] * (np.dot(dFlower_dper,F2lower.T) + np.dot(Flower,dF2lower_dper.T)) - 2*self.b[4]/self.period*np.dot(Flower,F2lower.T)

        dG_dper = 1./self.variance*(3*self.lengthscale**5/(400*np.sqrt(5))*dGint_dper + 0.5*dlower_terms_dper)
        dK_dper = mdot(dFX_dper,self.Gi,FX2.T) - mdot(FX,self.Gi,dG_dper,self.Gi,FX2.T) + mdot(FX,self.Gi,dFX2_dper.T)

        self.variance.gradient = np.sum(dK_dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dK_dlen*dL_dK)
        self.period.gradient = np.sum(dK_dper*dL_dK)
