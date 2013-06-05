# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
from GPy.util.linalg import mdot
from GPy.util.decorators import silence_errors

class periodic_Matern52(Kernpart):
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

    def __init__(self,input_dim=1,variance=1.,lengthscale=None,period=2*np.pi,n_freq=10,lower=0.,upper=4*np.pi):
        assert input_dim==1, "Periodic kernels are only defined for input_dim=1"
        self.name = 'periodic_Mat52'
        self.input_dim = input_dim
        if lengthscale is not None:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size == 1, "Wrong size: only one lengthscale needed"
        else:
            lengthscale = np.ones(1)
        self.lower,self.upper = lower, upper
        self.num_params = 3
        self.n_freq = n_freq
        self.n_basis = 2*n_freq
        self._set_params(np.hstack((variance,lengthscale,period)))

    def _cos(self,alpha,omega,phase):
        def f(x):
            return alpha*np.cos(omega*x+phase)
        return f

    @silence_errors
    def _cos_factorization(self,alpha,omega,phase):
        r1 = np.sum(alpha*np.cos(phase),axis=1)[:,None]
        r2 = np.sum(alpha*np.sin(phase),axis=1)[:,None]
        r =  np.sqrt(r1**2 + r2**2)
        psi = np.where(r1 != 0, (np.arctan(r2/r1) + (r1<0.)*np.pi),np.arcsin(r2))
        return r,omega[:,0:1], psi

    @silence_errors
    def _int_computation(self,r1,omega1,phi1,r2,omega2,phi2):
        Gint1 = 1./(omega1+omega2.T)*( np.sin((omega1+omega2.T)*self.upper+phi1+phi2.T) - np.sin((omega1+omega2.T)*self.lower+phi1+phi2.T)) + 1./(omega1-omega2.T)*( np.sin((omega1-omega2.T)*self.upper+phi1-phi2.T) - np.sin((omega1-omega2.T)*self.lower+phi1-phi2.T) )
        Gint2 = 1./(omega1+omega2.T)*( np.sin((omega1+omega2.T)*self.upper+phi1+phi2.T) - np.sin((omega1+omega2.T)*self.lower+phi1+phi2.T)) +  np.cos(phi1-phi2.T)*(self.upper-self.lower)
        #Gint2[0,0] = 2.*(self.upper-self.lower)*np.cos(phi1[0,0])*np.cos(phi2[0,0])
        Gint = np.dot(r1,r2.T)/2 * np.where(np.isnan(Gint1),Gint2,Gint1)
        return Gint

    def _get_params(self):
        """return the value of the parameters."""
        return np.hstack((self.variance,self.lengthscale,self.period))

    def _set_params(self,x):
        """set the value of the parameters."""
        assert x.size==3
        self.variance = x[0]
        self.lengthscale = x[1]
        self.period = x[2]

        self.a = [5*np.sqrt(5)/self.lengthscale**3, 15./self.lengthscale**2,3*np.sqrt(5)/self.lengthscale, 1.]
        self.b  = [9./8, 9*self.lengthscale**4/200., 3*self.lengthscale**2/5., 3*self.lengthscale**2/(5*8.), 3*self.lengthscale**2/(5*8.)]

        self.basis_alpha = np.ones((2*self.n_freq,))
        self.basis_omega = np.array(sum([[i*2*np.pi/self.period]*2 for i in  range(1,self.n_freq+1)],[]))
        self.basis_phi =   np.array(sum([[-np.pi/2, 0.]  for i in range(1,self.n_freq+1)],[]))

        self.G = self.Gram_matrix()
        self.Gi = np.linalg.inv(self.G)

    def _get_param_names(self):
        """return parameter names."""
        return ['variance','lengthscale','period']

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

    def K(self,X,X2,target):
        """Compute the covariance matrix between X and X2."""
        FX = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)
        if X2 is None:
            FX2 = FX
        else:
            FX2 = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X2)
        np.add(mdot(FX,self.Gi,FX2.T), target,target)

    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        FX  = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)
        np.add(target,np.diag(mdot(FX,self.Gi,FX.T)),target)

    @silence_errors
    def dK_dtheta(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to the parameters (shape is Nxnum_inducingxNparam)"""
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
        #IPPprim2[0,0] = 2*(self.upper**2 - self.lower**2)*np.cos(phi[0,0])*np.cos(phi1[0,0])
        IPPprim = np.where(np.isnan(IPPprim1),IPPprim2,IPPprim1)

        IPPint1 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.upper+phi-phi1.T-np.pi)
        IPPint1 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  +  1./(omega-omega1.T)**2*np.cos((omega-omega1.T)*self.lower+phi-phi1.T-np.pi)
        IPPint2 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  + 1./2*self.upper**2*np.cos(phi-phi1.T)
        IPPint2 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  + 1./2*self.lower**2*np.cos(phi-phi1.T)
        #IPPint2[0,0] = (self.upper**2 - self.lower**2)*np.cos(phi[0,0])*np.cos(phi1[0,0])
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

        # np.add(target[:,:,0],dK_dvar, target[:,:,0])
        target[0] += np.sum(dK_dvar*dL_dK)
        #np.add(target[:,:,1],dK_dlen, target[:,:,1])
        target[1] += np.sum(dK_dlen*dL_dK)
        #np.add(target[:,:,2],dK_dper, target[:,:,2])
        target[2] += np.sum(dK_dper*dL_dK)

    @silence_errors
    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        """derivative of the diagonal of the covariance matrix with respect to the parameters"""
        FX  = self._cos(self.basis_alpha[None,:],self.basis_omega[None,:],self.basis_phi[None,:])(X)

        La = np.column_stack((self.a[0]*np.ones((self.n_basis,1)), self.a[1]*self.basis_omega, self.a[2]*self.basis_omega**2, self.a[3]*self.basis_omega**3))
        Lo = np.column_stack((self.basis_omega, self.basis_omega, self.basis_omega, self.basis_omega))
        Lp = np.column_stack((self.basis_phi, self.basis_phi+np.pi/2, self.basis_phi+np.pi, self.basis_phi+np.pi*3/2))
        r,omega,phi =  self._cos_factorization(La,Lo,Lp)
        Gint = self._int_computation( r,omega,phi, r,omega,phi)

        Flower = np.array(self._cos(self.basis_alpha,self.basis_omega,self.basis_phi)(self.lower))[:,None]
        F1lower = np.array(self._cos(self.basis_alpha*self.basis_omega,self.basis_omega,self.basis_phi+np.pi/2)(self.lower))[:,None]
        F2lower = np.array(self._cos(self.basis_alpha*self.basis_omega**2,self.basis_omega,self.basis_phi+np.pi)(self.lower))[:,None]

        #dK_dvar
        dK_dvar = 1. / self.variance * mdot(FX, self.Gi, FX.T)

        #dK_dlen
        da_dlen = [-3*self.a[0]/self.lengthscale, -2*self.a[1]/self.lengthscale, -self.a[2]/self.lengthscale, 0.]
        db_dlen = [0., 4*self.b[1]/self.lengthscale, 2*self.b[2]/self.lengthscale, 2*self.b[3]/self.lengthscale, 2*self.b[4]/self.lengthscale]
        dLa_dlen =  np.column_stack((da_dlen[0]*np.ones((self.n_basis,1)), da_dlen[1]*self.basis_omega, da_dlen[2]*self.basis_omega**2, da_dlen[3]*self.basis_omega**3))
        r1,omega1,phi1 = self._cos_factorization(dLa_dlen,Lo,Lp)
        dGint_dlen = self._int_computation(r1,omega1,phi1, r,omega,phi)
        dGint_dlen = dGint_dlen + dGint_dlen.T
        dlower_terms_dlen = db_dlen[0]*np.dot(Flower,Flower.T) + db_dlen[1]*np.dot(F2lower,F2lower.T) + db_dlen[2]*np.dot(F1lower,F1lower.T) + db_dlen[3]*np.dot(F2lower,Flower.T) + db_dlen[4]*np.dot(Flower,F2lower.T)
        dG_dlen = 15*self.lengthscale**4/(400*np.sqrt(5))*Gint + 3*self.lengthscale**5/(400*np.sqrt(5))*dGint_dlen + dlower_terms_dlen
        dK_dlen = -mdot(FX,self.Gi,dG_dlen/self.variance,self.Gi,FX.T)

        #dK_dper
        dFX_dper  = self._cos(-self.basis_alpha[None,:]*self.basis_omega[None,:]/self.period*X ,self.basis_omega[None,:],self.basis_phi[None,:]+np.pi/2)(X)

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
        IPPint2 =  1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.upper+phi+phi1.T-np.pi)  + .5*self.upper**2*np.cos(phi-phi1.T)
        IPPint2 -= 1./(omega+omega1.T)**2*np.cos((omega+omega1.T)*self.lower+phi+phi1.T-np.pi)  + .5*self.lower**2*np.cos(phi-phi1.T)
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
        dK_dper = 2*mdot(dFX_dper,self.Gi,FX.T) - mdot(FX,self.Gi,dG_dper,self.Gi,FX.T)

        target[0] += np.sum(np.diag(dK_dvar)*dL_dKdiag)
        target[1] += np.sum(np.diag(dK_dlen)*dL_dKdiag)
        target[2] += np.sum(np.diag(dK_dper)*dL_dKdiag)
