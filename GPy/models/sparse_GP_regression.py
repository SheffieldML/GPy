# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from .. import kern
from ..inference.likelihoods import likelihood
from GP_regression import GP_regression

#Still TODO:
# make use of slices properly (kernel can now do this)
# enable heteroscedatic noise (kernel will need to compute psi2 as a (NxMxM) array)

class sparse_GP_regression(GP_regression):
    """
    Variational sparse GP model (Regression)

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param Y: observed data
    :type Y: np.ndarray of observations (N x D)
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param X_uncertainty: The uncertainty in the measurements of X (Gaussian variance)
    :type X_uncertainty: np.ndarray (N x Q) | None
    :param Zslices: slices for the inducing inputs (see slicing TODO: link)
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param beta: noise precision. TODO> ignore beta if doing EP
    :type beta: float
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self,X,Y,kernel=None, X_uncertainty=None, beta=100., Z=None,Zslices=None,M=10,normalize_X=False,normalize_Y=False):
        self.scale_factor = 100.0
        self.beta = beta
        if Z is None:
            self.Z = np.random.permutation(X.copy())[:M]
            self.M = M
        else:
            assert Z.shape[1]==X.shape[1]
            self.Z = Z
            self.M = Z.shape[0]
        if X_uncertainty is None:
            self.has_uncertain_inputs=False
        else:
            assert X_uncertainty.shape==X.shape
            self.has_uncertain_inputs=True
            self.X_uncertainty = X_uncertainty

        GP_regression.__init__(self, X, Y, kernel=kernel, normalize_X=normalize_X, normalize_Y=normalize_Y)
        self.trYYT = np.sum(np.square(self.Y))

        #normalise X uncertainty also
        if self.has_uncertain_inputs:
            self.X_uncertainty /= np.square(self._Xstd)

    def _computations(self):
        # TODO find routine to multiply triangular matrices
        #TODO: slices for psi statistics (easy enough)

        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            self.psi0 = self.kern.psi0(self.Z,self.X, self.X_uncertainty).sum()
            self.psi1 = self.kern.psi1(self.Z,self.X, self.X_uncertainty).T
            self.psi2 = self.kern.psi2(self.Z,self.X, self.X_uncertainty)
            self.psi2_beta_scaled = (self.psi2*(self.beta/self.scale_factor**2)).sum(0)
        else:
            self.psi0 = self.kern.Kdiag(self.X,slices=self.Xslices).sum()
            self.psi1 = self.kern.K(self.Z,self.X)
            #self.psi2 = np.dot(self.psi1,self.psi1.T)
            #self.psi2 = self.psi1.T[:,:,None]*self.psi1.T[:,None,:]
            tmp = self.psi1/(self.scale_factor/np.sqrt(self.beta))
            self.psi2_beta_scaled = np.dot(tmp,tmp.T)

        sf = self.scale_factor
        sf2 = sf**2

        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)#+np.eye(self.M)*1e-3)

        self.V = (self.beta/self.scale_factor)*self.Y
        self.A = mdot(self.Lmi, self.psi2_beta_scaled, self.Lmi.T)
        self.B = np.eye(self.M)/sf2 + self.A

        self.Bi, self.LB, self.LBi, self.B_logdet = pdinv(self.B)

        self.psi1V = np.dot(self.psi1, self.V)
        self.psi1VVpsi1 = np.dot(self.psi1V, self.psi1V.T)
        self.C = mdot(self.Lmi.T, self.Bi, self.Lmi)
        self.E = mdot(self.C, self.psi1VVpsi1/sf2, self.C.T)

        # Compute dL_dpsi
        self.dL_dpsi0 = - 0.5 * self.D * self.beta * np.ones(self.N)
        self.dL_dpsi1 = mdot(self.V, self.psi1V.T,self.C).T
        self.dL_dpsi2 = 0.5 * self.beta * self.D * self.Kmmi[None,:,:] # dB
        self.dL_dpsi2 += - 0.5 * self.beta/sf2 * self.D * self.C[None,:,:] # dC
        self.dL_dpsi2 += - 0.5 * self.beta * self.E[None,:,:] # dD

        # Compute dL_dKmm
        self.dL_dKmm = -0.5 * self.D * mdot(self.Lmi.T, self.A, self.Lmi)*sf2 # dB
        self.dL_dKmm += -0.5 * self.D * (- self.C/sf2 - 2.*mdot(self.C, self.psi2_beta_scaled, self.Kmmi) + self.Kmmi) # dC
        self.dL_dKmm +=  np.dot(np.dot(self.E*sf2, self.psi2_beta_scaled) - np.dot(self.C, self.psi1VVpsi1), self.Kmmi) + 0.5*self.E # dD


    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        self.beta = p[self.M*self.Q]
        self.kern._set_params(p[self.Z.size + 1:])
        self._computations()

    def _get_params(self):
        return np.hstack([self.Z.flatten(),self.beta,self.kern._get_params_transformed()])

    def _get_param_names(self):
        return sum([['iip_%i_%i'%(i,j) for i in range(self.Z.shape[0])] for j in range(self.Z.shape[1])],[]) + ['noise_precision']+self.kern._get_param_names_transformed()

        
    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        sf2 = self.scale_factor**2
        A = -0.5*self.N*self.D*(np.log(2.*np.pi) - np.log(self.beta)) -0.5*self.beta*self.trYYT
        B = -0.5*self.D*(self.beta*self.psi0-np.trace(self.A)*sf2)
        C = -0.5*self.D * (self.B_logdet + self.M*np.log(sf2))
        D = +0.5*np.sum(self.psi1VVpsi1 * self.C)
        return A+B+C+D

    def _log_likelihood_gradients(self):
        return np.hstack([self.dL_dZ().flatten(), self.dL_dbeta(), self.dL_dtheta()])
    
    def dL_dbeta(self):
        """
        Compute the gradient of the log likelihood wrt beta.
        """
        #TODO: suport heteroscedatic noise
        sf2 = self.scale_factor**2
        dA_dbeta =   0.5 * self.N*self.D/self.beta - 0.5 * self.trYYT
        dB_dbeta = - 0.5 * self.D * (self.psi0 - np.trace(self.A)/self.beta*sf2)
        dC_dbeta = - 0.5 * self.D * np.sum(self.Bi*self.A)/self.beta
        dD_dbeta = np.sum((self.C - 0.5 * mdot(self.C,self.psi2_beta_scaled,self.C) ) * self.psi1VVpsi1 )/self.beta

        return np.squeeze(dA_dbeta + dB_dbeta + dC_dbeta + dD_dbeta)

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm,self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z,self.X,self.X_uncertainty)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T,self.Z,self.X, self.X_uncertainty)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2,self.dL_dpsi1.T, self.Z,self.X, self.X_uncertainty) # for multiple_beta, dL_dpsi2 will be a different shape
        else:
            #re-cast computations in psi2 back to psi1:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(self.dL_dpsi2.sum(0),self.psi1)
            dL_dtheta += self.kern.dK_dtheta(dL_dpsi1,self.Z,self.X)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)

        return dL_dtheta

    def dL_dZ(self):
        """
        The derivative of the bound wrt the inducing inputs Z
        """
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm,self.Z)#factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1,self.Z,self.X, self.X_uncertainty)
            dL_dZ += 2.*self.kern.dpsi2_dZ(self.dL_dpsi2,self.Z,self.X, self.X_uncertainty) # 'stripes'
        else:
            #re-cast computations in psi2 back to psi1:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(self.dL_dpsi2.sum(0),self.psi1)
            dL_dZ += self.kern.dK_dX(dL_dpsi1,self.Z,self.X)
        return dL_dZ

    def _raw_predict(self, Xnew, slices, full_cov=False):
        """Internal helper function for making predictions, does not account for normalisation"""

        Kx = self.kern.K(self.Z, Xnew)
        mu = mdot(Kx.T, self.C/self.scale_factor, self.psi1V)

        if full_cov:
            Kxx = self.kern.K(Xnew)
            var = Kxx - mdot(Kx.T, (self.Kmmi - self.C/self.scale_factor**2), Kx) + np.eye(Xnew.shape[0])/self.beta # TODO: This beta doesn't belong here in the EP case.
        else:
            Kxx = self.kern.Kdiag(Xnew)
            var = Kxx - np.sum(Kx*np.dot(self.Kmmi - self.C/self.scale_factor**2, Kx),0) + 1./self.beta # TODO: This beta doesn't belong here in the EP case.

        return mu,var

    def plot(self, *args, **kwargs):
        """
        Plot the fitted model: just call the GP_regression plot function and then add inducing inputs
        """
        GP_regression.plot(self,*args,**kwargs)
        if self.Q==1:
            pb.plot(self.Z,self.Z*0+pb.ylim()[0],'k|',mew=1.5,markersize=12)
            if self.has_uncertain_inputs:
                pb.errorbar(self.X[:,0], pb.ylim()[0]+np.zeros(self.N), xerr=2*np.sqrt(self.X_uncertainty.flatten()))
        if self.Q==2:
            pb.plot(self.Z[:,0],self.Z[:,1],'wo')
