# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv, trace_dot
from ..util.plot import gpplot
from .. import kern
from GP import GP
from scipy import linalg

#Still TODO:
# make use of slices properly (kernel can now do this)
# enable heteroscedatic noise (kernel will need to compute psi2 as a (NxMxM) array)

class sparse_GP(GP):
    """
    Variational sparse GP model

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP)
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (N x Q) | None
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param Zslices: slices for the inducing inputs (see slicing TODO: link)
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, likelihood, kernel, Z, X_variance=None, Xslices=None,Zslices=None, normalize_X=False):
        self.scale_factor = 100.0# a scaling factor to help keep the algorithm stable
        self.auto_scale_factor = False
        self.Z = Z
        self.Zslices = Zslices
        self.Xslices = Xslices
        self.M = Z.shape[0]
        self.likelihood = likelihood

        if X_variance is None:
            self.has_uncertain_inputs=False
        else:
            assert X_variance.shape==X.shape
            self.has_uncertain_inputs=True
            self.X_variance = X_variance

        if not self.likelihood.is_heteroscedastic:
            self.likelihood.trYYT = np.trace(np.dot(self.likelihood.Y, self.likelihood.Y.T)) # TODO: something more elegant here?

        GP.__init__(self, X, likelihood, kernel=kernel, normalize_X=normalize_X, Xslices=Xslices)

        #normalize X uncertainty also
        if self.has_uncertain_inputs:
            self.X_variance /= np.square(self._Xstd)


    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            self.psi0 = self.kern.psi0(self.Z,self.X, self.X_variance)
            self.psi1 = self.kern.psi1(self.Z,self.X, self.X_variance).T
            self.psi2 = self.kern.psi2(self.Z,self.X, self.X_variance)
        else:
            self.psi0 = self.kern.Kdiag(self.X,slices=self.Xslices)
            self.psi1 = self.kern.K(self.Z,self.X)
            self.psi2 = None

    def _computations(self):
        #TODO: find routine to multiply triangular matrices
        #TODO: slices for psi statistics (easy enough)

        sf = self.scale_factor
        sf2 = sf**2

        #The rather complex computations of psi2_beta_scaled
        if self.likelihood.is_heteroscedastic:
            assert self.likelihood.D == 1 #TODO: what if the likelihood is heterscedatic and there are multiple independent outputs?
            if self.has_uncertain_inputs:
                self.psi2_beta_scaled = (self.psi2*(self.likelihood.precision.flatten().reshape(self.N,1,1)/sf2)).sum(0)
            else:
                tmp = self.psi1*(np.sqrt(self.likelihood.precision.flatten().reshape(1,self.N))/sf)
                self.psi2_beta_scaled = np.dot(tmp,tmp.T)
        else:
            if self.has_uncertain_inputs:
                self.psi2_beta_scaled = (self.psi2*(self.likelihood.precision/sf2)).sum(0)
            else:
                tmp = self.psi1*(np.sqrt(self.likelihood.precision)/sf)
                self.psi2_beta_scaled = np.dot(tmp,tmp.T)

        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)

        self.V = (self.likelihood.precision/self.scale_factor)*self.likelihood.Y

        #Compute A = L^-1 psi2 beta L^-T
        #self. A = mdot(self.Lmi,self.psi2_beta_scaled,self.Lmi.T)
        tmp = linalg.lapack.flapack.dtrtrs(self.Lm,self.psi2_beta_scaled.T,lower=1)[0]
        self.A = linalg.lapack.flapack.dtrtrs(self.Lm,np.asarray(tmp.T,order='F'),lower=1)[0]

        self.B = np.eye(self.M)/sf2 + self.A

        self.Bi, self.LB, self.LBi, self.B_logdet = pdinv(self.B)

        self.psi1V = np.dot(self.psi1, self.V)
        #tmp = np.dot(self.Lmi.T, self.LBi.T)
        tmp = linalg.lapack.clapack.dtrtrs(self.Lm.T,np.asarray(self.LBi.T,order='C'),lower=0)[0]
        self.C = np.dot(tmp,tmp.T) #TODO: tmp is triangular. replace with dtrmm (blas) when available
        self.Cpsi1V = np.dot(self.C,self.psi1V)
        self.Cpsi1VVpsi1 = np.dot(self.Cpsi1V,self.psi1V.T)
        #self.E = np.dot(self.Cpsi1VVpsi1,self.C)/sf2
        self.E = np.dot(self.Cpsi1V/sf,self.Cpsi1V.T/sf)

        # Compute dL_dpsi # FIXME: this is untested for the heterscedastic + uncertin inputs case
        self.dL_dpsi0 = - 0.5 * self.D * (self.likelihood.precision * np.ones([self.N,1])).flatten()
        self.dL_dpsi1 = np.dot(self.Cpsi1V,self.V.T)
        if self.likelihood.is_heteroscedastic:
            if self.has_uncertain_inputs:
                self.dL_dpsi2 = 0.5 * self.likelihood.precision[:,None,None] * self.D * self.Kmmi[None,:,:] # dB
                self.dL_dpsi2 += - 0.5 * self.likelihood.precision[:,None,None]/sf2 * self.D * self.C[None,:,:] # dC
                self.dL_dpsi2 += - 0.5 * self.likelihood.precision[:,None,None]* self.E[None,:,:] # dD
            else:
                self.dL_dpsi1 += mdot(self.Kmmi,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)) #dB
                self.dL_dpsi1 += -mdot(self.C,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)/sf2) #dC
                self.dL_dpsi1 += -mdot(self.E,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)) #dD
                self.dL_dpsi2 = None

        else:
            self.dL_dpsi2 = 0.5 * self.likelihood.precision * self.D * self.Kmmi # dB
            self.dL_dpsi2 += - 0.5 * self.likelihood.precision/sf2 * self.D * self.C # dC
            self.dL_dpsi2 += - 0.5 * self.likelihood.precision * self.E # dD
            if self.has_uncertain_inputs:
                #repeat for each of the N psi_2 matrices
                self.dL_dpsi2 = np.repeat(self.dL_dpsi2[None,:,:],self.N,axis=0)
            else:
                self.dL_dpsi1 += 2.*np.dot(self.dL_dpsi2,self.psi1)
                self.dL_dpsi2 = None


        # Compute dL_dKmm
        #self.dL_dKmm_old = -0.5 * self.D * mdot(self.Lmi.T, self.A, self.Lmi)*sf2 # dB
        #self.dL_dKmm += -0.5 * self.D * (- self.C/sf2 - 2.*mdot(self.C, self.psi2_beta_scaled, self.Kmmi) + self.Kmmi) # dC
        #self.dL_dKmm +=  np.dot(np.dot(self.E*sf2, self.psi2_beta_scaled) - self.Cpsi1VVpsi1, self.Kmmi) + 0.5*self.E # dD
        tmp = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(self.A),lower=1,trans=1)[0]
        self.dL_dKmm = -0.5*self.D*sf2*linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(tmp.T),lower=1,trans=1)[0] #dA
        self.dL_dKmm += 0.5*(self.D*(self.C/sf2 -self.Kmmi) + self.E) + np.dot(np.dot(self.D*self.C + self.E*sf2,self.psi2_beta_scaled) - self.Cpsi1VVpsi1,self.Kmmi) # d(C+D)

        #the partial derivative vector for the likelihood
        if self.likelihood.Nparams ==0:
            #save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
            #self.partial_for_likelihood = - 0.5 * self.D*self.likelihood.precision + 0.5 * (self.likelihood.Y**2).sum(1)*self.likelihood.precision**2 #dA
            #self.partial_for_likelihood +=  0.5 * self.D * (self.psi0*self.likelihood.precision**2 - (self.psi2*self.Kmmi[None,:,:]*self.likelihood.precision[:,None,None]**2).sum(1).sum(1)/sf2) #dB
            #self.partial_for_likelihood +=  0.5 * self.D * np.sum(self.Bi*self.A)*self.likelihood.precision #dC
            #self.partial_for_likelihood += -np.diag(np.dot((self.C - 0.5 * mdot(self.C,self.psi2_beta_scaled,self.C) ) , self.psi1VVpsi1 ))*self.likelihood.precision #dD
        else:
            #likelihood is not heterscedatic
            self.partial_for_likelihood =   - 0.5 * self.N*self.D*self.likelihood.precision + 0.5 * np.sum(np.square(self.likelihood.Y))*self.likelihood.precision**2
            self.partial_for_likelihood += 0.5 * self.D * (self.psi0.sum()*self.likelihood.precision**2 - np.trace(self.A)*self.likelihood.precision*sf2)
            self.partial_for_likelihood += 0.5 * self.D * trace_dot(self.Bi,self.A)*self.likelihood.precision
            self.partial_for_likelihood += self.likelihood.precision*(0.5*trace_dot(self.psi2_beta_scaled,self.E*sf2) - np.trace(self.Cpsi1VVpsi1))



    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        sf2 = self.scale_factor**2
        if self.likelihood.is_heteroscedastic:
            A = -0.5*self.N*self.D*np.log(2.*np.pi) +0.5*np.sum(np.log(self.likelihood.precision)) -0.5*np.sum(self.V*self.likelihood.Y)
            B = -0.5*self.D*(np.sum(self.likelihood.precision.flatten()*self.psi0) - np.trace(self.A)*sf2)
        else:
            A = -0.5*self.N*self.D*(np.log(2.*np.pi) + np.log(self.likelihood._variance)) -0.5*self.likelihood.precision*self.likelihood.trYYT
            B = -0.5*self.D*(np.sum(self.likelihood.precision*self.psi0) - np.trace(self.A)*sf2)
        C = -0.5*self.D * (self.B_logdet + self.M*np.log(sf2))
        D = 0.5*np.trace(self.Cpsi1VVpsi1)
        return A+B+C+D

    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        self.kern._set_params(p[self.Z.size:self.Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size+self.kern.Nparam:])
        self._compute_kernel_matrices()
        if self.auto_scale_factor:
            self.scale_factor = np.sqrt(self.psi2.sum(0).mean()*self.likelihood.precision)
        #if self.auto_scale_factor:
        #    if self.likelihood.is_heteroscedastic:
        #        self.scale_factor = max(1,np.sqrt(self.psi2_beta_scaled.sum(0).mean()))
        #    else:
        #        self.scale_factor = np.sqrt(self.psi2.sum(0).mean()*self.likelihood.precision)
        self._computations()

    def _get_params(self):
        return np.hstack([self.Z.flatten(),GP._get_params(self)])

    def _get_param_names(self):
        return sum([['iip_%i_%i'%(i,j) for j in range(self.Z.shape[1])] for i in range(self.Z.shape[0])],[]) + GP._get_param_names(self)

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian (or direct: TODO) likelihood, no iteration is required:
        this function does nothing
        """
        if self.has_uncertain_inputs:
            raise NotImplementedError, "EP approximation not implemented for uncertain inputs"
        else:
            self.likelihood.fit_DTC(self.Kmm,self.psi1)
            #self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
            self._set_params(self._get_params()) # update the GP


    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm,self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z,self.X,self.X_variance)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T,self.Z,self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2, self.Z,self.X, self.X_variance)
        else:
            dL_dtheta += self.kern.dK_dtheta(self.dL_dpsi1,self.Z,self.X)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)

        return dL_dtheta

    def dL_dZ(self):
        """
        The derivative of the bound wrt the inducing inputs Z
        """
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm, self.Z)  # factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1,self.Z,self.X, self.X_variance)
            dL_dZ += self.kern.dpsi2_dZ(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dZ += self.kern.dK_dX(self.dL_dpsi1,self.Z,self.X)
        return dL_dZ

    def _raw_predict(self, Xnew, slices, full_cov=False):
        """Internal helper function for making predictions, does not account for normalization"""

        Kx = self.kern.K(self.Z, Xnew)
        mu = mdot(Kx.T, self.C/self.scale_factor, self.psi1V)
        if full_cov:
            Kxx = self.kern.K(Xnew)
            var = Kxx - mdot(Kx.T, (self.Kmmi - self.C/self.scale_factor**2), Kx) #NOTE this won't work for plotting
        else:
            Kxx = self.kern.Kdiag(Xnew)
            var = Kxx - np.sum(Kx*np.dot(self.Kmmi - self.C/self.scale_factor**2, Kx),0)

        return mu,var[:,None]
