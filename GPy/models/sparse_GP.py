# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv, trace_dot, tdot
from ..util.plot import gpplot
from .. import kern
from GP import GP
from scipy import linalg

def backsub_both_sides(L,X):
    """ Return L^-T * X * L^-1, assumuing X is symmetrical and L is lower cholesky"""
    tmp,_ = linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(X),lower=1,trans=1)
    return linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(tmp.T),lower=1,trans=1)[0].T

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
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False):
        self.scale_factor = 100.0# a scaling factor to help keep the algorithm stable
        self.auto_scale_factor = False
        self.Z = Z
        self.M = Z.shape[0]
        self.likelihood = likelihood

        if X_variance is None:
            self.has_uncertain_inputs=False
        else:
            assert X_variance.shape==X.shape
            self.has_uncertain_inputs=True
            self.X_variance = X_variance

        GP.__init__(self, X, likelihood, kernel=kernel, normalize_X=normalize_X)

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
            self.psi0 = self.kern.Kdiag(self.X)
            self.psi1 = self.kern.K(self.Z,self.X)
            self.psi2 = None

    def _computations(self):
        #TODO: find routine to multiply triangular matrices

        sf = self.scale_factor
        sf2 = sf**2

        #invert Kmm
        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)

        #The rather complex computations of self.A
        if self.likelihood.is_heteroscedastic:
            assert self.likelihood.D == 1 #TODO: what if the likelihood is heterscedatic and there are multiple independent outputs?
            if self.has_uncertain_inputs:
                psi2_beta_scaled = (self.psi2*(self.likelihood.precision.flatten().reshape(self.N,1,1)/sf2)).sum(0)
                evals, evecs = linalg.eigh(psi2_beta_scaled)
                clipped_evals = np.clip(evals,0.,1e6) # TODO: make clipping configurable
                if not np.allclose(evals, clipped_evals):
                    print "Warning: clipping posterior eigenvalues"
                tmp = evecs*np.sqrt(clipped_evals)
                tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(tmp),lower=1)
                self.A = tdot(tmp)
            else:
                tmp = self.psi1*(np.sqrt(self.likelihood.precision.flatten().reshape(1,self.N))/sf)
                #self.psi2_beta_scaled = tdot(tmp)
                tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(tmp),lower=1)
                self.A = tdot(tmp)
        else:
            if self.has_uncertain_inputs:
                psi2_beta_scaled = (self.psi2*(self.likelihood.precision/sf2)).sum(0)
                evals, evecs = linalg.eigh(psi2_beta_scaled)
                clipped_evals = np.clip(evals,0.,1e6) # TODO: make clipping configurable
                if not np.allclose(evals, clipped_evals):
                    print "Warning: clipping posterior eigenvalues"
                tmp = evecs*np.sqrt(clipped_evals)
                #self.psi2_beta_scaled = tdot(tmp)
                tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(tmp),lower=1)
                self.A = tdot(tmp)
            else:
                tmp = self.psi1*(np.sqrt(self.likelihood.precision)/sf)
                #self.psi2_beta_scaled = tdot(tmp)
                tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(tmp),lower=1)
                self.A = tdot(tmp)

        #invert B and compute C. C is the posterior covariance of u
        self.B = np.eye(self.M)/sf2 + self.A
        self.Bi, self.LB, self.LBi, self.B_logdet = pdinv(self.B)
        tmp = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(self.Bi),lower=1,trans=1)[0]
        self.C = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(tmp.T),lower=1,trans=1)[0]

        self.V = (self.likelihood.precision/self.scale_factor)*self.likelihood.Y
        self.psi1V = np.dot(self.psi1, self.V)

        #back substutue C into psi1V
        tmp,info1 = linalg.lapack.flapack.dtrtrs(self.Lm,np.asfortranarray(self.psi1V),lower=1,trans=0)
        self._LBi_Lmi_psi1V,_ = linalg.lapack.flapack.dtrtrs(self.LB,np.asfortranarray(tmp),lower=1,trans=0)
        self._P = tdot(tmp)
        tmp,info2 = linalg.lapack.flapack.dpotrs(self.LB,tmp,lower=1)
        self.Cpsi1V,info3 = linalg.lapack.flapack.dtrtrs(self.Lm,tmp,lower=1,trans=1)
        #self.Cpsi1V = np.dot(self.C,self.psi1V)

        self.E = tdot(self.Cpsi1V/sf)



        # Compute dL_dpsi # FIXME: this is untested for the heterscedastic + uncertin inputs case
        self.dL_dpsi0 = - 0.5 * self.D * (self.likelihood.precision * np.ones([self.N,1])).flatten()
        self.dL_dpsi1 = np.dot(self.Cpsi1V,self.V.T)
        if self.likelihood.is_heteroscedastic:
            if self.has_uncertain_inputs:
                #self.dL_dpsi2 = 0.5 * self.likelihood.precision[:,None,None] * self.D * self.Kmmi[None,:,:] # dB
                #self.dL_dpsi2 += - 0.5 * self.likelihood.precision[:,None,None]/sf2 * self.D * self.C[None,:,:] # dC
                #self.dL_dpsi2 += - 0.5 * self.likelihood.precision[:,None,None]* self.E[None,:,:] # dD
                self.dL_dpsi2 = 0.5*self.likelihood.precision[:,None,None]*(self.D*(self.Kmmi - self.C/sf2) -self.E)[None,:,:]
            else:
                #self.dL_dpsi1 += mdot(self.Kmmi,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)) #dB
                #self.dL_dpsi1 += -mdot(self.C,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)/sf2) #dC
                #self.dL_dpsi1 += -mdot(self.E,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)) #dD
                self.dL_dpsi1 += np.dot(self.Kmmi - self.C/sf2 -self.E,self.psi1*self.likelihood.precision.reshape(1,self.N))
                self.dL_dpsi2 = None

        else:
            self.dL_dpsi2 = 0.5*self.likelihood.precision*(self.D*(self.Kmmi - self.C/sf2) -self.E)
            if self.has_uncertain_inputs:
                #repeat for each of the N psi_2 matrices
                self.dL_dpsi2 = np.repeat(self.dL_dpsi2[None,:,:],self.N,axis=0)
            else:
                #subsume back into psi1 (==Kmn)
                self.dL_dpsi1 += 2.*np.dot(self.dL_dpsi2,self.psi1)
                self.dL_dpsi2 = None


        # Compute dL_dKmm
        tmp = tdot(self._LBi_Lmi_psi1V)
        self.DBi_plus_BiPBi = backsub_both_sides(self.LB, self.D*np.eye(self.M) + tmp)
        tmp = -0.5*self.DBi_plus_BiPBi/sf2
        tmp += -0.5*self.B*sf2*self.D
        tmp += self.D*np.eye(self.M)
        self.dL_dKmm = backsub_both_sides(self.Lm,tmp)

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
            self.partial_for_likelihood =   - 0.5 * self.N*self.D*self.likelihood.precision + 0.5 * self.likelihood.trYYT*self.likelihood.precision**2
            self.partial_for_likelihood += 0.5 * self.D * (self.psi0.sum()*self.likelihood.precision**2 - np.trace(self.A)*self.likelihood.precision*sf2)
            #self.partial_for_likelihood += 0.5 * self.D * trace_dot(self.Bi,self.A)*self.likelihood.precision
            #self.partial_for_likelihood += self.likelihood.precision*(0.5*trace_dot(self.psi2_beta_scaled,self.E*sf2) - np.sum(np.square(self._LBi_Lmi_psi1V)))
            self.partial_for_likelihood += self.likelihood.precision*(0.5*trace_dot(self.A,self.DBi_plus_BiPBi) - np.sum(np.square(self._LBi_Lmi_psi1V)))



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
        D = 0.5*np.sum(np.square(self._LBi_Lmi_psi1V))
        return A+B+C+D

    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        self.kern._set_params(p[self.Z.size:self.Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size+self.kern.Nparam:])
        self._compute_kernel_matrices()
        #if self.auto_scale_factor:
        #    self.scale_factor = np.sqrt(self.psi2.sum(0).mean()*self.likelihood.precision)
        #if self.auto_scale_factor:
            #if self.likelihood.is_heteroscedastic:
                #self.scale_factor = max(100,np.sqrt(self.psi2_beta_scaled.sum(0).mean()))
            #else:
                #self.scale_factor = np.sqrt(self.psi2.sum(0).mean()*self.likelihood.precision)
        self.scale_factor = 1.
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

    def _raw_predict(self, Xnew, which_parts='all', full_cov=False):
        """Internal helper function for making predictions, does not account for normalization"""

        Kx = self.kern.K(self.Z, Xnew)
        mu = mdot(Kx.T, self.C/self.scale_factor, self.psi1V)
        if full_cov:
            Kxx = self.kern.K(Xnew,which_parts=which_parts)
            var = Kxx - mdot(Kx.T, (self.Kmmi - self.C/self.scale_factor**2), Kx) #NOTE this won't work for plotting
        else:
            Kxx = self.kern.Kdiag(Xnew,which_parts=which_parts)
            var = Kxx - np.sum(Kx*np.dot(self.Kmmi - self.C/self.scale_factor**2, Kx),0)

        return mu,var[:,None]
