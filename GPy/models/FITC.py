# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, tdot, symmetrify,pdinv
#from ..util.linalg import mdot, jitchol, chol_inv, pdinv, trace_dot
from ..util.plot import gpplot
from .. import kern
from scipy import stats, linalg
from sparse_GP import sparse_GP

def backsub_both_sides(L,X):
    """ Return L^-T * X * L^-1, assumuing X is symmetrical and L is lower cholesky"""
    tmp,_ = linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(X),lower=1,trans=1)
    return linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(tmp.T),lower=1,trans=1)[0].T

class FITC(sparse_GP):
    def __init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False):

        self.Z = Z
        self.M = self.Z.shape[0]
        self.true_precision = likelihood.precision
        sparse_GP.__init__(self, X, likelihood, kernel=kernel, Z=self.Z, X_variance=None, normalize_X=False)

    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        self.kern._set_params(p[self.Z.size:self.Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size+self.kern.Nparam:])
        self._compute_kernel_matrices()
        self.scale_factor = 1.
        self._computations()

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian (or direct: TODO) likelihood, no iteration is required:
        this function does nothing

        Diag(Knn - Qnn) is added to the noise term to use the tools already implemented in sparse_GP.
        The true precison is now 'true_precision' not 'precision'.
        """
        if self.has_uncertain_inputs:
            raise NotImplementedError, "FITC approximation not implemented for uncertain inputs"
        else:
            self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
            self._set_params(self._get_params()) # update the GP

    def _computations(self):
        #factor Kmm
        self.Lm = jitchol(self.Kmm)

        self.Lmi,info = linalg.lapack.flapack.dtrtrs(self.Lm,np.eye(self.M),lower=1)
        Lmipsi1 = np.dot(self.Lmi,self.psi1)
        self.Qnn = np.dot(Lmipsi1.T,Lmipsi1)
        self.Diag0 = self.psi0 - np.diag(self.Qnn)



        self.beta_star = self.likelihood.precision/(1. + self.likelihood.precision*self.Diag0[:,None]) #Includes Diag0 in the precision
        self.V_star = self.beta_star * self.likelihood.Y

        # The rather complex computations of self.A
        if self.has_uncertain_inputs:
                raise NotImplementedError
        else:
            if self.likelihood.is_heteroscedastic:
                assert self.likelihood.D == 1  # TODO: what if the likelihood is heterscedatic and there are multiple independent outputs?
            tmp = self.psi1 * (np.sqrt(self.beta_star.flatten().reshape(1, self.N)))
            tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm, np.asfortranarray(tmp), lower=1)
            self.A = tdot(tmp)





        # factor B
        self.B = np.eye(self.M) + self.A
        self.LB = jitchol(self.B)

        self.psi1V = np.dot(self.psi1, self.V_star)

        # back substutue C into psi1V
        tmp, info1 = linalg.lapack.flapack.dtrtrs(self.Lm, np.asfortranarray(self.psi1V), lower=1, trans=0)
        self._LBi_Lmi_psi1V, _ = linalg.lapack.flapack.dtrtrs(self.LB, np.asfortranarray(tmp), lower=1, trans=0)


        # dlogbeta_dtheta
        Kmmipsi1 = np.dot(self.Lmi.T,Lmipsi1)
        b_psi1_Ki = self.beta_star * Kmmipsi1.T
        Ki_pbp_Ki = np.dot(Kmmipsi1,b_psi1_Ki)
        dlogB_dpsi0 = -.5*self.kern.dKdiag_dtheta(self.beta_star,X=self.X)
        dlogB_dpsi1 = self.kern.dK_dtheta(b_psi1_Ki,self.X,self.Z)
        dlogB_dKmm = -.5*self.kern.dK_dtheta(Ki_pbp_Ki,X=self.Z)

        self.dlogB_dtheta = dlogB_dpsi0 + dlogB_dpsi1 + dlogB_dKmm


        # dyby_dtheta
        Kmmi = np.dot(self.Lmi.T,self.Lmi)
        VVT = np.outer(self.V_star,self.V_star)
        VV_p_Ki = np.dot(VVT,Kmmipsi1.T)
        Ki_pVVp_Ki = np.dot(Kmmipsi1,VV_p_Ki)
        dyby_dpsi0 = .5 * self.kern.dKdiag_dtheta(self.V_star**2,self.X)

        dyby_dpsi1 = 0
        dyby_dKmm = 0
        dyby_dtheta = dyby_dpsi0
        for psi1_n,V_n,X_n in zip(self.psi1.T,self.V_star,self.X):
            dyby_dpsi1 = -V_n**2 * np.dot(psi1_n[None,:],Kmmi)
            dyby_dtheta += self.kern.dK_dtheta(dyby_dpsi1,X_n[:,None],self.Z)

        for psi1_n,V_n,X_n in zip(self.psi1.T,self.V_star,self.X):
            psin_K = np.dot(psi1_n[None,:],Kmmi)
            tmp = np.dot(psin_K.T,psin_K)
            dyby_dKmm = .5*V_n**2 * tmp
            dyby_dtheta += self.kern.dK_dtheta(dyby_dKmm,self.Z)


        #self.dyby_dtheta = dyby_dpsi0 #+ dyby_dpsi1 + dyby_dKmm
        self.dyby_dtheta = dyby_dtheta



        # the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            # likelihood is not heterscedatic
            self.partial_for_likelihood = 0
            #self.partial_for_likelihood = -0.5 * self.N * self.D * self.likelihood.precision + 0.5 * self.likelihood.trYYT * self.likelihood.precision ** 2
            #self.partial_for_likelihood += 0.5 * self.D * (self.psi0.sum() * self.likelihood.precision ** 2 - np.trace(self.A) * self.likelihood.precision)
            #self.partial_for_likelihood += self.likelihood.precision * (0.5 * np.sum(self.A * self.DBi_plus_BiPBi) - np.sum(np.square(self._LBi_Lmi_psi1V)))



        """
        tmp, info2 = linalg.lapack.flapack.dpotrs(self.LB, tmp, lower=1)
        self.Cpsi1V, info3 = linalg.lapack.flapack.dtrtrs(self.Lm, tmp, lower=1, trans=1)

        # Compute dL_dKmm
        tmp = tdot(self._LBi_Lmi_psi1V)
        self.DBi_plus_BiPBi = backsub_both_sides(self.LB, self.D * np.eye(self.M) + tmp)
        tmp = -0.5 * self.DBi_plus_BiPBi
        tmp += -0.5 * self.B * self.D
        tmp += self.D * np.eye(self.M)
        self.dL_dKmm = backsub_both_sides(self.Lm, tmp)

        # Compute dL_dpsi # FIXME: this is untested for the heterscedastic + uncertain inputs case
        self.dL_dpsi0 = -0.5 * self.D * (self.likelihood.precision * np.ones([self.N, 1])).flatten()
        self.dL_dpsi1 = np.dot(self.Cpsi1V, self.likelihood.V.T)
        dL_dpsi2_beta = 0.5 * backsub_both_sides(self.Lm, self.D * np.eye(self.M) - self.DBi_plus_BiPBi)
        if self.likelihood.is_heteroscedastic:
            if self.has_uncertain_inputs:
                self.dL_dpsi2 = self.likelihood.precision[:, None, None] * dL_dpsi2_beta[None, :, :]
            else:
                self.dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, self.psi1 * self.likelihood.precision.reshape(1, self.N))
                self.dL_dpsi2 = None
        else:
            dL_dpsi2 = self.likelihood.precision * dL_dpsi2_beta
            if self.has_uncertain_inputs:
                # repeat for each of the N psi_2 matrices
                self.dL_dpsi2 = np.repeat(dL_dpsi2[None, :, :], self.N, axis=0)
            else:
                # subsume back into psi1 (==Kmn)
                self.dL_dpsi1 += 2.*np.dot(dL_dpsi2, self.psi1)
                self.dL_dpsi2 = None


        # the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            # likelihood is not heterscedatic
            self.partial_for_likelihood = -0.5 * self.N * self.D * self.likelihood.precision + 0.5 * self.likelihood.trYYT * self.likelihood.precision ** 2
            self.partial_for_likelihood += 0.5 * self.D * (self.psi0.sum() * self.likelihood.precision ** 2 - np.trace(self.A) * self.likelihood.precision)
            self.partial_for_likelihood += self.likelihood.precision * (0.5 * np.sum(self.A * self.DBi_plus_BiPBi) - np.sum(np.square(self._LBi_Lmi_psi1V)))

    """
    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        #A = -0.5 * self.N * self.D * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.beta_star))
        A = - 0.5 * np.sum(self.V_star * self.likelihood.Y)
        """
        A = -0.5 * self.N * self.D * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.beta_star)) - 0.5 * np.sum(self.V_star * self.likelihood.Y)
        #B = -0.5 * self.D * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self.A))
        C = -self.D * (np.sum(np.log(np.diag(self.LB))))
        D = 0.5 * np.sum(np.square(self._LBi_Lmi_psi1V))
        return A + C + D # +B
        """
        return A


    def _log_likelihood_gradients(self):
        pass
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        #dL_dtheta = self.dlogB_dtheta
        dL_dtheta = self.dyby_dtheta
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z, self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T, self.Z, self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dtheta += self.kern.dK_dtheta(self.dL_dpsi1, self.Z, self.X)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)
        """
        return dL_dtheta

    def dL_dZ(self):
        dL_dZ = np.zeros(self.M)
        """
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm, self.Z)  # factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1, self.Z, self.X, self.X_variance)
            dL_dZ += self.kern.dpsi2_dZ(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dZ += self.kern.dK_dX(self.dL_dpsi1, self.Z, self.X)
        """
        return dL_dZ







