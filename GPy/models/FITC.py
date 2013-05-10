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


        #ERASEME
        #N = likelihood.Y.size
        #self.beta_star = np.random.rand(N,1)
        #self.Kmm_ = kernel.K(self.Z).copy()
        #self.Kmmi_,a,b,c = pdinv(self.Kmm_)
        #self.psi1_ = kernel.K(self.Z,X).copy()

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

        #TODO eraseme
        """
        self.psi1 = self.psi1_
        self.Lm = jitchol(self.Kmm_)
        self.Lmi,info = linalg.lapack.flapack.dtrtrs(self.Lm,np.eye(self.M),lower=1)
        Lmipsi1 = np.dot(self.Lmi,self.psi1)
        #self.true_psi1 = self.kern.K(self.Z,self.X)
        #self.Qnn = mdot(self.true_psi1.T,self.Lmi.T,self.Lmi,self.true_psi1)
        self.Kmmi, a,b,c = pdinv(self.Kmm)
        self.Qnn = mdot(self.psi1.T,self.Kmmi,self.psi1)
        #self.Diag0 = self.psi0 #- np.diag(self.Qnn)
        self.Diag0 = - np.diag(self.Qnn)
        #Kmmi,Lm,Lmi,logdetK = pdinv(self.Kmm)
        #self.Lambda = self.Kmmi_ + mdot(self.Kmmi_,self.psi1_,self.beta_star*self.psi1_.T,self.Kmmi_) + np.eye(self.M)*100
        #self.Lambdai, LLm, LLmi, self.logdetLambda = pdinv(self.Lambda)
        """

        #TODO uncomment
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
        self.LBi,info = linalg.lapack.flapack.dtrtrs(self.LB,np.eye(self.M),lower=1)
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
        self.dlogbeta_dtheta = dlogB_dpsi0 + dlogB_dpsi1 + dlogB_dKmm

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
        self.dyby_dtheta = dyby_dtheta

        # dlogB_dtheta : C

        #C_B
        dC_B = -.5*Kmmi
        C_B = self.kern.dK_dtheta(dC_B,self.Z) #check

        #C_A
        LBiLmi = np.dot(self.LBi,self.Lmi)
        LBL_inv = np.dot(LBiLmi.T,LBiLmi)
        dC_AA = .5*LBL_inv
        C_AA = self.kern.dK_dtheta(dC_AA,self.Z) #check

        #C_AB
        psi1beta = self.psi1*self.beta_star.T
        dC_ABA = mdot(LBL_inv,psi1beta,Kmmipsi1.T)
        C_ABA = self.kern.dK_dtheta(dC_ABA,self.Z)
        dC_ABB = -np.dot(psi1beta.T,LBL_inv) #check
        C_ABB = self.kern.dK_dtheta(dC_ABB,self.X,self.Z) #check

        # C_ABC
        betapsi1TLmiLBi = np.dot(psi1beta.T,LBiLmi.T)
        alpha = np.array([np.dot(a.T,a) for a in betapsi1TLmiLBi])[:,None]
        dC_ABCA = .5 *alpha
        C_ABCA = self.kern.dKdiag_dtheta(dC_ABCA,self.X) #check

        C_ABCB = 0
        for psi1_n,alpha_n,X_n in zip(self.psi1.T,alpha,self.X):
            dC_ABCB_n = - alpha_n * np.dot(psi1_n[None,:],Kmmi)
            C_ABCB += self.kern.dK_dtheta(dC_ABCB_n,X_n[:,None],self.Z) #check

        C_ABCC = 0
        for psi1_n,alpha_n,X_n in zip(self.psi1.T,alpha,self.X):
            psin_K = np.dot(psi1_n[None,:],Kmmi)
            tmp = np.dot(psin_K.T,psin_K)
            dC_ABCC = .5 * alpha_n * tmp
            C_ABCC += self.kern.dK_dtheta(dC_ABCC,self.Z) #check

        self.dlogB_dtheta = C_B + C_AA + C_ABA + C_ABB + C_ABCA + C_ABCB + C_ABCC


        # the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            # likelihood is not heterscedatic
            self.partial_for_likelihood = 0 #FIXME
            #self.partial_for_likelihood = -0.5 * self.N * self.D * self.likelihood.precision + 0.5 * self.likelihood.trYYT * self.likelihood.precision ** 2
            #self.partial_for_likelihood += 0.5 * self.D * (self.psi0.sum() * self.likelihood.precision ** 2 - np.trace(self.A) * self.likelihood.precision)
            #self.partial_for_likelihood += self.likelihood.precision * (0.5 * np.sum(self.A * self.DBi_plus_BiPBi) - np.sum(np.square(self._LBi_Lmi_psi1V)))



    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        A = -0.5 * self.N * self.D * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.beta_star)) - 0.5 * np.sum(self.V_star * self.likelihood.Y)
        C = -self.D * (np.sum(np.log(np.diag(self.LB))))
        """
        A = -0.5 * self.N * self.D * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.beta_star)) - 0.5 * np.sum(self.V_star * self.likelihood.Y)
        #B = -0.5 * self.D * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self.A))
        C = -self.D * (np.sum(np.log(np.diag(self.LB))))
        D = 0.5 * np.sum(np.square(self._LBi_Lmi_psi1V))
        return A + C + D # +B
        """
        return A+C


    def _log_likelihood_gradients(self):
        pass
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        #dL_dtheta = self.dlogB_dtheta
        #dL_dtheta = self.dyby_dtheta
        #dL_dtheta = self.dlogbeta_dtheta + self.dyby_dtheta
        dL_dtheta = self.dlogB_dtheta
        dL_dtheta = self.dlogbeta_dtheta + self.dyby_dtheta + self.dlogB_dtheta
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







