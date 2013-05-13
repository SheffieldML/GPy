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
        sparse_GP.__init__(self, X, likelihood, kernel=kernel, Z=Z, X_variance=None, normalize_X=False)

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
        self.Qnn = np.dot(Lmipsi1.T,Lmipsi1).copy()
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
            #tmp = self.psi1 * (np.sqrt(self.true_beta_star.flatten().reshape(1, self.N))) #TODO eraseme
            tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm, np.asfortranarray(tmp), lower=1)
            self.A = tdot(tmp)

        # factor B
        self.B = np.eye(self.M) + self.A
        self.LB = jitchol(self.B)
        self.LBi,info = linalg.lapack.flapack.dtrtrs(self.LB,np.eye(self.M),lower=1)
        self.psi1V = np.dot(self.psi1, self.V_star)
        #self.psi1V = np.dot(self.psi1, self.true_V_star) # TODO eraseme

        # back substutue C into psi1V
        tmp, info1 = linalg.lapack.flapack.dtrtrs(self.Lm, np.asfortranarray(self.psi1V), lower=1, trans=0)
        self._LBi_Lmi_psi1V, _ = linalg.lapack.flapack.dtrtrs(self.LB, np.asfortranarray(tmp), lower=1, trans=0)

        # A:
        # dlogbeta_dtheta
        Kmmipsi1 = np.dot(self.Lmi.T,Lmipsi1)
        b_psi1_Ki = self.beta_star * Kmmipsi1.T
        Ki_pbp_Ki = np.dot(Kmmipsi1,b_psi1_Ki)
        dA_dpsi0 = -0.5 * self.beta_star
        dA_dpsi1 = b_psi1_Ki
        dA_dKmm = -0.5 * np.dot(Kmmipsi1,b_psi1_Ki)

        dA_dtheta_1 =  self.kern.dKdiag_dtheta(dA_dpsi0,X=self.X) + self.kern.dK_dtheta(dA_dpsi1,self.X,self.Z) + self.kern.dK_dtheta(dA_dKmm,X=self.Z)
        dA_dX_1 =  self.kern.dK_dX(dA_dpsi1.T,self.Z,self.X) + 2. * self.kern.dK_dX(dA_dKmm,X=self.Z)

        # dyby_dtheta
        Kmmi = np.dot(self.Lmi.T,self.Lmi)
        VVT = np.outer(self.V_star,self.V_star)
        VV_p_Ki = np.dot(VVT,Kmmipsi1.T)
        Ki_pVVp_Ki = np.dot(Kmmipsi1,VV_p_Ki)
        dyby_dpsi0 = .5 * self.kern.dKdiag_dtheta(self.V_star**2,self.X)

        dA_dpsi0 = .5 * self.V_star**2
        dA_dpsi0_theta = self.kern.dKdiag_dtheta(dA_dpsi0,X=self.X)

        dA_dpsi1_theta = 0
        dA_dpsi1_X = 0
        for psi1_n,V_n,X_n in zip(self.psi1.T,self.V_star,self.X):
            dA_dpsi1 = -V_n**2 * np.dot(psi1_n[None,:],Kmmi)
            dA_dpsi1_theta += self.kern.dK_dtheta(dA_dpsi1,X_n[None,:],self.Z)
            dA_dpsi1_X += self.kern.dK_dX(dA_dpsi1.T,self.Z,X_n[None,:])

        dA_dKmm_theta = 0
        dA_dKmm_X = 0
        for psi1_n,V_n,X_n in zip(self.psi1.T,self.V_star,self.X):
            psin_K = np.dot(psi1_n[None,:],Kmmi)
            tmp = np.dot(psin_K.T,psin_K)
            dA_dKmm = .5*V_n**2 * tmp
            dA_dKmm_theta += self.kern.dK_dtheta(dA_dKmm,self.Z)
            dA_dKmm_X += 2.*self.kern.dK_dX(dA_dKmm,self.Z)

        dA_dtheta_2 =  dA_dpsi0_theta + dA_dpsi1_theta + dA_dKmm_theta
        dA_dX_2 =  dA_dpsi1_X + dA_dKmm_X

        self.dA_dtheta = dA_dtheta_1 + dA_dtheta_2
        self.dA_dX = dA_dX_1 + dA_dX_2

        # dlogB_dtheta : C
        #C_B
        dC_dKmm = -.5*Kmmi

        #C_A
        LBiLmi = np.dot(self.LBi,self.Lmi)
        LBL_inv = np.dot(LBiLmi.T,LBiLmi)
        dC_dKmm += .5*LBL_inv

        #C_AB
        psi1beta = self.psi1*self.beta_star.T
        dC_dKmm += mdot(LBL_inv,psi1beta,Kmmipsi1.T)
        dC_dpsi1 = -np.dot(psi1beta.T,LBL_inv)

        # C_ABC
        betapsi1TLmiLBi = np.dot(psi1beta.T,LBiLmi.T)
        alpha = np.array([np.dot(a.T,a) for a in betapsi1TLmiLBi])[:,None]
        dC_dpsi0 = .5 *alpha

        _dC_dpsi1_dtheta = 0
        _dC_dpsi1_dX = 0
        for psi1_n,alpha_n,X_n in zip(self.psi1.T,alpha,self.X):
            _dC_dpsi1 = - alpha_n * np.dot(psi1_n[None,:],Kmmi)
            _dC_dpsi1_dtheta += self.kern.dK_dtheta(_dC_dpsi1,X_n[None,:],self.Z) #check
            _dC_dpsi1_dX += self.kern.dK_dX(_dC_dpsi1.T,self.Z,X_n[None,:])

        _dC_dKmm_dtheta = 0
        _dC_dKmm_dX = 0
        for psi1_n,alpha_n,X_n in zip(self.psi1.T,alpha,self.X):
            psin_K = np.dot(psi1_n[None,:],Kmmi)
            _dC_dKmm = .5 * alpha_n * np.dot(psin_K.T,psin_K)
            _dC_dKmm_dtheta += self.kern.dK_dtheta(_dC_dKmm,self.Z) #check
            _dC_dKmm_dX += 2.*self.kern.dK_dX(_dC_dKmm,self.Z)

        #self.dlogB_dtheta = dCB_dKmm_theta + dCAA_dKmm_theta + dCABA_dKmm_theta + dCABB_dpsi1_theta + dCABCA_dpsi0_theta + dCABCB_dpsi1_theta + dCABCC_dKmm_theta
        self.dlogB_dtheta = self.kern.dK_dtheta(dC_dKmm,self.Z) + self.kern.dK_dtheta(dC_dpsi1,self.X,self.Z) + self.kern.dKdiag_dtheta(dC_dpsi0,self.X) + _dC_dpsi1_dtheta + _dC_dKmm_dtheta
        self.dlogB_dX = 2.*self.kern.dK_dX(dC_dKmm,self.Z) + self.kern.dK_dX(dC_dpsi1.T,self.Z,self.X) + _dC_dpsi1_dX + _dC_dKmm_dX

        # dD_dtheta
        H = self.Kmm + mdot(self.psi1,self.beta_star*self.psi1.T)
        Hi, LH, LHi, logdetH = pdinv(H)

        # D_B
        gamma_1 = mdot(VVT,self.psi1.T,Hi)
        dD_B = gamma_1
        D_B = self.kern.dK_dtheta(dD_B,self.X,self.Z)

        dD_dpsi1 = gamma_1

        # D_C
        dD_CA = -.5 * mdot(Hi,self.psi1,gamma_1)
        D_CA = self.kern.dK_dtheta(dD_CA,self.Z)

        dD_dKmm = -.5 * mdot(Hi,self.psi1,gamma_1)

        # D_CB
        dD_CBA = - mdot(psi1beta.T,Hi,self.psi1,gamma_1)
        D_CBA = self.kern.dK_dtheta(dD_CBA,self.X,self.Z)

        dD_dpsi1 += -mdot(psi1beta.T,Hi,self.psi1,gamma_1)

        # D_CBB
        pHip = mdot(self.psi1.T,Hi,self.psi1)
        gamma_2 = mdot(self.beta_star*pHip,self.V_star)
        D_CBBA = .5 * self.kern.dKdiag_dtheta(gamma_2**2,self.X)

        dD_dpsi0 = 0.5*mdot(self.beta_star*pHip,self.V_star)**2

        _dD_dpsi1_dtheta_1 = 0
        _dD_dpsi1_dX_1 = 0
        for psi1_n,gamma_n,X_n in zip(self.psi1.T,gamma_2,self.X):
            _dD_dpsi1 = - gamma_n**2 * np.dot(psi1_n[None,:],Kmmi)
            _dD_dpsi1_dtheta_1 += self.kern.dK_dtheta(_dD_dpsi1,X_n[None,:],self.Z)
            _dD_dpsi1_dX_1 += self.kern.dK_dX(_dD_dpsi1.T,self.Z,X_n[None,:])

        _dD_dKmm_dtheta_1 = 0
        _dD_dKmm_dX_1 = 0
        for psi1_n,gamma_n,X_n in zip(self.psi1.T,gamma_2,self.X):
            psin_K = np.dot(psi1_n[None,:],Kmmi)
            _dD_dKmm = .5*gamma_n**2 * np.dot(psin_K.T,psin_K)
            _dD_dKmm_dtheta_1 += self.kern.dK_dtheta(_dD_dKmm,self.Z)
            _dD_dKmm_dX_1 += 2.*self.kern.dK_dX(_dD_dKmm,self.Z)

        # D_A
        gamma_3 = self.V_star * mdot(self.V_star.T,pHip*self.beta_star).T
        dD_AA = - gamma_3
        D_AA = self.kern.dKdiag_dtheta(dD_AA,self.X)

        dD_dpsi0 += -self.V_star * mdot(self.V_star.T,pHip*self.beta_star).T

        _dD_dpsi1_dtheta_2 = 0
        _dD_dpsi1_dX_2 = 0
        for psi1_n,gamma_n,X_n in zip(self.psi1.T,gamma_3,self.X):
            _dD_dpsi = 2. * gamma_n * np.dot(psi1_n[None,:],Kmmi)
            _dD_dpsi1_dtheta_2 += self.kern.dK_dtheta(_dD_dpsi,X_n[None,:],self.Z)
            _dD_dpsi1_dX_2 += self.kern.dK_dX(_dD_dpsi.T,self.Z,X_n[None,:])

        _dD_dKmm_dtheta_2 = 0
        _dD_dKmm_dX_2 = 0
        for psi1_n,gamma_n,X_n in zip(self.psi1.T,gamma_3,self.X):
            psin_K = np.dot(psi1_n[None,:],Kmmi)
            tmp = np.dot(psin_K.T,psin_K)
            dD_AC = - gamma_n * tmp
            _dD_dKmm_dtheta_2 += self.kern.dK_dtheta(dD_AC,self.Z)
            _dD_dKmm_dX_2 += 2.*self.kern.dK_dX(dD_AC,self.Z)

        self.dD_dtheta =  D_AA + _dD_dpsi1_dtheta_2 + _dD_dKmm_dtheta_2 + D_B + D_CA + D_CBA + D_CBBA + _dD_dpsi1_dtheta_1 + _dD_dKmm_dtheta_1
        self.dD_dtheta =  self.kern.dKdiag_dtheta(dD_dpsi0,self.X) + self.kern.dK_dtheta(dD_dKmm,self.Z) + self.kern.dK_dtheta(dD_dpsi1.T,self.Z,self.X) + _dD_dpsi1_dtheta_2 + _dD_dKmm_dtheta_2 +  _dD_dpsi1_dtheta_1 + _dD_dKmm_dtheta_1
        self.dD_dX =  2.*self.kern.dK_dX(dD_dKmm,self.Z) + self.kern.dK_dX(dD_dpsi1.T,self.Z,self.X) + _dD_dpsi1_dX_2 + _dD_dKmm_dX_2 + _dD_dpsi1_dX_1 + _dD_dKmm_dX_1


        # the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            # likelihood is not heterscedatic
            dbstar_dnoise = self.likelihood.precision * (self.beta_star**2 * self.Diag0[:,None] - self.beta_star) #check
            #dbstar_dnoise = self.likelihood.precision * (self.true_beta_star**2 * self.Diag0[:,None] - self.true_beta_star) #TODO erase
            Lmi_psi1 = mdot(self.Lmi,self.psi1)
            LBiLmipsi1 = np.dot(self.LBi,Lmi_psi1)
            aux_0 = np.dot(self._LBi_Lmi_psi1V.T,LBiLmipsi1)
            aux_1 = self.likelihood.Y.T * np.dot(self._LBi_Lmi_psi1V.T,LBiLmipsi1)
            aux_2 = np.dot(LBiLmipsi1.T,self._LBi_Lmi_psi1V)

            dA_dnoise = 0.5 * self.D * (dbstar_dnoise/self.beta_star).sum() - 0.5 * self.D * np.sum(self.likelihood.Y**2 * dbstar_dnoise) # check
            dC_dnoise = -0.5 * np.sum(mdot(self.LBi.T,self.LBi,Lmi_psi1) *  Lmi_psi1 * dbstar_dnoise.T) #check
            dC_dnoise = -0.5 * np.sum(mdot(self.LBi.T,self.LBi,Lmi_psi1) *  Lmi_psi1 * dbstar_dnoise.T) #check

            dD_dnoise_1 =  mdot(self.V_star*LBiLmipsi1.T,LBiLmipsi1*dbstar_dnoise.T*self.likelihood.Y.T) #check
            #dD_dnoise_1 =  mdot(self.true_V_star*LBiLmipsi1.T,LBiLmipsi1*dbstar_dnoise.T*self.likelihood.Y.T) #TODO eraseme

            alpha = mdot(LBiLmipsi1,self.V_star)
            alpha_ = mdot(LBiLmipsi1.T,alpha)
            dD_dnoise_2 = -0.5 * self.D * np.sum(alpha_**2 * dbstar_dnoise ) #check

            dD_dnoise = dD_dnoise_1 + dD_dnoise_2

            self.partial_for_likelihood = dA_dnoise + dC_dnoise + dD_dnoise

    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        A = -0.5 * self.N * self.D * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.beta_star)) - 0.5 * np.sum(self.V_star * self.likelihood.Y)
        #B = -0.5 * self.D * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self.A))
        C = -self.D * (np.sum(np.log(np.diag(self.LB))))
        D = 0.5 * np.sum(np.square(self._LBi_Lmi_psi1V))
        return A + C + D

    def _log_likelihood_gradients(self):
        pass
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        if self.has_uncertain_inputs:
            raise NotImplementedError, "FITC approximation not implemented for uncertain inputs"
        else:
            #dL_dtheta = dL_dtheta = self.dlogbeta_dtheta + self.dyby_dtheta #+ self.dlogB_dtheta + self.dD_dtheta
            dL_dtheta = self.dA_dtheta + self.dlogB_dtheta + self.dD_dtheta
        return dL_dtheta

    def dL_dZ(self):
        if self.has_uncertain_inputs:
            raise NotImplementedError, "FITC approximation not implemented for uncertain inputs"
        else:
            dL_dZ = self.dA_dX + self.dlogB_dX + self.dD_dX
        return dL_dZ

    def _raw_predict(self, Xnew, which_parts, full_cov=False):


        Iplus_Dprod_i = 1./(1.+ self.Diag0 * self.likelihood.precision.flatten())
        self.Diag = self.Diag0 * Iplus_Dprod_i
        self.P = Iplus_Dprod_i[:,None] * self.psi1.T
        self.RPT0 = np.dot(self.Lmi,self.psi1)
        self.L = np.linalg.cholesky(np.eye(self.M) + np.dot(self.RPT0,((1. - Iplus_Dprod_i)/self.Diag0)[:,None]*self.RPT0.T))
        self.R,info = linalg.flapack.dtrtrs(self.L,self.Lmi,lower=1)
        self.RPT = np.dot(self.R,self.P.T)
        self.Sigma = np.diag(self.Diag) + np.dot(self.RPT.T,self.RPT)
        self.w = self.Diag * self.likelihood.v_tilde
        self.gamma = np.dot(self.R.T, np.dot(self.RPT,self.likelihood.v_tilde))
        self.mu = self.w + np.dot(self.P,self.gamma)

        if self.likelihood.is_heteroscedastic:
            """
            Make a prediction for the generalized FITC model

            Arguments
            ---------
            X : Input prediction data - Nx1 numpy array (floats)
            """
            # q(u|f) = N(u| R0i*mu_u*f, R0i*C*R0i.T)

            # Ci = I + (RPT0)Di(RPT0).T
            # C = I - [RPT0] * (D+[RPT0].T*[RPT0])^-1*[RPT0].T
            #   = I - [RPT0] * (D + self.Qnn)^-1 * [RPT0].T
            #   = I - [RPT0] * (U*U.T)^-1 * [RPT0].T
            #   = I - V.T * V
            U = np.linalg.cholesky(np.diag(self.Diag0) + self.Qnn)
            V,info = linalg.flapack.dtrtrs(U,self.RPT0.T,lower=1)
            C = np.eye(self.M) - np.dot(V.T,V)
            mu_u = np.dot(C,self.RPT0)*(1./self.Diag0[None,:])
            #self.C = C
            #self.RPT0 = np.dot(self.R0,self.Knm.T) P0.T
            #self.mu_u = mu_u
            #self.U = U
            # q(u|y) = N(u| R0i*mu_H,R0i*Sigma_H*R0i.T)
            mu_H = np.dot(mu_u,self.mu)
            self.mu_H = mu_H
            Sigma_H = C + np.dot(mu_u,np.dot(self.Sigma,mu_u.T))
            # q(f_star|y) = N(f_star|mu_star,sigma2_star)
            Kx = self.kern.K(self.Z, Xnew, which_parts=which_parts)
            KR0T = np.dot(Kx.T,self.Lmi.T)
            mu_star = np.dot(KR0T,mu_H)
            if full_cov:
                Kxx = self.kern.K(Xnew,which_parts=which_parts)
                var = Kxx + np.dot(KR0T,np.dot(Sigma_H - np.eye(self.M),KR0T.T))
            else:
                Kxx = self.kern.Kdiag(Xnew,which_parts=which_parts)
                Kxx_ = self.kern.K(Xnew,which_parts=which_parts) # TODO: RA, is this line needed?
                var_ = Kxx_ + np.dot(KR0T,np.dot(Sigma_H - np.eye(self.M),KR0T.T)) # TODO: RA, is this line needed?
                var = (Kxx + np.sum(KR0T.T*np.dot(Sigma_H - np.eye(self.M),KR0T.T),0))[:,None]
            return mu_star[:,None],var
        else:
            raise NotImplementedError, "homoscedastic fitc not implemented"
            """
            Kx = self.kern.K(self.Z, Xnew)
            mu = mdot(Kx.T, self.C/self.scale_factor, self.psi1V)
            if full_cov:
                Kxx = self.kern.K(Xnew)
                var = Kxx - mdot(Kx.T, (self.Kmmi - self.C/self.scale_factor**2), Kx) #NOTE this won't work for plotting
            else:
                Kxx = self.kern.Kdiag(Xnew)
                var = Kxx - np.sum(Kx*np.dot(self.Kmmi - self.C/self.scale_factor**2, Kx),0)
            return mu,var[:,None]
            """
