# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, tdot, symmetrify, pdinv, dtrtrs
from ..util.plot import gpplot
from .. import kern
from scipy import stats
from sparse_gp import SparseGP

class FITC(SparseGP):
    """
    sparse FITC approximation

    :param X: inputs
    :type X: np.ndarray (num_data x Q)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP)
    :param kernel : the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, likelihood, kernel, Z, normalize_X=False):
        SparseGP.__init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False)
        assert self.output_dim == 1, "FITC model is not defined for handling multiple outputs"

    def update_likelihood_approximation(self):
        """
        Approximates a non-Gaussian likelihood using Expectation Propagation

        For a Gaussian likelihood, no iteration is required:
        this function does nothing
        """
        self.likelihood.restart()
        self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
        self._set_params(self._get_params())

    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        self.psi0 = self.kern.Kdiag(self.X)
        self.psi1 = self.kern.K(self.Z, self.X)
        self.psi2 = None

    def _computations(self):
        #factor Kmm
        self.Lm = jitchol(self.Kmm)
        self.Lmi,info = dtrtrs(self.Lm,np.eye(self.num_inducing),lower=1)
        Lmipsi1 = np.dot(self.Lmi,self.psi1)
        self.Qnn = np.dot(Lmipsi1.T,Lmipsi1).copy()
        self.Diag0 = self.psi0 - np.diag(self.Qnn)
        self.beta_star = self.likelihood.precision/(1. + self.likelihood.precision*self.Diag0[:,None]) #NOTE: beta_star contains Diag0 and the precision
        self.V_star = self.beta_star * self.likelihood.Y

        # The rather complex computations of self.A
        tmp = self.psi1 * (np.sqrt(self.beta_star.flatten().reshape(1, self.num_data)))
        tmp, _ = dtrtrs(self.Lm, np.asfortranarray(tmp), lower=1)
        self.A = tdot(tmp)

        # factor B
        self.B = np.eye(self.num_inducing) + self.A
        self.LB = jitchol(self.B)
        self.LBi = chol_inv(self.LB)
        self.psi1V = np.dot(self.psi1, self.V_star)

        Lmi_psi1V, info = dtrtrs(self.Lm, np.asfortranarray(self.psi1V), lower=1, trans=0)
        self._LBi_Lmi_psi1V, _ = dtrtrs(self.LB, np.asfortranarray(Lmi_psi1V), lower=1, trans=0)

        Kmmipsi1 = np.dot(self.Lmi.T,Lmipsi1)
        b_psi1_Ki = self.beta_star * Kmmipsi1.T
        Ki_pbp_Ki = np.dot(Kmmipsi1,b_psi1_Ki)
        Kmmi = np.dot(self.Lmi.T,self.Lmi)
        LBiLmi = np.dot(self.LBi,self.Lmi)
        LBL_inv = np.dot(LBiLmi.T,LBiLmi)
        VVT = np.outer(self.V_star,self.V_star)
        VV_p_Ki = np.dot(VVT,Kmmipsi1.T)
        Ki_pVVp_Ki = np.dot(Kmmipsi1,VV_p_Ki)
        psi1beta = self.psi1*self.beta_star.T
        H = self.Kmm + mdot(self.psi1,psi1beta.T)
        LH = jitchol(H)
        LHi = chol_inv(LH)
        Hi = np.dot(LHi.T,LHi)

        betapsi1TLmiLBi = np.dot(psi1beta.T,LBiLmi.T)
        alpha = np.array([np.dot(a.T,a) for a in betapsi1TLmiLBi])[:,None]
        gamma_1 = mdot(VVT,self.psi1.T,Hi)
        pHip = mdot(self.psi1.T,Hi,self.psi1)
        gamma_2 = mdot(self.beta_star*pHip,self.V_star)
        gamma_3 = self.V_star * gamma_2

        self._dL_dpsi0 = -0.5 * self.beta_star#dA_dpsi0: logdet(self.beta_star)
        self._dL_dpsi0 += .5 * self.V_star**2 #dA_psi0: yT*beta_star*y
        self._dL_dpsi0 += .5 *alpha #dC_dpsi0
        self._dL_dpsi0 += 0.5*mdot(self.beta_star*pHip,self.V_star)**2 - self.V_star * mdot(self.V_star.T,pHip*self.beta_star).T #dD_dpsi0

        self._dL_dpsi1 = b_psi1_Ki.copy() #dA_dpsi1: logdet(self.beta_star)
        self._dL_dpsi1 += -np.dot(psi1beta.T,LBL_inv) #dC_dpsi1
        self._dL_dpsi1 += gamma_1 - mdot(psi1beta.T,Hi,self.psi1,gamma_1) #dD_dpsi1

        self._dL_dKmm = -0.5 * np.dot(Kmmipsi1,b_psi1_Ki) #dA_dKmm: logdet(self.beta_star)
        self._dL_dKmm += .5*(LBL_inv - Kmmi) + mdot(LBL_inv,psi1beta,Kmmipsi1.T) #dC_dKmm
        self._dL_dKmm += -.5 * mdot(Hi,self.psi1,gamma_1) #dD_dKmm

        self._dpsi1_dtheta = 0
        self._dpsi1_dX = 0
        self._dKmm_dtheta = 0
        self._dKmm_dX = 0

        self._dpsi1_dX_jkj = 0
        self._dpsi1_dtheta_jkj = 0

        for i,V_n,alpha_n,gamma_n,gamma_k in zip(range(self.num_data),self.V_star,alpha,gamma_2,gamma_3):
            K_pp_K = np.dot(Kmmipsi1[:,i:(i+1)],Kmmipsi1[:,i:(i+1)].T)
            _dpsi1 = (-V_n**2 - alpha_n + 2.*gamma_k - gamma_n**2) * Kmmipsi1.T[i:(i+1),:]
            _dKmm = .5*(V_n**2 + alpha_n + gamma_n**2 - 2.*gamma_k) * K_pp_K #Diag_dD_dKmm
            self._dpsi1_dtheta += self.kern.dK_dtheta(_dpsi1,self.X[i:i+1,:],self.Z)
            self._dKmm_dtheta += self.kern.dK_dtheta(_dKmm,self.Z)
            self._dKmm_dX += 2.*self.kern.dK_dX(_dKmm ,self.Z)
            self._dpsi1_dX += self.kern.dK_dX(_dpsi1.T,self.Z,self.X[i:i+1,:])

        # the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented."
        else:
            # likelihood is not heterscedatic
            dbstar_dnoise = self.likelihood.precision * (self.beta_star**2 * self.Diag0[:,None] - self.beta_star)
            Lmi_psi1 = mdot(self.Lmi,self.psi1)
            LBiLmipsi1 = np.dot(self.LBi,Lmi_psi1)
            aux_0 = np.dot(self._LBi_Lmi_psi1V.T,LBiLmipsi1)
            aux_1 = self.likelihood.Y.T * np.dot(self._LBi_Lmi_psi1V.T,LBiLmipsi1)
            aux_2 = np.dot(LBiLmipsi1.T,self._LBi_Lmi_psi1V)

            dA_dnoise = 0.5 * self.input_dim * (dbstar_dnoise/self.beta_star).sum() - 0.5 * self.input_dim * np.sum(self.likelihood.Y**2 * dbstar_dnoise)
            dC_dnoise = -0.5 * np.sum(mdot(self.LBi.T,self.LBi,Lmi_psi1) *  Lmi_psi1 * dbstar_dnoise.T)
            dC_dnoise = -0.5 * np.sum(mdot(self.LBi.T,self.LBi,Lmi_psi1) *  Lmi_psi1 * dbstar_dnoise.T)

            dD_dnoise_1 =  mdot(self.V_star*LBiLmipsi1.T,LBiLmipsi1*dbstar_dnoise.T*self.likelihood.Y.T)
            alpha = mdot(LBiLmipsi1,self.V_star)
            alpha_ = mdot(LBiLmipsi1.T,alpha)
            dD_dnoise_2 = -0.5 * self.input_dim * np.sum(alpha_**2 * dbstar_dnoise )

            dD_dnoise_1 = mdot(self.V_star.T,self.psi1.T,self.Lmi.T,self.LBi.T,self.LBi,self.Lmi,self.psi1,dbstar_dnoise*self.likelihood.Y)
            dD_dnoise_2 = 0.5*mdot(self.V_star.T,self.psi1.T,Hi,self.psi1,dbstar_dnoise*self.psi1.T,Hi,self.psi1,self.V_star)
            dD_dnoise = dD_dnoise_1 + dD_dnoise_2

            self.partial_for_likelihood = dA_dnoise + dC_dnoise + dD_dnoise

    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        A = -0.5 * self.num_data * self.output_dim * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.beta_star)) - 0.5 * np.sum(self.V_star * self.likelihood.Y)
        C = -self.output_dim * (np.sum(np.log(np.diag(self.LB))))
        D = 0.5 * np.sum(np.square(self._LBi_Lmi_psi1V))
        return A + C + D

    def _log_likelihood_gradients(self):
        pass
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        dL_dtheta = self.kern.dKdiag_dtheta(self._dL_dpsi0,self.X)
        dL_dtheta += self.kern.dK_dtheta(self._dL_dpsi1,self.X,self.Z)
        dL_dtheta += self.kern.dK_dtheta(self._dL_dKmm,X=self.Z)
        dL_dtheta += self._dKmm_dtheta
        dL_dtheta += self._dpsi1_dtheta
        return dL_dtheta

    def dL_dZ(self):
        dL_dZ = self.kern.dK_dX(self._dL_dpsi1.T,self.Z,self.X)
        dL_dZ += 2. * self.kern.dK_dX(self._dL_dKmm,X=self.Z)
        dL_dZ += self._dpsi1_dX
        dL_dZ += self._dKmm_dX
        return dL_dZ

    def _raw_predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False):
        assert X_variance_new is None, "FITC model is not defined for handling uncertain inputs."

        if self.likelihood.is_heteroscedastic:
            Iplus_Dprod_i = 1./(1.+ self.Diag0 * self.likelihood.precision.flatten())
            self.Diag = self.Diag0 * Iplus_Dprod_i
            self.P = Iplus_Dprod_i[:,None] * self.psi1.T
            self.RPT0 = np.dot(self.Lmi,self.psi1)
            self.L = np.linalg.cholesky(np.eye(self.num_inducing) + np.dot(self.RPT0,((1. - Iplus_Dprod_i)/self.Diag0)[:,None]*self.RPT0.T))
            self.R,info = dtrtrs(self.L,self.Lmi,lower=1)
            self.RPT = np.dot(self.R,self.P.T)
            self.Sigma = np.diag(self.Diag) + np.dot(self.RPT.T,self.RPT)
            self.w = self.Diag * self.likelihood.v_tilde
            self.Gamma = np.dot(self.R.T, np.dot(self.RPT,self.likelihood.v_tilde))
            self.mu = self.w + np.dot(self.P,self.Gamma)

            """
            Make a prediction for the generalized FITC model

            Arguments
            ---------
            X : Input prediction data - Nx1 numpy array (floats)
            """
            # q(u|f) = N(u| R0i*mu_u*f, R0i*C*R0i.T)

            # Ci = I + (RPT0)Di(RPT0).T
            # C = I - [RPT0] * (input_dim+[RPT0].T*[RPT0])^-1*[RPT0].T
            #   = I - [RPT0] * (input_dim + self.Qnn)^-1 * [RPT0].T
            #   = I - [RPT0] * (U*U.T)^-1 * [RPT0].T
            #   = I - V.T * V
            U = np.linalg.cholesky(np.diag(self.Diag0) + self.Qnn)
            V,info = dtrtrs(U,self.RPT0.T,lower=1)
            C = np.eye(self.num_inducing) - np.dot(V.T,V)
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
                var = Kxx + np.dot(KR0T,np.dot(Sigma_H - np.eye(self.num_inducing),KR0T.T))
            else:
                Kxx = self.kern.Kdiag(Xnew,which_parts=which_parts)
                var = (Kxx + np.sum(KR0T.T*np.dot(Sigma_H - np.eye(self.num_inducing),KR0T.T),0))[:,None]
            return mu_star[:,None],var
        else:
            raise NotImplementedError, "Heteroscedastic case not implemented."
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
