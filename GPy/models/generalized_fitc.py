# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv, trace_dot
from ..util.plot import gpplot
from .. import kern
from scipy import stats, linalg
from ..core import SparseGP

def backsub_both_sides(L,X):
    """ Return L^-T * X * L^-1, assumuing X is symmetrical and L is lower cholesky"""
    tmp,_ = linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(X),lower=1,trans=1)
    return linalg.lapack.flapack.dtrtrs(L,np.asfortranarray(tmp.T),lower=1,trans=1)[0].T


class GeneralizedFITC(SparseGP):
    """
    Naish-Guzman, A. and Holden, S. (2008) implemantation of EP with FITC.

    :param X: inputs
    :type X: np.ndarray (N x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP)
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param X_variance: The variance in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (N x input_dim) | None
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param num_inducing : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type num_inducing: int
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False):
        self.Z = Z
        self.num_inducing = self.Z.shape[0]
        self.true_precision = likelihood.precision

        super(GeneralizedFITC, self).__init__(X, likelihood, kernel=kernel, Z=self.Z, X_variance=X_variance, normalize_X=normalize_X)
        self._set_params(self._get_params())

    def _set_params(self, p):
        self.Z = p[:self.num_inducing*self.input_dim].reshape(self.num_inducing, self.input_dim)
        self.kern._set_params(p[self.Z.size:self.Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size+self.kern.Nparam:])
        self._compute_kernel_matrices()
        self._computations()
        self._FITC_computations()

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian (or direct: TODO) likelihood, no iteration is required:
        this function does nothing

        Diag(Knn - Qnn) is added to the noise term to use the tools already implemented in SparseGP.
        The true precison is now 'true_precision' not 'precision'.
        """
        if self.has_uncertain_inputs:
            raise NotImplementedError, "FITC approximation not implemented for uncertain inputs"
        else:
            self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
            self.true_precision = self.likelihood.precision # Save the true precision
            self.likelihood.precision = self.true_precision/(1. + self.true_precision*self.Diag0[:,None]) # Add the diagonal element of the FITC approximation
            self._set_params(self._get_params()) # update the GP

    def _FITC_computations(self):
        """
        FITC approximation doesn't have the correction term in the log-likelihood bound,
        but adds a diagonal term to the covariance matrix: diag(Knn - Qnn).
        This function:
            - computes the FITC diagonal term
            - removes the extra terms computed in the SparseGP approximation
            - computes the likelihood gradients wrt the true precision.
        """
        #NOTE the true precison is now 'true_precision' not 'precision'
        if self.likelihood.is_heteroscedastic:

            # Compute generalized FITC's diagonal term of the covariance
            self.Lmi,info = linalg.lapack.flapack.dtrtrs(self.Lm,np.eye(self.num_inducing),lower=1)
            Lmipsi1 = np.dot(self.Lmi,self.psi1)
            self.Qnn = np.dot(Lmipsi1.T,Lmipsi1)
            #self.Kmmi, Lm, Lmi, Kmm_logdet = pdinv(self.Kmm)
            #self.Qnn = mdot(self.psi1.T,self.Kmmi,self.psi1)
            #a = kj
            self.Diag0 = self.psi0 - np.diag(self.Qnn)
            Iplus_Dprod_i = 1./(1.+ self.Diag0 * self.true_precision.flatten())
            self.Diag = self.Diag0 * Iplus_Dprod_i

            self.P = Iplus_Dprod_i[:,None] * self.psi1.T
            self.RPT0 = np.dot(self.Lmi,self.psi1)
            self.L = np.linalg.cholesky(np.eye(self.num_inducing) + np.dot(self.RPT0,((1. - Iplus_Dprod_i)/self.Diag0)[:,None]*self.RPT0.T))
            self.R,info = linalg.flapack.dtrtrs(self.L,self.Lmi,lower=1)
            self.RPT = np.dot(self.R,self.P.T)
            self.Sigma = np.diag(self.Diag) + np.dot(self.RPT.T,self.RPT)
            self.w = self.Diag * self.likelihood.v_tilde
            self.Gamma = np.dot(self.R.T, np.dot(self.RPT,self.likelihood.v_tilde))
            self.mu = self.w + np.dot(self.P,self.Gamma)

            # Remove extra term from dL_dpsi1
            self.dL_dpsi1 -= mdot(self.Lmi.T,Lmipsi1*self.likelihood.precision.flatten().reshape(1,self.N))
            #self.Kmmi, Lm, Lmi, Kmm_logdet = pdinv(self.Kmm)
            #self.dL_dpsi1 -= mdot(self.Kmmi,self.psi1*self.likelihood.precision.flatten().reshape(1,self.N)) #dB

            #########333333
            #self.Bi, self.LB, self.LBi, self.B_logdet = pdinv(self.B)
            #########333333



        else:
            raise NotImplementedError, "homoscedastic fitc not implemented"
            # Remove extra term from dL_dpsi1
            #self.dL_dpsi1 += -mdot(self.Kmmi,self.psi1*self.likelihood.precision) #dB

        sf = self.scale_factor
        sf2 = sf**2

        # Remove extra term from dL_dKmm
        self.dL_dKmm += 0.5 * self.input_dim * mdot(self.Lmi.T, self.A, self.Lmi)*sf2 # dB
        self.dL_dpsi0 = None

        #the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedastic derivates not implemented"
        else:
            raise NotImplementedError, "homoscedastic derivatives not implemented"
            #likelihood is not heterscedatic
            #self.partial_for_likelihood =   - 0.5 * self.N*self.input_dim*self.likelihood.precision + 0.5 * np.sum(np.square(self.likelihood.Y))*self.likelihood.precision**2
            #self.partial_for_likelihood += 0.5 * self.input_dim * trace_dot(self.Bi,self.A)*self.likelihood.precision
            #self.partial_for_likelihood += self.likelihood.precision*(0.5*trace_dot(self.psi2_beta_scaled,self.E*sf2) - np.trace(self.Cpsi1VVpsi1))
        #TODO partial derivative vector for the likelihood not implemented

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm,self.Z)
        if self.has_uncertain_inputs:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            #NOTE in SparseGP this would include the gradient wrt psi0
            dL_dtheta += self.kern.dK_dtheta(self.dL_dpsi1,self.Z,self.X)
        return dL_dtheta


    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        sf2 = self.scale_factor**2
        if self.likelihood.is_heteroscedastic:
            A = -0.5*self.N*self.input_dim*np.log(2.*np.pi) +0.5*np.sum(np.log(self.likelihood.precision)) -0.5*np.sum(self.V*self.likelihood.Y)
        else:
            A = -0.5*self.N*self.input_dim*(np.log(2.*np.pi) + np.log(self.likelihood._variance)) -0.5*self.likelihood.precision*self.likelihood.trYYT
        C = -self.input_dim * (np.sum(np.log(np.diag(self.LB))) + 0.5*self.num_inducing*np.log(sf2))
        #C = -0.5*self.input_dim * (self.B_logdet + self.num_inducing*np.log(sf2))
        D = 0.5*np.sum(np.square(self._LBi_Lmi_psi1V))
        #self.Cpsi1VVpsi1 = np.dot(self.Cpsi1V,self.psi1V.T)
        #D_ = 0.5*np.trace(self.Cpsi1VVpsi1)
        return A+C+D

    def _raw_predict(self, Xnew, which_parts, full_cov=False):
        if self.likelihood.is_heteroscedastic:
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
            V,info = linalg.flapack.dtrtrs(U,self.RPT0.T,lower=1)
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
                Kxx_ = self.kern.K(Xnew,which_parts=which_parts) # TODO: RA, is this line needed?
                var_ = Kxx_ + np.dot(KR0T,np.dot(Sigma_H - np.eye(self.num_inducing),KR0T.T)) # TODO: RA, is this line needed?
                var = (Kxx + np.sum(KR0T.T*np.dot(Sigma_H - np.eye(self.num_inducing),KR0T.T),0))[:,None]
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
