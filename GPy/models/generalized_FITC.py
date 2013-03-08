# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from scipy import linalg
from .. import kern
from sparse_GP import sparse_GP

"""
import numpy as np
import pylab as pb
from scipy import stats, linalg
from .. import kern
from ..core import model
from ..util.linalg import pdinv,mdot
from ..util.plot import gpplot
#from ..inference.Expectation_Propagation import FITC
from ..likelihoods.EP import FITC
from ..likelihoods import likelihood,probit
"""

class generalized_FITC(sparse_GP):
    def __init__(self, X, likelihood, kernel, Z, X_uncertainty=None, Xslices=None,Zslices=None, normalize_X=False):
    #def __init__(self, X, likelihood, kernel=None, inducing=10, epsilon_ep=1e-3, powerep=[1.,1.]):
        """
        Naish-Guzman, A. and Holden, S. (2008) implemantation of EP with FITC.

        :param X: input observations
        :param likelihood: Output's likelihood (likelihood class)
        :param kernel: a GPy kernel
        :param Z:  Either an array specifying the inducing points location or a scalar defining their number.
        """

        if type(Z) == int:
            self.M = Z
            self.Z = (np.random.random_sample(self.D*self.M)*(self.X.max()-self.X.min())+self.X.min()).reshape(self.M,-1)
        elif type(Z) == np.ndarray:
            self.Z = Z
            self.M = self.Z.shape[0]

        self._precision = likelihood.precision

        sparse_GP.__init__(self, X, likelihood, kernel=kernel, Z=self.Z, X_uncertainty=None, Xslices=None,Zslices=None, normalize_X=False)
        self.scale_factor = 100.

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian (or direct: TODO) likelihood, no iteration is required:
        this function does nothing
        """
        if self.has_uncertain_inputs:
            raise NotImplementedError, "FITC approximation not implemented for uncertain inputs"
        else:
            self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
            self._precision = self.likelihood.precision # Save the true precision
            self.likelihood.precision = self.likelihood.precision/(1. + self.likelihood.precision*self.Diag0[:,None]) # Add the diagonal element of the FITC approximation
            self._set_params(self._get_params()) # update the GP

    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        self.kern._set_params(p[self.Z.size:self.Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size+self.kern.Nparam:])
        self._compute_kernel_matrices()
        self._computations()
        self._FITC_computations()

    def _FITC_computations(self):
        """
        FITC approximation doesn't have the correction term in the log-likelihood bound,
        but adds a diagonal term to the covariance matrix.
        This function:
            - computes the diagonal term
            - eliminates the extra terms computed in the sparse_GP approximation
            - computes the likelihood gradients wrt the true precision.
        """
        # Compute FITC's diagonal term of the covariance
        sf = self.scale_factor
        sf2 = sf**2
        self.Qnn = mdot(self.psi1.T,self.Kmmi,self.psi1)
        self.Diag0 = self.psi0 - np.diag(self.Qnn)

        self.Diag = self.Diag0/(1.+ self.Diag0 * self._precision.flatten())
        self.P = (self.Diag / self.Diag0)[:,None] * self.psi1.T
        self.RPT0 = np.dot(self.Lmi,self.psi1)
        self.L = np.linalg.cholesky(np.eye(self.M) + np.dot(self.RPT0,(1./self.Diag0 - self.Diag/(self.Diag0**2))[:,None]*self.RPT0.T))
        self.R,info = linalg.flapack.dtrtrs(self.L,self.Lmi,lower=1)
        self.RPT = np.dot(self.R,self.P.T)
        self.Sigma = np.diag(self.Diag) + np.dot(self.RPT.T,self.RPT)
        self.w = self.Diag * self.likelihood.v_tilde
        self.gamma = np.dot(self.R.T, np.dot(self.RPT,self.likelihood.v_tilde))
        self.mu = self.w + np.dot(self.P,self.gamma)
        self.mu_tilde = (self.likelihood.v_tilde/self.likelihood.tau_tilde)[:,None]

        # Remove extra term from dL_dpsi
        self.dL_dpsi0 = np.zeros(self.N)
        # Remove extra term from dL_dKmm
        self.dL_dKmm = +0.5 * self.D * mdot(self.Lmi.T, self.A, self.Lmi)*sf2 # dB
        #the partial derivative vector for the likelihood with the true precision
        if self.likelihood.Nparams ==0:
            #save computation here
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            beta = self.likelihood._precision # NOTE the true precison is now '_precison' not 'precision'
            dbeta =   0.5 * self.N*self.D/beta - 0.5 * np.sum(np.square(self.likelihood.Y))
            #dbeta += - 0.5 * self.D * (self.psi0.sum() - np.trace(self.A)/beta*sf2)
            dbeta += - 0.5 * self.D * np.sum(self.Bi*self.A)/beta
            dbeta += np.sum((self.C - 0.5 * mdot(self.C,self.psi2_beta_scaled,self.C) ) * self.psi1VVpsi1 )/beta
            self.partial_for_likelihood = -dbeta*self.likelihood.precision**2




    def _raw_predict(self, Xnew, slices, full_cov=True):
        """
        Make a prediction for the vsGP model

        Arguments
        ---------
        X : Input prediction data - Nx1 numpy array (floats)
        """
        Kx = self.kern.K(self.Z, Xnew)
        #K_x = self.kernel.K(self.Z,X)
        if full_cov:
            Kxx = self.kern.K(Xnew)
        else:
            Kxx = self.kern.K(Xnew)#FIXME
            #raise NotImplementedError
            #Kxx = self.kern.Kdiag(Xnew)

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
        KR0T = np.dot(Kx.T,self.Lmi.T)
        mu_star = np.dot(KR0T,mu_H)
        sigma2_star = Kxx + np.dot(KR0T,np.dot(Sigma_H - np.eye(self.M),KR0T.T))
        vdiag = np.diag(sigma2_star)
        return mu_star[:,None],vdiag[:,None]
