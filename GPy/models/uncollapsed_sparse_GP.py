# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from .. import kern
from ..inference.likelihoods import likelihood
from sparse_GP_regression import sparse_GP_regression

class uncollapsed_sparse_GP(sparse_GP_regression):
    """
    Variational sparse GP model (Regression), where the approximating distribution q(u) is represented explicitly

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param Y: observed data
    :type Y: np.ndarray of observations (N x D)
    :param q_u: canonical parameters of the distribution squasehd into a 1D array
    :type q_u: np.ndarray
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param Zslices: slices for the inducing inputs (see slicing TODO: link)
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param beta: noise precision. TODO> ignore beta if doing EP
    :type beta: float
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, Y, q_u=None, *args, **kwargs)
        D = Y.shape[1]
        if q_u is None:
            if Z is None:
                M = Z.shape[0]
            else:
                M=M
            self.set_vb_param(np.hstack((np.ones(M*D)),np.eye(M).flatten()))
        sparse_GP_regression.__init__(self, X, Y, *args, **kwargs)

    def _computations(self):
        self.V = self.beta*self.Y
        self.psi1V = np.dot(self.psi1, self.V)
        self.psi1VVpsi1 = np.dot(self.psi1V, self.psi1V.T)
        self.Lm = jitchol(self.Kmm)
        self.Lmi = chol_inv(self.Lm)
        self.Kmmi = np.dot(self.Lmi.T, self.Lmi)
        self.A = mdot(self.Lmi, self.psi2, self.Lmi.T)
        self.B = np.eye(self.M) + self.beta * self.A
        self.Lambda = mdot(self.Lmi.T,self.B,sel.Lmi)

        # Compute dL_dpsi
        self.dL_dpsi0 = - 0.5 * self.D * self.beta * np.ones(self.N)
        self.dL_dpsi1 = 
        self.dL_dpsi2 = 

        # Compute dL_dKmm
        self.dL_dKmm = 
        self.dL_dKmm += 
        self.dL_dKmm += 

    def log_likelihood(self):
        """
        Compute the (lower bound on the) log marginal likelihood
        """
        A = -0.5*self.N*self.D*(np.log(2.*np.pi) - np.log(self.beta))
        B = -0.5*self.beta*self.D*self.trace_K
        C = -self.D *(self.Kmm_hld +0.5*np.sum(self.Lambda * self.mmT_S) + self.M/2.)
        E = -0.5*self.beta*self.trYYT
        F = np.sum(np.dot(self.V.T,self.projected_mean))
        return A+B+C+D+E+F

    def dL_dbeta(self):
        """
        Compute the gradient of the log likelihood wrt beta.
        TODO: suport heteroscedatic noise
        """
        dA_dbeta =   0.5 * self.N*self.D/self.beta
        dB_dbeta = - 0.5 * self.D * self.trace_K
        dC_dbeta = - 0.5 * self.D * #TODO
        dD_dbeta = - 0.5 * self.trYYT

        return np.squeeze(dA_dbeta + dB_dbeta + dC_dbeta + dD_dbeta + dE_dbeta)

    def _raw_predict(self, Xnew, slices):
        """Internal helper function for making predictions, does not account for normalisation"""

        #TODO
        return mu,var

    def set_vb_param(self,vb_param):
        """set the distribution q(u) from the canonical parameters"""
        self.q_u_prec = -2.*vb_param[self.M*self.D:].reshape(self.M,self.M)
        self.q_u_prec_L = jitchol(self.q_u_prec)
        self.q_u_cov_L = chol_inv(self.q_u_prec_L)
        self.q_u_cov = np.dot(self.q_u_cov_L,self.q_u_cov_L.T)
        self.q_u_mean = -2.*np.dot(self.q_u_cov,vb_param[:self.M*self.D].reshape(self.M,self.D))

        self.q_u_expectation = (self.q_u_mean, np.dot(self.q_u_mean,self.q_u_mean.T)+self.q_u_cov)

        self.q_u_canonical = (np.dot(self.q_u_prec, self.q_u_mean),-0.5*self.q_u_prec)
        #TODO: computations now?

    def get_vb_param(self):
        """
        Return the canonical parameters of the distribution q(u)
        """
        return np.hstack([e.flatten() for e in self.q_u_canonical])

    def vb_grad_natgrad(self):
        """
        Compute the gradients of the lower bound wrt the canonical and
        Expectation parameters of u.

        Note that the natural gradient in either is given by the gradient in the other (See Hensman et al 2012 Fast Variational inference in the conjugate exponential Family)
        """
        foobar #TODO

    def plot(self, *args, **kwargs):
        """
        add the distribution q(u) to the plot from sparse_GP_regression
        """
        sparse_GP_regression.plot(self,*args,**kwargs)
        #TODO: plot the q(u) dist.
