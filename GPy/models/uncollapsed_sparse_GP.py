# Copyright (c) 2012 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from .. import kern
from ..likelihoods import likelihood
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

    def __init__(self, X, Y, q_u=None, M=10, *args, **kwargs):
        self.D = Y.shape[1]
        if q_u is None:
            if 'Z' in kwargs.keys():
                print kwargs['Z']
                self.M = kwargs['Z'].shape[0]
                print self.M
            else:
                self.M = M
            q_u = np.hstack((np.random.randn(self.M*self.D),-0.5*np.eye(self.M).flatten()))
        self.set_vb_param(q_u)
        sparse_GP_regression.__init__(self, X, Y, M=self.M,*args, **kwargs)

    def _computations(self):
        self.V = self.beta*self.Y
        self.VmT = np.dot(self.V,self.q_u_expectation[0].T)
        self.psi1V = np.dot(self.psi1, self.V)
        self.psi1VVpsi1 = np.dot(self.psi1V, self.psi1V.T)
        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)
        self.A = mdot(self.Lmi, self.beta*self.psi2, self.Lmi.T)
        self.B = np.eye(self.M) + self.A
        self.Lambda = mdot(self.Lmi.T,self.B,self.Lmi)
        self.trace_K = self.psi0 - np.trace(self.A)/self.beta
        self.projected_mean = mdot(self.psi1.T,self.Kmmi,self.q_u_expectation[0])

        # Compute dL_dpsi
        self.dL_dpsi0 = - 0.5 * self.D * self.beta * np.ones(self.N)
        self.dL_dpsi1 = np.dot(self.VmT,self.Kmmi).T # This is the correct term for E I think...
        self.dL_dpsi2 = 0.5 * self.beta * self.D * (self.Kmmi - mdot(self.Kmmi,self.q_u_expectation[1],self.Kmmi))

        # Compute dL_dKmm
        tmp = self.beta*mdot(self.psi2,self.Kmmi,self.q_u_expectation[1]) -np.dot(self.q_u_expectation[0],self.psi1V.T)
        tmp += tmp.T
        tmp += self.D*(-self.beta*self.psi2 - self.Kmm + self.q_u_expectation[1])
        self.dL_dKmm = 0.5*mdot(self.Kmmi,tmp,self.Kmmi)

    def log_likelihood(self):
        """
        Compute the (lower bound on the) log marginal likelihood
        """
        A = -0.5*self.N*self.D*(np.log(2.*np.pi) - np.log(self.beta))
        B = -0.5*self.beta*self.D*self.trace_K
        C = -0.5*self.D *(self.Kmm_logdet-self.q_u_logdet + np.sum(self.Lambda * self.q_u_expectation[1]) - self.M)
        D = -0.5*self.beta*self.trYYT
        E = np.sum(np.dot(self.V.T,self.projected_mean))
        return A+B+C+D+E

    def dL_dbeta(self):
        """
        Compute the gradient of the log likelihood wrt beta.
        TODO: suport heteroscedatic noise
        """
        dA_dbeta =   0.5 * self.N*self.D/self.beta
        dB_dbeta = - 0.5 * self.D * self.trace_K
        dC_dbeta = - 0.5 * self.D * np.sum(self.q_u_expectation[1]*mdot(self.Kmmi,self.psi2,self.Kmmi))
        dD_dbeta = - 0.5 * self.trYYT
        dE_dbeta = np.sum(np.dot(self.Y.T,self.projected_mean))

        return np.squeeze(dA_dbeta + dB_dbeta + dC_dbeta + dD_dbeta + dE_dbeta)

    def _raw_predict(self, Xnew, slices,full_cov=False):
        """Internal helper function for making predictions, does not account for normalisation"""
        Kx = self.kern.K(Xnew,self.Z)
        mu = mdot(Kx,self.Kmmi,self.q_u_expectation[0])

        tmp = self.Kmmi- mdot(self.Kmmi,self.q_u_cov,self.Kmmi)
        if full_cov:
            Kxx = self.kern.K(Xnew)
            var = Kxx - mdot(Kx,tmp,Kx.T) + np.eye(Xnew.shape[0])/self.beta
        else:
            Kxx = self.kern.Kdiag(Xnew)
            var = Kxx - np.sum(Kx*np.dot(Kx,tmp),1) + 1./self.beta
        return mu,var


    def set_vb_param(self,vb_param):
        """set the distribution q(u) from the canonical parameters"""
        self.q_u_prec = -2.*vb_param[self.M*self.D:].reshape(self.M,self.M)
        self.q_u_cov, q_u_Li, q_u_L, tmp = pdinv(self.q_u_prec)
        self.q_u_logdet = -tmp
        self.q_u_mean = np.dot(self.q_u_cov,vb_param[:self.M*self.D].reshape(self.M,self.D))

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
        dL_dmmT_S = -0.5*self.Lambda-self.q_u_canonical[1]
        #dL_dm = np.dot(self.Kmmi,self.psi1V) - np.dot(self.Lambda,self.q_u_mean)
        dL_dm = np.dot(self.Kmmi,self.psi1V) - self.q_u_canonical[0]

        #dL_dSim =
        #dL_dmhSi =

        return np.hstack((dL_dm.flatten(),dL_dmmT_S.flatten()))  # natgrad only, grad TODO


    def plot(self, *args, **kwargs):
        """
        add the distribution q(u) to the plot from sparse_GP_regression
        """
        sparse_GP_regression.plot(self,*args,**kwargs)
        if self.Q==1:
            pb.errorbar(self.Z[:,0],self.q_u_expectation[0][:,0],yerr=2.*np.sqrt(np.diag(self.q_u_cov)),fmt=None,ecolor='b')

