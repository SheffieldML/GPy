# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from .. import kern
from GP import GP
from ..inference.EP import Full,DTC,FITC
from ..inference.likelihoods import likelihood,probit,poisson,gaussian

#Still TODO:
# make use of slices properly (kernel can now do this)
# enable heteroscedatic noise (kernel will need to compute psi2 as a (NxMxM) array)

class sparse_GP(GP):
    """
    Variational sparse GP model (Regression)

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param Y: observed data
    :type Y: np.ndarray of observations (N x D)
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param X_uncertainty: The uncertainty in the measurements of X (Gaussian variance)
    :type X_uncertainty: np.ndarray (N x Q) | None
    :param Zslices: slices for the inducing inputs (see slicing TODO: link)
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param beta: noise precision. TODO> ignore beta if doing EP
    :type beta: float
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    :parm likelihood: a GPy likelihood, defaults to gaussian
    :param method_ep: sparse approximation used by Expectation Propagation algorithm, defaults to DTC
    :type M: string (Full|DTC|FITC)
    :param epsilon_ep: convergence criterion for the Expectation Propagation algorithm, defaults to 0.1
    :param powerep: power-EP parameters [$\eta$,$\delta$], defaults to [1.,1.]
    :type powerep: list
    """

    def __init__(self,X,Y=None,kernel=None,X_uncertainty=None,beta=100.,Z=None,Zslices=None,M=10,normalize_X=False,normalize_Y=False,likelihood=None,method_ep='DTC',epsilon_ep=1e-3,power_ep=[1.,1.]):

        if Z is None:
            self.Z = np.random.permutation(X.copy())[:M]
            self.M = M
        else:
            assert Z.shape[1]==X.shape[1]
            self.Z = Z
            self.M = Z.shape[0]
        if X_uncertainty is None:
            self.has_uncertain_inputs=False
        else:
            assert X_uncertainty.shape==X.shape
            self.has_uncertain_inputs=True
            self.X_uncertainty = X_uncertainty

        GP.__init__(self, X=X, Y=Y, kernel=kernel, normalize_X=normalize_X, normalize_Y=normalize_Y,likelihood=likelihood,epsilon_ep=epsilon_ep,power_ep=power_ep)

        #normalise X uncertainty also
        if self.has_uncertain_inputs:
            self.X_uncertainty /= np.square(self._Xstd)

        if not self.EP:
            self.trYYT = np.sum(np.square(self.Y))
        else:
            self.method_ep = method_ep


    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        if not self.EP:
            self.beta = p[self.M*self.Q]
            #self.beta = np.repeat(p[self.M*self.Q],self.N)[:,None]
            self.kern._set_params(p[self.Z.size + 1:])
            self.beta2 = self.beta**2
        else:
            self.kern._set_params(p[self.Z.size:])
            if self.Y is None:
                self.Y = np.ones([self.N,1])
        self._compute_kernel_matrices()
        self._computations() #NOTE At this point computations of dL are not needed

    def _get_params(self):
        if not self.EP:
            return np.hstack([self.Z.flatten(),self.beta,self.kern._get_params_transformed()])
        else:
            return np.hstack([self.Z.flatten(),self.kern._get_params_transformed()])

    def _get_param_names(self):
        if not self.EP:
            return sum([['iip_%i_%i'%(i,j) for i in range(self.Z.shape[0])] for j in range(self.Z.shape[1])],[]) + ['noise_precision']+self.kern._get_param_names_transformed()
        else:
            return sum([['iip_%i_%i'%(i,j) for i in range(self.Z.shape[0])] for j in range(self.Z.shape[1])],[]) + self.kern._get_param_names_transformed()

    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        #TODO: slices for psi statistics (easy enough)
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            if not self.EP:
                self.psi0 = self.kern.psi0(self.Z,self.X, self.X_uncertainty)#.sum()
                self.psi1 = self.kern.psi1(self.Z,self.X, self.X_uncertainty).T
                self.psi2 = self.kern.psi2(self.Z,self.X, self.X_uncertainty)#FIXME add beta vector
            else:
                raise NotImplementedError, "uncertain_inputs not yet supported for EP"
        else:
            self.psi0 = self.kern.Kdiag(self.X,slices=self.Xslices)#.sum() FIXME
            self.psi1 = self.kern.K(self.Z,self.X)
            self.psi2 = np.dot(self.psi1,self.psi1.T)
            self.psi2_beta_scaled = np.dot(self.psi1,self.beta*self.psi1.T)

    def _computations(self):
        # TODO find routine to multiply triangular matrices
        self.V = self.beta*self.Y
        self.psi1V = np.dot(self.psi1, self.V)
        self.psi1VVpsi1 = np.dot(self.psi1V, self.psi1V.T)
        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)
        self.A = mdot(self.Lmi, self.psi2_beta_scaled, self.Lmi.T)
        self.B = np.eye(self.M) + self.A
        self.Bi, self.LB, self.LBi, self.B_logdet = pdinv(self.B)
        self.LLambdai = np.dot(self.LBi, self.Lmi)
        self.trace_K = self.psi0.sum() - np.trace(self.A)
        self.LBL_inv = mdot(self.Lmi.T, self.Bi, self.Lmi)
        self.C = mdot(self.LLambdai, self.psi1V)
        self.G =  mdot(self.LBL_inv, self.psi1VVpsi1, self.LBL_inv.T)

        # Compute dL_dpsi
        self.dL_dpsi0 = - 0.5 * self.D * self.beta.flatten() * np.ones(self.N)
        self.dL_dpsi1 = mdot(self.LLambdai.T,self.C,self.V.T)
        #self.dL_dpsi2 = - 0.5 * self.beta * (self.D*(self.LBL_inv - self.Kmmi) + self.G)
        self.dL_dpsi2 = - 0.5 * (self.D*(self.LBL_inv - self.Kmmi) + self.G)

        # Compute dL_dKmm
        self.dL_dKmm = -0.5 * self.D * mdot(self.Lmi.T, self.A, self.Lmi) # dB
        #self.dL_dKmm += -0.5 * self.D * (- self.LBL_inv - 2.*self.beta*mdot(self.LBL_inv, self.psi2, self.Kmmi) + self.Kmmi) # dC
        self.dL_dKmm += -0.5 * self.D * (- self.LBL_inv - 2.*mdot(self.LBL_inv, self.psi2_beta_scaled, self.Kmmi) + self.Kmmi) # dC
        #self.dL_dKmm +=  np.dot(np.dot(self.G,self.beta*self.psi2) - np.dot(self.LBL_inv, self.psi1VVpsi1), self.Kmmi) + 0.5*self.G # dE
        self.dL_dKmm +=  np.dot(np.dot(self.G,self.psi2_beta_scaled) - np.dot(self.LBL_inv, self.psi1VVpsi1), self.Kmmi) + 0.5*self.G # dE

    def approximate_likelihood(self):
        assert not isinstance(self.likelihood, gaussian), "EP is only available for non-gaussian likelihoods"
        if self.method_ep == 'DTC':
            self.ep_approx = DTC(self.Kmm,self.likelihood,self.psi1,epsilon=self.epsilon_ep,power_ep=[self.eta,self.delta])
        elif self.method_ep == 'FITC':
            self.ep_approx = FITC(self.Kmm,self.likelihood,self.psi1,self.psi0,epsilon=self.epsilon_ep,power_ep=[self.eta,self.delta])
        else:
            self.ep_approx = Full(self.X,self.likelihood,self.kernel,inducing=None,epsilon=self.epsilon_ep,power_ep=[self.eta,self.delta])
        self.beta, self.Y, self.Z_ep = self.ep_approx.fit_EP()
        print "Aqui toy"
        self.trbetaYYT = np.sum(np.square(self.Y)*self.beta)
        self._computations()

    def log_likelihood(self):
        """
        Compute the (lower bound on the) log marginal likelihood
        """
        if not self.EP:
            A = -0.5*self.N*self.D*(np.log(2.*np.pi) - np.log(self.beta))
            D = -0.5*self.beta*self.trYYT
        else:
            A = -0.5*self.D*(self.N*np.log(2.*np.pi) - np.sum(np.log(self.beta)))
            D = -0.5*self.trbetaYYT
        B = -0.5*self.D*self.trace_K
        C = -0.5*self.D * self.B_logdet
        E = +0.5*np.sum(self.psi1VVpsi1 * self.LBL_inv)
        return A+B+C+D+E

    def dL_dbeta(self):
        """
        Compute the gradient of the log likelihood wrt beta.
        """
        #TODO: suport heteroscedatic noise
        dA_dbeta =   0.5 * self.N*self.D/self.beta
        dB_dbeta = - 0.5 * self.D * self.trace_K
        dC_dbeta = - 0.5 * self.D * np.sum(self.Bi*self.A)/self.beta
        dD_dbeta = - 0.5 * self.trYYT
        tmp = mdot(self.LBi.T, self.LLambdai, self.psi1V)
        dE_dbeta = (np.sum(np.square(self.C)) - 0.5 * np.sum(self.A * np.dot(tmp, tmp.T)))/self.beta

        return np.squeeze(dA_dbeta + dB_dbeta + dC_dbeta + dD_dbeta + dE_dbeta)

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm,self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z,self.X,self.X_uncertainty)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T,self.Z,self.X, self.X_uncertainty)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2,self.Z,self.X, self.X_uncertainty) # for multiple_beta, dL_dpsi2 will be a different shape
        else:
            #re-cast computations in psi2 back to psi1:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(self.dL_dpsi2,self.psi1)
            dL_dtheta += self.kern.dK_dtheta(dL_dpsi1,self.Z,self.X)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)

        return dL_dtheta

    def dL_dZ(self):
        """
        The derivative of the bound wrt the inducing inputs Z
        """
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm,self.Z,)#factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1.T,self.Z,self.X, self.X_uncertainty)
            dL_dZ += self.kern.dpsi2_dZ(self.dL_dpsi2,self.Z,self.X, self.X_uncertainty)
        else:
            #re-cast computations in psi2 back to psi1:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(self.dL_dpsi2,self.psi1)
            dL_dZ += self.kern.dK_dX(dL_dpsi1,self.Z,self.X)
        return dL_dZ

    def _log_likelihood_gradients(self):
        if not self.EP:
            return np.hstack([self.dL_dZ().flatten(), self.dL_dbeta(), self.dL_dtheta()])
        else:
            return np.hstack([self.dL_dZ().flatten(), self.dL_dtheta()])

    def _raw_predict(self, Xnew, slices, full_cov=False):
        """Internal helper function for making predictions, does not account for normalisation"""
        Kx = self.kern.K(self.Z, Xnew)
        mu = mdot(Kx.T, self.LBL_inv, self.psi1V)
        phi = None
        if full_cov:
            Kxx = self.kern.K(Xnew)
            var = Kxx - mdot(Kx.T, (self.Kmmi - self.LBL_inv), Kx)
            if not self.EP:
                var += np.eye(Xnew.shape[0])/self.beta # TODO: This beta doesn't belong here in the EP case.
            else:
                raise NotImplementedError, "full_cov = True not implemented for EP"
                #var = np.diag(var)[:,None]
                #phi = self.likelihood.predictive_mean(mu,var)
        else:
            Kxx = self.kern.Kdiag(Xnew)
            var = Kxx - np.sum(Kx*np.dot(self.Kmmi - self.LBL_inv, Kx),0)
            if not self.EP:
                var += 1./self.beta # TODO: This beta doesn't belong here in the EP case.
            else:
                phi = self.likelihood.predictive_mean(mu,var)
        return mu,var,phi

    def plot(self, *args, **kwargs):
        """
        Plot the fitted model: just call the GP_regression plot function and then add inducing inputs
        """
        #GP_regression.plot(self,*args,**kwargs)
        GP.plot(self,*args,**kwargs)
        if self.Q==1:
            pb.plot(self.Z,self.Z*0+pb.ylim()[0],'k|',mew=1.5,markersize=12)
            if self.has_uncertain_inputs:
                pb.errorbar(self.X[:,0], pb.ylim()[0]+np.zeros(self.N), xerr=2*np.sqrt(self.X_uncertainty.flatten()))
        if self.Q==2:
            pb.plot(self.Z[:,0],self.Z[:,1],'wo')
