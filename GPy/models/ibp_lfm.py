# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import kern
from ..util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri, pdinv
from ..util import diag
from ..core.parameterization import Param
from ..likelihoods import Gaussian
from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
from ..inference.latent_function_inference.posterior import Posterior
from GPy.core.parameterization.variational import VariationalPrior
from ..core.parameterization.parameterized import Parameterized
from paramz.transformations import Logexp, Logistic, __fixed__
log_2_pi = np.log(2*np.pi)

class VarDTC_minibatch_IBPLFM(VarDTC_minibatch):
    '''
    Modifications of VarDTC_minibatch for IBP LFM
    '''
    
    def __init__(self, batchsize=None, limit=3, mpi_comm=None):
        super(VarDTC_minibatch_IBPLFM, self).__init__(batchsize, limit, mpi_comm)

    def gatherPsiStat(self, kern, X, Z, Y, beta, Zp):

        het_noise = beta.size > 1
        
        assert beta.size == 1
    
        trYYT = self.get_trYYT(Y)
        if self.Y_speedup and not het_noise:
            Y = self.get_YYTfactor(Y)
    
        num_inducing = Z.shape[0]
        num_data, output_dim = Y.shape
        batchsize = num_data if self.batchsize is None else self.batchsize
    
        psi2_full = np.zeros((num_inducing, num_inducing)) # MxM
        psi1Y_full = np.zeros((output_dim, num_inducing)) # DxM
        psi0_full = 0.
        YRY_full = 0.
    
        for n_start in range(0, num_data, batchsize):
            n_end = min(batchsize+n_start, num_data)
            if batchsize == num_data:
                Y_slice = Y
                X_slice = X
            else:
                Y_slice = Y[n_start:n_end]
                X_slice = X[n_start:n_end]
            
            if het_noise:
                b = beta[n_start]
                YRY_full += np.inner(Y_slice, Y_slice)*b
            else:
                b = beta
    
            psi0 = kern.Kdiag(X_slice) #Kff^q
            psi1 = kern.K(X_slice, Z) #Kfu

            indX = X_slice.values
            indX = np.int_(np.round(indX[:, -1]))

            Zp = Zp.gamma.values
            # Extend Zp across columns
            indZ = Z.values
            indZ = np.int_(np.round(indZ[:, -1])) - Zp.shape[0]
            Zpq = Zp[:, indZ]

            for d in np.unique(indX):
                indd = indX == d
                psi1d = psi1[indd, :]
                Zpd = Zp[d, :]
                Zp2 = Zpd[:, None]*Zpd[None, :] - np.diag(np.power(Zpd, 2)) + np.diag(Zpd)
                psi2_full += (np.dot(psi1d.T, psi1d)*Zp2[np.ix_(indZ, indZ)])*b #Zp2*Kufd*Kfud*beta

            psi0_full += np.sum(psi0*Zp[indX, :])*b
            psi1Y_full += np.dot(Y_slice.T, psi1*Zpq[indX, :])*b
    
        if not het_noise:
            YRY_full = trYYT*beta
    
        if self.mpi_comm is not None:
            from mpi4py import MPI
            psi0_all = np.array(psi0_full)
            psi1Y_all = psi1Y_full.copy()
            psi2_all = psi2_full.copy()
            YRY_all = np.array(YRY_full)
            self.mpi_comm.Allreduce([psi0_full, MPI.DOUBLE], [psi0_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([psi1Y_full, MPI.DOUBLE], [psi1Y_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([psi2_full, MPI.DOUBLE], [psi2_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([YRY_full, MPI.DOUBLE], [YRY_all, MPI.DOUBLE])
            return psi0_all, psi1Y_all, psi2_all, YRY_all
    
        return psi0_full, psi1Y_full, psi2_full, YRY_full


    def inference_likelihood(self, kern, X, Z, likelihood, Y, Zp):
        """
        The first phase of inference:
        Compute: log-likelihood, dL_dKmm

        Cached intermediate results: Kmm, KmmInv,
        """
        
        num_data, output_dim = Y.shape
        input_dim = Z.shape[0]
        if self.mpi_comm is not None:
            from mpi4py import MPI
            num_data_all = np.array(num_data,dtype=np.int32)
            self.mpi_comm.Allreduce([np.int32(num_data), MPI.INT], [num_data_all, MPI.INT])
            num_data = num_data_all

        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        if het_noise:
            self.batchsize = 1

        psi0_full, psi1Y_full, psi2_full, YRY_full = self.gatherPsiStat(kern, X, Z, Y, beta, Zp)

        #======================================================================
        # Compute Common Components
        #======================================================================

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        if not np.isfinite(Kmm).all():
            print(Kmm)
        Lm = jitchol(Kmm)
        LmInv = dtrtri(Lm)

        LmInvPsi2LmInvT = np.dot(LmInv, np.dot(psi2_full, LmInv.T))
        Lambda = np.eye(Kmm.shape[0])+LmInvPsi2LmInvT
        LL = jitchol(Lambda)
        LLInv = dtrtri(LL)
        logdet_L = 2.*np.sum(np.log(np.diag(LL)))
        LmLLInv = np.dot(LLInv, LmInv)
        
        b = np.dot(psi1Y_full, LmLLInv.T)
        bbt = np.sum(np.square(b))
        v = np.dot(b, LmLLInv).T
        LLinvPsi1TYYTPsi1LLinvT = tdot(b.T)

        tmp = -np.dot(np.dot(LLInv.T, LLinvPsi1TYYTPsi1LLinvT + output_dim*np.eye(input_dim)), LLInv)
        dL_dpsi2R = .5*np.dot(np.dot(LmInv.T, tmp + output_dim*np.eye(input_dim)), LmInv)

        # Cache intermediate results
        self.midRes['dL_dpsi2R'] = dL_dpsi2R
        self.midRes['v'] = v

        #======================================================================
        # Compute log-likelihood
        #======================================================================
        if het_noise:
            logL_R = -np.sum(np.log(beta))
        else:
            logL_R = -num_data*np.log(beta)
        logL = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-np.trace(LmInvPsi2LmInvT))+YRY_full-bbt)*.5 - output_dim*logdet_L*.5

        #======================================================================
        # Compute dL_dKmm
        #======================================================================

        dL_dKmm = dL_dpsi2R - .5*output_dim*np.dot(np.dot(LmInv.T, LmInvPsi2LmInvT), LmInv)

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        if not self.Y_speedup or het_noise:
            wd_inv = backsub_both_sides(Lm, np.eye(input_dim)- backsub_both_sides(LL, np.identity(input_dim), transpose='left'), transpose='left')
            post = Posterior(woodbury_inv=wd_inv, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=Lm)
        else:
            post = None

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================

        if not het_noise:
            dL_dthetaL = .5*(YRY_full*beta + beta*output_dim*psi0_full - num_data*output_dim*beta) - beta*(dL_dpsi2R*psi2_full).sum() - beta*(v.T*psi1Y_full).sum()
            self.midRes['dL_dthetaL'] = dL_dthetaL

        return logL, dL_dKmm, post
        
    def inference_minibatch(self, kern, X, Z, likelihood, Y, Zp):
        """
        The second phase of inference: Computing the derivatives over a minibatch of Y
        Compute: dL_dpsi0, dL_dpsi1, dL_dpsi2, dL_dthetaL
        return a flag showing whether it reached the end of Y (isEnd)
        """

        num_data, output_dim = Y.shape

        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        #self.YYTfactor = beta*self.get_YYTfactor(Y)
        if self.Y_speedup and not het_noise:
            YYT_factor = self.get_YYTfactor(Y)
        else:
            YYT_factor = Y

        n_start = self.batch_pos
        batchsize = num_data if self.batchsize is None else self.batchsize
        n_end = min(batchsize+n_start, num_data)
        if n_end == num_data:
            isEnd = True
            self.batch_pos = 0
        else:
            isEnd = False
            self.batch_pos = n_end

        if batchsize == num_data:
            Y_slice = YYT_factor
            X_slice = X
        else:
            Y_slice = YYT_factor[n_start:n_end]
            X_slice = X[n_start:n_end]

        psi0 = kern.Kdiag(X_slice) #Kffdiag
        psi1 = kern.K(X_slice, Z) #Kfu
        betapsi1 = np.einsum('n,nm->nm', beta, psi1)

        X_slice = X_slice.values
        Z = Z.values

        Zp = Zp.gamma.values
        indX = np.int_(X_slice[:, -1])
        indZ = np.int_(Z[:, -1]) - Zp.shape[0]

        betaY = beta*Y_slice

        #======================================================================
        # Load Intermediate Results
        #======================================================================

        dL_dpsi2R = self.midRes['dL_dpsi2R']
        v = self.midRes['v']

        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi0 = -.5*output_dim*(beta * Zp[indX, :]) #XxQ #TODO: Check this gradient

        dL_dpsi1 = np.dot(betaY, v.T)
        dL_dEZp  = psi1*dL_dpsi1
        dL_dpsi1 = Zp[np.ix_(indX, indZ)]*dL_dpsi1
        dL_dgamma = np.zeros(Zp.shape)
        for d in np.unique(indX):
            indd = indX == d
            betapsi1d = betapsi1[indd, :]
            psi1d = psi1[indd, :]
            Zpd = Zp[d, :]
            Zp2 = Zpd[:, None]*Zpd[None, :] - np.diag(np.power(Zpd, 2)) + np.diag(Zpd)
            dL_dpsi1[indd, :] += np.dot(betapsi1d, Zp2[np.ix_(indZ, indZ)] * dL_dpsi2R)*2.

            dL_EZp2 = dL_dpsi2R * (np.dot(psi1d.T, psi1d) * beta)*2.  # Zpd*Kufd*Kfud*beta
            #Gradient of Likelihood wrt gamma is calculated here
            EZ = Zp[d, indZ]
            for q in range(Zp.shape[1]):
                EZt = EZ.copy()
                indq = indZ == q
                EZt[indq] = .5
                dL_dgamma[d, q] = np.sum(dL_dEZp[np.ix_(indd, indq)]) + np.sum(dL_EZp2[:, indq]*EZt[:, None]) -\
                    .5*beta*(np.sum(psi0[indd, q]))

        #======================================================================
        # Compute dL_dthetaL
        #======================================================================
        if isEnd:
            dL_dthetaL = self.midRes['dL_dthetaL']
        else:
            dL_dthetaL = 0.

        grad_dict = {'dL_dKdiag': dL_dpsi0,
                     'dL_dKnm': dL_dpsi1,
                     'dL_dthetaL': dL_dthetaL,
                     'dL_dgamma': dL_dgamma}

        return isEnd, (n_start, n_end), grad_dict


def update_gradients(model, mpi_comm=None):
    if mpi_comm is None:
        Y = model.Y
        X = model.X
    else:
        Y = model.Y_local
        X = model.X[model.N_range[0]:model.N_range[1]]

    model._log_marginal_likelihood, dL_dKmm, model.posterior = model.inference_method.inference_likelihood(model.kern, X, model.Z, model.likelihood, Y, model.Zp)

    het_noise = model.likelihood.variance.size > 1

    if het_noise:
        dL_dthetaL = np.empty((model.Y.shape[0],))
    else:
        dL_dthetaL = np.float64(0.)

    kern_grad = model.kern.gradient.copy()
    kern_grad[:] = 0.
    model.Z.gradient = 0.
    gamma_gradient = model.Zp.gamma.copy()
    gamma_gradient[:] = 0.

    isEnd = False
    while not isEnd:
        isEnd, n_range, grad_dict = model.inference_method.inference_minibatch(model.kern, X, model.Z, model.likelihood, Y, model.Zp)

        if (n_range[1]-n_range[0]) == X.shape[0]:
            X_slice = X
        elif mpi_comm is None:
            X_slice = model.X[n_range[0]:n_range[1]]
        else:
            X_slice = model.X[model.N_range[0]+n_range[0]:model.N_range[0]+n_range[1]]

        #gradients w.r.t. kernel
        model.kern.update_gradients_diag(grad_dict['dL_dKdiag'], X_slice)
        kern_grad += model.kern.gradient

        model.kern.update_gradients_full(grad_dict['dL_dKnm'], X_slice, model.Z)
        kern_grad += model.kern.gradient

        #gradients w.r.t. Z
        model.Z.gradient += model.kern.gradients_X(grad_dict['dL_dKnm'].T, model.Z, X_slice)

        #gradients w.r.t. posterior parameters of Zp
        gamma_gradient += grad_dict['dL_dgamma']

        if het_noise:
            dL_dthetaL[n_range[0]:n_range[1]] = grad_dict['dL_dthetaL']
        else:
            dL_dthetaL += grad_dict['dL_dthetaL']

    # Gather the gradients from multiple MPI nodes
    if mpi_comm is not None:
        from mpi4py import MPI
        if het_noise:
            raise "het_noise not implemented!"
        kern_grad_all = kern_grad.copy()
        Z_grad_all = model.Z.gradient.copy()
        gamma_grad_all = gamma_gradient.copy()
        mpi_comm.Allreduce([kern_grad, MPI.DOUBLE], [kern_grad_all, MPI.DOUBLE])
        mpi_comm.Allreduce([model.Z.gradient, MPI.DOUBLE], [Z_grad_all, MPI.DOUBLE])
        mpi_comm.Allreduce([gamma_gradient, MPI.DOUBLE], [gamma_grad_all, MPI.DOUBLE])
        kern_grad = kern_grad_all
        model.Z.gradient = Z_grad_all
        gamma_gradient = gamma_grad_all

    #gradients w.r.t. kernel
    model.kern.update_gradients_full(dL_dKmm, model.Z, None)
    model.kern.gradient += kern_grad

    #gradients w.r.t. Z
    model.Z.gradient += model.kern.gradients_X(dL_dKmm, model.Z)

    #gradient w.r.t. gamma
    model.Zp.gamma.gradient = gamma_gradient

    # Update Log-likelihood
    KL_div = model.variational_prior.KL_divergence(model.Zp)
    # update for the KL divergence
    model.variational_prior.update_gradients_KL(model.Zp)

    model._log_marginal_likelihood += KL_div

    # dL_dthetaL
    model.likelihood.update_gradients(dL_dthetaL)


class IBPPosterior(Parameterized):
    '''
    The IBP distribution for variational approximations.
    '''
    def __init__(self, binary_prob, tau=None, name='Sensitivity space', *a, **kw):
        """
        binary_prob : the probability of including a latent function over an output.
        """
        super(IBPPosterior, self).__init__(name=name, *a, **kw)
        self.gamma = Param("binary_prob", binary_prob, Logistic(1e-10, 1. - 1e-10))
        self.link_parameter(self.gamma)
        if tau is not None:
            assert tau.size == 2*self.gamma_.shape[1]
            self.tau = Param("tau", tau, Logexp())
        else:
            self.tau = Param("tau", np.ones((2, self.gamma.shape[1])), Logexp())
        self.link_parameter(self.tau)

    def set_gradients(self, grad):
        self.gamma.gradient, self.tau.gradient = grad

    def __getitem__(self, s):
        pass
    #     if isinstance(s, (int, slice, tuple, list, np.ndarray)):
    #         import copy
    #         n = self.__new__(self.__class__, self.name)
    #         dc = self.__dict__.copy()
    #         dc['binary_prob'] = self.binary_prob[s]
    #         dc['tau'] = self.tau
    #         dc['parameters'] = copy.copy(self.parameters)
    #         n.__dict__.update(dc)
    #         n.parameters[dc['binary_prob']._parent_index_] = dc['binary_prob']
    #         n.parameters[dc['tau']._parent_index_] = dc['tau']
    #         n._gradient_array_ = None
    #         oversize = self.size - self.gamma.size - self.tau.size
    #         n.size = n.gamma.size + n.tau.size + oversize
    #         return n
    #     else:
    #         return super(IBPPosterior, self).__getitem__(s)

class IBPPrior(VariationalPrior):
    def __init__(self, rank, alpha=2., name='IBPPrior', **kw):
        super(IBPPrior, self).__init__(name=name, **kw)
        from paramz.transformations import __fixed__  
        self.rank = rank
        self.alpha = Param('alpha', alpha, __fixed__)
        self.link_parameter(self.alpha)

    def KL_divergence(self, variational_posterior):
        from scipy.special import gamma, psi
        
        eta, tau = variational_posterior.gamma.values, variational_posterior.tau.values
        
        sum_eta = np.sum(eta, axis=0) #sum_d gamma(d,q)
        D_seta = eta.shape[0] - sum_eta
        ad = self.alpha/eta.shape[1]
        psitau1 = psi(tau[0, :])
        psitau2 = psi(tau[1, :])
        sumtau = np.sum(tau, axis=0)
        psitau = psi(sumtau)
        # E[log p(z)]
        part1 = np.sum(sum_eta*psitau1 + D_seta*psitau2 - eta.shape[0]*psitau)
            
        # E[log p(pi)]
        part1 += (ad - 1.)*np.sum(psitau1 - psitau) + eta.shape[1]*np.log(ad)
        
        #H(z)
        part2 = np.sum(-(1.-eta)*np.log(1.-eta) - eta*np.log(eta))
        #H(pi)
        part2 += np.sum(np.log(gamma(tau[0, :])*gamma(tau[1, :])/gamma(sumtau))-(tau[0, :]-1.)*psitau1-(tau[1, :]-1.)*psitau2\
                 + (sumtau-2.)*psitau)
        
        return part1+part2

    def update_gradients_KL(self, variational_posterior):
        eta, tau = variational_posterior.gamma.values, variational_posterior.tau.values

        from scipy.special import psi, polygamma
        dgamma = np.log(1. - eta) - np.log(eta) + psi(tau[0, :]) - psi(tau[1, :])
        variational_posterior.gamma.gradient += dgamma
        ad = self.alpha/self.rank
        sumeta = np.sum(eta, axis=0)
        sumtau = np.sum(tau, axis=0)
        common = (-eta.shape[0] - (ad - 1.) + (sumtau - 2.))*polygamma(1, sumtau)
        variational_posterior.tau.gradient[0, :] = (sumeta + ad - tau[0, :])*polygamma(1, tau[0, :]) + common
        variational_posterior.tau.gradient[1, :] = ((eta.shape[0] - sumeta) - (tau[1, :] - 1.))*polygamma(1, tau[1, :])\
                                                   + common


class IBPLFM(SparseGP_MPI):
    """
    Indian Buffet Process for Latent Force Models

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param X: input data (np.ndarray) [X:values, X:index], index refers to the number of the output
    :type X: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    : param rank: number of latent functions

    """
    def __init__(self, X, Y, input_dim=2, output_dim=1, rank=1, Gamma=None, num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='IBP for LFM', alpha=2., beta=2., connM=None, tau=None, mpi_comm=None, normalizer=False, variational_prior=None,**kwargs):

        if kernel is None:
            kernel = kern.EQ_ODE2(input_dim, output_dim, rank)
        
        if Gamma is None:
            gamma = np.empty((output_dim, rank)) # The posterior probabilities of the binary variable in the variational approximation
            gamma[:] = 0.5 + 0.1 * np.random.randn(output_dim, rank)
            gamma[gamma>1.-1e-9] = 1.-1e-9
            gamma[gamma<1e-9] = 1e-9
        else:
            gamma = Gamma.copy()
        
        #TODO: create a vector of inducing points
        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]
        
        if likelihood is None:
            likelihood = Gaussian()

        if inference_method is None:
            inference_method = VarDTC_minibatch_IBPLFM(mpi_comm=mpi_comm)
        
        #Definition of variational terms
        self.variational_prior = IBPPrior(rank=rank, alpha=alpha) if variational_prior is None else variational_prior
        self.Zp = IBPPosterior(gamma, tau=tau)

        super(IBPLFM, self).__init__(X, Y, Z, kernel, likelihood, variational_prior=self.variational_prior, inference_method=inference_method, name=name, mpi_comm=mpi_comm, normalizer=normalizer, **kwargs)
        self.link_parameter(self.Zp, index=0)
        
    def set_Zp_gradients(self, Zp, Zp_grad):
        """Set the gradients of the posterior distribution of Zp in its specific form."""
        Zp.gamma.gradient = Zp_grad
    
    def get_Zp_gradients(self, Zp):
        """Get the gradients of the posterior distribution of Zp in its specific form."""
        return Zp.gamma.gradient

    def _propogate_Zp_val(self):
        pass

    def parameters_changed(self):
        #super(IBPLFM,self).parameters_changed()
        if isinstance(self.inference_method, VarDTC_minibatch_IBPLFM):
            update_gradients(self,  mpi_comm=self.mpi_comm)
            return

        # Add the KL divergence term
        self._log_marginal_likelihood += self.variational_prior.KL_divergence(self.Zp)
        #TODO Change the following according to this variational distribution
        #self.Zp.gamma.gradient = self.

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.Zp)