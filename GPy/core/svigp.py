# Copyright (c) 2012, James Hensman and Nicolo' Fusi
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from .. import kern
from ..util.linalg import pdinv, mdot, tdot, dpotrs, dtrtrs, jitchol, backsub_both_sides
from ..likelihoods import EP
from gp_base import GPBase
from model import Model
import time
import sys


class SVIGP(GPBase):
    """

    Stochastic Variational inference in a Gaussian Process

    :param X: inputs
    :type X: np.ndarray (num_data x num_inputs)
    :param Y: observed data
    :type Y: np.ndarray of observations (num_data x output_dim)
    :param batchsize: the size of a minibatch
    :param q_u: canonical parameters of the distribution squasehd into a 1D array
    :type q_u: np.ndarray
    :param kernel: the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param Z: inducing inputs
    :type Z: np.ndarray (num_inducing x num_inputs)

    """

    def __init__(self, X, likelihood, kernel, Z, q_u=None, batchsize=10, X_variance=None):
        GPBase.__init__(self, X, likelihood, kernel, normalize_X=False)
        self.batchsize=batchsize
        self.Y = self.likelihood.Y.copy()
        self.Z = Z
        self.num_inducing = Z.shape[0]
        self.batchcounter = 0
        self.epochs = 0
        self.iterations = 0

        self.vb_steplength = 0.05
        self.param_steplength = 1e-5
        self.momentum = 0.9

        if X_variance is None:
            self.has_uncertain_inputs = False
        else:
            self.has_uncertain_inputs = True
            self.X_variance = X_variance


        if q_u is None:
             q_u = np.hstack((np.random.randn(self.num_inducing*self.output_dim),-.5*np.eye(self.num_inducing).flatten()))
        self.set_vb_param(q_u)

        self._permutation = np.random.permutation(self.num_data)
        self.load_batch()

        self._param_trace = []
        self._ll_trace = []
        self._grad_trace = []

        #set the adaptive steplength parameters
        self.hbar_t = 0.0
        self.tau_t = 100.0
        self.gbar_t = 0.0
        self.gbar_t1 = 0.0
        self.gbar_t2 = 0.0
        self.hbar_tp = 0.0
        self.tau_tp = 10000.0
        self.gbar_tp = 0.0
        self.adapt_param_steplength = True
        self.adapt_vb_steplength = True
        self._param_steplength_trace = []
        self._vb_steplength_trace = []

        self.ensure_default_constraints()

    def getstate(self):
        steplength_params = [self.hbar_t, self.tau_t, self.gbar_t, self.gbar_t1, self.gbar_t2, self.hbar_tp, self.tau_tp, self.gbar_tp, self.adapt_param_steplength, self.adapt_vb_steplength, self.vb_steplength, self.param_steplength]
        return GPBase.getstate(self) + \
            [self.get_vb_param(),
             self.Z,
             self.num_inducing,
             self.has_uncertain_inputs,
             self.X_variance,
             self.X_batch,
             self.X_variance_batch,
             steplength_params,
             self.batchcounter,
             self.batchsize,
             self.epochs,
             self.momentum,
             self.data_prop,
             self._param_trace,
             self._param_steplength_trace,
             self._vb_steplength_trace,
             self._ll_trace,
             self._grad_trace,
             self.Y,
             self._permutation,
             self.iterations
            ]

    def setstate(self, state):
        self.iterations = state.pop()
        self._permutation = state.pop()
        self.Y = state.pop()
        self._grad_trace = state.pop()
        self._ll_trace = state.pop()
        self._vb_steplength_trace = state.pop()
        self._param_steplength_trace = state.pop()
        self._param_trace = state.pop()
        self.data_prop = state.pop()
        self.momentum = state.pop()
        self.epochs = state.pop()
        self.batchsize = state.pop()
        self.batchcounter = state.pop()
        steplength_params = state.pop()
        (self.hbar_t, self.tau_t, self.gbar_t, self.gbar_t1, self.gbar_t2, self.hbar_tp, self.tau_tp, self.gbar_tp, self.adapt_param_steplength, self.adapt_vb_steplength, self.vb_steplength, self.param_steplength) = steplength_params
        self.X_variance_batch = state.pop()
        self.X_batch = state.pop()
        self.X_variance = state.pop()
        self.has_uncertain_inputs = state.pop()
        self.num_inducing = state.pop()
        self.Z = state.pop()
        vb_param = state.pop()
        GPBase.setstate(self, state)
        self.set_vb_param(vb_param)

    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            self.psi0 = self.kern.psi0(self.Z, self.X_batch, self.X_variance_batch)
            self.psi1 = self.kern.psi1(self.Z, self.X_batch, self.X_variance_batch)
            self.psi2 = self.kern.psi2(self.Z, self.X_batch, self.X_variance_batch)
        else:
            self.psi0 = self.kern.Kdiag(self.X_batch)
            self.psi1 = self.kern.K(self.X_batch, self.Z)
            self.psi2 = None

    def dL_dtheta(self):
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z, self.X_batch, self.X_variance_batch)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1, self.Z, self.X_batch, self.X_variance_batch)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2, self.Z, self.X_batch, self.X_variance_batch)
        else:
            dL_dtheta += self.kern.dK_dtheta(self.dL_dpsi1, self.X_batch, self.Z)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X_batch)
        return dL_dtheta

    def _set_params(self, p, computations=True):
        self.kern._set_params_transformed(p[:self.kern.num_params])
        self.likelihood._set_params(p[self.kern.num_params:])
        if computations:
            self._compute_kernel_matrices()
            self._computations()

    def _get_params(self):
        return np.hstack((self.kern._get_params_transformed() , self.likelihood._get_params()))

    def _get_param_names(self):
        return self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    def load_batch(self):
        """
        load a batch of data (set self.X_batch and self.likelihood.Y from self.X, self.Y)
        """

        #if we've seen all the data, start again with them in a new random order
        if self.batchcounter+self.batchsize > self.num_data:
            self.batchcounter = 0
            self.epochs += 1
            self._permutation = np.random.permutation(self.num_data)

        this_perm = self._permutation[self.batchcounter:self.batchcounter+self.batchsize]

        self.X_batch = self.X[this_perm]
        self.likelihood.set_data(self.Y[this_perm])
        if self.has_uncertain_inputs:
            self.X_variance_batch = self.X_variance[this_perm]

        self.batchcounter += self.batchsize

        self.data_prop = float(self.batchsize)/self.num_data

        self._compute_kernel_matrices()
        self._computations()

    def _computations(self,do_Kmm=True, do_Kmm_grad=True):
        """
        All of the computations needed. Some are optional, see kwargs.
        """

        if do_Kmm:
            self.Lm = jitchol(self.Kmm)

        # The rather complex computations of self.A
        if self.has_uncertain_inputs:
            if self.likelihood.is_heteroscedastic:
                psi2_beta = (self.psi2 * (self.likelihood.precision.flatten().reshape(self.batchsize, 1, 1))).sum(0)
            else:
                psi2_beta = self.psi2.sum(0) * self.likelihood.precision
            evals, evecs = np.linalg.eigh(psi2_beta)
            clipped_evals = np.clip(evals, 0., 1e6) # TODO: make clipping configurable
            tmp = evecs * np.sqrt(clipped_evals)
        else:
            if self.likelihood.is_heteroscedastic:
                tmp = self.psi1.T * (np.sqrt(self.likelihood.precision.flatten().reshape(1, self.batchsize)))
            else:
                tmp = self.psi1.T * (np.sqrt(self.likelihood.precision))
        tmp, _ = dtrtrs(self.Lm, np.asfortranarray(tmp), lower=1)
        self.A = tdot(tmp)

        self.V = self.likelihood.precision*self.likelihood.Y
        self.VmT = np.dot(self.V,self.q_u_expectation[0].T)
        self.psi1V = np.dot(self.psi1.T, self.V)

        self.B = np.eye(self.num_inducing)*self.data_prop + self.A
        self.Lambda = backsub_both_sides(self.Lm, self.B.T)
        self.LQL = backsub_both_sides(self.Lm,self.q_u_expectation[1].T,transpose='right')

        self.trace_K = self.psi0.sum() - np.trace(self.A)/self.likelihood.precision
        self.Kmmi_m, _ = dpotrs(self.Lm, self.q_u_expectation[0], lower=1)
        self.projected_mean = np.dot(self.psi1,self.Kmmi_m)

        # Compute dL_dpsi
        self.dL_dpsi0 = - 0.5 * self.output_dim * self.likelihood.precision * np.ones(self.batchsize)
        self.dL_dpsi1, _ = dpotrs(self.Lm,np.asfortranarray(self.VmT.T),lower=1)
        self.dL_dpsi1 = self.dL_dpsi1.T

        dL_dpsi2 = -0.5 * self.likelihood.precision * backsub_both_sides(self.Lm, self.LQL - self.output_dim * np.eye(self.num_inducing))
        if self.has_uncertain_inputs:
            self.dL_dpsi2 = np.repeat(dL_dpsi2[None,:,:],self.batchsize,axis=0)
        else:
            self.dL_dpsi1 += 2.*np.dot(dL_dpsi2,self.psi1.T).T
            self.dL_dpsi2 = None

        # Compute dL_dKmm
        if do_Kmm_grad:
            tmp = np.dot(self.LQL,self.A) - backsub_both_sides(self.Lm,np.dot(self.q_u_expectation[0],self.psi1V.T),transpose='right')
            tmp += tmp.T
            tmp += -self.output_dim*self.B
            tmp += self.data_prop*self.LQL
            self.dL_dKmm = 0.5*backsub_both_sides(self.Lm,tmp)

        #Compute the gradient of the log likelihood wrt noise variance
        self.partial_for_likelihood =  -0.5*(self.batchsize*self.output_dim - np.sum(self.A*self.LQL))*self.likelihood.precision
        self.partial_for_likelihood +=  (0.5*self.output_dim*self.trace_K + 0.5 * self.likelihood.trYYT - np.sum(self.likelihood.Y*self.projected_mean))*self.likelihood.precision**2


    def log_likelihood(self):
        """
        As for uncollapsed sparse GP, but account for the proportion of data we're looking at right now.

        NB. self.batchsize is the size of the batch, not the size of X_all
        """
        assert not self.likelihood.is_heteroscedastic
        A = -0.5*self.batchsize*self.output_dim*(np.log(2.*np.pi) - np.log(self.likelihood.precision))
        B = -0.5*self.likelihood.precision*self.output_dim*self.trace_K
        Kmm_logdet = 2.*np.sum(np.log(np.diag(self.Lm)))
        C = -0.5*self.output_dim*self.data_prop*(Kmm_logdet-self.q_u_logdet - self.num_inducing)
        C += -0.5*np.sum(self.LQL * self.B)
        D = -0.5*self.likelihood.precision*self.likelihood.trYYT
        E = np.sum(self.V*self.projected_mean)
        return (A+B+C+D+E)/self.data_prop

    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))/self.data_prop

    def vb_grad_natgrad(self):
        """
        Compute the gradients of the lower bound wrt the canonical and
        Expectation parameters of u.

        Note that the natural gradient in either is given by the gradient in the other (See Hensman et al 2012 Fast Variational inference in the conjugate exponential Family)
        """

        # Gradient for eta
        dL_dmmT_S = -0.5*self.Lambda/self.data_prop + 0.5*self.q_u_prec
        Kmmipsi1V,_ = dpotrs(self.Lm,self.psi1V,lower=1)
        dL_dm = (Kmmipsi1V - np.dot(self.Lambda,self.q_u_mean))/self.data_prop

        # Gradients for theta
        S = self.q_u_cov
        Si = self.q_u_prec
        m = self.q_u_mean
        dL_dSi = -mdot(S,dL_dmmT_S, S)

        dL_dmhSi = -2*dL_dSi
        dL_dSim = np.dot(dL_dSi,m) + np.dot(Si, dL_dm)

        return np.hstack((dL_dm.flatten(),dL_dmmT_S.flatten())) , np.hstack((dL_dSim.flatten(), dL_dmhSi.flatten()))


    def optimize(self, iterations, print_interval=10, callback=lambda:None, callback_interval=5):

        param_step = 0.

        #Iterate!
        for i in range(iterations):
            
            #store the current configuration for plotting later
            self._param_trace.append(self._get_params())
            self._ll_trace.append(self.log_likelihood() + self.log_prior())

            #load a batch and do the appropriate computations (kernel matrices, etc)
            self.load_batch()

            #compute the (stochastic) gradient
            natgrads = self.vb_grad_natgrad()
            grads = self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients())
            self._grad_trace.append(grads)

            #compute the steps in all parameters
            vb_step = self.vb_steplength*natgrads[0]
            #only move the parameters after the first epoch and only if the steplength is nonzero
            if (self.epochs>=1) and (self.param_steplength > 0):
                param_step = self.momentum*param_step + self.param_steplength*grads
            else:
                param_step = 0.

            self.set_vb_param(self.get_vb_param() + vb_step)
            #Note: don't recompute everything here, wait until the next iteration when we have a new batch
            self._set_params(self._untransform_params(self._get_params_transformed() + param_step), computations=False)

            #print messages if desired
            if i and (not i%print_interval):
                print i, np.mean(self._ll_trace[-print_interval:]) #, self.log_likelihood()
                print np.round(np.mean(self._grad_trace[-print_interval:],0),3)
                sys.stdout.flush()

            #callback
            if i and not i%callback_interval:
                callback(self) # Change this to callback()
                time.sleep(0.01)

            if self.epochs > 10:
                self._adapt_steplength()
            self._vb_steplength_trace.append(self.vb_steplength)
            self._param_steplength_trace.append(self.param_steplength)

            self.iterations += 1


    def _adapt_steplength(self):
        if self.adapt_vb_steplength:
            # self._adaptive_vb_steplength()
            self._adaptive_vb_steplength_KL()
        #self._vb_steplength_trace.append(self.vb_steplength)
        assert self.vb_steplength >= 0

        if self.adapt_param_steplength:
            self._adaptive_param_steplength()
            # self._adaptive_param_steplength_log()
            # self._adaptive_param_steplength_from_vb()
        #self._param_steplength_trace.append(self.param_steplength)

    def _adaptive_param_steplength(self):
        if hasattr(self, 'adapt_param_steplength_decr'):
            decr_factor = self.adapt_param_steplength_decr
        else:
            decr_factor = 0.02
        g_tp = self._transform_gradients(self._log_likelihood_gradients())
        self.gbar_tp = (1-1/self.tau_tp)*self.gbar_tp + 1/self.tau_tp * g_tp
        self.hbar_tp = (1-1/self.tau_tp)*self.hbar_tp + 1/self.tau_tp * np.dot(g_tp.T, g_tp)
        new_param_steplength = np.dot(self.gbar_tp.T, self.gbar_tp) / self.hbar_tp
        #- hack
        new_param_steplength *= decr_factor
        self.param_steplength = (self.param_steplength + new_param_steplength)/2
        #-
        self.tau_tp = self.tau_tp*(1-self.param_steplength) + 1

    def _adaptive_param_steplength_log(self):
        stp = np.logspace(np.log(0.0001), np.log(1e-6), base=np.e, num=18000)
        self.param_steplength = stp[self.iterations]

    def _adaptive_param_steplength_log2(self):
        self.param_steplength = (self.iterations + 0.001)**-0.5

    def _adaptive_param_steplength_from_vb(self):
        self.param_steplength = self.vb_steplength * 0.01

    def _adaptive_vb_steplength(self):
        decr_factor = 0.1
        g_t = self.vb_grad_natgrad()[0]
        self.gbar_t = (1-1/self.tau_t)*self.gbar_t + 1/self.tau_t * g_t
        self.hbar_t = (1-1/self.tau_t)*self.hbar_t + 1/self.tau_t * np.dot(g_t.T, g_t)
        new_vb_steplength = np.dot(self.gbar_t.T, self.gbar_t) / self.hbar_t
        #- hack
        new_vb_steplength *= decr_factor
        self.vb_steplength = (self.vb_steplength + new_vb_steplength)/2
        #-
        self.tau_t = self.tau_t*(1-self.vb_steplength) + 1

    def _adaptive_vb_steplength_KL(self):
        decr_factor = 0.1
        natgrad = self.vb_grad_natgrad()
        g_t1 = natgrad[0]
        g_t2 = natgrad[1]
        self.gbar_t1 = (1-1/self.tau_t)*self.gbar_t1 + 1/self.tau_t * g_t1
        self.gbar_t2 = (1-1/self.tau_t)*self.gbar_t2 + 1/self.tau_t * g_t2
        self.hbar_t = (1-1/self.tau_t)*self.hbar_t + 1/self.tau_t * np.dot(g_t1.T, g_t2)
        self.vb_steplength = np.dot(self.gbar_t1.T, self.gbar_t2) / self.hbar_t
        self.vb_steplength *= decr_factor
        self.tau_t = self.tau_t*(1-self.vb_steplength) + 1

    def _raw_predict(self, X_new, X_variance_new=None, which_parts='all',full_cov=False):
        """Internal helper function for making predictions, does not account for normalization"""

        #TODO: make this more efficient!
        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)
        tmp = self.Kmmi- mdot(self.Kmmi,self.q_u_cov,self.Kmmi)

        if X_variance_new is None:
            Kx = self.kern.K(X_new,self.Z)
            mu = np.dot(Kx,self.Kmmi_m)
            if full_cov:
                Kxx = self.kern.K(X_new)
                var = Kxx - mdot(Kx,tmp,Kx.T)
            else:
                Kxx = self.kern.Kdiag(X_new)
                var = (Kxx - np.sum(Kx*np.dot(Kx,tmp),1))[:,None]
            return mu, var
        else:
            assert X_variance_new.shape == X_new.shape
            Kx = self.kern.psi1(self.Z,X_new, X_variance_new)
            mu = np.dot(Kx,self.Kmmi_m)
            Kxx = self.kern.psi0(self.Z,X_new,X_variance_new)
            psi2 = self.kern.psi2(self.Z,X_new,X_variance_new)
            diag_var = Kxx - np.sum(np.sum(psi2*tmp[None,:,:],1),1)
            if full_cov:
                raise NotImplementedError
            else:
                return mu, diag_var[:,None]

    def predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False, sampling=False, num_samples=15000):
        # normalize X values
        Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        if X_variance_new is not None:
            X_variance_new = X_variance_new / self._Xscale ** 2

        # here's the actual prediction by the GP model
        mu, var = self._raw_predict(Xnew, X_variance_new, full_cov=full_cov, which_parts=which_parts)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov, sampling=sampling, num_samples=num_samples)

        return mean, var, _025pm, _975pm


    def set_vb_param(self,vb_param):
        """set the distribution q(u) from the canonical parameters"""
        self.q_u_canonical_flat = vb_param.copy()
        self.q_u_canonical = self.q_u_canonical_flat[:self.num_inducing*self.output_dim].reshape(self.num_inducing,self.output_dim),self.q_u_canonical_flat[self.num_inducing*self.output_dim:].reshape(self.num_inducing,self.num_inducing)

        self.q_u_prec = -2.*self.q_u_canonical[1]
        self.q_u_cov, q_u_Li, q_u_L, tmp = pdinv(self.q_u_prec)
        self.q_u_Li = q_u_Li
        self.q_u_logdet = -tmp
        self.q_u_mean, _ = dpotrs(q_u_Li, np.asfortranarray(self.q_u_canonical[0]),lower=1)

        self.q_u_expectation = (self.q_u_mean, np.dot(self.q_u_mean,self.q_u_mean.T)+self.q_u_cov*self.output_dim)


    def get_vb_param(self):
        """
        Return the canonical parameters of the distribution q(u)
        """
        return self.q_u_canonical_flat


    def plot(self, ax=None, fignum=None, Z_height=None, **kwargs):

        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        #horrible hack here:
        data = self.likelihood.data.copy()
        self.likelihood.data = self.Y
        GPBase.plot(self, ax=ax, **kwargs)
        self.likelihood.data = data

        Zu = self.Z * self._Xscale + self._Xoffset
        if self.input_dim==1:
            ax.plot(self.X_batch, self.likelihood.data, 'gx',mew=2)
            if Z_height is None:
                Z_height = ax.get_ylim()[0]
            ax.plot(Zu, np.zeros_like(Zu) + Z_height, 'r|', mew=1.5, markersize=12)

        if self.input_dim==2:
            ax.scatter(self.X[:,0], self.X[:,1], 20., self.Y[:,0], linewidth=0, cmap=pb.cm.jet)
            ax.plot(Zu[:,0], Zu[:,1], 'w^')

    def plot_traces(self):
        pb.figure()
        t = np.array(self._param_trace)
        pb.subplot(2,1,1)
        for l,ti in zip(self._get_param_names(),t.T):
            if not l[:3]=='iip':
                pb.plot(ti,label=l)
        pb.legend(loc=0)

        pb.subplot(2,1,2)
        pb.plot(np.asarray(self._ll_trace),label='stochastic likelihood')
        pb.legend(loc=0)
