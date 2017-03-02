'''
Created on 6 Nov 2013

@author: maxz
'''

import numpy as np
from .parameterized import Parameterized
from .param import Param
from paramz.transformations import Logexp, Logistic,__fixed__

class VariationalPrior(Parameterized):
    def __init__(self, name='latent prior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)

    def KL_divergence(self, variational_posterior):
        raise NotImplementedError("override this for variational inference of latent space")

    def update_gradients_KL(self, variational_posterior):
        """
        updates the gradients for mean and variance **in place**
        """
        raise NotImplementedError("override this for variational inference of latent space")

class NormalPrior(VariationalPrior):
    def __init__(self, name='normal_prior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)

    def KL_divergence(self, variational_posterior):
        var_mean = np.square(variational_posterior.mean).sum()
        var_S = (variational_posterior.variance - np.log(variational_posterior.variance)).sum()
        return 0.5 * (var_mean + var_S) - 0.5 * variational_posterior.input_dim * variational_posterior.num_data

    def update_gradients_KL(self, variational_posterior):
        # dL:
        # print (1. - (1. / (variational_posterior.variance))) * 0.5
        variational_posterior.mean.gradient -= variational_posterior.mean
        variational_posterior.variance.gradient -= (1. - (1. / (variational_posterior.variance))) * 0.5
        # print variational_posterior.mean
        # print variational_posterior.variance.gradient

class GmmNormalPrior(VariationalPrior):
    def __init__(self, px_mu, px_lmatrix, pi, wi, n_component, variational_wi, name="GMMNormalPrior", **kw):
        super(GmmNormalPrior, self).__init__(name=name, **kw)
        self.n_component = n_component
        
        self.px_mu = Param('mu_k', px_mu)
        self.px_lmatrix = Param('lmatrix_k', px_lmatrix)

        # Make sure they sum to one
        # variational_pi = variational_pi / np.sum(variational_pi)
        # variational_wi = variational_wi /variational_wi.sum(axis=0)
        self.pi = pi
        self.wi = wi # p(x) mixing coeffients 
        self.variational_wi = Param('variational_wi', variational_wi) # variational mixing coefficients 
        self.check_all_weights()

        self.link_parameter(self.px_mu)
        self.link_parameter(self.px_lmatrix) 
        self.link_parameter(self.variational_wi)
        # self.variational_wi = self.variational_wi/ self.variational_wi.sum(axis=0)
        # self.variational_pi.constrain_bounded(0.0, 1.0)
        #self.variational_wi.constrain_positive()

        # self.stop = 5

    def KL_divergence(self, variational_posterior):
        # Lagrange multiplier maybe also needed here

        self.pi = np.exp(self.variational_wi)/np.exp(self.variational_wi).sum(axis = 0)
        # self.variational_wi -= self.variational_wi.max(axis=0)
        mu = variational_posterior.mean
        S = variational_posterior.variance

        cov_inv = np.zeros((self.px_lmatrix.shape))
        cov_k = np.zeros((self.px_lmatrix.shape))
        ######################################################
        for k in range(self.px_lmatrix.shape[0]):
            cov_inv[k,:,:] = np.linalg.inv(self.px_lmatrix[k,:,:]).T.dot(np.linalg.inv(self.px_lmatrix[k,:,:]))
            cov_k[k,:,:] = np.dot(self.px_lmatrix[k,:,:], self.px_lmatrix[k,:,:].T)
        #######################################################
        # variational_wets = self.variational_wi
        # wets = self.wi

        variational_wets = np.exp(self.variational_wi)/ np.exp(self.variational_wi).sum(axis = 0)
        wets = np.exp(self.wi)/ np.exp(self.wi).sum(axis = 0)

        total_n = variational_posterior.input_dim * variational_posterior.num_data
        # cov_diag = np.diagonal(np.linalg.inv(self.px_var).T)
        cov_diag = np.diagonal(cov_inv.T)
        mu_minus = self.px_mu[:, np.newaxis, :] - mu[np.newaxis, :, :]

        term_1 = (variational_wets * np.log(np.linalg.det(cov_k))[:, np.newaxis]).sum()- np.log(S).sum()
        term_2 = (variational_wets * np.log(wets/variational_wets)).sum()
        term_3 = (variational_wets[:,:,np.newaxis] * cov_diag[:, np.newaxis, :]*S[np.newaxis, :, :]).sum()
        term_4 = np.zeros((mu_minus.shape[0], mu_minus.shape[1]))
        for k in range(mu_minus.shape[0]):
            for i in range(mu_minus.shape[1]):
                term_4[k,i] = variational_wets[k, i]*np.trace(np.dot(cov_inv[k, :,:], np.dot(mu_minus[k,i,:][:,None], mu_minus[k,i,:][None,:])).T)

        return 0.5 *(term_1-total_n + term_3 + term_4.sum())- term_2
            
    def update_gradients_KL(self, variational_posterior):
        # import pdb; pdb.set_trace() # breakpoint 1
        # print("Updating Gradients")
        # print (self.variational_wi)
        # if self.stop<1:
        #     return
        # self.stop-=1
        # dL:
        # variational_posterior.mean.gradient -= variational_posterior.mean
        # variational_posterior.variance.gradient -= (1. -  (1. / (variational_posterior.variance))) * 0.5
        self.px_mu.gradient = 0
        self.px_lmatrix.gradient = 0
        self.variational_wi.gradient = 0
        # print self.variational_wi
        #self.variational_wi -= self.variational_wi.max(axis = 0)[None,:]
        
        # self.variational_wi = self.variational_wi/(self.variational_wi).sum(axis=0)

        mu = variational_posterior.mean
        S = variational_posterior.variance

        cov_inv = np.zeros((self.px_lmatrix.shape))
        cov_k = np.zeros((self.px_lmatrix.shape))
        ######################################################
        for k in range(self.px_lmatrix.shape[0]):
            cov_inv[k,:,:] = np.linalg.inv(self.px_lmatrix[k,:,:]).T.dot(np.linalg.inv(self.px_lmatrix[k,:,:]))
            cov_k[k,:,:] = np.dot(self.px_lmatrix[k,:,:], self.px_lmatrix[k,:,:].T)
        #######################################################

        # variational_wets = self.variational_wi
        # wets = self.wi

        # wi_max = self.variational_wi - self.variational_wi.max(axis = 0)#
        # variational_wets = np.exp(wi_max)/ np.exp(wi_max).sum(axis = 0)
        variational_wets = np.exp(self.variational_wi)/ np.exp(self.variational_wi).sum(axis = 0)
        wets = np.exp(self.wi)/ np.exp(self.wi).sum(axis = 0)

        mu_minus = self.px_mu[:, np.newaxis, :] - mu[np.newaxis, :, :]
        sigma_mu = np.zeros((mu_minus.shape))
        sigma2_S = np.zeros((mu_minus.shape[0],mu_minus.shape[1],mu_minus.shape[2],mu_minus.shape[2]))
        sigma_S = np.zeros((sigma2_S.shape))
        sigma_S_sigma = np.zeros((sigma2_S.shape))        
        mu_sigma_mu = np.zeros((mu_minus.shape[0],mu_minus.shape[1]))
        sigma_diag = np.diagonal(cov_inv.T)
        # sigma_inv1 = np.linalg.inv(self.px_var) #equal to cov_inv


        for k in range(mu_minus.shape[0]):
            for i in range(mu_minus.shape[1]):
                sigma_mu[k,i,:] = np.dot(cov_inv[k,:,:], mu_minus[k,i,:])
                sigma_S[k,i,:,:] = np.dot(cov_inv[k,:,:], np.diag(S[i,:]))
                sigma2_S[k,i,:,:] = np.dot(np.diag(S[i,:]), np.matrix(cov_inv[k,:,:])**2)
                sigma_S_sigma[k,i,:,:] = np.dot(sigma_mu[k,i,:][:,None],sigma_mu[k,i,:][None,:] )
                mu_sigma_mu[k,i] = np.dot(mu_minus[k,i,:][None,:], sigma_mu[k,i,:][:,None])


        variational_posterior.mean.gradient += (variational_wets[:,:,np.newaxis] * sigma_mu).sum(axis = 0)
        variational_posterior.variance.gradient += 0.5 * (1. /S - (variational_wets[:, :, np.newaxis] * sigma_diag[:,np.newaxis,:]).sum(axis=0))
        self.px_mu.gradient -= (variational_wets[:,:,np.newaxis] * sigma_mu).sum(axis=1)
        # self.px_var.gradient -= 0.5 * (variational_wets[:,:,np.newaxis, np.newaxis] * ((np.linalg.inv(self.px_var))[:,np.newaxis, :,:] - sigma2_S
        #                                         - sigma_S_sigma) ).sum(axis=1)
        dL_dcov = 0.5 * (variational_wets[:,:,np.newaxis, np.newaxis] * (cov_inv[:,np.newaxis, :,:] - sigma2_S
                                                - sigma_S_sigma) ).sum(axis=1)
        dL_dlmatrix = np.zeros((dL_dcov.shape))
        for k in range(mu_minus.shape[0]):
            dL_dlmatrix[k,:,:] = 2 * np.dot(dL_dcov[k,:,:], self.px_lmatrix[k,:,:])
        self.px_lmatrix.gradient -=  dL_dlmatrix
        # print self.px_lmatrix
        # print 'test'
        # print dL_dlmatrix

        dL_dw = np.zeros((self.variational_wi.shape))
        ew = np.exp(self.variational_wi)
        # ew = np.exp(wi_max)
        sumew = ew.sum(axis=0)
        dL_dq = ((0.5*(np.log(np.linalg.det(cov_k))[:,np.newaxis] + (sigma_S).sum(axis=2).sum(axis=2) + mu_sigma_mu) - (np.log(wets/variational_wets) - 1)))
        dq_dwi = ((sumew - ew) * ew ) / (sumew**2)
        for i in range(mu_minus.shape[1]):
            dq_dw = np.diag(dq_dwi[:,i])
            for j in range(mu_minus.shape[0]):
                for k in range(mu_minus.shape[0]):
                    if j != k:
                        dq_dw[j, k] = -ew[j, i] * ew[k,i] / (sumew[i]**2)
            dL_dw[:,i] = np.dot(dq_dw, dL_dq[:,i])
        self.variational_wi.gradient -= dL_dw
        # print dL_dw
        # for k in range(mu_minus.shape[1]):
        #     dq_dw_ij = np.zeros((mu_minus.shape[0],mu_minus.shape[0]))
        #     # print k
        #     for i in range(mu_minus.shape[0]):
        #         for j in range(mu_minus.shape[0]):
        #             if i == j:
        #                 dq_dw_ij[i,j] = ew[i,j]/sumew[k] - (ew[i,j]/sumew[k])**2
        #             else :
        #                 dq_dw_ij[i,j] = - ew[i,k] * ew[j,k] / sumew[k]**2
        #     # print dq_dw_ij
        #     dL_dw[:, k] = np.dot(dq_dw_ij, dL_dq[:,k])

        # print (dL_dw)
        
        # self.variational_wi.gradient -= dq_dw
        # self.variational_wi.gradient -= (0.5*(np.log(np.linalg.det(self.px_var))[:,np.newaxis] + (sigma_S).sum(axis=2).sum(axis=2) + mu_sigma_mu) - (np.log)(wets/variational_wets) - 1)*(
        #                                 (np.exp(self.variational_wi).sum(axis = 0) - np.exp(self.variational_wi))  * np.exp(self.variational_wi)/ (self.variational_wi.sum(axis=0))**2)
        # print  (np.exp(self.variational_wi).sum(axis = 0) - np.exp(self.variational_wi))  * np.exp(self.variational_wi)/ (np.exp(self.variational_wi).sum(axis=0))**2

        # print self.variational_wi.gradient


    def check_weights(self, weights):
        assert weights.min() >= -64.0
        assert weights.max() <= 64.0
        # assert weights.sum() == 1.0

    def check_all_weights(self):
        self.check_weights(self.variational_wi)
        # self.check_weights(self.pi)


class SpikeAndSlabPrior(VariationalPrior):
    def __init__(self, pi=None, learnPi=False, variance = 1.0, group_spike=False, name='SpikeAndSlabPrior', **kw):
        super(SpikeAndSlabPrior, self).__init__(name=name, **kw)
        self.group_spike = group_spike
        self.variance = Param('variance',variance)
        self.learnPi = learnPi
        if learnPi:
            self.pi = Param('Pi', pi, Logistic(1e-10,1.-1e-10))
        else:
            self.pi = Param('Pi', pi, __fixed__)
        self.link_parameter(self.pi)


    def KL_divergence(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        if self.group_spike:
            gamma = variational_posterior.gamma.values[0]
        else:
            gamma = variational_posterior.gamma.values
        if len(self.pi.shape)==2:
            idx = np.unique(variational_posterior.gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        var_gamma = (gamma*np.log(gamma/pi)).sum()+((1-gamma)*np.log((1-gamma)/(1-pi))).sum()
        return var_gamma+ (gamma* (np.log(self.variance)-1. +var_mean + var_S)).sum()/2.

    def update_gradients_KL(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        if self.group_spike:
            gamma = variational_posterior.gamma.values[0]
        else:
            gamma = variational_posterior.gamma.values
        if len(self.pi.shape)==2:
            idx = np.unique(variational_posterior.gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        if self.group_spike:
            dgamma = np.log((1-pi)/pi*gamma/(1.-gamma))/variational_posterior.num_data
        else:
            dgamma = np.log((1-pi)/pi*gamma/(1.-gamma))
        variational_posterior.binary_prob.gradient -= dgamma+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
        mu.gradient -= gamma*mu/self.variance
        S.gradient -= (1./self.variance - 1./S) * gamma /2.
        if self.learnPi:
            if len(self.pi)==1:
                self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum()
            elif len(self.pi.shape)==1:
                self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum(axis=0)
            else:
                self.pi[idx].gradient = (gamma/self.pi[idx] - (1.-gamma)/(1.-self.pi[idx]))

class VariationalPosterior(Parameterized):
    def __init__(self, means=None, variances=None, name='latent space', *a, **kw):
        super(VariationalPosterior, self).__init__(name=name, *a, **kw)
        self.mean = Param("mean", means)
        self.variance = Param("variance", variances, Logexp())
        self.ndim = self.mean.ndim
        self.shape = self.mean.shape
        self.num_data, self.input_dim = self.mean.shape
        self.link_parameters(self.mean, self.variance)
        self.num_data, self.input_dim = self.mean.shape
        if self.has_uncertain_inputs():
            assert self.variance.shape == self.mean.shape, "need one variance per sample and dimenion"

    def set_gradients(self, grad):
        self.mean.gradient, self.variance.gradient = grad

    def _raveled_index(self):
        index = np.empty(dtype=int, shape=0)
        size = 0
        for p in self.parameters:
            index = np.hstack((index, p._raveled_index()+size))
            size += p._realsize_ if hasattr(p, '_realsize_') else p.size
        return index

    def has_uncertain_inputs(self):
        return not self.variance is None

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size
            n.size = n.mean.size + n.variance.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(VariationalPosterior, self).__getitem__(s)

class NormalPosterior(VariationalPosterior):
    '''
    NormalPosterior distribution for variational approximations.

    holds the means and variances for a factorizing multivariate normal distribution
    '''

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self, *args, **kwargs)

    def KL(self, other):
        """Compute the KL divergence to another NormalPosterior Object. This only holds, if the two NormalPosterior objects have the same shape, as we do computational tricks for the multivariate normal KL divergence.
        """
        return .5*(
            np.sum(self.variance/other.variance)
            + ((other.mean-self.mean)**2/other.variance).sum()
            - self.num_data * self.input_dim
            + np.sum(np.log(other.variance)) - np.sum(np.log(self.variance))
            )

class SpikeAndSlabPosterior(VariationalPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, group_spike=False, sharedX=False, name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        super(SpikeAndSlabPosterior, self).__init__(means, variances, name)
        self.group_spike = group_spike
        self.sharedX = sharedX
        if sharedX:
            self.mean.fix(warning=False)
            self.variance.fix(warning=False)
        if group_spike:
            self.gamma_group = Param("binary_prob_group",binary_prob.mean(axis=0),Logistic(1e-10,1.-1e-10))
            self.gamma = Param("binary_prob",binary_prob, __fixed__)
            self.link_parameters(self.gamma_group,self.gamma)
        else:
            self.gamma = Param("binary_prob",binary_prob,Logistic(1e-10,1.-1e-10))
            self.link_parameter(self.gamma)

    def propogate_val(self):
        if self.group_spike:
            self.gamma.values[:] = self.gamma_group.values

    def collate_gradient(self):
        if self.group_spike:
            self.gamma_group.gradient = self.gamma.gradient.reshape(self.gamma.shape).sum(axis=0)

    def set_gradients(self, grad):
        self.mean.gradient, self.variance.gradient, self.gamma.gradient = grad

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['binary_prob'] = self.binary_prob[s]
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n.parameters[dc['binary_prob']._parent_index_] = dc['binary_prob']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size - self.gamma.size
            n.size = n.mean.size + n.variance.size + n.gamma.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(SpikeAndSlabPosterior, self).__getitem__(s)

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot_SpikeSlab(self,*args, **kwargs)
