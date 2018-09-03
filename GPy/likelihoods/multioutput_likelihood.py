# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# Multioutput likelihood structure is similar to the 
# corresponding structure in GPstuff. If building complex
# multioutput models on top of this class and need a reference,
# check GPstuff project.

import numpy as np
from scipy import stats, special
from . import link_functions
from .likelihood import Likelihood
from .mixed_noise import MixedNoise
from .gaussian import Gaussian
from ..core.parameterization import Param
from paramz.transformations import Logexp
from ..core.parameterization import Parameterized
from GPy.util.multioutput import index_to_slices
import itertools

class MultioutputLikelihood(MixedNoise):
    '''
    CombinedLikelihood is used to combine different likelihoods for 
    multioutput models, where different outputs have different observation models.
    
    As input the likelihood takes a list of likelihoods used. The likelihood
    uses "output_index" in Y_metadata to connect observations to likelihoods.
    '''
    def __init__(self, likelihoods_list, name='multioutput_likelihood'):
        super(Likelihood, self).__init__(name=name)
        
        indices, inverse = self._unique_likelihoods(likelihoods_list)
        self.link_parameters(*[likelihoods_list[i] for i in indices])
        
        self.index_map = [indices[i] for i in inverse]
        self.inverse = inverse 
        
        self.gradient_sizes = [likelihoods_list[i].size for i in indices]
        self.gradient_index = np.cumsum(self.gradient_sizes) - self.gradient_sizes[0]
        
        self.likelihoods_list = likelihoods_list
        
        self.gp_link = None
        self.log_concave = False
        self.not_block_really = False
    
    def _unique_likelihoods(self, likelihoods_list):
        indices = []
        inverse = []
        for i in range(len(likelihoods_list)):
            for j in indices:
                if likelihoods_list[i] is likelihoods_list[j]:
                    inverse += [j]
                    break
            if len(inverse) <= i:
                indices += [i]
                inverse += [i]
        return indices, inverse
    
    def moments_match_ep(self, data_i, tau_i, v_i, Y_metadata_i):
        return self.likelihoods_list[Y_metadata_i["output_index"][0]].moments_match_ep(data_i, tau_i, v_i, Y_metadata_i)

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = [self.index_map[i] for i in Y_metadata['output_index'].flatten()]
        return np.array([dL_dKdiag[ind==i].sum() for i in np.unique(self.index_map)])
    
    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gk', boost_grad=1.):
        ind = [self.index_map[i] for i in Y_metadata['output_index'].flatten()]
        slices = index_to_slices(ind)
        grads = np.zeros((self.size))
        for i in range(len(slices)):
            if self.likelihoods_list[i].size > 0:
                ii = self.inverse[i] ## index in our gradient_sizes and gradient_index -lists
                for j in range(len(slices[i])): 
                    grads[self.gradient_index[ii]:self.gradient_index[ii]+self.gradient_sizes[ii]] += self.likelihoods_list[i].ep_gradients(Y[slices[i][j],:], cav_tau[slices[i][j]], cav_v[slices[i][j]], dL_dKdiag = dL_dKdiag[slices[i][j]], Y_metadata=Y_metadata, quad_mode=quad_mode, boost_grad=boost_grad)
        return grads

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        mu_new = np.zeros(mu.shape )
        var_new = np.zeros(var.shape )
        for j in outputs:
            m, v = self.likelihoods_list[j].predictive_values(mu[ind==j,:], var[ind==j,:], full_cov, Y_metadata=None)
            mu_new[ind==j,:] = m
            var_new[ind==j,:] = v
        return mu_new, var_new

    def predictive_variance(self, mu, sigma, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        var = np.zeros( (sigma.size) )
        for j in outputs:
            v = self.likelihoods_list[j].predictive_variance(mu[ind==j,:],
                sigma[ind==j,:],Y_metadata=None)
            var[ind==j,:] = np.hstack(v)
        return [v[:,None] for v in var.T]

    def pdf(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        pdf = np.zeros(y.shape)
        for j in outputs:
            pdf[ind==j,:] = self.likelihoods_list[j].pdf(f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return pdf

    def pdf_link(self, inv_link_f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        pdf_link = np.zeros(y.shape)
        for j in outputs:
            pdf_link[ind==j,:] = self.likelihoods_list[j].pdf_link(inv_link_f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return pdf_link

    def logpdf(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        if ind.shape[0]==1:
            ind = ind[0]*np.ones(f.shape[0])
            y = y*np.ones(f.shape)
        lpdf = np.zeros(f.shape)
        for j in outputs:
            lpdf[np.where(ind==j)[0],:] = self.likelihoods_list[j].logpdf(f[np.where(ind==j)[0],:], y[np.where(ind==j)[0],:], Y_metadata=None)
        return lpdf

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        logpdf_link = np.zeros(y.shape)
        for j in outputs:
            logpdf_link[ind==j,:] = self.likelihoods_list[j].logpdf_link(inv_link_f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return logpdf_link
    
    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        dlogpdf_dlink = np.zeros(y.shape)
        for j in outputs:
            dlogpdf_dlink[ind==j,:] = self.likelihoods_list[j].dlogpdf_dlink(inv_link_f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return dlogpdf_dlink
    
    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        d2logpdf_dlink2 = np.zeros(y.shape)
        for j in outputs:
            d2logpdf_dlink2[ind==j,:] = self.likelihoods_list[j].d2logpdf_dlink2(inv_link_f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return d2logpdf_dlink2
     
    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        d3logpdf_dlink3 = np.zeros(y.shape)
        for j in outputs:
            d3logpdf_dlink3[ind==j,:] = self.likelihoods_list[j].d3logpdf_dlink3(inv_link_f[ind==j,:], y[ind==j,:], Y_metadata=None)
        return d3logpdf_dlink3
        
    def log_predictive_density(self, y_test, mu_star, var_star, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        log_pred = np.zeros(y_test.shape)
        for j in outputs:
            log_pred[ind==j,:] = self.likelihoods_list[j].log_predictive_density(y_test[ind==j,:], mu_star[ind==j,:], var_star[ind==j,:], Y_metadata=None)
        return log_pred

    def dlogpdf_dtheta(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        if ind.shape[0]==1:
            ind = ind[0]*np.ones(f.shape[0])
            y = y*np.ones(f.shape)
        slices = index_to_slices(ind)
        pdf = np.zeros((self.size, f.shape[0], f.shape[1]) )
        for i in range(len(slices)):
            if self.likelihoods_list[i].size > 0:
                ii = self.inverse[i]
                for j in range(len(slices[i])):
                    pdf[self.gradient_index[ii]:self.gradient_index[ii]+self.gradient_sizes[ii], slices[i][j],:] = self.likelihoods_list[i].dlogpdf_dtheta(f[slices[i][j],:], y[slices[i][j],:], Y_metadata=None)
        return pdf

    def d2logpdf_df2(self, f, y, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros(f.shape)
        for j in outputs:
            Q[ind==j,:] = self.likelihoods_list[j].d2logpdf_df2(f[ind==j,:],
                y[ind==j,:],Y_metadata=None)
        return Q

    def dlogpdf_df(self, f, y, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros(f.shape)
        for j in outputs:
            Q[ind==j,:] = self.likelihoods_list[j].dlogpdf_df(f[ind==j,:],
                y[ind==j,:],Y_metadata=None)
        return Q

    def d3logpdf_df3(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros(f.shape)
        for j in outputs:
            Q[ind==j,:] = self.likelihoods_list[j].d3logpdf_df3(f[ind==j,:],
                y[ind==j,:],Y_metadata=None)
        return Q
    
    def dlogpdf_df_dtheta(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        if ind.shape[0]==1:
            ind = ind[0]*np.ones(f.shape[0])
            y = y*np.ones(f.shape)
        slices = index_to_slices(ind)
        pdf = np.zeros((self.size, f.shape[0], f.shape[1]) )
        for i in range(len(slices)):
            if self.likelihoods_list[i].size > 0:
                ii = self.inverse[i]
                for j in range(len(slices[i])):
                    pdf[self.gradient_index[ii]:self.gradient_index[ii]+self.gradient_sizes[ii], slices[i][j],:] = self.likelihoods_list[i].dlogpdf_df_dtheta(f[slices[i][j],:], y[slices[i][j],:], Y_metadata=None)
        return pdf
    
    def d2logpdf_df2_dtheta(self, f, y, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        if ind.shape[0]==1:
            ind = ind[0]*np.ones(f.shape[0])
            y = y*np.ones(f.shape)
        slices = index_to_slices(ind)
        pdf = np.zeros((self.size, f.shape[0], f.shape[1]) )
        for i in range(len(slices)):
            if self.likelihoods_list[i].size > 0:
                ii = self.inverse[i]
                for j in range(len(slices[i])):
                    pdf[self.gradient_index[ii]:self.gradient_index[ii]+self.gradient_sizes[ii], slices[i][j],:] = self.likelihoods_list[i].d2logpdf_df2_dtheta(f[slices[i][j],:], y[slices[i][j],:], Y_metadata=None)
        return pdf