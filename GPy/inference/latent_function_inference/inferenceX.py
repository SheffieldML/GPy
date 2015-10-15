# Copyright (c) 2014, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ...core import Model
from GPy.core.parameterization import variational
from ...util.linalg import tdot

def infer_newX(model, Y_new, optimize=True, init='L2'):
    """
    Infer the distribution of X for the new observed data *Y_new*.

    :param model: the GPy model used in inference
    :type model: GPy.core.Model
    :param Y_new: the new observed data for inference
    :type Y_new: numpy.ndarray
    :param optimize: whether to optimize the location of new X (True by default)
    :type optimize: boolean
    :return: a tuple containing the estimated posterior distribution of X and the model that optimize X
    :rtype: (GPy.core.parameterization.variational.VariationalPosterior, GPy.core.Model)
    """
    infr_m = InferenceX(model, Y_new, init=init)

    if optimize:
        infr_m.optimize()

    return infr_m.X, infr_m

class InferenceX(Model):
    """
    The model class for inference of new X with given new Y. (replacing the "do_test_latent" in Bayesian GPLVM)
    It is a tiny inference model created from the original GP model. The kernel, likelihood (only Gaussian is supported at the moment) 
    and posterior distribution are taken from the original model.
    For Regression models and GPLVM, a point estimate of the latent variable X will be inferred. 
    For Bayesian GPLVM, the variational posterior of X will be inferred. 
    X is inferred through a gradient optimization of the inference model.

    :param model: the GPy model used in inference
    :type model: GPy.core.Model
    :param Y: the new observed data for inference
    :type Y: numpy.ndarray
    :param init: the distance metric of Y for initializing X with the nearest neighbour.
    :type init: 'L2', 'NCC' and 'rand'
    """
    def __init__(self, model, Y, name='inferenceX', init='L2'):
        if np.isnan(Y).any() or getattr(model, 'missing_data', False):
            assert Y.shape[0]==1, "The current implementation of inference X only support one data point at a time with missing data!"
            self.missing_data = True
            self.valid_dim = np.logical_not(np.isnan(Y[0]))
            self.ninan = getattr(model, 'ninan', None)
        else:
            self.missing_data = False
        super(InferenceX, self).__init__(name)
        self.likelihood = model.likelihood.copy()
        self.kern = model.kern.copy()
#         if model.kern.useGPU:
#             from ...models import SSGPLVM
#             if isinstance(model, SSGPLVM):
#                 self.kern.GPU_SSRBF(True)
#             else:
#                 self.kern.GPU(True)
        from copy import deepcopy
        self.posterior = deepcopy(model.posterior)
        if isinstance(model.X, variational.VariationalPosterior):
            self.uncertain_input = True
            from ...models.ss_gplvm import IBPPrior
            from ...models.ss_mrd import IBPPrior_SSMRD
            if isinstance(model.variational_prior, IBPPrior) or isinstance(model.variational_prior, IBPPrior_SSMRD):
                self.variational_prior = variational.SpikeAndSlabPrior(pi=0.5, learnPi=False, group_spike=False)
            else:
                self.variational_prior = model.variational_prior.copy()
        else:
            self.uncertain_input = False
        if hasattr(model, 'Z'):
            self.sparse_gp = True
            self.Z = model.Z.copy()
        else:
            self.sparse_gp = False
            self.uncertain_input = False
            self.Z = model.X.copy()
        self.Y = Y
        self.X = self._init_X(model, Y, init=init)
        self.compute_dL()

        self.link_parameter(self.X)

    def _init_X(self, model, Y_new, init='L2'):
        # Initialize the new X by finding the nearest point in Y space.

        Y = model.Y
        if self.missing_data:
            Y = Y[:,self.valid_dim]
            Y_new = Y_new[:,self.valid_dim]
            dist = -2.*Y_new.dot(Y.T) + np.square(Y_new).sum(axis=1)[:,None]+ np.square(Y).sum(axis=1)[None,:]
        else:
            if init=='L2':
                dist = -2.*Y_new.dot(Y.T) + np.square(Y_new).sum(axis=1)[:,None]+ np.square(Y).sum(axis=1)[None,:]
            elif init=='NCC':
                dist = Y_new.dot(Y.T)
            elif init=='rand':
                dist = np.random.rand(Y_new.shape[0],Y.shape[0])
        idx = dist.argmin(axis=1)

        from ...models import SSGPLVM
        if isinstance(model, SSGPLVM):
            X = variational.SpikeAndSlabPosterior((model.X.mean[idx].values), (model.X.variance[idx].values), (model.X.gamma[idx].values))
            if model.group_spike:
                X.gamma.fix()
        else:
            if self.uncertain_input and self.sparse_gp:
                X = variational.NormalPosterior((model.X.mean[idx].values), (model.X.variance[idx].values))
            else:
                from ...core import Param
                X = Param('latent mean',(model.X[idx].values).copy())

        return X

    def compute_dL(self):
        # Common computation
        beta = 1./np.fmax(self.likelihood.variance, 1e-6)
        output_dim = self.Y.shape[-1]
        wv = self.posterior.woodbury_vector
        if self.missing_data:
            wv = wv[:,self.valid_dim]
            output_dim = self.valid_dim.sum()
            if self.ninan is not None:
                self.dL_dpsi2 = beta/2.*(self.posterior.woodbury_inv[:,:,self.valid_dim] - tdot(wv)[:, :, None]).sum(-1)
            else:
                self.dL_dpsi2 = beta/2.*(output_dim*self.posterior.woodbury_inv - tdot(wv))
            self.dL_dpsi1 = beta*np.dot(self.Y[:,self.valid_dim], wv.T)
            self.dL_dpsi0 = - beta/2.* np.ones(self.Y.shape[0])
        else:
            self.dL_dpsi2 = beta*(output_dim*self.posterior.woodbury_inv - tdot(wv))/2. #np.einsum('md,od->mo',wv, wv)
            self.dL_dpsi1 = beta*np.dot(self.Y, wv.T)
            self.dL_dpsi0 = -beta/2.*output_dim* np.ones(self.Y.shape[0])

    def parameters_changed(self):
        if self.uncertain_input:
            psi0 = self.kern.psi0(self.Z, self.X)
            psi1 = self.kern.psi1(self.Z, self.X)
            psi2 = self.kern.psi2(self.Z, self.X)
        else:
            psi0 = self.kern.Kdiag(self.X)
            psi1 = self.kern.K(self.X, self.Z)
            psi2 = np.dot(psi1.T,psi1)

        self._log_marginal_likelihood = (self.dL_dpsi2*psi2).sum()+(self.dL_dpsi1*psi1).sum()+(self.dL_dpsi0*psi0).sum()

        if self.uncertain_input:
            X_grad = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.dL_dpsi0, dL_dpsi1=self.dL_dpsi1, dL_dpsi2=self.dL_dpsi2)
            self.X.set_gradients(X_grad)
        else:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(psi1,self.dL_dpsi2)
            X_grad = self.kern.gradients_X_diag(self.dL_dpsi0, self.X)
            X_grad += self.kern.gradients_X(dL_dpsi1, self.X, self.Z)
            self.X.gradient = X_grad

        if self.uncertain_input:
            if isinstance(self.variational_prior, variational.SpikeAndSlabPrior):
                # Update Log-likelihood
                KL_div = self.variational_prior.KL_divergence(self.X)
                # update for the KL divergence
                self.variational_prior.update_gradients_KL(self.X)
            else:
                # Update Log-likelihood
                KL_div = self.variational_prior.KL_divergence(self.X)
                # update for the KL divergence
                self.variational_prior.update_gradients_KL(self.X)
            self._log_marginal_likelihood += -KL_div

    def log_likelihood(self):
        return self._log_marginal_likelihood

