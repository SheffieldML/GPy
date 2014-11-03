"""
"""
import numpy as np
from ...core import Model
from ...core.parameterization import variational

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
    The class for inference of new X with given new Y. (do_test_latent)
    
    :param model: the GPy model used in inference
    :type model: GPy.core.Model
    :param Y: the new observed data for inference
    :type Y: numpy.ndarray
    """
    def __init__(self, model, Y, name='inferenceX', init='L2'):
        if np.isnan(Y).any():
            assert Y.shape[0]==1, "The current implementation of inference X only support one data point at a time with missing data!"
            self.missing_data = True
            self.valid_dim = np.logical_not(np.isnan(Y[0]))
        else:
            self.missing_data = False
        super(InferenceX, self).__init__(name)
        self.likelihood = model.likelihood.copy()
        self.kern = model.kern.copy()
        if model.kern.useGPU:
            from ...models import SSGPLVM
            if isinstance(model, SSGPLVM):
                self.kern.GPU_SSRBF(True)
            else:
                self.kern.GPU(True)
        from copy import deepcopy
        self.posterior = deepcopy(model.posterior)
        self.variational_prior = model.variational_prior.copy()
        self.Z = model.Z.copy()
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
        idx = dist.argmin(axis=1)
        
        from ...models import SSGPLVM
        from ...util.misc import param_to_array
        if isinstance(model, SSGPLVM):
            X = variational.SpikeAndSlabPosterior(param_to_array(model.X.mean[idx]), param_to_array(model.X.variance[idx]), param_to_array(model.X.gamma[idx]))
            if model.group_spike:
                X.gamma.fix()
        else:
            X = variational.NormalPosterior(param_to_array(model.X.mean[idx]), param_to_array(model.X.variance[idx]))
        
        return X
        
    def compute_dL(self):
        # Common computation
        beta = 1./np.fmax(self.likelihood.variance, 1e-6)
        output_dim = self.Y.shape[-1]
        wv = self.posterior.woodbury_vector
        if self.missing_data:
            wv = wv[:,self.valid_dim]
            output_dim = self.valid_dim.sum()
            self.dL_dpsi2 = beta*(output_dim*self.posterior.woodbury_inv - np.einsum('md,od->mo',wv, wv))/2.
            self.dL_dpsi1 = beta*np.dot(self.Y[:,self.valid_dim], wv.T)
            self.dL_dpsi0 = - beta/2.* np.ones(self.Y.shape[0])
        else:
            self.dL_dpsi2 = beta*(output_dim*self.posterior.woodbury_inv - np.einsum('md,od->mo',wv, wv))/2.
            self.dL_dpsi1 = beta*np.dot(self.Y, wv.T)
            self.dL_dpsi0 = -beta/2.* np.ones(self.Y.shape[0])            #self.dL_dpsi0[:] = 0
                
    def parameters_changed(self):
        psi0 = self.kern.psi0(self.Z, self.X)
        psi1 = self.kern.psi1(self.Z, self.X)
        psi2 = self.kern.psi2(self.Z, self.X)

        self._log_marginal_likelihood = (self.dL_dpsi2*psi2).sum()+(self.dL_dpsi1*psi1).sum()+(self.dL_dpsi0*psi0).sum()
        X_grad = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.dL_dpsi0, dL_dpsi1=self.dL_dpsi1, dL_dpsi2=self.dL_dpsi2)
        self.X.set_gradients(X_grad)
        
        from ...core.parameterization.variational import SpikeAndSlabPrior
        if isinstance(self.variational_prior, SpikeAndSlabPrior):
            # Update Log-likelihood
            KL_div = self.variational_prior.KL_divergence(self.X, N=self.Y.shape[0])
            # update for the KL divergence
            self.variational_prior.update_gradients_KL(self.X, N=self.Y.shape[0])
        else:
            # Update Log-likelihood
            KL_div = self.variational_prior.KL_divergence(self.X)
            # update for the KL divergence
            self.variational_prior.update_gradients_KL(self.X)
        self._log_marginal_likelihood += -KL_div
        
    def log_likelihood(self):
        return self._log_marginal_likelihood

