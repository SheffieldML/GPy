"""
"""
import numpy as np
from ...core import Model
from ...core.parameterization import variational

def inference_newX(model, Y_new, optimize=True, init='L2'):
    infr_m = Inference_X(model, Y_new, init=init)
    
    if optimize:
        infr_m.optimize()
        
    return infr_m.X, infr_m

class Inference_X(Model):
    """The class for inference of new X with given new Y. (do_test_latent)"""
    def __init__(self, model, Y, name='inference_X', init='L2'):
        """TODO: give comments"""
        if np.isnan(Y).any():
            assert Y.shape[0]==1, "The current implementation of inference X only support one data point at a time with missing data!"
            self.missing_data = True
            self.valid_dim = np.logical_not(np.isnan(Y[0]))
        else:
            self.missing_data = False
        super(Inference_X, self).__init__(name)
        self.likelihood = model.likelihood.copy()
        self.kern = model.kern.copy()
        if model.kern.useGPU:
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
                #[X.gamma[:,i].tie_together() for i in xrange(X.gamma.shape[1])] # Tie columns together
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
            self.dL_dpsi0 = -output_dim * beta/2.* np.ones(self.Y.shape[0])
        else:
            self.dL_dpsi2 = beta*(output_dim*self.posterior.woodbury_inv - np.einsum('md,od->mo',wv, wv))/2.
            self.dL_dpsi1 = beta*np.dot(self.Y, wv.T)
            self.dL_dpsi0 = -output_dim * beta/2.* np.ones(self.Y.shape[0])
                
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
        
    def log_likelihood(self):
        return self._log_marginal_likelihood

