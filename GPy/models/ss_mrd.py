"""
The Maniforld Relevance Determination model with the spike-and-slab prior
"""

import numpy as np
from ..core import Model
from .ss_gplvm import SSGPLVM
from ..core.parameterization.variational import SpikeAndSlabPrior
from ..util.misc import param_to_array
from ..kern import RBF
from numpy.linalg.linalg import LinAlgError

class SSMRD(Model):
    
    def __init__(self, Ylist, input_dim, X=None, X_variance=None, Gammas=None, initx = 'PCA_concat', initz = 'permute', 
                 num_inducing=10, Zs=None, kernel=None, inference_methods=None, likelihoods=None, 
                 pi=0.5, name='ss_mrd', Ynames=None, mpi_comm=None):
        super(SSMRD, self).__init__(name)
        self.mpi_comm = mpi_comm
        
        # initialize X for individual models
        X, X_variance, Gammas, fracs = self._init_X(Ylist, input_dim, X, X_variance, Gammas, initx)
        
        if kernel is None:
            kernel = RBF(input_dim, lengthscale=1./fracs, ARD=True)
        if Zs is None:
            Zs = [None]* len(Ylist)
        if likelihoods is None:
            likelihoods = [None]* len(Ylist)
        if inference_methods is None:
            inference_methods = [None]* len(Ylist)
        
        self.var_priors = [VarPrior_SSMRD(nModels=len(Ylist),pi=pi,learnPi=False, group_spike=True) for i in xrange(len(Ylist))]
        self.models = [SSGPLVM(y, input_dim, X=X, X_variance=X_variance, Gamma=Gammas[i], num_inducing=num_inducing,Z=Zs[i], learnPi=False, group_spike=True,
                               kernel=kernel.copy(),inference_method=inference_methods[i],likelihood=likelihoods[i], variational_prior=self.var_priors[i],
                               name='model_'+str(i), mpi_comm=mpi_comm) for i,y in enumerate(Ylist)]
        self.link_parameters(*(self.models))
        
        self.models[0].X.mean.tie_vector(*[m.X.mean for m in self.models[1:]])
        self.models[0].X.variance.tie_vector(*[m.X.variance for m in self.models[1:]])
        self.models[0].kern.tie_vector(*[m.kern for m in self.models[1:]])
        
    def parameters_changed(self):
        varp_list = [m.X for m in self.models]
        [vp._update_inernal(varp_list) for vp in self.var_priors]
        super(SSMRD, self).parameters_changed()        
        self._log_marginal_likelihood = sum([m._log_marginal_likelihood for m in self.models])

    def log_likelihood(self):
        return self._log_marginal_likelihood
    
    def _init_X(self, Ylist, input_dim, X=None, X_variance=None, Gammas=None, initx='PCA_concat'):
        
        # Divide latent dimensions
        idx = np.empty((input_dim,),dtype=np.int)
        residue = (input_dim)%(len(Ylist))
        for i in xrange(len(Ylist)):
            if i < residue:
                size = input_dim/len(Ylist)+1
                idx[i*size:(i+1)*size] = i
            else:
                size = input_dim/len(Ylist)
                idx[i*size+residue:(i+1)*size+residue] = i
        
        if X is None:
            X = np.empty((Ylist[0].shape[0],input_dim))
            fracs = np.empty((input_dim,))
            from ..util.initialization import initialize_latent
            for i in xrange(len(Ylist)):
                Y = Ylist[i]
                dim = (idx==i).sum()
                if dim>0:
                    x, fr = initialize_latent('PCA', dim, Y)
                    X[:,idx==i] = x
                    fracs[idx==i] = fr
        else:
            fracs = np.ones(input_dim)
    
        if X_variance is None: # The variance of the variational approximation (S)
            X_variance = np.random.uniform(0,.1,X.shape)
            
        if Gammas is None:
            Gammas = []
            for x in X:
                gamma = np.empty_like(X) # The posterior probabilities of the binary variable in the variational approximation
                gamma[:] = 0.5 + 0.1 * np.random.randn(X.shape[0], input_dim)
                gamma[gamma>1.-1e-9] = 1.-1e-9
                gamma[gamma<1e-9] = 1e-9
                Gammas.append(gamma)
        return X, X_variance, Gammas, fracs

    @Model.optimizer_array.setter
    def optimizer_array(self, p):
        if self.mpi_comm != None:
            if self._IN_OPTIMIZATION_ and self.mpi_comm.rank==0:
                self.mpi_comm.Bcast(np.int32(1),root=0)
            self.mpi_comm.Bcast(p, root=0)        
        Model.optimizer_array.fset(self,p)
        
    def optimize(self, optimizer=None, start=None, **kwargs):
        self._IN_OPTIMIZATION_ = True
        if self.mpi_comm==None:
            super(SSMRD, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==0:
            super(SSMRD, self).optimize(optimizer,start,**kwargs)
            self.mpi_comm.Bcast(np.int32(-1),root=0)
        elif self.mpi_comm.rank>0:
            x = self.optimizer_array.copy()
            flag = np.empty(1,dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag,root=0)
                if flag==1:
                    try:
                        self.optimizer_array = x
                        self._fail_count = 0
                    except (LinAlgError, ZeroDivisionError, ValueError):
                        if self._fail_count >= self._allowed_failures:
                            raise
                        self._fail_count += 1
                elif flag==-1:
                    break
                else:
                    self._IN_OPTIMIZATION_ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self._IN_OPTIMIZATION_ = False




class VarPrior_SSMRD(SpikeAndSlabPrior):
    def __init__(self, nModels, pi=None, learnPi=False, group_spike=True, variance = 1.0, name='SSMRDPrior', **kw):
        self.nModels = nModels
        self._b_prob_all = 0.5
        super(VarPrior_SSMRD, self).__init__(pi=pi,learnPi=learnPi,group_spike=group_spike,variance=variance, name=name, **kw)
    
    def _update_inernal(self, varp_list):
        """Make an update of the internal status by gathering the variational posteriors for all the individual models."""
        # The probability for the binary variable for the same latent dimension of any of the models is on.
        if self.group_spike:
            self._b_prob_all = 1.-param_to_array(varp_list[0].binary_prob[0])
            [np.multiply(self._b_prob_all, 1.-vp.binary_prob[0], self._b_prob_all) for vp in varp_list[1:]]
        else:
            self._b_prob_all = 1.-param_to_array(varp_list[0].binary_prob)
            [np.multiply(self._b_prob_all, 1.-vp.binary_prob, self._b_prob_all) for vp in varp_list[1:]]
            

    def KL_divergence(self, variational_posterior, N=None):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        if len(self.pi.shape)==2:
            idx = np.unique(gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        ml = self._highest_parent_
        if hasattr(ml, 'models'):
            varp_list = [m.X for m in ml.models]
            [vp._update_inernal(varp_list) for vp in ml.var_priors]

        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        if self.group_spike:
            assert N is not None
            var_gamma = ((gamma*np.log(gamma/pi)).sum()+((1-gamma)*np.log((1-gamma)/(1-pi))).sum())/N
        else:
            var_gamma = (gamma*np.log(gamma/pi)).sum()+((1-gamma)*np.log((1-gamma)/(1-pi))).sum()
        return var_gamma +((1.-self._b_prob_all)*(np.log(self.variance)-1. +var_mean + var_S)).sum()/(2.*self.nModels)

    def update_gradients_KL(self, variational_posterior, N=None):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        if len(self.pi.shape)==2:
            idx = np.unique(gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        if self.group_spike:
            assert N is not None
            tmp = self._b_prob_all/(1.-gamma[0])
            gamma.gradient -= np.log((1-pi)/pi*gamma/(1.-gamma))/N +tmp*((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
        else:
            gamma.gradient -= np.log((1-pi)/pi*gamma/(1.-gamma))+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
        mu.gradient -= (1.-self._b_prob_all)*mu/(self.variance*self.nModels)
        S.gradient -= (1./self.variance - 1./S) * (1.-self._b_prob_all) /(2.*self.nModels)
        if self.learnPi:
            raise 'Not Supported!'

