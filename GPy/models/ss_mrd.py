"""
The Maniforld Relevance Determination model with the spike-and-slab prior
"""

from ..core import Model
from .ss_gplvm import SSGPLVM

class SSMRD(Model):
    
    def __init__(self, Ylist, input_dim, X=None, X_variance=None,
                 initx = 'PCA', initz = 'permute',
                 num_inducing=10, Z=None, kernel=None,
                 inference_method=None, likelihoods=None, name='ss_mrd', Ynames=None):
        super(SSMRD, self).__init__(name)
        
        self.updates = False
        self.models = [SSGPLVM(y, input_dim, X=X, X_variance=X_variance, num_inducing=num_inducing,Z=Z,init=initx,
                               kernel=kernel if kernel else None,inference_method=inference_method,likelihood=likelihoods,
                               name='model_'+str(i)) for i,y in enumerate(Ylist)]
        self.add_parameters(*(self.models))
        self.updates = True
        
        [[self.models[j].X.mean.flat[i:i+1].tie('mean_'+str(i)) for j in xrange(len(self.models))] for i in xrange(self.models[0].X.mean.size)]
        [[self.models[j].X.variance.flat[i:i+1].tie('var_'+str(i)) for j in xrange(len(self.models))] for i in xrange(self.models[0].X.variance.size)]

    def parameters_changed(self):
        super(SSMRD, self).parameters_changed()        
        self._log_marginal_likelihood = sum([m._log_marginal_likelihood for m in self.models])

    def log_likelihood(self):
        return self._log_marginal_likelihood