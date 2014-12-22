# Copyright (c) 2014, James Hensman, Alex Matthews
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
from ..util import choleskies
from sparse_gp import SparseGP
from parameterization.param import Param

class SVGP(SparseGP):
    def __init__(self, X, Y, Z, kernel, likelihood, name='SVGP', Y_metadata=None):
        """
        Stochastic Variational GP.

        For Gaussian Likelihoods, this implements

        Gaussian Processes for Big data, Hensman, Fusi and Lawrence, UAI 2013,

        But without natural gradients. We'll use the lower-triangluar
        representation of the covariance matrix to ensure
        positive-definiteness.

        For Non Gaussian Likelihoods, this implements

        Hensman, Matthews and Ghahramani, Scalable Variational GP Classification, ArXiv 1411.2005
        """

        #create the SVI inference method
        from ..inference.latent_function_inference import SVGP as svgp_inf
        inf_method = svgp_inf()

        SparseGP.__init__(self,X, Y, Z, kernel, likelihood, inference_method=inf_method,
                 name=name, Y_metadata=Y_metadata, normalizer=False)

        #?? self.set_data(X, Y)

        self.m = Param('q_u_mean', np.zeros((self.num_inducing, Y.shape[1])))
        chol = choleskies.triang_to_flat(np.tile(np.eye(self.num_inducing)[:,:,None], (1,1,Y.shape[1])))
        self.chol = Param('q_u_chol', chol)
        self.link_parameter(self.chol)
        self.link_parameter(self.m)

        #self.batch_scale = 1. # how to rescale the batch likelihood in case of minibatches


    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.q_u_mean, self.q_u_chol, self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata)

        #update the kernel gradients
        self.kern.update_gradients_full(self.grad_dict['dL_dKmm'], self.Z)
        grad = self.kern.gradient.copy()
        self.kern.update_gradients_full(self.grad_dict['dL_dKmn'], self.Z, self.X)
        grad += self.kern.gradient
        self.kern.update_gradients_diag(self.grad_dict['dL_dKdiag'], self.X)
        self.kern.gradient += grad
        if not self.Z.is_fixed:# only compute these expensive gradients if we need them
            self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z) + self.kern.gradients_X(self.grad_dict['dL_dKmn'], self.Z, self.X)


        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        #update the variational parameter gradients:
        self.m.gradient = self.grad_dict['dL_dm']
        self.chol.gradient = self.grad_dict['dL_dchol']


    #def set_data(self, X, Y):
        #assert X.shape[1]==self.Z.shape[1]
        #self.X, self.Y = GPy.core.ObsAr(X), Y

    def optimizeWithFreezingZ(self):
        self.Z.fix()
        self.kern.fix()
        self.optimize('bfgs')
        self.Z.unfix()
        self.kern.constrain_positive()
        self.optimize('bfgs')

#class SPGPC_stoch(SPGPC):
    #def __init__(self, X, Y, Z, kern=None, likelihood=None, batchsize=10):
        #SPGPC.__init__(self, X[:1], Y[:1], Z, kern, likelihood)
        #self.X_all, self.Y_all = X, Y
        #self.batchsize = batchsize
        #self.batch_scale = float(self.X_all.shape[0])/float(self.batchsize)
#
    #def stochastic_grad(self, w):
        #i = np.random.permutation(self.X_all.shape[0])[:self.batchsize]
        #self.set_data(self.X_all[i], self.Y_all[i])
        #return self._grads(w)




