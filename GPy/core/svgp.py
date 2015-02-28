# Copyright (c) 2014, James Hensman, Alex Matthews
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
from ..util import choleskies
from .sparse_gp import SparseGP
from .parameterization.param import Param
from ..inference.latent_function_inference import SVGP as svgp_inf


class SVGP(SparseGP):
    def __init__(self, X, Y, Z, kernel, likelihood, name='SVGP', Y_metadata=None, batchsize=None):
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
        if batchsize is None:
            batchsize = X.shape[0]

        self.X_all, self.Y_all = X, Y
        # how to rescale the batch likelihood in case of minibatches
        self.batchsize = batchsize
        batch_scale = float(self.X_all.shape[0])/float(self.batchsize)
        #KL_scale = 1./np.float64(self.mpi_comm.size)
        KL_scale = 1.0

        import climin.util
        #Make a climin slicer to make drawing minibatches much quicker. Annoyingly, this doesn;t pickle.
        self.slicer = climin.util.draw_mini_slices(self.X_all.shape[0], self.batchsize)
        X_batch, Y_batch = self.new_batch()

        #create the SVI inference method
        inf_method = svgp_inf()

        SparseGP.__init__(self, X_batch, Y_batch, Z, kernel, likelihood, inference_method=inf_method,
                 name=name, Y_metadata=Y_metadata, normalizer=False)

        self.m = Param('q_u_mean', np.zeros((self.num_inducing, Y.shape[1])))
        chol = choleskies.triang_to_flat(np.tile(np.eye(self.num_inducing)[:,:,None], (1,1,Y.shape[1])))
        self.chol = Param('q_u_chol', chol)
        self.link_parameter(self.chol)
        self.link_parameter(self.m)

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.q_u_mean, self.q_u_chol, self.kern, self.X, self.Z, self.likelihood, self.Y, self.Y_metadata, KL_scale=1.0, batch_scale=float(self.X_all.shape[0])/float(self.X.shape[0]))

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

    def set_data(self, X, Y):
        """
        Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients
        """
        assert X.shape[1]==self.Z.shape[1]
        self.X, self.Y = X, Y

    def new_batch(self):
        """
        Return a new batch of X and Y by taking a chunk of data from the complete X and Y
        """
        i = self.slicer.next()
        return self.X_all[i], self.Y_all[i]

    def stochastic_grad(self, parameters):
        self.set_data(*self.new_batch())
        return self._grads(parameters)

    def optimizeWithFreezingZ(self):
        self.Z.fix()
        self.kern.fix()
        self.optimize('bfgs')
        self.Z.unfix()
        self.kern.constrain_positive()
        self.optimize('bfgs')
