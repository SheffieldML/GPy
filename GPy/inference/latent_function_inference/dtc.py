# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from ...util.linalg import jitchol, tdot, dtrtrs, dpotri, pdinv
import numpy as np
log_2_pi = np.log(2*np.pi)

class DTC(object):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    NB. It's not recommended to use this function! It's here for historical purposes. 

    """
    def __init__(self):
        self.const_jitter = 1e-6

    def inference(self, kern, X, X_variance, Z, likelihood, Y):
        assert X_variance is None, "cannot use X_variance with DTC. Try varDTC."

        #TODO: MAX! fix this!
        from ...util.misc import param_to_array
        Y = param_to_array(Y)

        num_inducing, _ = Z.shape
        num_data, output_dim = Y.shape

        #make sure the noise is not hetero
        beta = 1./np.squeeze(likelihood.variance)
        if beta.size <1:
            raise NotImplementedError, "no hetero noise with this implementatino of DTC"

        Kmm = kern.K(Z)
        Knn = kern.Kdiag(X)
        Knm = kern.K(X, Z)
        U = Knm
        Uy = np.dot(U.T,Y)

        #factor Kmm 
        Kmmi, L, Li, _ = pdinv(Kmm)

        # Compute A
        #LiUT, _ = dtrtrs(L, U.T*np.sqrt(beta), lower=1)
        LiUT = np.dot(Li, U.T)*np.sqrt(beta)
        A = tdot(LiUT) + np.eye(num_inducing)

        # factor A
        LA = jitchol(A)

        # back substutue to get b, P, v
        tmp, _ = dtrtrs(L, Uy, lower=1)
        b, _ = dtrtrs(LA, tmp*beta, lower=1)
        tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
        v, _ = dtrtrs(L, tmp, lower=1, trans=1)
        tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
        P = tdot(tmp.T)

        #compute log marginal
        log_marginal = -0.5*num_data*output_dim*np.log(2*np.pi) + \
                       -np.sum(np.log(np.diag(LA)))*output_dim + \
                       0.5*num_data*output_dim*np.log(beta) + \
                       -0.5*beta*np.sum(np.square(Y)) + \
                       0.5*np.sum(np.square(b))

        # Compute dL_dKmm
        vvT_P = tdot(v.reshape(-1,1)) + P
        dL_dK = 0.5*(Kmmi - vvT_P)

        # Compute dL_dU
        vY = np.dot(v.reshape(-1,1),Y.T)
        dL_dU = vY - np.dot(vvT_P, U.T)
        dL_dU *= beta

        #compute dL_dR
        Uv = np.dot(U, v)
        dL_dR = 0.5*(np.sum(U*np.dot(U,P), 1) - 1./beta + np.sum(np.square(Y), 1) - 2.*np.sum(Uv*Y, 1) + np.sum(np.square(Uv), 1))*beta**2

        grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag':np.zeros_like(Knn), 'dL_dKnm':dL_dU.T}

        #update gradients
        kern.update_gradients_sparse(X=X, Z=Z, **grad_dict)
        likelihood.update_gradients(dL_dR)

        #construct a posterior object
        post = Posterior(woodbury_inv=Kmmi-P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)

        return post, log_marginal, grad_dict


