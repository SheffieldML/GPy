# Copyright (c) 2012-2014, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import Posterior
from ...util.linalg import jitchol, tdot, dtrtrs, dpotri, pdinv
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class DTC(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    NB. It's not recommended to use this function! It's here for historical purposes. 

    """
    def __init__(self):
        self.const_jitter = 1e-6

    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
        assert mean_function is None, "inference with a mean function not implemented"
        assert X_variance is None, "cannot use X_variance with DTC. Try varDTC."

        num_inducing, _ = Z.shape
        num_data, output_dim = Y.shape

        #make sure the noise is not hetero
        precision = 1./likelihood.gaussian_variance(Y_metadata)
        if precision.size > 1:
            raise NotImplementedError("no hetero noise with this implementation of DTC")

        Kmm = kern.K(Z)
        Knn = kern.Kdiag(X)
        Knm = kern.K(X, Z)
        U = Knm
        Uy = np.dot(U.T,Y)

        #factor Kmm
        Kmmi, L, Li, _ = pdinv(Kmm)

        # Compute A
        LiUTbeta = np.dot(Li, U.T)*np.sqrt(precision)
        A = tdot(LiUTbeta) + np.eye(num_inducing)

        # factor A
        LA = jitchol(A)

        # back substutue to get b, P, v
        tmp, _ = dtrtrs(L, Uy, lower=1)
        b, _ = dtrtrs(LA, tmp*precision, lower=1)
        tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
        v, _ = dtrtrs(L, tmp, lower=1, trans=1)
        tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
        P = tdot(tmp.T)

        #compute log marginal
        log_marginal = -0.5*num_data*output_dim*np.log(2*np.pi) + \
                       -np.sum(np.log(np.diag(LA)))*output_dim + \
                       0.5*num_data*output_dim*np.log(precision) + \
                       -0.5*precision*np.sum(np.square(Y)) + \
                       0.5*np.sum(np.square(b))

        # Compute dL_dKmm
        vvT_P = tdot(v.reshape(-1,1)) + P
        dL_dK = 0.5*(Kmmi - vvT_P)

        # Compute dL_dU
        vY = np.dot(v.reshape(-1,1),Y.T)
        dL_dU = vY - np.dot(vvT_P, U.T)
        dL_dU *= precision

        #compute dL_dR
        Uv = np.dot(U, v)
        dL_dR = 0.5*(np.sum(U*np.dot(U,P), 1) - 1./precision + np.sum(np.square(Y), 1) - 2.*np.sum(Uv*Y, 1) + np.sum(np.square(Uv), 1))*precision**2

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)

        grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag':np.zeros_like(Knn), 'dL_dKnm':dL_dU.T, 'dL_dthetaL':dL_dthetaL}

        #construct a posterior object
        post = Posterior(woodbury_inv=Kmmi-P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)

        return post, log_marginal, grad_dict

class vDTC(object):
    def __init__(self):
        self.const_jitter = 1e-6

    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
        assert mean_function is None, "inference with a mean function not implemented"
        assert X_variance is None, "cannot use X_variance with DTC. Try varDTC."

        num_inducing, _ = Z.shape
        num_data, output_dim = Y.shape

        #make sure the noise is not hetero
        precision = 1./likelihood.gaussian_variance(Y_metadata)
        if precision.size > 1:
            raise NotImplementedError("no hetero noise with this implementation of DTC")

        Kmm = kern.K(Z)
        Knn = kern.Kdiag(X)
        Knm = kern.K(X, Z)
        U = Knm
        Uy = np.dot(U.T,Y)

        #factor Kmm
        Kmmi, L, Li, _ = pdinv(Kmm)

        # Compute A
        LiUTbeta = np.dot(Li, U.T)*np.sqrt(precision)
        A_ = tdot(LiUTbeta)
        trace_term = -0.5*(np.sum(Knn)*precision - np.trace(A_))
        A = A_ + np.eye(num_inducing)

        # factor A
        LA = jitchol(A)

        # back substutue to get b, P, v
        tmp, _ = dtrtrs(L, Uy, lower=1)
        b, _ = dtrtrs(LA, tmp*precision, lower=1)
        tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
        v, _ = dtrtrs(L, tmp, lower=1, trans=1)
        tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
        P = tdot(tmp.T)
        stop

        #compute log marginal
        log_marginal = -0.5*num_data*output_dim*np.log(2*np.pi) + \
                       -np.sum(np.log(np.diag(LA)))*output_dim + \
                       0.5*num_data*output_dim*np.log(precision) + \
                       -0.5*precision*np.sum(np.square(Y)) + \
                       0.5*np.sum(np.square(b)) + \
                       trace_term

        # Compute dL_dKmm
        vvT_P = tdot(v.reshape(-1,1)) + P
        LAL = Li.T.dot(A).dot(Li)
        dL_dK = Kmmi - 0.5*(vvT_P + LAL)

        # Compute dL_dU
        vY = np.dot(v.reshape(-1,1),Y.T)
        #dL_dU = vY - np.dot(vvT_P, U.T)
        dL_dU = vY - np.dot(vvT_P - Kmmi, U.T)
        dL_dU *= precision

        #compute dL_dR
        Uv = np.dot(U, v)
        dL_dR = 0.5*(np.sum(U*np.dot(U,P), 1) - 1./precision + np.sum(np.square(Y), 1) - 2.*np.sum(Uv*Y, 1) + np.sum(np.square(Uv), 1) )*precision**2
        dL_dR -=precision*trace_term/num_data

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)
        grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag':np.zeros_like(Knn) + -0.5*precision, 'dL_dKnm':dL_dU.T, 'dL_dthetaL':dL_dthetaL}

        #construct a posterior object
        post = Posterior(woodbury_inv=Kmmi-P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)


        return post, log_marginal, grad_dict


