# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

# This implementation of converting GPs to state space models is based on the article:

#@article{Gilboa:2015,
#  title={Scaling multidimensional inference for structured Gaussian processes},
#  author={Gilboa, Elad and Saat{\c{c}}i, Yunus and Cunningham, John P},
#  journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
#  volume={37},
#  number={2},
#  pages={424--436},
#  year={2015},
#  publisher={IEEE}
#}

from ssm_posterior import SsmPosterior
from ...util.linalg import pdinv, dpotrs, tdot
from ...util import diag
import numpy as np
import math as mt
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)


class GaussianSSMInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LL^T = YY^T.

        Note that L may have fewer columns than Y, else L=Y.
        """
        N, D = Y.shape
        if (N>D):
            return Y
        else:
            #if Y in self.cache, return self.Cache[Y], else store Y in cache and return L.
            #print "WARNING: N>D of Y, we need caching of L, such that L*L^T = Y, returning Y still!"
            return Y

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):

        """
        Returns a Posterior class containing essential quantities of the posterior
        """
        order = kern.order
        K = X.shape[0]
        log_likelihood = 0
        results = np.zeros((K,4),dtype=object)
        H = np.zeros((1,order))
        H[0][0] = 1
        v_0 = kern.Phi_of_r(-1)
        mu_0 = np.zeros((order, 1))
        noise_var = likelihood.variance + 1e-8

        # carry out forward filtering
        for t in range(K):
            if (t == 0):
                prior_m = np.dot(H,mu_0)
                prior_v = np.dot(np.dot(H, v_0), H.T) + noise_var

                log_likelihood = -0.5*(log_2_pi + mt.log(prior_v) + ((Y[0] - prior_m)**2)/prior_v)

                kalman_gain = np.dot(v_0, H.T) / prior_v
                mu = mu_0 + kalman_gain*(Y[0] - prior_m)


                V = np.dot(np.eye(order) - np.dot(kalman_gain,H), v_0)
                results[0][0] = mu
                results[0][1] = V
            else:
                delta = X[t] - X[t-1]
                Q = kern.Q_of_r(delta)
                Phi = kern.Phi_of_r(delta)
                P = np.dot(np.dot(Phi, V), Phi.T) + Q
                PhiMu = np.dot(Phi, mu)
                prior_m = np.dot(H, PhiMu)
                prior_v = np.dot(np.dot(H, P), H.T) + noise_var

                log_likelihood_i = -0.5*(log_2_pi + mt.log(prior_v) + ((Y[t] - prior_m)**2)/prior_v)
                log_likelihood += log_likelihood_i

                kalman_gain = np.dot(P, H.T)/prior_v
                mu = PhiMu + kalman_gain*(Y[t] - prior_m)
                V = np.dot((np.eye(order) - np.dot(kalman_gain, H)), P)
                
                results[t-1][2] = Phi
                results[t-1][3] = P
                results[t][0] = mu
                results[t][1] = V

        # carry out backwards smoothing

        W = np.dot((np.eye(order) - np.dot(kalman_gain,H)),(np.dot(Phi,results[K-2][1])))

        mu_s = results[K-1][0]
        V_s = results[K-1][1]

        posterior_mean = np.zeros((K,1))
        posterior_var = np.zeros((K,1))
        E = np.zeros((K,4), dtype='object')

        posterior_mean[K-1] = np.dot(H, mu_s)
        posterior_var[K-1] = np.dot(np.dot(H, V_s), H.T)
        E[K-1][0] = mu_s
        E[K-1][1] = V_s

        for t in range(K-2, -1, -1):
            mu = results[t][0]
            V = results[t][1]
            Phi = results[t][2]
            P = results[t][3]
            
            L = np.dot(np.dot(V, Phi.T), np.linalg.solve(P, np.eye(order))) # forward substitution
            mu_s = mu + np.dot(L, mu_s - np.dot(Phi, mu))
            V_s = V + np.dot(np.dot(L, V_s - P), L.T)
            posterior_mean[t] = np.dot(H, mu_s)
            posterior_var[t] = np.dot(np.dot(H, V_s), H.T)

            if (t < K-2):
                W = np.dot(results[t+1][1], L.T) + np.dot(E[t+1][2], np.dot(W - np.dot(results[t+1][2], results[t+1][1]), L.T))

            E[t][0] = mu_s
            E[t][1] = V_s
            E[t][2] = L
            E[t][3] = W

        return SsmPosterior(mu_f=results[:,0], V_f=results[:,1], mu_s=E[:,0], V_s=E[:,1], expectations=E), log_likelihood
