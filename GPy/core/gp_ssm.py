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

import numpy as np
import scipy.linalg as sp
from gp import GP
from parameterization.param import Param
from ..inference.latent_function_inference import gaussian_ssm_inference
from .. import likelihoods
from ..inference import optimization
from parameterization.transformations import Logexp
from GPy.inference.latent_function_inference.posterior import Posterior

class GpSSM(GP):
    """
    A GP model for sorted one-dimensional inputs

    This model allows the representation of a Gaussian Process as a Gauss-Markov State Machine.

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance

    """

    def __init__(self, X, Y, kernel, likelihood, inference_method=None,
                 name='gp ssm', Y_metadata=None, normalizer=False):
        #pick a sensible inference method

        inference_method = gaussian_ssm_inference.GaussianSSMInference()

        GP.__init__(self, X, Y, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=normalizer)

        self.posterior = None

    def optimize(self, optimizer=None, start=None, **kwargs):
        prevLikelihood = 0
        count = 0
        change = 1
        while ((change > 0.001) and (count < 50)):
            posterior, likelihood = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y, self.Y_metadata)
            expectations = posterior.expectations
            self.optimize_params(expectations=expectations)
            change = np.abs(likelihood - prevLikelihood)
            prevLikelihood = likelihood
            count = count + 1

    def optimize_params(self, optimizer=None, start=None, expectations=None, **kwargs):
        if self.is_fixed:
            print("nothing to optimize")
        if self.size == 0:
            print("nothing to optimize")

        if not self.update_model():
            print("Updates were off, setting updates on again")
            self.update_model(True)

        if start == None:
            start = self.optimizer_array

        if optimizer is None:
            optimizer = self.preferred_optimizer

        if isinstance(optimizer, optimization.Optimizer):
            opt = optimizer
            opt.model = self
        else:
            optimizer = optimization.get_optimizer(optimizer)
            opt = optimizer(start, model=self, **kwargs)

        opt.max_iters = 1

        opt.run(f_fp=self.param_maximisation_step, args=(self.X, self.Y, expectations))
        self.optimization_runs.append(opt)
        self.optimizer_array = opt.x_opt

    def param_maximisation_step(self, loghyper, X, Y, E, *args):

        loghyper = np.log(np.exp(loghyper) + 1) - 1e-20
        lam = loghyper[0]
        sig = loghyper[1]
        noise = loghyper[2]

        kern = self.kern
        order = kern.order

        K = len(X)
        mu_0 = np.zeros((order, 1))
        v_0 = kern.Phi_of_r(-1)
        dvSig = v_0/sig
        dvLam = kern.dQ(-1)[0]

        mu = E[0][0]
        V = E[0][1]
        E11 = V + np.dot(mu, mu.T)
        V0_inv = np.linalg.solve(v_0, np.eye(len(mu)))
        Ub = np.log(np.linalg.det(v_0)) + np.trace(np.dot(V0_inv, E11))
        dUb_lam = np.trace(np.dot(V0_inv, dvLam.T)) - np.trace(np.dot(np.dot(np.dot(V0_inv, dvLam.T),V0_inv), E11))
        dUb_sig = np.trace(np.dot(V0_inv, dvSig.T)) - np.trace(np.dot(np.dot(np.dot(V0_inv, dvSig.T),V0_inv), E11))

        for t in range(1, K):
            delta = X[t] - X[t-1]
            Q = kern.Q_of_r(delta)
            Phi = kern.Phi_of_r(delta)
            dPhi = kern.dPhidLam(delta)
            [dLam, dSig] = kern.dQ(delta)
            mu_prev = E[t-1][0]
            V_prev = E[t-1][1]
            mu = E[t][0]
            V = E[t][1]
            Ett_prev = V_prev + np.dot(mu_prev, mu_prev.T)
            Eadj = E[t-1][3] + np.dot(mu, mu_prev.T)
            Ett = V + np.dot(mu, mu.T)
            CC = sp.cholesky(Q, lower=True)
            Q_inv = np.linalg.solve(CC.T, np.linalg.solve(CC, np.eye(len(mu))))

            Ub = Ub + np.log(np.linalg.det(Q)) + np.trace(np.dot(Q_inv, Ett)) - 2*np.trace(np.dot(np.dot(Phi.T, Q_inv), Eadj)) + np.trace(np.dot(np.dot(np.dot(Phi.T, Q_inv), Phi), Ett_prev))
            A = np.dot(dPhi.T, Q_inv) - np.dot(Phi.T, np.dot(np.dot(Q_inv, dLam), Q_inv))
            dUb_lam = dUb_lam + np.trace(np.dot(Q_inv, dLam.T)) - np.trace(np.dot(np.dot(np.dot(Q_inv, dLam.T), Q_inv), Ett)) - 2*np.trace(np.dot(A,Eadj)) + np.trace(np.dot(np.dot(np.dot(Phi.T, Q_inv), dPhi) + np.dot(A, Phi), Ett_prev))        
            A = -1 * np.dot(Phi.T, np.dot(np.dot(Q_inv, dSig), Q_inv))
            dUb_sig = dUb_sig + np.trace(np.dot(Q_inv, dSig.T)) - np . trace(np.dot(np.dot(np.dot(Q_inv, dSig.T), Q_inv), Ett)) - 2*np.trace(np.dot(A, Eadj)) + np.trace(np.dot(np.dot(A, Phi), Ett_prev))

        dUb_noise = 0
        for t in xrange(K):
            mu = E[t][0]
            V = E[t][1]
            Ett = V + np.dot(mu, mu.T)
            Ub = Ub + np.log(noise) + (Y[t]**2 - 2*Y[t]*mu[0] + Ett[0][0])/noise
            dUb_noise = dUb_noise + 1/noise - (Y[t]**2 - 2*Y[t]*mu[0] + Ett[0][0])/(noise**2)

        dUb = np.array([lam, sig, noise]) * np.array([dUb_lam.item(), dUb_sig.item(), dUb_noise.item()])
        
        return Ub.item(), dUb


    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method reperforms inference, recalculating the posterior and log marginal likelihood

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.

        We override the method in the parent class since we do not handle updates to the standard gradients.
        """
        self.posterior, self._log_marginal_likelihood = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.Y_metadata)

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values
        N.B. It is assumed that input points are sorted
        """
        if kern is None:
            kern = self.kern

        X = self.X
        K = X.shape[0]
        K_new = Xnew.shape[0]
        order = kern.order
        mean_pred = np.zeros(K_new)
        var_pred = np.zeros(K_new)
        H = np.zeros((1,order))
        H[0][0] = 1

        count = 0
        for t in xrange(K_new):
            while ((count < K) and (Xnew[t] > X[count])):
                count += 1

            if (count == 0):
                mu = np.zeros((order, 1))
                V = kern.Phi_of_r(-1)

                delta = np.abs(Xnew[t] - X[count])
                Phi = kern.Phi_of_r(delta)
                Q = kern.Q_of_r(delta)
                P = np.dot(np.dot(Phi, V), Phi.T) + Q

                mu_s = self.posterior.mu_s[count]
                V_s = self.posterior.V_s[count]
                L = np.dot(np.dot(V, Phi.T), np.linalg.solve(P, np.eye(len(P))))
                mu_s = mu + np.dot(L, mu_s - np.dot(Phi, mu))
                V_s = V + np.dot(np.dot(L, V_s - P), L.T)

                mean_pred[t] = np.dot(H, mu_s)
                var_pred[t] = np.dot(np.dot(H, V_s), H.T)

            elif (count == K):
                # forwards phase
                delta = np.abs(Xnew[t] - X[count-1])
                Phi = kern.Phi_of_r(delta)
                Q = kern.Q_of_r(delta)
                mu_f = self.posterior.mu_f[count-1]
                V_f = self.posterior.V_f[count-1]

                mu = np.dot(Phi, mu_f)
                P = np.dot(np.dot(Phi, V_f), Phi.T) + Q
                V = P

                mean_pred[t] = np.dot(H,mu)
                var_pred[t] = np.dot(np.dot(H, V), H.T)

            else:
                # forwards phase
                delta = np.abs(Xnew[t] - X[count-1])
                Phi = kern.Phi_of_r(delta)
                Q = kern.Q_of_r(delta)
                mu_f = self.posterior.mu_f[count-1]
                V_f = self.posterior.V_f[count-1]
                mu = np.dot(Phi, mu_f)
                P = np.dot(np.dot(Phi, V_f), Phi.T) + Q
                V = P

                delta = np.abs(Xnew[t] - X[count])
                Phi = kern.Phi_of_r(delta)
                Q = kern.Q_of_r(delta)
                P = np.dot(np.dot(Phi, V), Phi.T) + Q

                # backwards phase
                mu_s = self.posterior.mu_s[count]
                V_s = self.posterior.V_s[count]

                L = np.dot(np.dot(V, Phi.T), np.linalg.solve(P, np.eye(len(P))))
                mu_s = mu + np.dot(L, mu_s - np.dot(Phi, mu))
                V_s = V + np.dot(np.dot(L, V_s - P), L.T)

                mean_pred[t] = np.dot(H, mu_s)
                var_pred[t] = np.dot(np.dot(H, V_s), H.T)


        mean_pred = mean_pred.reshape(-1, 1)
        var_pred = var_pred.reshape(-1, 1)

        return mean_pred, var_pred
