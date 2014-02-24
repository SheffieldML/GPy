import numpy as np
from scipy import stats
from ..util.linalg import pdinv,mdot,jitchol,chol_inv,DSYR,tdot,dtrtrs
from likelihood import likelihood

class EP(object):
    def __init__(self, epsilon=1e-6, eta=1., delta=1.):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006.

        :param epsilon: Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :type epsilon: float
        :param eta: Power EP thing TODO: Ricardo: what, exactly?
        :type eta: float64
        :param delta: Power EP thing TODO: Ricardo: what, exactly?
        :type delta: float64
        """
        self.epsilon, self.eta, self.delta = epsilon, eta, delta
        self.reset()

    def reset(self):
        self.old_mutilde, self.old_vtilde = None, None

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):

        K = kern.K(X)

        mu_tilde, tau_tilde = self.expectation_propagation()


    def expectation_propagation(self, K, Y, Y_metadata, likelihood)

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"


        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(self.num_data)
        Sigma = K.copy()

        #Initial values - Marginal moments
        Z_hat = np.empty(num_data,dtype=np.float64)
        mu_hat = np.empty(num_data,dtype=np.float64)
        sigma2_hat = np.empty(num_data,dtype=np.float64)

        #initial values - Gaussian factors
        if self.old_mutilde is None:
            tau_tilde, mu_tilde, v_tilde = np.zeros((3, num_data, num_data))
        else:
            assert old_mutilde.size == num_data, "data size mis-match: did you change the data? try resetting!"
            mu_tilde, v_tilde = self.old_mutilde, self.old_vtilde
            tau_tilde = v_tilde/mu_tilde

        #Approximation
        epsilon_np1 = self.epsilon + 1.
        epsilon_np2 = self.epsilon + 1.
       	iterations = 0
        while (epsilon_np1 > self.epsilon) or (epsilon_np2 > self.epsilon):
            update_order = np.random.permutation(num_data)
            for i in update_order:
                #Cavity distribution parameters
                tau_cav = 1./Sigma[i,i] - self.eta*tau_tilde[i]
                v_cav = mu[i]/Sigma[i,i] - self.eta*v_tilde[i]
                #Marginal moments
                Z_hat[i], mu_hat[i], sigma2_hat[i] = likelihood.moments_match(Y[i], tau_cav, v_cav, Y_metadata=(None if Y_metadata is None else Y_metadata[i]))
                #Site parameters update
                delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma[i,i])
                delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma[i,i])
                tau_tilde[i] += delta_tau
                v_tilde[i] += delta_v
                #Posterior distribution parameters update
                DSYR(Sigma, Sigma[:,i].copy(), -Delta_tau/(1.+ Delta_tau*Sigma[i,i]))
                mu = np.dot(Sigma, v_tilde)
                iterations += 1

            #(re) compute Sigma and mu using full Cholesky decompy
            tau_tilde_root = np.sqrt(tau_tilde)
            Sroot_tilde_K = tau_tilde_root[:,None] * K
            B = np.eye(num_data) + Sroot_tilde_K * tau_tilde_root[None,:]
            L = jitchol(B)
            V, _ = dtrtrs(L, Sroot_tilde_K, lower=1)
            Sigma = K - np.dot(V.T,V)
            mu = np.dot(Sigma,v_tilde)

            #monitor convergence
            epsilon_np1 = np.mean(np.square(tau_tilde-tau_tilde_old))
            epsilon_np2 = np.mean(np.square(v_tilde-v_tilde_old))
            tau_tilde_old = tau_tilde.copy()
            v_tilde_old = v_tilde.copy()

        return mu, Sigma, mu_tilde, tau_tilde

