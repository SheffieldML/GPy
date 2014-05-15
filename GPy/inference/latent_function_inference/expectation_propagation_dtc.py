import numpy as np
from ...util.linalg import pdinv,jitchol,DSYR,tdot,dtrtrs, dpotrs
from expectation_propagation import EP
from posterior import Posterior
log_2_pi = np.log(2*np.pi)

class EPDTC(EP):
    def __init__(self, epsilon=1e-6, eta=1., delta=1.):
        self.epsilon, self.eta, self.delta = epsilon, eta, delta
        self.reset()

    def reset(self):
        self.old_mutilde, self.old_vtilde = None, None
        self._ep_approximation = None

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None):
        num_data, output_dim = X.shape
        assert output_dim ==1, "ep in 1D only (for now!)"

        Kmm = kern.K(Z)
        Kmn = kern.K(Z,X)

        Lm = jitchol(Kmm)
        Lmi = dtrtrs(Lm,np.eye(Lm.shape[0]))[0]
        Kmmi = np.dot(Lmi.T,Lmi)
        KmmiKmn = np.dot(Kmmi,Kmn)
        K = np.dot(Kmn.T,KmmiKmn)

        if self._ep_approximation is None:
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
        else:
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation

        Wi, LW, LWi, W_logdet = pdinv(K + np.diag(1./tau_tilde))

        alpha, _ = dpotrs(LW, mu_tilde, lower=1)

        log_marginal =  0.5*(-num_data * log_2_pi - W_logdet - np.sum(alpha * mu_tilde)) # TODO: add log Z_hat??

        dL_dK = 0.5 * (tdot(alpha[:,None]) - Wi)

        dL_dthetaL = np.zeros(likelihood.size)#TODO: derivatives of the likelihood parameters

        return Posterior(woodbury_inv=Wi, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}



    def expectation_propagation(self, Kmm, Kmn, Y, likelihood, Y_metadata):

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"

        KmnKnm = np.dot(Kmn,Kmn.T)
        Lm = jitchol(Kmm)
        Lmi = dtrtrs(Lm,np.eye(Lm.shape[0]))[0] #chol_inv(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        KmmiKmn = np.dot(Kmmi,Kmn)
        Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        LLT0 = Kmm.copy()

        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(num_data)
        LLT = Kmm.copy() #Sigma = K.copy()
        Sigma_diag = Qnn_diag.copy()

        #Initial values - Marginal moments
        Z_hat = np.empty(num_data,dtype=np.float64)
        mu_hat = np.empty(num_data,dtype=np.float64)
        sigma2_hat = np.empty(num_data,dtype=np.float64)

        #initial values - Gaussian factors
        if self.old_mutilde is None:
            tau_tilde, mu_tilde, v_tilde = np.zeros((3, num_data))
        else:
            assert old_mutilde.size == num_data, "data size mis-match: did you change the data? try resetting!"
            mu_tilde, v_tilde = self.old_mutilde, self.old_vtilde
            tau_tilde = v_tilde/mu_tilde

        #Approximation
        tau_diff = self.epsilon + 1.
        v_diff = self.epsilon + 1.
       	iterations = 0
        while (tau_diff > self.epsilon) or (v_diff > self.epsilon):
            update_order = np.random.permutation(num_data)
            for i in update_order:
                #Cavity distribution parameters
                tau_cav = 1./Sigma_diag[i] - self.eta*tau_tilde[i]
                v_cav = mu[i]/Sigma_diag[i] - self.eta*v_tilde[i]
                #Marginal moments
                Z_hat[i], mu_hat[i], sigma2_hat[i] = likelihood.moments_match_ep(Y[i], tau_cav, v_cav)#, Y_metadata=None)#=(None if Y_metadata is None else Y_metadata[i]))
                #Site parameters update
                delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                tau_tilde[i] += delta_tau
                v_tilde[i] += delta_v
                #Posterior distribution parameters update

                #DSYR(Sigma, Sigma[:,i].copy(), -delta_tau/(1.+ delta_tau*Sigma[i,i]))
                DSYR(LLT,Kmn[:,i].copy(),delta_tau)
                L = jitchol(LLT)

                V,info = dtrtrs(L,Kmn,lower=1)
                Sigma_diag = np.sum(V*V,-2)
                si = np.sum(V.T*V[:,i],-1)
                mu += (delta_v-delta_tau*mu[i])*si
                #mu = np.dot(Sigma, v_tilde)

            #(re) compute Sigma and mu using full Cholesky decompy
            LLT = LLT0 + np.dot(Kmn*tau_tilde[None,:],Kmn.T)
            L = jitchol(LLT)
            V,info = dtrtrs(L,Kmn,lower=1)
            V2,info = dtrtrs(L.T,V,lower=0)
            #Sigma_diag = np.sum(V*V,-2)
            #Knmv_tilde = np.dot(Kmn,v_tilde)
            #mu = np.dot(V2.T,Knmv_tilde)
            Sigma = np.dot(V2.T,V2)
            mu = np.dot(Sigma,v_tilde)

            #monitor convergence
            if iterations>0:
                tau_diff = np.mean(np.square(tau_tilde-tau_tilde_old))
                v_diff = np.mean(np.square(v_tilde-v_tilde_old))
            tau_tilde_old = tau_tilde.copy()
            v_tilde_old = v_tilde.copy()

            tau_diff = 0
            v_diff = 0
            iterations += 1

        mu_tilde = v_tilde/tau_tilde
        return mu, Sigma, mu_tilde, tau_tilde, Z_hat
