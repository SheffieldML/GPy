import numpy as np
from scipy import stats, linalg
from ..util.linalg import pdinv,mdot,jitchol
from likelihood import likelihood

class EP(likelihood):
    def __init__(self,data,likelihood_function,epsilon=1e-3,power_ep=[1.,1.]):
        """
        Expectation Propagation

        Arguments
        ---------
        epsilon : Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        likelihood_function : a likelihood function (see likelihood_functions.py)
        """
        self.likelihood_function = likelihood_function
        self.epsilon = epsilon
        self.eta, self.delta = power_ep
        self.is_heteroscedastic = True

        #Initial values - Likelihood approximation parameters:
        #p(y|f) = t(f|tau_tilde,v_tilde)
        self.tau_tilde = np.zeros(self.N)
        self.v_tilde = np.zeros(self.N)

    def _compute_GP_variables(self):
        #Variables to be called from GP
        mu_tilde = self.v_tilde/self.tau_tilde #When calling EP, this variable is used instead of Y in the GP model
        sigma_sum = 1./self.tau_ + 1./self.tau_tilde
        mu_diff_2 = (self.v_/self.tau_ - mu_tilde)**2
        self.Z = np.sum(np.log(self.Z_hat)) + 0.5*np.sum(np.log(sigma_sum)) + 0.5*np.sum(mu_diff_2/sigma_sum) #Normalization constant, aka Z_ep

        self.Y =  mu_tilde[:,None]
        self.precsion = self.tau_tilde
        self.covariance_matrix = np.diag(1./self.precision)

    def fit_full(self,K):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006.
        """
        #Prior distribution parameters: p(f|X) = N(f|0,K)

        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        self.mu = np.zeros(self.N)
        self.Sigma = K.copy()

        """
        Initial values - Cavity distribution parameters:
        q_(f|mu_,sigma2_) = Product{q_i(f|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.N,dtype=float)
        self.v_ = np.empty(self.N,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.N,dtype=float)
        self.Z_hat = np.empty(self.N,dtype=float)
        phi = np.empty(self.N,dtype=float)
        mu_hat = np.empty(self.N,dtype=float)
        sigma2_hat = np.empty(self.N,dtype=float)

        #Approximation
        epsilon_np1 = self.epsilon + 1.
        epsilon_np2 = self.epsilon + 1.
       	self.iterations = 0
        self.np1 = [self.tau_tilde.copy()]
        self.np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.random.permutation(self.N)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./self.Sigma[i,i] - self.eta*self.tau_tilde[i]
                self.v_[i] = self.mu[i]/self.Sigma[i,i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.likelihood.moments_match(i,self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./self.Sigma[i,i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - self.mu[i]/self.Sigma[i,i])
                self.tau_tilde[i] = self.tau_tilde[i] + Delta_tau
                self.v_tilde[i] = self.v_tilde[i] + Delta_v
                #Posterior distribution parameters update
                si=self.Sigma[:,i].reshape(self.N,1)
                self.Sigma = self.Sigma - Delta_tau/(1.+ Delta_tau*self.Sigma[i,i])*np.dot(si,si.T)
                self.mu = np.dot(self.Sigma,self.v_tilde)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            Sroot_tilde_K = np.sqrt(self.tau_tilde)[:,None]*K
            B = np.eye(self.N) + np.sqrt(self.tau_tilde)[None,:]*Sroot_tilde_K
            L = jitchol(B)
            V,info = linalg.flapack.dtrtrs(L,Sroot_tilde_K,lower=1)
            self.Sigma = K - np.dot(V.T,V)
            self.mu = np.dot(self.Sigma,self.v_tilde)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.N
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

        return self._compute_GP_variables()

    def fit_DTC(self, Knn_diag, Kmn, Kmm):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see ... 2013.
        """

        #TODO: this doesn;t work with uncertain inputs! 

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = Qnn = Knm*Kmmi*Kmn
        """
        Kmmi, Lm, Lmi, Kmm_logdet = pdinv(Kmm)
        KmnKnm = np.dot(Kmn, Kmn.T)
        KmmiKmn = np.dot(Kmmi,self.Kmn)
        Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        LLT0 = Kmm.copy()

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*gamma
        """
        mu = np.zeros(self.N)
        LLT = Kmm.copy()
        Sigma_diag = Qnn_diag.copy()

        """
        Initial values - Cavity distribution parameters:
        q_(g|mu_,sigma2_) = Product{q_i(g|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        tau_ = np.empty(self.N,dtype=float)
        v_ = np.empty(self.N,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.N,dtype=float)
        Z_hat = np.empty(self.N,dtype=float)
        phi = np.empty(self.N,dtype=float)
        mu_hat = np.empty(self.N,dtype=float)
        sigma2_hat = np.empty(self.N,dtype=float)

        #Approximation
        epsilon_np1 = 1
        epsilon_np2 = 1
       	self.iterations = 0
        np1 = [tau_tilde.copy()]
        np2 = [v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.random.permutation(self.N)
            for i in update_order:
                #Cavity distribution parameters
                tau_[i] = 1./Sigma_diag[i] - self.eta*tau_tilde[i]
                v_[i] = mu[i]/Sigma_diag[i] - self.eta*v_tilde[i]
                #Marginal moments
                Z_hat[i], mu_hat[i], sigma2_hat[i] = self.likelihood_function.moments_match(self.data[i],tau_[i],v_[i])
                #Site parameters update
                Delta_tau = delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                tau_tilde[i] = tau_tilde[i] + Delta_tau
                v_tilde[i] = v_tilde[i] + Delta_v
                #Posterior distribution parameters update
                LLT = LLT + np.outer(Kmn[:,i],Kmn[:,i])*Delta_tau
                L = jitchol(LLT)
                V,info = linalg.flapack.dtrtrs(L,Kmn,lower=1)
                Sigma_diag = np.sum(V*V,-2)
                si = np.sum(V.T*V[:,i],-1)
                mu = mu + (Delta_v-Delta_tau*mu[i])*si
                self.iterations += 1
            #Sigma recomputation with Cholesky decompositon
            LLT0 = LLT0 + np.dot(Kmn*tau_tilde[None,:],Kmn.T)
            L = jitchol(LLT)
            V,info = linalg.flapack.dtrtrs(L,Kmn,lower=1)
            V2,info = linalg.flapack.dtrtrs(L.T,V,lower=0)
            Sigma_diag = np.sum(V*V,-2)
            Knmv_tilde = np.dot(Kmn,v_tilde)
            mu = np.dot(V2.T,Knmv_tilde)
            epsilon_np1 = sum((tau_tilde-np1[-1])**2)/self.N
            epsilon_np2 = sum((v_tilde-np2[-1])**2)/self.N
            np1.append(tau_tilde.copy())
            np2.append(v_tilde.copy())

        self._compute_GP_variables()

    def fit_FITC(self, Knn_diag, Kmn):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see Naish-Guzman and Holden, 2008.
        """

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = diag(Knn-Qnn) + Qnn, Qnn = Knm*Kmmi*Kmn
        """
        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)
        self.P0 = self.Kmn.T
        self.KmnKnm = np.dot(self.P0.T, self.P0)
        self.KmmiKmn = np.dot(self.Kmmi,self.P0.T)
        self.Qnn_diag = np.sum(self.P0.T*self.KmmiKmn,-2)
        self.Diag0 = self.Knn_diag - self.Qnn_diag
        self.R0 = jitchol(self.Kmmi).T

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*gamma
        """
        self.w = np.zeros(self.N)
        self.gamma = np.zeros(self.M)
        self.mu = np.zeros(self.N)
        self.P = self.P0.copy()
        self.R = self.R0.copy()
        self.Diag = self.Diag0.copy()
        self.Sigma_diag = self.Knn_diag

        """
        Initial values - Cavity distribution parameters:
        q_(g|mu_,sigma2_) = Product{q_i(g|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.N,dtype=float)
        self.v_ = np.empty(self.N,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.N,dtype=float)
        self.Z_hat = np.empty(self.N,dtype=float)
        phi = np.empty(self.N,dtype=float)
        mu_hat = np.empty(self.N,dtype=float)
        sigma2_hat = np.empty(self.N,dtype=float)

        #Approximation
        epsilon_np1 = 1
        epsilon_np2 = 1
       	self.iterations = 0
        self.np1 = [self.tau_tilde.copy()]
        self.np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.random.permutation(self.N)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./self.Sigma_diag[i] - self.eta*self.tau_tilde[i]
                self.v_[i] = self.mu[i]/self.Sigma_diag[i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.likelihood.moments_match(i,self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./self.Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - self.mu[i]/self.Sigma_diag[i])
                self.tau_tilde[i] = self.tau_tilde[i] + Delta_tau
                self.v_tilde[i] = self.v_tilde[i] + Delta_v
                #Posterior distribution parameters update
                dtd1 = Delta_tau*self.Diag[i] + 1.
                dii = self.Diag[i]
                self.Diag[i] = dii - (Delta_tau * dii**2.)/dtd1
                pi_ = self.P[i,:].reshape(1,self.M)
                self.P[i,:] = pi_ - (Delta_tau*dii)/dtd1 * pi_
                Rp_i = np.dot(self.R,pi_.T)
                RTR = np.dot(self.R.T,np.dot(np.eye(self.M) - Delta_tau/(1.+Delta_tau*self.Sigma_diag[i]) * np.dot(Rp_i,Rp_i.T),self.R))
                self.R = jitchol(RTR).T
                self.w[i] = self.w[i] + (Delta_v - Delta_tau*self.w[i])*dii/dtd1
                self.gamma = self.gamma + (Delta_v - Delta_tau*self.mu[i])*np.dot(RTR,self.P[i,:].T)
                self.RPT = np.dot(self.R,self.P.T)
                self.Sigma_diag = self.Diag + np.sum(self.RPT.T*self.RPT.T,-1)
                self.mu = self.w + np.dot(self.P,self.gamma)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            self.Diag = self.Diag0/(1.+ self.Diag0 * self.tau_tilde)
            self.P = (self.Diag / self.Diag0)[:,None] * self.P0
            self.RPT0 = np.dot(self.R0,self.P0.T)
            L = jitchol(np.eye(self.M) + np.dot(self.RPT0,(1./self.Diag0 - self.Diag/(self.Diag0**2))[:,None]*self.RPT0.T))
            self.R,info = linalg.flapack.dtrtrs(L,self.R0,lower=1)
            self.RPT = np.dot(self.R,self.P.T)
            self.Sigma_diag = self.Diag + np.sum(self.RPT.T*self.RPT.T,-1)
            self.w = self.Diag * self.v_tilde
            self.gamma = np.dot(self.R.T, np.dot(self.RPT,self.v_tilde))
            self.mu = self.w + np.dot(self.P,self.gamma)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.N
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

        return self._compute_GP_variables()
