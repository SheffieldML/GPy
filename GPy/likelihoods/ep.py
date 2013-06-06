import numpy as np
from scipy import stats
from ..util.linalg import pdinv,mdot,jitchol,chol_inv,DSYR,tdot,dtrtrs
from likelihood import likelihood

class EP(likelihood):
    def __init__(self,data,LikelihoodFunction,epsilon=1e-3,power_ep=[1.,1.]):
        """
        Expectation Propagation

        Arguments
        ---------
        epsilon : Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        LikelihoodFunction : a likelihood function (see likelihood_functions.py)
        """
        self.LikelihoodFunction = LikelihoodFunction
        self.epsilon = epsilon
        self.eta, self.delta = power_ep
        self.data = data
        self.N, self.output_dim = self.data.shape
        self.is_heteroscedastic = True
        self.Nparams = 0
        self._transf_data = self.LikelihoodFunction._preprocess_values(data)

        #Initial values - Likelihood approximation parameters:
        #p(y|f) = t(f|tau_tilde,v_tilde)
        self.tau_tilde = np.zeros(self.N)
        self.v_tilde = np.zeros(self.N)

        #initial values for the GP variables
        self.Y = np.zeros((self.N,1))
        self.covariance_matrix = np.eye(self.N)
        self.precision = np.ones(self.N)[:,None]
        self.Z = 0
        self.YYT = None
        self.V = self.precision * self.Y

    def restart(self):
        self.tau_tilde = np.zeros(self.N)
        self.v_tilde = np.zeros(self.N)
        self.Y = np.zeros((self.N,1))
        self.covariance_matrix = np.eye(self.N)
        self.precision = np.ones(self.N)[:,None]
        self.Z = 0
        self.YYT = None
        self.V = self.precision * self.Y

    def predictive_values(self,mu,var,full_cov):
        if full_cov:
            raise NotImplementedError, "Cannot make correlated predictions with an EP likelihood"
        return self.LikelihoodFunction.predictive_values(mu,var)

    def _get_params(self):
        return np.zeros(0)
    def _get_param_names(self):
        return []
    def _set_params(self,p):
        pass # TODO: the EP likelihood might want to take some parameters...
    def _gradients(self,partial):
        return np.zeros(0) # TODO: the EP likelihood might want to take some parameters...

    def _compute_GP_variables(self):
        #Variables to be called from GP
        mu_tilde = self.v_tilde/self.tau_tilde #When calling EP, this variable is used instead of Y in the GP model
        sigma_sum = 1./self.tau_ + 1./self.tau_tilde
        mu_diff_2 = (self.v_/self.tau_ - mu_tilde)**2
        self.Z = np.sum(np.log(self.Z_hat)) + 0.5*np.sum(np.log(sigma_sum)) + 0.5*np.sum(mu_diff_2/sigma_sum) #Normalization constant, aka Z_ep

        self.Y =  mu_tilde[:,None]
        self.YYT = np.dot(self.Y,self.Y.T)
        self.covariance_matrix = np.diag(1./self.tau_tilde)
        self.precision = self.tau_tilde[:,None]
        self.V = self.precision * self.Y

    def fit_full(self,K):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006.
        """
        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(self.N)
        Sigma = K.copy()

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
                self.tau_[i] = 1./Sigma[i,i] - self.eta*self.tau_tilde[i]
                self.v_[i] = mu[i]/Sigma[i,i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.LikelihoodFunction.moments_match(self._transf_data[i],self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma[i,i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma[i,i])
                self.tau_tilde[i] += Delta_tau
                self.v_tilde[i] += Delta_v
                #Posterior distribution parameters update
                DSYR(Sigma,Sigma[:,i].copy(), -float(Delta_tau/(1.+ Delta_tau*Sigma[i,i])))
                mu = np.dot(Sigma,self.v_tilde)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            Sroot_tilde_K = np.sqrt(self.tau_tilde)[:,None]*K
            B = np.eye(self.N) + np.sqrt(self.tau_tilde)[None,:]*Sroot_tilde_K
            L = jitchol(B)
            V,info = dtrtrs(L,Sroot_tilde_K,lower=1)
            Sigma = K - np.dot(V.T,V)
            mu = np.dot(Sigma,self.v_tilde)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.N
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

        return self._compute_GP_variables()

    def fit_DTC(self, Kmm, Kmn):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see ... 2013.
        """
        num_inducing = Kmm.shape[0]

        #TODO: this doesn't work with uncertain inputs!

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = Qnn = Knm*Kmmi*Kmn
        """
        KmnKnm = np.dot(Kmn,Kmn.T)
        Lm = jitchol(Kmm)
        Lmi = chol_inv(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        KmmiKmn = np.dot(Kmmi,Kmn)
        Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        LLT0 = Kmm.copy()

        #Kmmi, Lm, Lmi, Kmm_logdet = pdinv(Kmm)
        #KmnKnm = np.dot(Kmn, Kmn.T)
        #KmmiKmn = np.dot(Kmmi,Kmn)
        #Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        #LLT0 = Kmm.copy()

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*Gamma
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
        np1 = [self.tau_tilde.copy()]
        np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.random.permutation(self.N)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./Sigma_diag[i] - self.eta*self.tau_tilde[i]
                self.v_[i] = mu[i]/Sigma_diag[i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.LikelihoodFunction.moments_match(self._transf_data[i],self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                self.tau_tilde[i] += Delta_tau
                self.v_tilde[i] += Delta_v
                #Posterior distribution parameters update
                DSYR(LLT,Kmn[:,i].copy(),Delta_tau) #LLT = LLT + np.outer(Kmn[:,i],Kmn[:,i])*Delta_tau
                L = jitchol(LLT)
                #cholUpdate(L,Kmn[:,i]*np.sqrt(Delta_tau))
                V,info = dtrtrs(L,Kmn,lower=1)
                Sigma_diag = np.sum(V*V,-2)
                si = np.sum(V.T*V[:,i],-1)
                mu += (Delta_v-Delta_tau*mu[i])*si
                self.iterations += 1
            #Sigma recomputation with Cholesky decompositon
            LLT = LLT0 + np.dot(Kmn*self.tau_tilde[None,:],Kmn.T)
            L = jitchol(LLT)
            V,info = dtrtrs(L,Kmn,lower=1)
            V2,info = dtrtrs(L.T,V,lower=0)
            Sigma_diag = np.sum(V*V,-2)
            Knmv_tilde = np.dot(Kmn,self.v_tilde)
            mu = np.dot(V2.T,Knmv_tilde)
            epsilon_np1 = sum((self.tau_tilde-np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-np2[-1])**2)/self.N
            np1.append(self.tau_tilde.copy())
            np2.append(self.v_tilde.copy())

        self._compute_GP_variables()

    def fit_FITC(self, Kmm, Kmn, Knn_diag):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see Naish-Guzman and Holden, 2008.
        """
        num_inducing = Kmm.shape[0]

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = diag(Knn-Qnn) + Qnn, Qnn = Knm*Kmmi*Kmn
        """
        Lm = jitchol(Kmm)
        Lmi = chol_inv(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        P0 = Kmn.T
        KmnKnm = np.dot(P0.T, P0)
        KmmiKmn = np.dot(Kmmi,P0.T)
        Qnn_diag = np.sum(P0.T*KmmiKmn,-2)
        Diag0 = Knn_diag - Qnn_diag
        R0 = jitchol(Kmmi).T

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*Gamma
        """
        self.w = np.zeros(self.N)
        self.Gamma = np.zeros(num_inducing)
        mu = np.zeros(self.N)
        P = P0.copy()
        R = R0.copy()
        Diag = Diag0.copy()
        Sigma_diag = Knn_diag
        RPT0 = np.dot(R0,P0.T)

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
                self.tau_[i] = 1./Sigma_diag[i] - self.eta*self.tau_tilde[i]
                self.v_[i] = mu[i]/Sigma_diag[i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.LikelihoodFunction.moments_match(self._transf_data[i],self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                self.tau_tilde[i] += Delta_tau
                self.v_tilde[i] += Delta_v
                #Posterior distribution parameters update
                dtd1 = Delta_tau*Diag[i] + 1.
                dii = Diag[i]
                Diag[i] = dii - (Delta_tau * dii**2.)/dtd1
                pi_ = P[i,:].reshape(1,num_inducing)
                P[i,:] = pi_ - (Delta_tau*dii)/dtd1 * pi_
                Rp_i = np.dot(R,pi_.T)
                RTR = np.dot(R.T,np.dot(np.eye(num_inducing) - Delta_tau/(1.+Delta_tau*Sigma_diag[i]) * np.dot(Rp_i,Rp_i.T),R))
                R = jitchol(RTR).T
                self.w[i] += (Delta_v - Delta_tau*self.w[i])*dii/dtd1
                self.Gamma += (Delta_v - Delta_tau*mu[i])*np.dot(RTR,P[i,:].T)
                RPT = np.dot(R,P.T)
                Sigma_diag = Diag + np.sum(RPT.T*RPT.T,-1)
                mu = self.w + np.dot(P,self.Gamma)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            Iplus_Dprod_i = 1./(1.+ Diag0 * self.tau_tilde)
            Diag = Diag0 * Iplus_Dprod_i
            P = Iplus_Dprod_i[:,None] * P0
            safe_diag = np.where(Diag0 < self.tau_tilde, self.tau_tilde/(1.+Diag0*self.tau_tilde), (1. - Iplus_Dprod_i)/Diag0)
            L = jitchol(np.eye(num_inducing) + np.dot(RPT0,safe_diag[:,None]*RPT0.T))
            R,info = dtrtrs(L,R0,lower=1)
            RPT = np.dot(R,P.T)
            Sigma_diag = Diag + np.sum(RPT.T*RPT.T,-1)
            self.w = Diag * self.v_tilde
            self.Gamma = np.dot(R.T, np.dot(RPT,self.v_tilde))
            mu = self.w + np.dot(P,self.Gamma)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.N
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

        return self._compute_GP_variables()
