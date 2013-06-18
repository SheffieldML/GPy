import numpy as np
import scipy as sp
import GPy
from scipy.linalg import inv, cho_solve, det
from numpy.linalg import cond
from likelihood import likelihood
from ..util.linalg import pdinv, mdot, jitchol, chol_inv, det_ln_diag, pddet
from scipy.linalg.lapack import dtrtrs
import random
#import pylab as plt

class Laplace(likelihood):
    """Laplace approximation to a posterior"""

    def __init__(self, data, likelihood_function, extra_data=None, rasm=True):
        """
        Laplace Approximation

        First find the moments \hat{f} and the hessian at this point (using Newton-Raphson)
        then find the z^{prime} which allows this to be a normalised gaussian instead of a
        non-normalized gaussian

        Finally we must compute the GP variables (i.e. generate some Y^{squiggle} and z^{squiggle}
        which makes a gaussian the same as the laplace approximation

        Arguments
        ---------

        :data: array of data the likelihood function is approximating
        :likelihood_function: likelihood function - subclass of likelihood_function
        :extra_data: additional data used by some likelihood functions, for example survival likelihoods need censoring data
        :rasm: Flag of whether to use rasmussens numerically stable mode finding or simple ncg optimisation

        """
        self.data = data
        self.likelihood_function = likelihood_function
        self.extra_data = extra_data
        self.rasm = rasm

        #Inital values
        self.N, self.D = self.data.shape
        self.is_heteroscedastic = True
        self.Nparams = 0

        self.NORMAL_CONST = ((0.5 * self.N) * np.log(2 * np.pi))

        #Initial values for the GP variables
        self.Y = np.zeros((self.N, 1))
        self.covariance_matrix = np.eye(self.N)
        self.precision = np.ones(self.N)[:, None]
        self.Z = 0
        self.YYT = None

    def predictive_values(self, mu, var, full_cov):
        if full_cov:
            raise NotImplementedError("Cannot make correlated predictions with an Laplace likelihood")
        return self.likelihood_function.predictive_values(mu, var)

    def _get_params(self):
        return np.asarray(self.likelihood_function._get_params())

    def _get_param_names(self):
        return self.likelihood_function._get_param_names()

    def _set_params(self, p):
        return self.likelihood_function._set_params(p)

    def _shared_gradients_components(self):
        #FIXME: Careful of side effects! And make sure W and K are up to date!
        d3lik_d3fhat = self.likelihood_function.d3lik_d3f(self.data, self.f_hat)
        dL_dfhat = -0.5*(np.diag(self.Ki_W_i)[:, None]*d3lik_d3fhat)

        Wi_K_i = self.W_12*self.Bi*self.W_12.T #same as rasms R

        I_KW_i = np.eye(self.N) - np.dot(self.K, Wi_K_i)
        return dL_dfhat, I_KW_i, Wi_K_i

    def _Kgradients(self, dK_dthetaK, X):
        """
        Gradients with respect to prior kernel parameters
        """
        dL_dfhat, I_KW_i, Wi_K_i = self._shared_gradients_components()
        dlp = self.likelihood_function.dlik_df(self.data, self.f_hat)

        #Implicit
        impl = mdot(dlp, dL_dfhat.T, I_KW_i)
        expl_a = - mdot(self.Ki_f, self.Ki_f.T)
        expl_b = Wi_K_i
        expl = 0.5*expl_a - 0.5*expl_b
        dL_dthetaK_exp = dK_dthetaK(expl, X)
        dL_dthetaK_imp = dK_dthetaK(impl, X)
        dL_dthetaK = -(dL_dthetaK_imp + dL_dthetaK_exp)

        #dL_dthetaK = np.zeros(dK_dthetaK.shape)
        #for thetaK_i, dK_dthetaK_i in enumerate(dK_dthetaK):
            ##Explicit
            #f_Ki_dK_dtheta_Ki_f = mdot(self.Ki_f.T, dK_dthetaK_i, self.Ki_f)
            #dL_dthetaK[thetaK_i] = 0.5*f_Ki_dK_dtheta_Ki_f - 0.5*np.trace(Wi_K_i*dK_dthetaK_i)
            ##Implicit
            #df_hat_dthetaK = mdot(I_KW_i, dK_dthetaK_i, dlp)
            #dL_dthetaK[thetaK_i] += np.dot(dL_dfhat.T, df_hat_dthetaK)

        return dL_dthetaK

    def _gradients(self, partial):
        """
        Gradients with respect to likelihood parameters
        """
        #return np.zeros(1)
        dL_dfhat, I_KW_i, Wi_K_i = self._shared_gradients_components()
        dlik_dthetaL, dlik_grad_dthetaL, dlik_hess_dthetaL = self.likelihood_function._gradients(self.data, self.f_hat)

        num_params = len(dlik_dthetaL)
        dL_dthetaL = np.zeros(num_params) # make space for one derivative for each likelihood parameter
        for thetaL_i in range(num_params):
            #Explicit
            #dL_dthetaL[thetaL_i] = np.sum(dlik_dthetaL[thetaL_i]) - 0.5*np.trace(np.dot(Ki_W_i.T, np.diagflat(dlik_hess_dthetaL[thetaL_i])))
            #dL_dthetaL[thetaL_i] = np.sum(dlik_dthetaL[thetaL_i]) + 0.5*np.dot(Ki_W_i.T, dlik_hess_dthetaL[thetaL_i][:, None])
            #                                               might be +
            dL_dthetaL[thetaL_i] = np.sum(dlik_dthetaL[thetaL_i]) - 0.5*np.dot(np.diag(self.Ki_W_i), dlik_hess_dthetaL[thetaL_i])
            #Implicit
            df_hat_dthetaL = mdot(I_KW_i, self.K, dlik_grad_dthetaL[thetaL_i])
            dL_dthetaL[thetaL_i] += np.dot(dL_dfhat.T, df_hat_dthetaL)

        return dL_dthetaL #should be array of length *params-being optimized*, for student t just optimising 1 parameter, this is (1,)

    def _compute_GP_variables(self):
        """
        Generates data Y which would give the normal distribution identical to the laplace approximation

        GPy expects a likelihood to be gaussian, so need to caluclate the points Y^{squiggle} and Z^{squiggle}
        that makes the posterior match that found by a laplace approximation to a non-gaussian likelihood

        Given we are approximating $p(y|f)p(f)$ with a normal distribution (given $p(y|f)$ is not normal)
        then we have a rescaled normal distibution z*N(f|f_hat,hess_hat^-1) with the same area as p(y|f)p(f)
        due to the z rescaling.

        at the moment the data Y correspond to the normal approximation z*N(f|f_hat,hess_hat^1)

        This function finds the data D=(Y_tilde,X) that would produce z*N(f|f_hat,hess_hat^1)
        giving a normal approximation of z_tilde*p(Y_tilde|f,X)p(f)

        $$\tilde{Y} = \tilde{\Sigma} Hf$$
        where
        $$\tilde{\Sigma}^{-1} = H - K^{-1}$$
        i.e. $$\tilde{\Sigma}^{-1} = diag(\nabla\nabla \log(y|f))$$
        since $diag(\nabla\nabla \log(y|f)) = H - K^{-1}$
        and $$\ln \tilde{z} = \ln z + \frac{N}{2}\ln 2\pi + \frac{1}{2}\tilde{Y}\tilde{\Sigma}^{-1}\tilde{Y}$$
        $$\tilde{\Sigma} = W^{-1}$$

        """
        #Wi(Ki + W) = WiKi + I = KW_i + I = L_Lt_W_i + I = Wi_Lit_Li + I = Lt_W_i_Li + I
        #dtritri -> L -> L_i
        #dtrtrs -> L.T*W, L_i -> (L.T*W)_i*L_i
        #((L.T*w)_i + I)f_hat = y_tilde
        L = jitchol(self.K)
        Li = chol_inv(L)
        Lt_W = L.T*self.W.T

        Lt_W_i_Li = dtrtrs(Lt_W, Li, lower=False)[0]
        self.Wi__Ki_W = Lt_W_i_Li + np.eye(self.N)

        Y_tilde = np.dot(self.Wi__Ki_W, self.f_hat)

        ln_W_det = det_ln_diag(self.W)
        yf_W_yf = mdot((Y_tilde - self.f_hat).T, np.diagflat(self.W), (Y_tilde - self.f_hat))

        #Z_tilde = (+ self.NORMAL_CONST
                   #+ self.ln_z_hat
                   #+ 0.5*self.ln_I_KW_det
                   #- 0.5*ln_W_det
                   #+ 0.5*self.f_Ki_f
                   #+ 0.5*yf_W_yf
                   #)

        self.Sigma_tilde = np.diagflat(1.0/self.W)

        Ki, _, _, K_det = pdinv(self.K)
        ln_det_K_Wi__Bi = self.ln_I_KW_det + pddet(self.Sigma_tilde + self.K)
        W = np.diagflat(self.W)
        Wi = self.Sigma_tilde
        W12i = np.sqrt(Wi)
        D = Ki - mdot((Ki + W), W12i, self.Bi, W12i, (Ki + W))
        fDf = mdot(self.f_hat.T, D, self.f_hat)
        l = self.likelihood_function.link_function(self.data, self.f_hat, extra_data=self.extra_data)
        Z_tilde = (+ self.NORMAL_CONST
                   + l
                   + 0.5*ln_det_K_Wi__Bi
                   - 0.5*fDf
                  )

        #Convert to float as its (1, 1) and Z must be a scalar
        self.Z = np.float64(Z_tilde)
        self.Y = Y_tilde
        self.YYT = np.dot(self.Y, self.Y.T)
        self.covariance_matrix = self.Sigma_tilde
        self.precision = 1.0 / np.diag(self.covariance_matrix)[:, None]

    def fit_full(self, K):
        """
        The laplace approximation algorithm, find K and expand hessian
        For nomenclature see Rasmussen & Williams 2006 - modified for numerical stability
        :K: Covariance matrix
        """
        self.K = K.copy()

        #Find mode
        if self.rasm:
            self.f_hat = self.rasm_mode(K)
        else:
            self.f_hat = self.ncg_mode(K)

        #Compute hessian and other variables at mode
        self._compute_likelihood_variables()

    def _compute_likelihood_variables(self):
        #At this point get the hessian matrix (or vector as W is diagonal)
        self.W = -self.likelihood_function.d2lik_d2f(self.data, self.f_hat, extra_data=self.extra_data)

        if not self.likelihood_function.log_concave:
            self.W[self.W < 0] = 1e-6  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                       #If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                       #To cause the posterior to become less certain than the prior and likelihood,
                                       #This is a property only held by non-log-concave likelihoods

        #TODO: Could save on computation when using rasm by returning these, means it isn't just a "mode finder" though
        self.B, self.B_chol, self.W_12 = self._compute_B_statistics(self.K, self.W)
        self.Bi, _, _, B_det = pdinv(self.B)

        #Do the computation again at f to get Ki_f which is useful
        b = self.W*self.f_hat + self.likelihood_function.dlik_df(self.data, self.f_hat, extra_data=self.extra_data)
        solve_chol = cho_solve((self.B_chol, True), np.dot(self.W_12*self.K, b))
        a = b - self.W_12*solve_chol
        self.Ki_f = a

        self.f_Ki_f = np.dot(self.f_hat.T, self.Ki_f)
        self.Ki_W_i = self.K - mdot(self.K, self.W_12*self.Bi*self.W_12.T, self.K)

        #For det, |I + KW| == |I + W_12*K*W_12|
        self.ln_I_KW_det = pddet(np.eye(self.N) + self.W_12*self.K*self.W_12.T)

        #self.ln_I_KW_det = pddet(np.eye(self.N) + np.dot(self.K, self.W))
        self.ln_z_hat = (- 0.5*self.f_Ki_f
                         - self.ln_I_KW_det
                         + self.likelihood_function.link_function(self.data, self.f_hat, extra_data=self.extra_data)
                         )

        return self._compute_GP_variables()

    def _compute_B_statistics(self, K, W):
        """Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal element and can be easyily inverted

        :K: Covariance matrix
        :W: Negative hessian at a point (diagonal matrix)
        :returns: (B, L)
        """
        #W is diagonal so its sqrt is just the sqrt of the diagonal elements
        W_12 = np.sqrt(W)
        B = np.eye(self.N) + W_12*K*W_12.T
        L = jitchol(B)
        return (B, L, W_12)

    def ncg_mode(self, K):
        """
        Find the mode using a normal ncg optimizer and inversion of K (numerically unstable but intuative)
        :K: Covariance matrix
        :returns: f_mode
        """
        self.Ki, _, _, self.ln_K_det = pdinv(K)

        f = np.zeros((self.N, 1))

        #FIXME: Can we get rid of this horrible reshaping?
        #ONLY WORKS FOR 1D DATA
        def obj(f):
            res = -1 * (self.likelihood_function.link_function(self.data[:, 0], f, extra_data=self.extra_data) - 0.5 * np.dot(f.T, np.dot(self.Ki, f))
                        - self.NORMAL_CONST)
            return float(res)

        def obj_grad(f):
            res = -1 * (self.likelihood_function.dlik_df(self.data[:, 0], f, extra_data=self.extra_data) - np.dot(self.Ki, f))
            return np.squeeze(res)

        def obj_hess(f):
            res = -1 * (--np.diag(self.likelihood_function.d2lik_d2f(self.data[:, 0], f, extra_data=self.extra_data)) - self.Ki)
            return np.squeeze(res)

        f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess, disp=False)
        return f_hat[:, None]

    def rasm_mode(self, K, MAX_ITER=500000, MAX_RESTART=50):
        """
        Rasmussens numerically stable mode finding
        For nomenclature see Rasmussen & Williams 2006

        :K: Covariance matrix
        :MAX_ITER: Maximum number of iterations of newton-raphson before forcing finish of optimisation
        :MAX_RESTART: Maximum number of restarts (reducing step_size) before forcing finish of optimisation
        :returns: f_mode
        """
        f = np.zeros((self.N, 1))
        new_obj = -np.inf
        old_obj = np.inf

        def obj(a, f):
            #Careful of shape of data!
            return -0.5*np.dot(a.T, f) + self.likelihood_function.link_function(self.data, f, extra_data=self.extra_data)

        difference = np.inf
        epsilon = 1e-6
        step_size = 1
        rs = 0
        i = 0
        while difference > epsilon and i < MAX_ITER and rs < MAX_RESTART:
            #f_old = f.copy()
            W = -self.likelihood_function.d2lik_d2f(self.data, f, extra_data=self.extra_data)
            if not self.likelihood_function.log_concave:
                W[W < 0] = 1e-6     # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                    # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                    # To cause the posterior to become less certain than the prior and likelihood,
                                    # This is a property only held by non-log-concave likelihoods
            B, L, W_12 = self._compute_B_statistics(K, W)

            W_f = W*f
            grad = self.likelihood_function.dlik_df(self.data, f, extra_data=self.extra_data)
            #Find K_i_f
            b = W_f + grad
            b = step_size*b

            #Need this to find the f we have a stepsize which we need to move in, rather than a full unit movement
            #c = np.dot(K, W_f) + f*(1-step_size) + step_size*np.dot(K, grad)
            #solve_L = cho_solve((L, True), W_12*c)
            #f = c - np.dot(K, W_12*solve_L)

            #FIXME: Can't we get rid of this? Don't we want to evaluate obj(c,f) and this is our new_obj?
            #Why did I choose to evaluate the objective function at the new f with the old hessian? I'm sure there was a good reason,
            #Document it!
            solve_L = cho_solve((L, True), W_12*np.dot(K, b))
            a = b - W_12*solve_L
            f = np.dot(K, a)

            tmp_old_obj = old_obj
            old_obj = new_obj
            new_obj = obj(a, f)
            difference = new_obj - old_obj
            if difference < 0:
                #print "Objective function rose", difference
                #If the objective function isn't rising, restart optimization
                step_size *= 0.9
                #print "Reducing step-size to {ss:.3} and restarting optimization".format(ss=step_size)
                #objective function isn't increasing, try reducing step size
                #f = f_old #it's actually faster not to go back to old location and just zigzag across the mode
                old_obj = tmp_old_obj
                rs += 1

            difference = abs(difference)
            i += 1

        self.i = i
        return f
