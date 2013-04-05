import numpy as np
import scipy as sp
import GPy
from scipy.linalg import cholesky, eig, inv, det, cho_solve
from GPy.likelihoods.likelihood import likelihood
from GPy.util.linalg import pdinv, mdot, jitchol
#import numpy.testing.assert_array_equal

class Laplace(likelihood):
    """Laplace approximation to a posterior"""

    def __init__(self, data, likelihood_function, rasm=True):
        """
        Laplace Approximation

        First find the moments \hat{f} and the hessian at this point (using Newton-Raphson)
        then find the z^{prime} which allows this to be a normalised gaussian instead of a
        non-normalized gaussian

        Finally we must compute the GP variables (i.e. generate some Y^{squiggle} and z^{squiggle}
        which makes a gaussian the same as the laplace approximation

        Arguments
        ---------

        :data: @todo
        :likelihood_function: @todo

        """
        self.data = data
        self.likelihood_function = likelihood_function
        self.rasm = rasm

        #Inital values
        self.N, self.D = self.data.shape
        self.is_heteroscedastic = True
        self.Nparams = 0

        self.NORMAL_CONST = -((0.5 * self.N) * np.log(2 * np.pi))

        #Initial values for the GP variables
        self.Y = np.zeros((self.N, 1))
        self.covariance_matrix = np.eye(self.N)
        self.precision = np.ones(self.N)[:, None]
        self.Z = 0
        self.YYT = None

    def predictive_values(self, mu, var, full_cov):
        if full_cov:
            raise NotImplementedError("Cannot make correlated predictions with an EP likelihood")
        return self.likelihood_function.predictive_values(mu, var)

    def _get_params(self):
        return np.zeros(0)

    def _get_param_names(self):
        return []

    def _set_params(self, p):
        pass # TODO: Laplace likelihood might want to take some parameters...

    def _gradients(self, partial):
        #return np.zeros(0) # TODO: Laplace likelihood might want to take some parameters...
        return np.zeros(0) # TODO: Laplace likelihood might want to take some parameters...
        raise NotImplementedError

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

        """
        self.Sigma_tilde_i = self.W
        #Check it isn't singular!
        epsilon = 1e-6
        if np.abs(det(self.Sigma_tilde_i)) < epsilon:
            print "WARNING: Transformed covariance matrix is signular!"
            #raise ValueError("inverse covariance must be non-singular to invert!")
        #Do we really need to inverse Sigma_tilde_i? :(
        if self.likelihood_function.log_concave:
            (self.Sigma_tilde, _, _, _) = pdinv(self.Sigma_tilde_i)
        else:
            self.Sigma_tilde = inv(self.Sigma_tilde_i)
        #f_hat? should be f but we must have optimized for them I guess?
        #Y_tilde = mdot(self.Sigma_tilde, self.hess_hat_i, self.f_hat)
        Y_tilde = mdot(self.Sigma_tilde, (self.Ki + self.W), self.f_hat)
        #KW = np.dot(self.K, self.W)
        #KW_i, _, _, _ = pdinv(KW)
        #Y_tilde = mdot((KW_i + np.eye(self.N)), self.f_hat)
        #Z_tilde = (self.ln_z_hat - self.NORMAL_CONST
                    #+ 0.5*mdot(self.f_hat.T, (self.hess_hat, self.f_hat))
                    #+ 0.5*mdot(Y_tilde.T, (self.Sigma_tilde_i, Y_tilde))
                    #- mdot(Y_tilde.T, (self.Sigma_tilde_i, self.f_hat))
                   #)
        _, _, _, ln_W12_Bi_W12_i = pdinv(mdot(self.W_12, self.Bi, self.W_12))
        f_Si_f = mdot(self.f_hat.T, self.Sigma_tilde_i, self.f_hat)
        Z_tilde = -self.NORMAL_CONST + self.ln_z_hat -0.5*ln_W12_Bi_W12_i - 0.5*self.f_Ki_f - 0.5*f_Si_f

        #Convert to float as its (1, 1) and Z must be a scalar
        self.Z = np.float64(Z_tilde)
        self.Y = Y_tilde
        self.YYT = np.dot(self.Y, self.Y.T)
        self.covariance_matrix = self.Sigma_tilde
        self.precision = 1 / np.diag(self.covariance_matrix)[:, None]

    def fit_full(self, K):
        """
        The laplace approximation algorithm
        For nomenclature see Rasmussen & Williams 2006
        :K: Covariance matrix
        """
        self.K = K.copy()
        print "Inverting K"
        #self.Ki, _, _, log_Kdet = pdinv(K)
        print "K inverted, optimising"
        if self.rasm:
            self.f_hat = self.rasm_mode(K)
        else:
            self.f_hat = self.ncg_mode(K)

        #At this point get the hessian matrix
        self.W = -np.diag(self.likelihood_function.link_hess(self.data, self.f_hat))

        if not self.likelihood_function.log_concave:
            self.W[self.W < 0] = 1e-6 #FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                   #If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                   #To cause the posterior to become less certain than the prior and likelihood,
                                   #This is a property only held by non-log-concave likelihoods
        #TODO: Could save on computation when using rasm by returning these, means it isn't just a "mode finder" though
        self.B, L, self.W_12 = self._compute_B_statistics(K, self.W)
        self.Bi, _, _, B_det = pdinv(self.B)
        #ln_W_det = np.linalg.det(self.W)
        #ln_B_det = np.linalg.det(self.B)
        ln_det = np.linalg.det(np.eye(self.N) - mdot(self.W_12, self.Bi, self.W_12, K))
        b = np.dot(self.W, self.f_hat) + self.likelihood_function.link_grad(self.data, self.f_hat)[:, None]
        #TODO: Check L is lower
        solve_L = cho_solve((L, True), mdot(self.W_12, (K, b)))
        a = b - mdot(self.W_12, solve_L)
        self.f_Ki_f = np.dot(self.f_hat.T, a)

        #self.hess_hat = self.Ki + self.W
        #(self.hess_hat, _, _, self.log_hess_hat_i_det) = pdinv(self.hess_hat)

        ##Check hess_hat is positive definite
        #try:
            #cholesky(self.hess_hat)
        #except:
            #raise ValueError("Must be positive definite")

        ##Check its eigenvalues are positive
        #eigenvalues = eig(self.hess_hat)
        #if not np.all(eigenvalues > 0):
            #raise ValueError("Eigen values not positive")

        #z_hat is how much we need to scale the normal distribution by to get the area of our approximation close to
        #the area of p(f)p(y|f) we do this by matching the height of the distributions at the mode
        #z_hat = -0.5*ln|H| - 0.5*ln|K| - 0.5*f_hat*K^{-1}*f_hat \sum_{n} ln p(y_n|f_n)
        #Unsure whether its log_hess or log_hess_i
        #self.ln_z_hat = (- 0.5*self.log_hess_hat_i_det
                         #+ 0.5*self.log_Kdet
                         #+ self.likelihood_function.link_function(self.data, self.f_hat)
                         ##+ self.likelihood_function.link_function(self.data, self.f_hat)
                         #- 0.5*mdot(self.f_hat.T, (self.Ki, self.f_hat))
                         #)
        self.ln_z_hat = (- 0.5*log_Kdet
                         - 0.5*self.f_Ki_f
                         + self.likelihood_function.link_function(self.data, self.f_hat)
                         + 0.5*ln_det
                         )

        return self._compute_GP_variables()

    def _compute_B_statistics(self, K, W):
        """Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal element and can be easyily inverted

        :K: Covariance matrix
        :W: Negative hessian at a point (diagonal matrix)
        :returns: (B, L)
        """
        #W is diagnoal so its sqrt is just the sqrt of the diagonal elements
        W_12 = np.sqrt(W)
        import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
        B = np.eye(K.shape[0]) + mdot(W_12, K, W_12)
        L = jitchol(B)
        return (B, L, W_12)

    def ncg_mode(self, K):
        """Find the mode using a normal ncg optimizer and inversion of K (numerically unstable but intuative)
        :K: Covariance matrix
        :returns: f_mode
        """
        f = np.zeros((self.N, 1))
        LOG_K_CONST = -(0.5 * self.log_Kdet)

        #FIXME: Can we get rid of this horrible reshaping?
        def obj(f):
            res = -1 * (self.likelihood_function.link_function(self.data[:, 0], f) - 0.5 * mdot(f.T, (self.Ki, f))
                        + self.NORMAL_CONST + LOG_K_CONST)
            return float(res)

        def obj_grad(f):
            res = -1 * (self.likelihood_function.link_grad(self.data[:, 0], f) - mdot(self.Ki, f))
            return np.squeeze(res)

        def obj_hess(f):
            res = -1 * (--np.diag(self.likelihood_function.link_hess(self.data[:, 0], f)) - self.Ki)
            return np.squeeze(res)

        f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess)
        return f_hat[:, None]

    def rasm_mode(self, K, MAX_ITER=500000, MAX_RESTART=50):
        """
        Rasmussens numerically stable mode finding
        For nomenclature see Rasmussen & Williams 2006

        :K: Covariance matrix
        :returns: f_mode
        """
        f = np.zeros((self.N, 1))
        new_obj = -np.inf
        old_obj = np.inf

        def obj(a, f):
            #Careful of shape of data!
            return -0.5*np.dot(a.T, f) + self.likelihood_function.link_function(self.data, f)

        difference = np.inf
        epsilon = 1e-6
        step_size = 1
        rs = 0
        i = 0
        while difference > epsilon:# and i < MAX_ITER and rs < MAX_RESTART:
            print "optimising"
            f_old = f.copy()
            W = -np.diag(self.likelihood_function.link_hess(self.data, f))
            if not self.likelihood_function.log_concave:
                #if np.any(W < 0):
                    #print "NEGATIVE VALUES :("
                    #pass
                W[W < 0] = 1e-6     #FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                    #If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                    #To cause the posterior to become less certain than the prior and likelihood,
                                    #This is a property only held by non-log-concave likelihoods
            print "Decomposing"
            B, L, W_12 = self._compute_B_statistics(K, W)
            print "Finding f"

            W_f = np.dot(W, f)#FIXME: Make this fast as W_12 is diagonal!
            grad = self.likelihood_function.link_grad(self.data, f)[:, None]
            #Find K_i_f
            b = W_f + grad
            #b = np.dot(W, f) + np.dot(self.Ki, f)*(1-step_size) + step_size*self.likelihood_function.link_grad(self.data, f)[:, None]
            #TODO: Check L is lower

            solve_L = cho_solve((L, True), mdot(W_12, (K, b)))#FIXME: Make this fast as W_12 is diagonal!
            a = b - mdot(W_12, solve_L)#FIXME: Make this fast as W_12 is diagonal!
            #f = np.dot(K, a)

            #a should be equal to Ki*f now so should be able to use it
            c = mdot(K, W_f) + f*(1-step_size) + step_size*np.dot(K, grad)
            solve_L = cho_solve((L, True), mdot(W_12, c))#FIXME: Make this fast as W_12 is diagonal!
            f = c - mdot(K, W_12, solve_L)#FIXME: Make this fast as W_12 is diagonal!

            #K_w_f = mdot(K, (W, f))
            #c = step_size*mdot(K, self.likelihood_function.link_grad(self.data, f)[:, None]) - step_size*f
            #d = f + K_w_f + c
            #solve_L = cho_solve((L, True), mdot(W_12, d))
            #f = c - mdot(K, (W_12, solve_L))
            #a = mdot(self.Ki, f)

            tmp_old_obj = old_obj
            old_obj = new_obj
            new_obj = obj(a, f)
            difference = new_obj - old_obj
            #print "Difference: ", difference
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
        #print "{i} steps".format(i=i)
        return f
