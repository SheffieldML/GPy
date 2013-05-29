import numpy as np
import scipy as sp
import GPy
from scipy.linalg import inv, cho_solve, det
from numpy.linalg import cond
from likelihood import likelihood
from ..util.linalg import pdinv, mdot, jitchol, chol_inv, det_ln_diag, pddet
from scipy.linalg.lapack import dtrtrs
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
        return np.asarray(self.likelihood_function._get_params())

    def _get_param_names(self):
        return self.likelihood_function._get_param_names()

    def _set_params(self, p):
        #print "Setting laplace param with: ", p
        return self.likelihood_function._set_params(p)

    def both_gradients(self, dL_d_K_Sigma, dK_dthetaK):
        """
        Find the gradients of the marginal likelihood w.r.t both thetaK and thetaL

        dL_dthetaK differs from that of normal likelihoods as it has additional terms coming from
        changes to y_tilde and changes to Sigma_tilde when the kernel parameters are adjusted

        Similar terms arise when finding the gradients with respect to changes in the liklihood
        parameters
        """
        return (self._Kgradients(dL_d_K_Sigma, dK_dthetaK), self._gradients(dL_d_K_Sigma))

    def _shared_gradients_components(self):

    def _Kgradients(self, dL_d_K_Sigma, dK_dthetaK):
        """
        Gradients with respect to prior kernel parameters
        """
        return dL_dthetaK

    def _gradients(self, partial):
        """
        Gradients with respect to likelihood parameters
        """
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
        epsilon = 1e14

        #Wi(Ki + W) = WiKi + I = KW_i + I = L_Lt_W_i + I = Wi_Lit_Li + I = Lt_W_i_Li + I
        #dtritri -> L -> L_i
        #dtrtrs -> L.T*W, L_i -> (L.T*W)_i*L_i
        #((L.T*w)_i + I)f_hat = y_tilde
        L = jitchol(self.K)
        Li = chol_inv(L)
        Lt_W = np.dot(L.T, self.W) #FIXME: Can make Faster

        ##Check it isn't singular!
        if cond(Lt_W) > epsilon:
            print "WARNING: L_inv.T * W matrix is singular,\nnumerical stability may be a problem"

        Lt_W_i_Li = dtrtrs(Lt_W, Li, lower=False)[0]
        self.Wi__Ki_W = Lt_W_i_Li + np.eye(self.N)
        Y_tilde = np.dot(self.Wi__Ki_W, self.f_hat)

        #f.T(Ki + W)f
        f_Ki_W_f = (np.dot(self.f_hat.T, cho_solve((L, True), self.f_hat))
                    + mdot(self.f_hat.T, self.W, self.f_hat)
                    )

        y_W_f = mdot(Y_tilde.T, self.W, self.f_hat)
        y_W_y = mdot(Y_tilde.T, self.W, Y_tilde)
        ln_W_det = det_ln_diag(self.W)
        Z_tilde = (- self.NORMAL_CONST
                   + 0.5*self.ln_K_det
                   + 0.5*ln_W_det
                   + 0.5*self.ln_Ki_W_i_det
                   + 0.5*f_Ki_W_f
                   + 0.5*y_W_y
                   - y_W_f
                   + self.ln_z_hat
                   )
        #Z_tilde = (self.NORMAL_CONST
                   #- 0.5*self.ln_K_det
                   #- 0.5*ln_W_det
                   #- 0.5*self.ln_Ki_W_i_det
                   #- 0.5*f_Ki_W_f
                   #- 0.5*y_W_y
                   #+ y_W_f
                   #+ self.ln_z_hat
                   #)
        self.Z_tilde = 0

        ##Check it isn't singular!
        if cond(self.W) > epsilon:
            print "WARNING: Transformed covariance matrix is singular,\nnumerical stability may be a problem"

        self.Sigma_tilde = inv(self.W)  # Damn

        #Convert to float as its (1, 1) and Z must be a scalar
        self.Z = np.float64(Z_tilde)
        self.Y = Y_tilde
        self.YYT = np.dot(self.Y, self.Y.T)
        self.covariance_matrix = self.Sigma_tilde
        self.precision = 1 / np.diag(self.covariance_matrix)[:, None]

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
        #At this point get the hessian matrix
        #print "Data: ", self.data
        #print "fhat: ", self.f_hat
        self.W = -np.diag(self.likelihood_function.link_hess(self.data, self.f_hat, extra_data=self.extra_data))

        if not self.likelihood_function.log_concave:
            self.W[self.W < 0] = 1e-6  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                       #If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                       #To cause the posterior to become less certain than the prior and likelihood,
                                       #This is a property only held by non-log-concave likelihoods

        #TODO: Could save on computation when using rasm by returning these, means it isn't just a "mode finder" though
        self.B, self.B_chol, self.W_12 = self._compute_B_statistics(self.K, self.W)
        self.Bi, _, _, B_det = pdinv(self.B)

        Ki_W_i = self.K - mdot(self.K, self.W_12, self.Bi, self.W_12, self.K)
        self.ln_Ki_W_i_det = np.linalg.det(Ki_W_i)

        b = np.dot(self.W, self.f_hat) + self.likelihood_function.link_grad(self.data, self.f_hat, extra_data=self.extra_data)[:, None]
        solve_chol = cho_solve((self.B_chol, True), mdot(self.W_12, (self.K, b)))
        a = b - mdot(self.W_12, solve_chol)
        self.Ki_f = a
        self.f_Ki_f = np.dot(self.f_hat.T, self.Ki_f)
        self.ln_K_det = pddet(self.K)

        self.ln_z_hat = (- 0.5*self.f_Ki_f
                         - 0.5*self.ln_K_det
                         + 0.5*self.ln_Ki_W_i_det
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
        #W is diagnoal so its sqrt is just the sqrt of the diagonal elements
        W_12 = np.sqrt(W)
        B = np.eye(K.shape[0]) + np.dot(W_12, np.dot(K, W_12))
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
                        + self.NORMAL_CONST)
            return float(res)

        def obj_grad(f):
            res = -1 * (self.likelihood_function.link_grad(self.data[:, 0], f, extra_data=self.extra_data) - np.dot(self.Ki, f))
            return np.squeeze(res)

        def obj_hess(f):
            res = -1 * (--np.diag(self.likelihood_function.link_hess(self.data[:, 0], f, extra_data=self.extra_data)) - self.Ki)
            return np.squeeze(res)

        f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess)
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
            W = -np.diag(self.likelihood_function.link_hess(self.data, f, extra_data=self.extra_data))
            if not self.likelihood_function.log_concave:
                W[W < 0] = 1e-6     # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                    # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                    # To cause the posterior to become less certain than the prior and likelihood,
                                    # This is a property only held by non-log-concave likelihoods
            B, L, W_12 = self._compute_B_statistics(K, W)

            W_f = np.dot(W, f)
            grad = self.likelihood_function.link_grad(self.data, f, extra_data=self.extra_data)[:, None]
            #Find K_i_f
            b = W_f + grad

            #a should be equal to Ki*f now so should be able to use it
            c = np.dot(K, W_f) + f*(1-step_size) + step_size*np.dot(K, grad)
            solve_L = cho_solve((L, True), np.dot(W_12, c))
            f = c - np.dot(K, np.dot(W_12, solve_L))

            solve_L = cho_solve((L, True), np.dot(W_12, np.dot(K, b)))
            a = b - np.dot(W_12, solve_L)
            #f = np.dot(K, a)

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
