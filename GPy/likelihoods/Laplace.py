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

    def __init__(self, data, likelihood_function, extra_data=None, opt='rasm'):
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
        :opt: Optimiser to use, rasm numerically stable, ncg or nelder-mead (latter only work with 1d data)

        """
        self.data = data
        self.likelihood_function = likelihood_function
        self.extra_data = extra_data
        self.opt = opt

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

        self.old_a = None

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
        dL_dfhat = -0.5*(np.diag(self.Ki_W_i)[:, None]*d3lik_d3fhat).T
        I_KW_i = np.eye(self.N) - np.dot(self.K, self.Wi_K_i)
        return dL_dfhat, I_KW_i

    def _Kgradients(self, dK_dthetaK, X):
        """
        Gradients with respect to prior kernel parameters
        """
        dL_dfhat, I_KW_i = self._shared_gradients_components()
        dlp = self.likelihood_function.dlik_df(self.data, self.f_hat)

        #Implicit
        impl = mdot(dlp, dL_dfhat, I_KW_i)
        expl_a = mdot(self.Ki_f, self.Ki_f.T)
        expl_b = self.Wi_K_i
        #print "expl_a: {}, expl_b: {}".format(expl_a, expl_b)
        expl = 0.5*expl_a + 0.5*expl_b # Might need to be -?
        dL_dthetaK_exp = dK_dthetaK(expl, X)
        dL_dthetaK_imp = dK_dthetaK(impl, X)
        print "dL_dthetaK_exp: {}     dL_dthetaK_implicit: {}".format(dL_dthetaK_exp, dL_dthetaK_imp)
        dL_dthetaK = dL_dthetaK_exp + dL_dthetaK_imp
        return dL_dthetaK

    def _gradients(self, partial):
        """
        Gradients with respect to likelihood parameters
        """
        dL_dfhat, I_KW_i = self._shared_gradients_components()
        dlik_dthetaL, dlik_grad_dthetaL, dlik_hess_dthetaL = self.likelihood_function._gradients(self.data, self.f_hat)

        num_params = len(dlik_dthetaL)
        dL_dthetaL = np.zeros(num_params) # make space for one derivative for each likelihood parameter
        for thetaL_i in range(num_params):
            #Explicit
            dL_dthetaL_exp = np.sum(dlik_dthetaL[thetaL_i]) - 0.5*np.dot(np.diag(self.Ki_W_i), dlik_hess_dthetaL[thetaL_i])
            #dL_dthetaL_exp = np.sum(dlik_dthetaL[thetaL_i]) - 0.5*np.trace(mdot(self.Bi, self.K, dlik_hess_dthetaL[thetaL_i]))
            #Implicit
            df_hat_dthetaL = mdot(I_KW_i, self.K, dlik_grad_dthetaL[thetaL_i])
            dL_dthetaL_imp = np.dot(dL_dfhat, df_hat_dthetaL)
            #print "dL_dthetaL_exp: {}     dL_dthetaL_implicit: {}".format(dL_dthetaL_exp, dL_dthetaL_imp)
            dL_dthetaL[thetaL_i] = dL_dthetaL_exp + dL_dthetaL_imp

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
        #L = jitchol(self.K)
        #Li = chol_inv(L)
        #Lt_W = L.T*self.W.T

        #Lt_W_i_Li = dtrtrs(Lt_W, Li, lower=True)[0]
        #self.Wi__Ki_W = Lt_W_i_Li + np.eye(self.N)
        #Y_tilde = np.dot(self.Wi__Ki_W, self.f_hat)

        Wi = 1.0/self.W
        self.Sigma_tilde = np.diagflat(Wi)

        Y_tilde = Wi*self.Ki_f + self.f_hat

        self.Wi_K_i = self.W_12*self.Bi*self.W_12.T #same as rasms R
        #self.Wi_K_i[self.Wi_K_i< 1e-6] = 1e-6

        self.ln_det_K_Wi__Bi = self.ln_I_KW_det + pddet(self.Sigma_tilde + self.K)
        self.lik = self.likelihood_function.link_function(self.data, self.f_hat, extra_data=self.extra_data)

        self.y_Wi_Ki_i_y = mdot(Y_tilde.T, self.Wi_K_i, Y_tilde)
        self.aA = 0.5*self.ln_det_K_Wi__Bi
        self.bB = - 0.5*self.f_Ki_f
        self.cC = 0.5*self.y_Wi_Ki_i_y
        Z_tilde = (+ 100*self.NORMAL_CONST
                   + self.lik
                   + 0.5*self.ln_det_K_Wi__Bi
                   - 0.5*self.f_Ki_f
                   + 0.5*self.y_Wi_Ki_i_y
                  )
        print "Ztilde: {} lik: {} a: {} b: {} c: {}".format(Z_tilde, self.lik, self.aA, self.bB, self.cC)
        print self.likelihood_function._get_params()

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
        self.f_hat = {
            'rasm': self.rasm_mode,
            'ncg': self.ncg_mode,
            'nelder': self.nelder_mode
        }[self.opt](self.K)

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
        #b = self.W*self.f_hat + self.likelihood_function.dlik_df(self.data, self.f_hat, extra_data=self.extra_data)
        #solve_chol = cho_solve((self.B_chol, True), np.dot(self.W_12*self.K, b))
        #a = b - self.W_12*solve_chol
        self.Ki_f = self.a

        self.f_Ki_f = np.dot(self.f_hat.T, self.Ki_f)
        self.Ki_W_i = self.K - mdot(self.K, self.W_12*self.Bi*self.W_12.T, self.K)

        #For det, |I + KW| == |I + W_12*K*W_12|
        self.ln_I_KW_det = pddet(np.eye(self.N) + self.W_12*self.K*self.W_12.T)

        #self.ln_I_KW_det = pddet(np.eye(self.N) + np.dot(self.K, self.W))
        #self.ln_z_hat = (- 0.5*self.f_Ki_f
                         #- self.ln_I_KW_det
                         #+ self.likelihood_function.link_function(self.data, self.f_hat, extra_data=self.extra_data)
                         #)

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

    def nelder_mode(self, K):
        f = np.zeros((self.N, 1))
        self.Ki, _, _, self.ln_K_det = pdinv(K)
        def obj(f):
            res = -1 * (self.likelihood_function.link_function(self.data[:, 0], f, extra_data=self.extra_data) - 0.5*np.dot(f.T, np.dot(self.Ki, f)))
            return float(res)

        res = sp.optimize.minimize(obj, f, method='nelder-mead', options={'xtol': 1e-7, 'maxiter': 25000, 'disp': True})
        f_new = res.x
        return f_new[:, None]

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
            res = -1 * (np.diag(self.likelihood_function.d2lik_d2f(self.data[:, 0], f, extra_data=self.extra_data)) - self.Ki)
            return np.squeeze(res)

        f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess, disp=False)
        return f_hat[:, None]

    def rasm_mode(self, K, MAX_ITER=100, MAX_RESTART=10):
        """
        Rasmussens numerically stable mode finding
        For nomenclature see Rasmussen & Williams 2006

        :K: Covariance matrix
        :MAX_ITER: Maximum number of iterations of newton-raphson before forcing finish of optimisation
        :MAX_RESTART: Maximum number of restarts (reducing step_size) before forcing finish of optimisation
        :returns: f_mode
        """
        self.old_before_s = self.likelihood_function._get_params()
        print "before: ", self.old_before_s
        #if self.old_before_s < 1e-5:
            #import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

        #old_a = np.zeros((self.N, 1))
        if self.old_a is None:
            old_a = np.zeros((self.N, 1))
            f = np.dot(K, old_a)
        else:
            old_a = self.old_a.copy()
            f = self.f_hat.copy()

        new_obj = -np.inf
        old_obj = np.inf

        def obj(a, f):
            return -0.5*np.dot(a.T, f) + self.likelihood_function.link_function(self.data, f, extra_data=self.extra_data)

        difference = np.inf
        epsilon = 1e-4
        step_size = 1
        rs = 0
        i = 0

        while difference > epsilon and i < MAX_ITER:# and rs < MAX_RESTART:
            W = -self.likelihood_function.d2lik_d2f(self.data, f, extra_data=self.extra_data)
            #W = np.maximum(W, 0)
            if not self.likelihood_function.log_concave:
                W[W < 0] = 1e-6     # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                    # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                    # To cause the posterior to become less certain than the prior and likelihood,
                                    # This is a property only held by non-log-concave likelihoods
            B, L, W_12 = self._compute_B_statistics(K, W.copy())

            W_f = W*f
            grad = self.likelihood_function.dlik_df(self.data, f, extra_data=self.extra_data)

            b = W_f + grad
            solve_L = cho_solve((L, True), W_12*np.dot(K, b))
            #Work out the DIRECTION that we want to move in, but don't choose the stepsize yet
            full_step_a = b - W_12*solve_L
            da = full_step_a - old_a

            #f_old = f.copy()
            #def inner_obj(step_size, old_a, da, K):
                #a = old_a + step_size*da
                #f = np.dot(K, a)
                #self.a = a.copy() # This is nasty, need to set something within an optimization though
                #self.f = f.copy()
                #return -obj(a, f)

            #from functools import partial
            #i_o = partial(inner_obj, old_a=old_a, da=da, K=K)
            ##new_obj = sp.optimize.brent(i_o, tol=1e-4, maxiter=20)
            #new_obj = sp.optimize.minimize_scalar(i_o, method='brent', tol=1e-4, options={'maxiter':20, 'disp':True}).fun
            #f = self.f.copy()
            #import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

            f_old = f.copy()
            update_passed = False
            while not update_passed:
                a = old_a + step_size*da
                f = np.dot(K, a)

                old_obj = new_obj
                new_obj = obj(a, f)
                difference = new_obj - old_obj
                print "difference: ",difference
                if difference < 0:
                    #print "Objective function rose", np.float(difference)
                    #If the objective function isn't rising, restart optimization
                    step_size *= 0.8
                    #print "Reducing step-size to {ss:.3} and restarting optimization".format(ss=step_size)
                    #objective function isn't increasing, try reducing step size
                    f = f_old.copy() #it's actually faster not to go back to old location and just zigzag across the mode
                    old_obj = new_obj
                    rs += 1
                else:
                    update_passed = True

            #difference = abs(new_obj - old_obj)
            #old_obj = new_obj.copy()
            difference = np.abs(np.sum(f - f_old))
            #old_a = self.a.copy() #a
            old_a = a.copy()
            i += 1
            #print "a max: {} a min: {} a var: {}".format(np.max(self.a), np.min(self.a), np.var(self.a))

        self.old_a = old_a.copy()
        #print "Positive difference obj: ", np.float(difference)
        #print "Iterations: {}, Step size reductions: {}, Final_difference: {}, step_size: {}".format(i, rs, difference, step_size)
        print "Iterations: {}, Final_difference: {}".format(i, difference)
        if difference > 1e-4:
            print "FAIL FAIL FAIL FAIL FAIL FAIL"
            import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
            if hasattr(self, 'X'):
                import pylab as pb
                pb.figure()
                pb.subplot(311)
                pb.title('old f_hat')
                pb.plot(self.X, self.f_hat)
                pb.subplot(312)
                pb.title('old ff')
                pb.plot(self.X, self.old_ff)
                pb.subplot(313)
                pb.title('new f_hat')
                pb.plot(self.X, f)

                pb.figure()
                pb.subplot(121)
                pb.title('old K')
                pb.imshow(np.diagflat(self.old_K), interpolation='none')
                pb.colorbar()
                pb.subplot(122)
                pb.title('new K')
                pb.imshow(np.diagflat(K), interpolation='none')
                pb.colorbar()

                pb.figure()
                pb.subplot(121)
                pb.title('old W')
                pb.imshow(np.diagflat(self.old_W), interpolation='none')
                pb.colorbar()
                pb.subplot(122)
                pb.title('new W')
                pb.imshow(np.diagflat(W), interpolation='none')
                pb.colorbar()

                import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
                pb.close('all')

        #FIXME: DELETE THESE
        self.old_W = W.copy()
        self.old_grad = grad.copy()
        self.old_B = B.copy()
        self.old_W_12 = W_12.copy()
        self.old_ff = f.copy()
        self.old_K = self.K.copy()
        self.old_s = self.likelihood_function._get_params()
        print "after: ", self.old_s
        #print "FINAL a max: {} a min: {} a var: {}".format(np.max(self.a), np.min(self.a), np.var(self.a))
        self.a = a
        #self.B, self.B_chol, self.W_12 = B, L, W_12
        #self.Bi, _, _, B_det = pdinv(self.B)
        return f
