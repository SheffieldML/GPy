# Copyright (c) 2013, 2014 Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
#Parts of this file were influenced by the Matlab GPML framework written by
#Carl Edward Rasmussen & Hannes Nickisch, however all bugs are our own.
#
#The GPML code is released under the FreeBSD License.
#Copyright (c) 2005-2013 Carl Edward Rasmussen & Hannes Nickisch. All rights reserved.
#
#The code and associated documentation is available from
#http://gaussianprocess.org/gpml/code.

import numpy as np
from ...util.linalg import mdot, jitchol, dpotrs, dtrtrs, dpotri, symmetrify, pdinv
from .posterior import Posterior
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
from scipy import optimize
from . import LatentFunctionInference
from scipy.integrate import quad

class Laplace(LatentFunctionInference):

    def __init__(self):
        """
        Laplace Approximation

        Find the moments \hat{f} and the hessian at this point
        (using Newton-Raphson) of the unnormalised posterior

        """

        self._mode_finding_tolerance = 1e-4
        self._mode_finding_max_iter = 30
        self.bad_fhat = False
        #Store whether it is the first run of the inference so that we can choose whether we need
        #to calculate things or reuse old variables
        self.first_run = True
        self._previous_Ki_fhat = None

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None, f_hat=None, W=None, Ki_W_i=None):
        """
        Leave one out log predictive density as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        Ki_f_init = np.zeros_like(Y)

        if K is None:
            K = kern.K(X)

        if f_hat is None:
            f_hat, _ = self.rasm_mode(K, Y, likelihood, Ki_f_init, Y_metadata=Y_metadata)

        if W is None:
            W = -likelihood.d2logpdf_df2(f_hat, Y, Y_metadata=Y_metadata)

        if Ki_W_i is None:
            _, _, _, Ki_W_i = self._compute_B_statistics(K, W, likelihood.log_concave)

        logpdf_dfhat = likelihood.dlogpdf_df(f_hat, Y, Y_metadata=Y_metadata)

        if W.shape[1] == 1:
            W = np.diagflat(W)

        #Eq 14, and 16
        var_site = 1./np.diag(W)[:, None]
        mu_site = f_hat + var_site*logpdf_dfhat
        prec_site = 1./var_site
        #Eq 19
        marginal_cov = Ki_W_i
        marginal_mu = marginal_cov.dot(np.diagflat(prec_site)).dot(mu_site)
        marginal_var = np.diag(marginal_cov)[:, None]
        #Eq 30 with using site parameters instead of Gaussian site parameters
        #(var_site instead of sigma^{2} )
        posterior_cav_var = 1./(1./marginal_var - 1./var_site)
        posterior_cav_mean = posterior_cav_var*((1./marginal_var)*marginal_mu - (1./var_site)*Y)

        flat_y = Y.flatten()
        flat_mu = posterior_cav_mean.flatten()
        flat_var = posterior_cav_var.flatten()

        if Y_metadata is not None:
            #Need to zip individual elements of Y_metadata aswell
            Y_metadata_flat = {}
            if Y_metadata is not None:
                for key, val in Y_metadata.items():
                    Y_metadata_flat[key] = np.atleast_1d(val).reshape(-1, 1)

            zipped_values = []

            for i in range(Y.shape[0]):
                y_m = {}
                for key, val in Y_metadata_flat.items():
                    if np.isscalar(val) or val.shape[0] == 1:
                        y_m[key] = val
                    else:
                        #Won't broadcast yet
                        y_m[key] = val[i]
                zipped_values.append((flat_y[i], flat_mu[i], flat_var[i], y_m))
        else:
            #Otherwise just pass along None's
            zipped_values = zip(flat_y, flat_mu, flat_var, [None]*Y.shape[0])

        def integral_generator(yi, mi, vi, yi_m):
            def f(fi_star):
                #More stable in the log space
                p_fi = np.exp(likelihood.logpdf(fi_star, yi, yi_m)
                              - 0.5*np.log(2*np.pi*vi)
                              - 0.5*np.square(mi-fi_star)/vi)
                return p_fi
            return f

        #Eq 30
        p_ystar, _ = zip(*[quad(integral_generator(y, m, v, yi_m), -np.inf, np.inf)
                           for y, m, v, yi_m in zipped_values])
        p_ystar = np.array(p_ystar).reshape(-1, 1)
        return np.log(p_ystar)

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """
        assert mean_function is None, "inference with a mean function not implemented"

        # Compute K
        K = kern.K(X)

        #Find mode
        if self.bad_fhat or self.first_run:
            Ki_f_init = np.zeros_like(Y)
            self.first_run = False
        else:
            Ki_f_init = self._previous_Ki_fhat

        Ki_f_init = np.zeros_like(Y)# FIXME: take this out

        f_hat, Ki_fhat = self.rasm_mode(K, Y, likelihood, Ki_f_init, Y_metadata=Y_metadata)

        #Compute hessian and other variables at mode
        log_marginal, woodbury_inv, dL_dK, dL_dthetaL = self.mode_computations(f_hat, Ki_fhat, K, Y, likelihood, kern, Y_metadata)

        self._previous_Ki_fhat = Ki_fhat.copy()
        return Posterior(woodbury_vector=Ki_fhat, woodbury_inv=woodbury_inv, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}

    def rasm_mode(self, K, Y, likelihood, Ki_f_init, Y_metadata=None, *args, **kwargs):
        """
        Rasmussen's numerically stable mode finding
        For nomenclature see Rasmussen & Williams 2006
        Influenced by GPML (BSD) code, all errors are our own

        :param K: Covariance matrix evaluated at locations X
        :type K: NxD matrix
        :param Y: The data
        :type Y: np.ndarray
        :param likelihood: the likelihood of the latent function value for the given data
        :type likelihood: a GPy.likelihood object
        :param Ki_f_init: the initial guess at the mode
        :type Ki_f_init: np.ndarray
        :param Y_metadata: information about the data, e.g. which likelihood to take from a multi-likelihood object
        :type Y_metadata: np.ndarray | None
        :returns: f_hat, mode on which to make laplace approxmiation
        :rtype: np.ndarray
        """

        Ki_f = Ki_f_init.copy()
        f = np.dot(K, Ki_f)

        #define the objective function (to be maximised)
        def obj(Ki_f, f):
            ll = -0.5*np.sum(np.dot(Ki_f.T, f)) + np.sum(likelihood.logpdf(f, Y, Y_metadata=Y_metadata))
            if np.isnan(ll):
                import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
                return -np.inf
            else:
                return ll


        difference = np.inf
        iteration = 0
        while difference > self._mode_finding_tolerance and iteration < self._mode_finding_max_iter:
            W = -likelihood.d2logpdf_df2(f, Y, Y_metadata=Y_metadata)
            if np.any(np.isnan(W)):
                raise ValueError('One or more element(s) of W is NaN')
            grad = likelihood.dlogpdf_df(f, Y, Y_metadata=Y_metadata)
            if np.any(np.isnan(grad)):
                raise ValueError('One or more element(s) of grad is NaN')

            W_f = W*f

            b = W_f + grad # R+W p46 line 6.
            W12BiW12, _, _, _ = self._compute_B_statistics(K, W, likelihood.log_concave, *args, **kwargs)
            W12BiW12Kb = np.dot(W12BiW12, np.dot(K, b))

            #Work out the DIRECTION that we want to move in, but don't choose the stepsize yet
            full_step_Ki_f = b - W12BiW12Kb # full_step_Ki_f = a in R&W p46 line 6.
            dKi_f = full_step_Ki_f - Ki_f

            #define an objective for the line search (minimize this one)
            def inner_obj(step_size):
                Ki_f_trial = Ki_f + step_size*dKi_f
                f_trial = np.dot(K, Ki_f_trial)
                return -obj(Ki_f_trial, f_trial)

            #use scipy for the line search, the compute new values of f, Ki_f
            step = optimize.brent(inner_obj, tol=1e-4, maxiter=12)
            Ki_f_new = Ki_f + step*dKi_f
            f_new = np.dot(K, Ki_f_new)
            #print "new {} vs old {}".format(obj(Ki_f_new, f_new), obj(Ki_f, f))
            old_obj = obj(Ki_f, f)
            new_obj = obj(Ki_f_new, f_new)
            if new_obj < old_obj:
                raise ValueError("Shouldn't happen, brent optimization failing")
            difference = np.abs(new_obj - old_obj)
            # difference = np.abs(np.sum(f_new - f)) + np.abs(np.sum(Ki_f_new - Ki_f))
            Ki_f = Ki_f_new
            f = f_new
            iteration += 1

        #Warn of bad fits
        if difference > self._mode_finding_tolerance:
            if not self.bad_fhat:
                warnings.warn("Not perfect mode found (f_hat). difference: {}, iteration: {} out of max {}".format(difference, iteration, self._mode_finding_max_iter))
            self.bad_fhat = True
        elif self.bad_fhat:
            self.bad_fhat = False
            warnings.warn("f_hat now fine again. difference: {}, iteration: {} out of max {}".format(difference, iteration, self._mode_finding_max_iter))

        return f, Ki_f

    def mode_computations(self, f_hat, Ki_f, K, Y, likelihood, kern, Y_metadata):
        """
        At the mode, compute the hessian and effective covariance matrix.

        returns: logZ : approximation to the marginal likelihood
                 woodbury_inv : variable required for calculating the approximation to the covariance matrix
                 dL_dthetaL : array of derivatives (1 x num_kernel_params)
                 dL_dthetaL : array of derivatives (1 x num_likelihood_params)
        """
        #At this point get the hessian matrix (or vector as W is diagonal)
        W = -likelihood.d2logpdf_df2(f_hat, Y, Y_metadata=Y_metadata)
        if np.any(np.isnan(W)):
            raise ValueError('One or more element(s) of W is NaN')

        K_Wi_i, logdet_I_KW, I_KW_i, Ki_W_i = self._compute_B_statistics(K, W, likelihood.log_concave)

        #compute the log marginal
        log_marginal = -0.5*np.sum(np.dot(Ki_f.T, f_hat)) + np.sum(likelihood.logpdf(f_hat, Y, Y_metadata=Y_metadata)) - 0.5*logdet_I_KW

        # Compute matrices for derivatives
        dW_df = -likelihood.d3logpdf_df3(f_hat, Y, Y_metadata=Y_metadata) # -d3lik_d3fhat
        if np.any(np.isnan(dW_df)):
            raise ValueError('One or more element(s) of dW_df is NaN')

        dL_dfhat = -0.5*(np.diag(Ki_W_i)[:, None]*dW_df) # s2 in R&W p126 line 9.
        #BiK, _ = dpotrs(L, K, lower=1)
        #dL_dfhat = 0.5*np.diag(BiK)[:, None]*dW_df
        I_KW_i = np.eye(Y.shape[0]) - np.dot(K, K_Wi_i)

        ####################
        #  compute dL_dK   #
        ####################
        if kern.size > 0 and not kern.is_fixed:
            #Explicit
            explicit_part = 0.5*(np.dot(Ki_f, Ki_f.T) - K_Wi_i)

            #Implicit
            implicit_part = np.dot(Ki_f, dL_dfhat.T).dot(I_KW_i)

            dL_dK = explicit_part + implicit_part
        else:
            dL_dK = np.zeros(likelihood.size)

        ####################
        #compute dL_dthetaL#
        ####################
        if likelihood.size > 0 and not likelihood.is_fixed:
            dlik_dthetaL, dlik_grad_dthetaL, dlik_hess_dthetaL = likelihood._laplace_gradients(f_hat, Y, Y_metadata=Y_metadata)

            num_params = likelihood.size
            # make space for one derivative for each likelihood parameter
            dL_dthetaL = np.zeros(num_params)
            for thetaL_i in range(num_params):
                #Explicit
                dL_dthetaL_exp = ( np.sum(dlik_dthetaL[thetaL_i,:, :])
                                # The + comes from the fact that dlik_hess_dthetaL == -dW_dthetaL
                                  + 0.5*np.sum(np.diag(Ki_W_i)*np.squeeze(dlik_hess_dthetaL[thetaL_i, :, :]))
                                )

                #Implicit
                dfhat_dthetaL = mdot(I_KW_i, K, dlik_grad_dthetaL[thetaL_i, :, :])
                #dfhat_dthetaL = mdot(Ki_W_i, dlik_grad_dthetaL[thetaL_i, :, :])
                dL_dthetaL_imp = np.dot(dL_dfhat.T, dfhat_dthetaL)
                dL_dthetaL[thetaL_i] = np.sum(dL_dthetaL_exp + dL_dthetaL_imp)

        else:
            dL_dthetaL = np.zeros(likelihood.size)

        #Cache some things for speedy LOO
        self.Ki_W_i = Ki_W_i
        self.K = K
        self.W = W
        self.f_hat = f_hat
        return log_marginal, K_Wi_i, dL_dK, dL_dthetaL

    def _compute_B_statistics(self, K, W, log_concave, *args, **kwargs):
        """
        Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal elements and can be easily inverted

        :param K: Prior Covariance matrix evaluated at locations X
        :type K: NxN matrix
        :param W: Negative hessian at a point (diagonal matrix)
        :type W: Vector of diagonal values of Hessian (1xN)
        :returns: (W12BiW12, L_B, Li_W12)
        """
        if not log_concave:
            #print "Under 1e-10: {}".format(np.sum(W < 1e-6))
            W = np.clip(W, 1e-6, 1e+30)
            # For student-T we can clip this more intelligently. If the
            # objective has hardly changed, we can increase the clipping limit
            # by ((v+1)/v)/sigma2
            # NOTE: when setting a parameter inside parameters_changed it will allways come to closed update circles!!!
            #W.__setitem__(W < 1e-6, 1e-6, update=False)  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                # To cause the posterior to become less certain than the prior and likelihood,
                                # This is a property only held by non-log-concave likelihoods
        if np.any(np.isnan(W)):
            raise ValueError('One or more element(s) of W is NaN')
        #W is diagonal so its sqrt is just the sqrt of the diagonal elements
        W_12 = np.sqrt(W)
        B = np.eye(K.shape[0]) + W_12*K*W_12.T
        L = jitchol(B)

        LiW12, _ = dtrtrs(L, np.diagflat(W_12), lower=1, trans=0)
        K_Wi_i = np.dot(LiW12.T, LiW12) # R = W12BiW12, in R&W p 126, eq 5.25

        #here's a better way to compute the required matrix.
        # you could do the model finding witha backsub, instead of a dot...
        #L2 = L/W_12
        #K_Wi_i_2 , _= dpotri(L2)
        #symmetrify(K_Wi_i_2)

        #compute vital matrices
        C = np.dot(LiW12, K)
        Ki_W_i = K - C.T.dot(C)

        I_KW_i = np.eye(K.shape[0]) - np.dot(K, K_Wi_i)
        logdet_I_KW = 2*np.sum(np.log(np.diag(L)))

        return K_Wi_i, logdet_I_KW, I_KW_i, Ki_W_i

class LaplaceBlock(Laplace):
    def rasm_mode(self, K, Y, likelihood, Ki_f_init, Y_metadata=None, *args, **kwargs):
        Ki_f = Ki_f_init.copy()
        f = np.dot(K, Ki_f)

        #define the objective function (to be maximised)
        def obj(Ki_f, f):
            ll = -0.5*np.dot(Ki_f.T, f) + np.sum(likelihood.logpdf_sum(f, Y, Y_metadata=Y_metadata))
            if np.isnan(ll):
                return -np.inf
            else:
                return ll

        difference = np.inf
        iteration = 0

        I = np.eye(K.shape[0])
        while difference > self._mode_finding_tolerance and iteration < self._mode_finding_max_iter:
            W = -likelihood.d2logpdf_df2(f, Y, Y_metadata=Y_metadata)

            W[np.diag_indices_from(W)] = np.clip(np.diag(W), 1e-6, 1e+30)

            W_f = np.dot(W, f)
            grad = likelihood.dlogpdf_df(f, Y, Y_metadata=Y_metadata)

            b = W_f + grad # R+W p46 line 6.
            K_Wi_i, _, _, _ = self._compute_B_statistics(K, W, likelihood.log_concave, *args, **kwargs)

            #Work out the DIRECTION that we want to move in, but don't choose the stepsize yet
            #a = (I - (K+Wi)i*K)*b
            full_step_Ki_f = np.dot(I - np.dot(K_Wi_i, K), b)
            dKi_f = full_step_Ki_f - Ki_f

            #define an objective for the line search (minimize this one)
            def inner_obj(step_size):
                Ki_f_trial = Ki_f + step_size*dKi_f
                f_trial = np.dot(K, Ki_f_trial)
                return -obj(Ki_f_trial, f_trial)

            #use scipy for the line search, the compute new values of f, Ki_f
            step = optimize.brent(inner_obj, tol=1e-4, maxiter=12)

            Ki_f_new = Ki_f + step*dKi_f
            f_new = np.dot(K, Ki_f_new)

            difference = np.abs(np.sum(f_new - f)) + np.abs(np.sum(Ki_f_new - Ki_f))
            Ki_f = Ki_f_new
            f = f_new
            iteration += 1

        #Warn of bad fits
        if difference > self._mode_finding_tolerance:
            if not self.bad_fhat:
                warnings.warn("Not perfect f_hat fit difference: {}".format(difference))
            self._previous_Ki_fhat = np.zeros_like(Y)
            self.bad_fhat = True
        elif self.bad_fhat:
            self.bad_fhat = False
            warnings.warn("f_hat now fine again")
        if iteration > self._mode_finding_max_iter:
            warnings.warn("didn't find the best")

        return f, Ki_f

    def mode_computations(self, f_hat, Ki_f, K, Y, likelihood, kern, Y_metadata):
        #At this point get the hessian matrix (or vector as W is diagonal)
        W = -likelihood.d2logpdf_df2(f_hat, Y, Y_metadata=Y_metadata)

        W[np.diag_indices_from(W)] = np.clip(np.diag(W), 1e-6, 1e+30)

        K_Wi_i, log_B_det, I_KW_i, Ki_W_i = self._compute_B_statistics(K, W, likelihood.log_concave)

        #compute the log marginal
        #FIXME: The derterminant should be output_dim*0.5 I think, gradients may now no longer check
        log_marginal = -0.5*np.dot(f_hat.T, Ki_f) + np.sum(likelihood.logpdf_sum(f_hat, Y, Y_metadata=Y_metadata)) - 0.5*log_B_det

        #Compute vival matrices for derivatives
        dW_df = -likelihood.d3logpdf_df3(f_hat, Y, Y_metadata=Y_metadata) # -d3lik_d3fhat

        #dL_dfhat = np.zeros((f_hat.shape[0]))
        #for i in range(f_hat.shape[0]):
            #dL_dfhat[i] = -0.5*np.trace(np.dot(Ki_W_i, dW_df[:,:,i]))

        dL_dfhat = -0.5*np.einsum('ij,ijk->k', Ki_W_i, dW_df)

        woodbury_vector = likelihood.dlogpdf_df(f_hat, Y, Y_metadata=Y_metadata)

        ####################
        #compute dL_dK#
        ####################
        if kern.size > 0 and not kern.is_fixed:
            #Explicit
            explicit_part = 0.5*(np.dot(Ki_f, Ki_f.T) - K_Wi_i)

            #Implicit
            implicit_part = woodbury_vector.dot(dL_dfhat[None,:]).dot(I_KW_i)
            #implicit_part = Ki_f.dot(dL_dfhat[None,:]).dot(I_KW_i)

            dL_dK = explicit_part + implicit_part
        else:
            dL_dK = np.zeros_like(K)

        ####################
        #compute dL_dthetaL#
        ####################
        if likelihood.size > 0 and not likelihood.is_fixed:
            raise NotImplementedError
        else:
            dL_dthetaL = np.zeros(likelihood.size)

        #self.K_Wi_i = K_Wi_i
        #self.Ki_W_i = Ki_W_i
        #self.W = W
        #self.K = K
        #self.dL_dfhat = dL_dfhat
        #self.explicit_part = explicit_part
        #self.implicit_part = implicit_part
        return log_marginal, K_Wi_i, dL_dK, dL_dthetaL

    def _compute_B_statistics(self, K, W, log_concave, *args, **kwargs):
        """
        Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal element and can be easyily inverted

        :param K: Prior Covariance matrix evaluated at locations X
        :type K: NxN matrix
        :param W: Negative hessian at a point (diagonal matrix)
        :type W: Vector of diagonal values of hessian (1xN)
        :returns: (K_Wi_i, L_B, not_provided)
        """
        #w = GPy.util.diag.view(W)
        #W[:] = np.where(w<1e-6, 1e-6, w)

        #B = I + KW
        B = np.eye(K.shape[0]) + np.dot(K, W)
        #Bi, L, Li, logdetB = pdinv(B)
        Bi = np.linalg.inv(B)

        #K_Wi_i = np.eye(K.shape[0]) - mdot(W, Bi, K)
        K_Wi_i = np.dot(W, Bi)

        #self.K_Wi_i_brute = np.linalg.inv(K + np.linalg.inv(W))
        #self.B = B
        #self.Bi = Bi
        Ki_W_i = np.dot(Bi, K)

        sign, logdetB = np.linalg.slogdet(B)
        return K_Wi_i, sign*logdetB, Bi, Ki_W_i
