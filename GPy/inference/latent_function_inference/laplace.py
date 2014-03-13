# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
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
from ...util.misc import param_to_array
from posterior import Posterior
import warnings
from scipy import optimize

class Laplace(object):

    def __init__(self):
        """
        Laplace Approximation

        Find the moments \hat{f} and the hessian at this point
        (using Newton-Raphson) of the unnormalised posterior

        """

        self._mode_finding_tolerance = 1e-7
        self._mode_finding_max_iter = 40
        self.bad_fhat = True
        self._previous_Ki_fhat = None

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        #make Y a normal array!
        Y = param_to_array(Y)

        # Compute K
        K = kern.K(X)

        #Find mode
        if self.bad_fhat:
            Ki_f_init = np.zeros_like(Y)
        else:
            Ki_f_init = self._previous_Ki_fhat

        f_hat, Ki_fhat = self.rasm_mode(K, Y, likelihood, Ki_f_init, Y_metadata=Y_metadata)

        self.f_hat = f_hat
        #Compute hessian and other variables at mode
        log_marginal, woodbury_vector, woodbury_inv, dL_dK, dL_dthetaL = self.mode_computations(f_hat, Ki_fhat, K, Y, likelihood, kern, Y_metadata)

        self._previous_Ki_fhat = Ki_fhat.copy()
        return Posterior(woodbury_vector=woodbury_vector, woodbury_inv=woodbury_inv, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}

    def rasm_mode(self, K, Y, likelihood, Ki_f_init, Y_metadata=None):
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
            return -0.5*np.dot(Ki_f.flatten(), f.flatten()) + likelihood.logpdf(f, Y, extra_data=Y_metadata)

        difference = np.inf
        iteration = 0
        while difference > self._mode_finding_tolerance and iteration < self._mode_finding_max_iter:
            W = -likelihood.d2logpdf_df2(f, Y, extra_data=Y_metadata)
            grad = likelihood.dlogpdf_df(f, Y, extra_data=Y_metadata)

            W_f = W*f

            b = W_f + grad # R+W p46 line 6.
            W12BiW12, _, _ = self._compute_B_statistics(K, W, likelihood.log_concave)
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

            difference = np.abs(np.sum(f_new - f)) + np.abs(np.sum(Ki_f_new - Ki_f))
            Ki_f = Ki_f_new
            f = f_new
            iteration += 1

        #Warn of bad fits
        if difference > self._mode_finding_tolerance:
            if not self.bad_fhat:
                warnings.warn("Not perfect f_hat fit difference: {}".format(difference))
            self.bad_fhat = True
        elif self.bad_fhat:
            self.bad_fhat = False
            warnings.warn("f_hat now fine again")

        return f, Ki_f

    def mode_computations(self, f_hat, Ki_f, K, Y, likelihood, kern, Y_metadata):
        """
        At the mode, compute the hessian and effective covariance matrix.

        returns: logZ : approximation to the marginal likelihood
                 woodbury_vector : variable required for calculating the approximation to the covariance matrix
                 woodbury_inv : variable required for calculating the approximation to the covariance matrix
                 dL_dthetaL : array of derivatives (1 x num_kernel_params)
                 dL_dthetaL : array of derivatives (1 x num_likelihood_params)
        """
        #At this point get the hessian matrix (or vector as W is diagonal)
        W = -likelihood.d2logpdf_df2(f_hat, Y, extra_data=Y_metadata)

        K_Wi_i, L, LiW12 = self._compute_B_statistics(K, W, likelihood.log_concave)

        #compute vital matrices
        C = np.dot(LiW12, K)
        Ki_W_i  = K - C.T.dot(C) #Could this be wrong?

        #compute the log marginal
        log_marginal = -0.5*np.dot(Ki_f.flatten(), f_hat.flatten()) + likelihood.logpdf(f_hat, Y, extra_data=Y_metadata) - np.sum(np.log(np.diag(L)))

        #Compute vival matrices for derivatives
        dW_df = -likelihood.d3logpdf_df3(f_hat, Y, extra_data=Y_metadata) # -d3lik_d3fhat
        woodbury_vector = likelihood.dlogpdf_df(f_hat, Y, extra_data=Y_metadata)
        dL_dfhat = -0.5*(np.diag(Ki_W_i)[:, None]*dW_df) #why isn't this -0.5? s2 in R&W p126 line 9.
        #BiK, _ = dpotrs(L, K, lower=1)
        #dL_dfhat = 0.5*np.diag(BiK)[:, None]*dW_df
        I_KW_i = np.eye(Y.shape[0]) - np.dot(K, K_Wi_i)

        ####################
        #compute dL_dK#
        ####################
        if kern.size > 0 and not kern.is_fixed:
            #Explicit
            explicit_part = 0.5*(np.dot(Ki_f, Ki_f.T) - K_Wi_i)

            #Implicit
            implicit_part = np.dot(woodbury_vector, dL_dfhat.T).dot(I_KW_i)

            dL_dK = explicit_part + implicit_part
        else:
            dL_dK = np.zeros(likelihood.size)

        ####################
        #compute dL_dthetaL#
        ####################
        if likelihood.size > 0 and not likelihood.is_fixed:
            dlik_dthetaL, dlik_grad_dthetaL, dlik_hess_dthetaL = likelihood._laplace_gradients(f_hat, Y, extra_data=Y_metadata)

            num_params = likelihood.size
            # make space for one derivative for each likelihood parameter
            dL_dthetaL = np.zeros(num_params)
            for thetaL_i in range(num_params):
                #Explicit
                dL_dthetaL_exp = ( np.sum(dlik_dthetaL[thetaL_i])
                                # The + comes from the fact that dlik_hess_dthetaL == -dW_dthetaL
                                + 0.5*np.sum(np.diag(Ki_W_i).flatten()*dlik_hess_dthetaL[:, thetaL_i].flatten())
                                )

                #Implicit
                dfhat_dthetaL = mdot(I_KW_i, K, dlik_grad_dthetaL[:, thetaL_i])
                #dfhat_dthetaL = mdot(Ki_W_i, dlik_grad_dthetaL[:, thetaL_i])
                dL_dthetaL_imp = np.dot(dL_dfhat.T, dfhat_dthetaL)
                dL_dthetaL[thetaL_i] = dL_dthetaL_exp + dL_dthetaL_imp

        else:
            dL_dthetaL = np.zeros(likelihood.size)

        return log_marginal, woodbury_vector, K_Wi_i, dL_dK, dL_dthetaL

    def _compute_B_statistics(self, K, W, log_concave):
        """
        Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal element and can be easyily inverted

        :param K: Prior Covariance matrix evaluated at locations X
        :type K: NxN matrix
        :param W: Negative hessian at a point (diagonal matrix)
        :type W: Vector of diagonal values of hessian (1xN)
        :returns: (W12BiW12, L_B, Li_W12)
        """
        if not log_concave:
            #print "Under 1e-10: {}".format(np.sum(W < 1e-6))
            W[W<1e-6] = 1e-6
            # NOTE: when setting a parameter inside parameters_changed it will allways come to closed update circles!!!
            #W.__setitem__(W < 1e-6, 1e-6, update=False)  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                # To cause the posterior to become less certain than the prior and likelihood,
                                # This is a property only held by non-log-concave likelihoods

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

        return K_Wi_i, L, LiW12

