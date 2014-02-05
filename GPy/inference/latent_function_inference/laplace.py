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
from ...util.linalg import mdot, jitchol, pddet, dpotrs
from functools import partial as partial_func
from posterior import Posterior
import warnings
from scipy import optimize

class LaplaceInference(object):

    def __init__(self):
        """
        Laplace Approximation

        Find the moments \hat{f} and the hessian at this point
        (using Newton-Raphson) of the unnormalised posterior

        """
        self.NORMAL_CONST = (0.5 * np.log(2 * np.pi))

        self._mode_finding_tolerance = 1e-7
        self._mode_finding_max_iter = 40
        self.bad_fhat = True


    def inference(self, kern, X, likelihood, Y, Y_metadata=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        # Compute K
        K = kern.K(X)

        #Find mode
        if self.bad_fhat:
            Ki_f_init = np.random.randn(*Y.shape)/50
        else:
            Ki_f_init = self._previous_Ki_fhat
        self.f_hat, self._previous_Ki_fhat = self.rasm_mode(K, Y, likelihood, Ki_f_init, Y_metadata=Y_metadata)

        stop

        #Compute hessian and other variables at mode
        self._compute_likelihood_variables()

        likelihood.gradient = self.likelihood_gradients()
        dL_dK = self._Kgradients()
        kern.update_gradients_full(dL_dK)

        return Posterior(mean=self.f_hat, cov=self.Sigma, K=self.K), log_marginal_approx, {'dL_dK':dL_dK}

    def _shared_gradients_components(self):
        """
        A helper function to compute some common quantities
        """
        d3lik_d3fhat = likelihood.d3logpdf_df3(self.f_hat, self.data, extra_data=self.extra_data)
        dL_dfhat = 0.5*(np.diag(self.Ki_W_i)[:, None]*d3lik_d3fhat).T #why isn't this -0.5?
        I_KW_i = np.eye(self.N) - np.dot(self.K, self.Wi_K_i)
        return dL_dfhat, I_KW_i

    def _Kgradients(self):
        """
        Gradients with respect to prior kernel parameters dL_dK to be chained
        with dK_dthetaK to give dL_dthetaK
        :returns: dL_dK matrix
        :rtype: Matrix (1 x num_kernel_params)
        """
        dL_dfhat, I_KW_i = self._shared_gradients_components()
        dlp = likelihood.dlogpdf_df(self.f_hat, Y, extra_data=None) # TODO: how will extra data work?

        #Explicit
        expl_a = np.dot(self.Ki_f, self.Ki_f.T)
        expl_b = self.Wi_K_i
        expl = 0.5*expl_a - 0.5*expl_b
        dL_dthetaK_exp = dK_dthetaK(expl, X)

        #Implicit
        impl = mdot(dlp, dL_dfhat, I_KW_i)

        dL_dK = expl + impl

        return dL_dK

    def likelihood_gradients(self):
        """
        Gradients with respect to likelihood parameters (dL_dthetaL)

        :rtype: array of derivatives (1 x num_likelihood_params)
        """
        dL_dfhat, I_KW_i = self._shared_gradients_components()
        dlik_dthetaL, dlik_grad_dthetaL, dlik_hess_dthetaL = likelihood._laplace_gradients(self.f_hat, self.data, extra_data=self.extra_data)

        num_params = len(self._get_param_names())
        # make space for one derivative for each likelihood parameter
        dL_dthetaL = np.zeros(num_params)
        for thetaL_i in range(num_params):
            #Explicit
            dL_dthetaL_exp = ( np.sum(dlik_dthetaL[:, thetaL_i])
                             #- 0.5*np.trace(mdot(self.Ki_W_i, (self.K, np.diagflat(dlik_hess_dthetaL[thetaL_i]))))
                             + np.dot(0.5*np.diag(self.Ki_W_i)[:,None].T, dlik_hess_dthetaL[:, thetaL_i])
                             )

            #Implicit
            dfhat_dthetaL = mdot(I_KW_i, self.K, dlik_grad_dthetaL[:, thetaL_i])
            dL_dthetaL_imp = np.dot(dL_dfhat, dfhat_dthetaL)
            dL_dthetaL[thetaL_i] = dL_dthetaL_exp + dL_dthetaL_imp

        return dL_dthetaL

    def _compute_likelihood_variables(self):
        """
        At the mode, compute the hessian and effective covaraince matrix.
        """
        #At this point get the hessian matrix (or vector as W is diagonal)
        self.W = -likelihood.d2logpdf_df2(self.f_hat, self.data, extra_data=self.extra_data)

        if not likelihood.log_concave:
            #print "Under 1e-10: {}".format(np.sum(self.W < 1e-6))
            self.W[self.W < 1e-6] = 1e-6  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur

        self.W12BiW12, self.ln_B_det = self._compute_B_statistics(self.K, self.W, np.eye(self.N), likelihood.log_concave)

        self.Ki_f = self.Ki_f
        self.f_Ki_f = np.dot(self.f_hat.T, self.Ki_f)
        self.Ki_W_i = self.K - mdot(self.K, self.W12BiW12, self.K)

    def _compute_B_statistics(self, K, W, a, log_concave):
        """
        Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal element and can be easyily inverted

        :param K: Prior Covariance matrix evaluated at locations X
        :type K: NxN matrix
        :param W: Negative hessian at a point (diagonal matrix)
        :type W: Vector of diagonal values of hessian (1xN)
        :param a: Matrix to calculate W12BiW12a
        :type a: Matrix NxN
        :returns: (W12BiW12, ln_B_det)
        """
        if not log_concave:
            #print "Under 1e-10: {}".format(np.sum(W < 1e-6))
            W[W < 1e-6] = 1e-6  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                # To cause the posterior to become less certain than the prior and likelihood,
                                # This is a property only held by non-log-concave likelihoods


        #W is diagonal so its sqrt is just the sqrt of the diagonal elements
        W_12 = np.sqrt(W)
        B = np.eye(K.shape[0]) + W_12*K*W_12.T
        L = jitchol(B)

        W12BiW12a = W_12*dpotrs(L, np.asfortranarray(W_12*a), lower=1)[0]
        ln_B_det = 2.*np.sum(np.log(np.diag(L)))
        return W12BiW12a, ln_B_det

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

        ##Start f's at zero originally or if we have gone off track, try restarting
        #if self.old_Ki_f is None or self.bad_fhat:
            #old_Ki_f = np.random.rand(self.N, 1)/50.0
            ##old_Ki_f = self.Y
            #f = np.dot(K, old_Ki_f)
        #else:
            ##Start at the old best point
            #old_Ki_f = self.old_Ki_f.copy()
            #f = self.f_hat.copy()

        Ki_f = Ki_f_init.copy()
        f = np.dot(K, Ki_f)


        #define the objective function (to be maximised)
        def obj(Ki_f, f):
            return -0.5*np.dot(Ki_f.T, f) + likelihood.logpdf(f, Y, extra_data=Y_metadata)

        difference = np.inf
        i = 0
        while difference > self._mode_finding_tolerance and i < self._mode_finding_max_iter:
            W = -likelihood.d2logpdf_df2(f, Y, extra_data=Y_metadata)

            W_f = W*f
            grad = likelihood.dlogpdf_df(f, Y, extra_data=Y_metadata)

            b = W_f + grad
            W12BiW12Kb, _ = self._compute_B_statistics(K, W.copy(), np.dot(K, b), likelihood.log_concave)

            #Work out the DIRECTION that we want to move in, but don't choose the stepsize yet
            full_step_Ki_f = b - W12BiW12Kb
            dKi_f = full_step_Ki_f - Ki_f

            #define an objective for the line search
            def inner_obj(step_size):
                Ki_f_trial = Ki_f + step_size*dKi_f
                f_trial = np.dot(K, Ki_f_trial)
                print -obj(Ki_f_trial, f_trial),
                return -obj(Ki_f_trial, f_trial)

            #use scipy for the line search, the compute new values of f, Ki_f
            step = optimize.brent(inner_obj, tol=1e-4, maxiter=12)
            Ki_f_new = Ki_f + step*dKi_f
            f_new = np.dot(K, Ki_f_new)

            print ""
            print obj(Ki_f, f), obj(Ki_f_new, f_new), step
            print ""

            #i_o = partial_func(inner_obj, old_Ki_f=old_Ki_f, dKi_f=dKi_f, K=K)
            #Find the stepsize that minimizes the objective function using a brent line search
            #The tolerance and maxiter matter for speed! Seems to be best to keep them low and make more full
            #steps than get this exact then make a step, if B was bigger it might be the other way around though
            #new_obj = sp.optimize.minimize_scalar(i_o, method='brent', tol=1e-4, options={'maxiter':5}).fun
            #new_obj = sp.optimize.brent(i_o, tol=1e-4, maxiter=10)
            #f = self.tmp_f.copy()
            #Ki_f = self.tmp_Ki_f.copy()

            difference = np.abs(np.sum(f_new - f)) + np.abs(np.sum(Ki_f_new - Ki_f))
            Ki_f = Ki_f_new
            f = f_new
            i += 1


        #Warn of bad fits
        if difference > self._mode_finding_tolerance:
            if not self.bad_fhat:
                warnings.warn("Not perfect f_hat fit difference: {}".format(difference))
            self.bad_fhat = True
        elif self.bad_fhat:
            self.bad_fhat = False
            warnings.warn("f_hat now fine again")

        return f, Ki_f

    def _compute_GP_variables(self):
        """
        Generate data Y which would give the normal distribution identical
        to the laplace approximation to the posterior, but normalised

        GPy expects a likelihood to be gaussian, so need to caluclate
        the data Y^{\tilde} that makes the posterior match that found
        by a laplace approximation to a non-gaussian likelihood but with
        a gaussian likelihood

        Firstly,
        The hessian of the unormalised posterior distribution is (K^{-1} + W)^{-1},
        i.e. z*N(f|f^{\hat}, (K^{-1} + W)^{-1}) but this assumes a non-gaussian likelihood,
        we wish to find the hessian \Sigma^{\tilde}
        that has the same curvature but using our new simulated data Y^{\tilde}
        i.e. we do N(Y^{\tilde}|f^{\hat}, \Sigma^{\tilde})N(f|0, K) = z*N(f|f^{\hat}, (K^{-1} + W)^{-1})
        and we wish to find what Y^{\tilde} and \Sigma^{\tilde}
        We find that Y^{\tilde} = W^{-1}(K^{-1} + W)f^{\hat} and \Sigma^{tilde} = W^{-1}

        Secondly,
        GPy optimizes the log marginal log p(y) = -0.5*ln|K+\Sigma^{\tilde}| - 0.5*Y^{\tilde}^{T}(K^{-1} + \Sigma^{tilde})^{-1}Y + lik.Z
        So we can suck up any differences between that and our log marginal likelihood approximation
        p^{\squiggle}(y) = -0.5*f^{\hat}K^{-1}f^{\hat} + log p(y|f^{\hat}) - 0.5*log |K||K^{-1} + W|
        which we want to optimize instead, by equating them and rearranging, the difference is added onto
        the log p(y) that GPy optimizes by default

        Thirdly,
        Since we have gradients that depend on how we move f^{\hat}, we have implicit components
        aswell as the explicit dL_dK, we hold these differences in dZ_dK and add them to dL_dK in the
        gp.py code
        """
        Wi = 1.0/self.W
        self.Sigma_tilde = np.diagflat(Wi)

        Y_tilde = Wi*self.Ki_f + self.f_hat

        self.Wi_K_i = self.W12BiW12
        ln_det_Wi_K = pddet(self.Sigma_tilde + self.K)
        lik = likelihood.logpdf(self.f_hat, self.data, extra_data=self.extra_data)
        y_Wi_K_i_y = mdot(Y_tilde.T, self.Wi_K_i, Y_tilde)

        Z_tilde = (+ lik
                   - 0.5*self.ln_B_det
                   + 0.5*ln_det_Wi_K
                   - 0.5*self.f_Ki_f
                   + 0.5*y_Wi_K_i_y
                   + self.NORMAL_CONST
                  )

        #Convert to float as its (1, 1) and Z must be a scalar
        self.Z = np.float64(Z_tilde)
        self.Y = Y_tilde
        self.YYT = np.dot(self.Y, self.Y.T)
        self.covariance_matrix = self.Sigma_tilde
        self.precision = 1.0 / np.diag(self.covariance_matrix)[:, None]

        #Compute dZ_dK which is how the approximated distributions gradients differ from the dL_dK computed for other likelihoods
        self.dZ_dK = self._Kgradients()
        #+ 0.5*self.Wi_K_i - 0.5*np.dot(self.Ki_f, self.Ki_f.T) #since we are not adding the K gradients explicit part theres no need to compute this again


