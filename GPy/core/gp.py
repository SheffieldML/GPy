# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from .. import kern
from ..util.linalg import pdinv, mdot, tdot, dpotrs, dtrtrs
#from ..util.plot import gpplot,  Tango
from ..likelihoods import EP
from gp_base import GPBase

class GP(GPBase):
    """
    Gaussian Process model for regression and EP

    :param X: input observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :parm likelihood: a GPy likelihood
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :rtype: model object
    :param epsilon_ep: convergence criterion for the Expectation Propagation algorithm, defaults to 0.1
    :param powerep: power-EP parameters [$\eta$,$\delta$], defaults to [1.,1.]
    :type powerep: list

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """
    def __init__(self, X, likelihood, kernel, normalize_X=False):
        GPBase.__init__(self, X, likelihood, kernel, normalize_X=normalize_X)
        self._set_params(self._get_params())

    def _set_params(self, p):
        self.kern._set_params_transformed(p[:self.kern.num_params_transformed()])
        self.likelihood._set_params(p[self.kern.num_params_transformed():])

        self.K = self.kern.K(self.X)
        self.K += self.likelihood.covariance_matrix

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        # the gradient of the likelihood wrt the covariance matrix
        if self.likelihood.YYT is None:
            #alpha = np.dot(self.Ki, self.likelihood.Y)
            alpha,_ = dpotrs(self.L, self.likelihood.Y,lower=1)

            self.dL_dK = 0.5 * (tdot(alpha) - self.output_dim * self.Ki)
        else:
            #tmp = mdot(self.Ki, self.likelihood.YYT, self.Ki)
            tmp, _ = dpotrs(self.L, np.asfortranarray(self.likelihood.YYT), lower=1)
            tmp, _ = dpotrs(self.L, np.asfortranarray(tmp.T), lower=1)
            self.dL_dK = 0.5 * (tmp - self.output_dim * self.Ki)

    def _get_params(self):
        return np.hstack((self.kern._get_params_transformed(), self.likelihood._get_params()))


    def _get_param_names(self):
        return self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian likelihood, no iteration is required:
        this function does nothing
        """
        self.likelihood.restart()
        self.likelihood.fit_full(self.kern.K(self.X))
        self._set_params(self._get_params())  # update the GP

    def _model_fit_term(self):
        """
        Computes the model fit using YYT if it's available
        """
        if self.likelihood.YYT is None:
            tmp, _ = dtrtrs(self.L, np.asfortranarray(self.likelihood.Y), lower=1)
            return -0.5 * np.sum(np.square(tmp))
            #return -0.5 * np.sum(np.square(np.dot(self.Li, self.likelihood.Y)))
        else:
            return -0.5 * np.sum(np.multiply(self.Ki, self.likelihood.YYT))

    def log_likelihood(self):
        """
        The log marginal likelihood of the GP.

        For an EP model,  can be written as the log likelihood of a regression
        model for a new variable Y* = v_tilde/tau_tilde, with a covariance
        matrix K* = K + diag(1./tau_tilde) plus a normalization term.
        """
        return -0.5 * self.output_dim * self.K_logdet + self._model_fit_term() + self.likelihood.Z


    def _log_likelihood_gradients(self):
        """
        The gradient of all parameters.

        Note, we use the chain rule: dL_dtheta = dL_dK * d_K_dtheta
        """
        return np.hstack((self.kern.dK_dtheta(dL_dK=self.dL_dK, X=self.X), self.likelihood._gradients(partial=np.diag(self.dL_dK))))

    def _raw_predict(self, _Xnew, which_parts='all', full_cov=False,stop=False):
        """
        Internal helper function for making predictions, does not account
        for normalization or likelihood
        """
        Kx = self.kern.K(_Xnew,self.X,which_parts=which_parts).T
        #KiKx = np.dot(self.Ki, Kx)
        KiKx, _ = dpotrs(self.L, np.asfortranarray(Kx), lower=1)
        mu = np.dot(KiKx.T, self.likelihood.Y)
        if full_cov:
            Kxx = self.kern.K(_Xnew, which_parts=which_parts)
            var = Kxx - np.dot(KiKx.T, Kx)
        else:
            Kxx = self.kern.Kdiag(_Xnew, which_parts=which_parts)
            var = Kxx - np.sum(np.multiply(KiKx, Kx), 0)
            var = var[:, None]
        if stop:
            debug_this # @UndefinedVariable
        return mu, var

    def predict(self, Xnew, which_parts='all', full_cov=False):
        """
        Predict the function(s) at the new point(s) Xnew.
        Arguments
        ---------
        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the folll covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.input_dim
        :rtype: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :rtype: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.input_dim


           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        # normalize X values
        Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        mu, var = self._raw_predict(Xnew, full_cov=full_cov, which_parts=which_parts)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov)

        return mean, var, _025pm, _975pm
