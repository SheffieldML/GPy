# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#TODO
"""
A lot of this code assumes that the link function is the identity.

I think laplace code is okay, but I'm quite sure that the EP moments will only work if the link is identity.

Furthermore, exact Guassian inference can only be done for the identity link, so we should be asserting so for all calls which relate to that.

James 11/12/13
"""

import numpy as np
from scipy import stats, special
import link_functions
from likelihood import Likelihood
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from scipy import stats

class Gaussian(Likelihood):
    """
    Gaussian likelihood

    .. math::
        \\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2}

    :param variance: variance value of the Gaussian distribution
    :param N: Number of data points
    :type N: int
    """
    def __init__(self, gp_link=None, variance=1., name='Gaussian_noise'):
        if gp_link is None:
            gp_link = link_functions.Identity()

        assert isinstance(gp_link, link_functions.Identity), "the likelihood only implemented for the identity link"

        super(Gaussian, self).__init__(gp_link, name=name)

        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)

        if isinstance(gp_link, link_functions.Identity):
            self.log_concave = True

    def betaY(self,Y,Y_metadata=None):
        #TODO: ~Ricardo this does not live here
        return Y/self.gaussian_variance(Y_metadata)

    def gaussian_variance(self, Y_metadata=None):
        return self.variance

    def update_gradients(self, grad):
        self.variance.gradient = grad

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        return dL_dKdiag.sum()

    def _preprocess_values(self, Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.
        """
        return Y

    def _moments_match_ep(self, data_i, tau_i, v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        sigma2_hat = 1./(1./self.variance + tau_i)
        mu_hat = sigma2_hat*(data_i/self.variance + v_i)
        sum_var = self.variance + 1./tau_i
        Z_hat = 1./np.sqrt(2.*np.pi*sum_var)*np.exp(-.5*(data_i - v_i/tau_i)**2./sum_var)
        return Z_hat, mu_hat, sigma2_hat

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        if full_cov:
            if var.ndim == 2:
                var += np.eye(var.shape[0])*self.variance
            if var.ndim == 3:
                var += np.atleast_3d(np.eye(var.shape[0])*self.variance)
        else:
            var += self.variance
        return mu, var

    def predictive_mean(self, mu, sigma):
        return mu

    def predictive_variance(self, mu, sigma, predictive_mean=None):
        return self.variance + sigma**2

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        return  [stats.norm.ppf(q/100.)*np.sqrt(var + self.variance) + mu for q in quantiles]

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        #Assumes no covariance, exp, sum, log for numerical stability
        return np.exp(np.sum(np.log(stats.norm.pdf(y, link_f, np.sqrt(self.variance)))))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log likelihood function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: log likelihood evaluated for this point
        :rtype: float
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        N = y.shape[0]
        ln_det_cov = N*np.log(self.variance)

        return -0.5*(np.sum((y-link_f)**2/self.variance) + ln_det_cov + N*np.log(2.*np.pi))

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\frac{1}{\\sigma^{2}}(y_{i} - \\lambda(f_{i}))

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        s2_i = (1.0/self.variance)
        grad = s2_i*y - s2_i*link_f
        return grad

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link_f, w.r.t link_f.
        i.e. second derivative logpdf at y given link(f_i) link(f_j)  w.r.t link(f_i) and link(f_j)

        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{2}f} = -\\frac{1}{\\sigma^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points link(f))
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        N = y.shape[0]
        hess = -(1.0/self.variance)*np.ones((N, 1))
        return hess

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{3}\\lambda(f)} = 0

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: third derivative of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        N = y.shape[0]
        d3logpdf_dlink3 = np.zeros((N,1))
        return d3logpdf_dlink3

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given link(f), w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\sigma^{2}} = -\\frac{N}{2\\sigma^{2}} + \\frac{(y_{i} - \\lambda(f_{i}))^{2}}{2\\sigma^{4}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: float
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        e = y - link_f
        s_4 = 1.0/(self.variance**2)
        N = y.shape[0]
        dlik_dsigma = -0.5*N/self.variance + 0.5*s_4*np.sum(np.square(e))
        return np.sum(dlik_dsigma) # Sure about this sum?

    def dlogpdf_dlink_dvar(self, link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)}) = \\frac{1}{\\sigma^{4}}(-y_{i} + \\lambda(f_{i}))

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        s_4 = 1.0/(self.variance**2)
        dlik_grad_dsigma = -s_4*y + s_4*link_f
        return dlik_grad_dsigma

    def d2logpdf_dlink2_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d^{2} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{2}\\lambda(f)}) = \\frac{1}{\\sigma^{4}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log hessian evaluated at points link(f_i) and link(f_j) w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert np.asarray(link_f).shape == np.asarray(y).shape
        s_4 = 1.0/(self.variance**2)
        N = y.shape[0]
        d2logpdf_dlink2_dvar = np.ones((N,1))*s_4
        return d2logpdf_dlink2_dvar

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dvar = self.dlogpdf_link_dvar(f, y, Y_metadata=Y_metadata)
        return np.asarray([[dlogpdf_dvar]])

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dlink_dvar = self.dlogpdf_dlink_dvar(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dlink_dvar

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        d2logpdf_dlink2_dvar = self.d2logpdf_dlink2_dvar(f, y, Y_metadata=Y_metadata)
        return d2logpdf_dlink2_dvar

    def _mean(self, gp):
        """
        Expected value of y under the Mass (or density) function p(y|f)

        .. math::
            E_{p(y|f)}[y]
        """
        return self.gp_link.transf(gp)

    def _variance(self, gp):
        """
        Variance of y under the Mass (or density) function p(y|f)

        .. math::
            Var_{p(y|f)}[y]
        """
        return self.variance

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #orig_shape = gp.shape
        gp = gp.flatten()
        Ysim = np.array([np.random.normal(self.gp_link.transf(gpj), scale=np.sqrt(self.variance), size=1) for gpj in gp])
        return Ysim.reshape(orig_shape)

    def log_predictive_density(self, y_test, mu_star, var_star):
        """
        assumes independence
        """
        v = var_star + self.variance
        return -0.5*np.log(2*np.pi) -0.5*np.log(v) - 0.5*np.square(y_test - mu_star)/v

