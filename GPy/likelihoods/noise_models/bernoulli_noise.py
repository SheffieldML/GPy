# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations
from noise_distributions import NoiseDistribution

class Bernoulli(NoiseDistribution):
    """
    Bernoulli likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

    .. Note::
        Y is expected to take values in {-1,1}
        Probit likelihood usually used
    """
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False):
        super(Bernoulli, self).__init__(gp_link,analytical_mean,analytical_variance)
        if isinstance(gp_link , (gp_transformations.Heaviside, gp_transformations.Probit)):
            self.log_concave = True

    def _preprocess_values(self,Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.

        ..Note:: Binary classification algorithm works better with classes {-1,1}
        """
        Y_prep = Y.copy()
        Y1 = Y[Y.flatten()==1].size
        Y2 = Y[Y.flatten()==0].size
        assert Y1 + Y2 == Y.size, 'Bernoulli likelihood is meant to be used only with outputs in {0,1}.'
        Y_prep[Y.flatten() == 0] = -1
        return Y_prep

    def _moments_match_analytical(self,data_i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        if data_i == 1:
            sign = 1.
        elif data_i == 0:
            sign = -1
        else:
            raise ValueError("bad value for Bernouilli observation (0,1)")
        if isinstance(self.gp_link,gp_transformations.Probit):
            z = sign*v_i/np.sqrt(tau_i**2 + tau_i)
            Z_hat = std_norm_cdf(z)
            phi = std_norm_pdf(z)
            mu_hat = v_i/tau_i + sign*phi/(Z_hat*np.sqrt(tau_i**2 + tau_i))
            sigma2_hat = 1./tau_i - (phi/((tau_i**2+tau_i)*Z_hat))*(z+phi/Z_hat)

        elif isinstance(self.gp_link,gp_transformations.Heaviside):
            a = sign*v_i/np.sqrt(tau_i)
            Z_hat = std_norm_cdf(a)
            N = std_norm_pdf(a)
            mu_hat = v_i/tau_i + sign*N/Z_hat/np.sqrt(tau_i)
            sigma2_hat = (1. - a*N/Z_hat - np.square(N/Z_hat))/tau_i
            if np.any(np.isnan([Z_hat, mu_hat, sigma2_hat])):
                stop
        else:
            raise ValueError("Exact moment matching not available for link {}".format(self.gp_link.gp_transformations.__name__))

        return Z_hat, mu_hat, sigma2_hat

    def _predictive_mean_analytical(self,mu,variance):

        if isinstance(self.gp_link,gp_transformations.Probit):
            return stats.norm.cdf(mu/np.sqrt(1+variance))

        elif isinstance(self.gp_link,gp_transformations.Heaviside):
            return stats.norm.cdf(mu/np.sqrt(variance))

        else:
            raise NotImplementedError

    def _predictive_variance_analytical(self,mu,variance, pred_mean):

        if isinstance(self.gp_link,gp_transformations.Heaviside):
            return 0.
        else:
            raise NotImplementedError

    def pdf_link(self, link_f, y, extra_data=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data not used in bernoulli
        :returns: likelihood evaluated for this point
        :rtype: float

        .. Note:
            Each y_i must be in {0,1}
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        objective = (link_f**y) * ((1.-link_f)**(1.-y))
        return np.exp(np.sum(np.log(objective)))

    def logpdf_link(self, link_f, y, extra_data=None):
        """
        Log Likelihood function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = y_{i}\\log\\lambda(f_{i}) + (1-y_{i})\\log (1-f_{i})

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data not used in bernoulli
        :returns: log likelihood evaluated at points link(f)
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        #objective = y*np.log(link_f) + (1.-y)*np.log(link_f)
        objective = np.where(y==1, np.log(link_f), np.log(1-link_f))
        return np.sum(objective)

    def dlogpdf_dlink(self, link_f, y, extra_data=None):
        """
        Gradient of the pdf at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\frac{y_{i}}{\\lambda(f_{i})} - \\frac{(1 - y_{i})}{(1 - \\lambda(f_{i}))}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data not used in bernoulli
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        grad = (y/link_f) - (1.-y)/(1-link_f)
        return grad

    def d2logpdf_dlink2(self, link_f, y, extra_data=None):
        """
        Hessian at y, given link_f, w.r.t link_f the hessian will be 0 unless i == j
        i.e. second derivative logpdf at y given link(f_i) link(f_j)  w.r.t link(f_i) and link(f_j)


        .. math::
            \\frac{d^{2}\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)^{2}} = \\frac{-y_{i}}{\\lambda(f)^{2}} - \\frac{(1-y_{i})}{(1-\\lambda(f))^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data not used in bernoulli
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points link(f))
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        d2logpdf_dlink2 = -y/(link_f**2) - (1-y)/((1-link_f)**2)
        return d2logpdf_dlink2

    def d3logpdf_dlink3(self, link_f, y, extra_data=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{2y_{i}}{\\lambda(f)^{3}} - \\frac{2(1-y_{i}}{(1-\\lambda(f))^{3}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data not used in bernoulli
        :returns: third derivative of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        d3logpdf_dlink3 = 2*(y/(link_f**3) - (1-y)/((1-link_f)**3))
        return d3logpdf_dlink3

    def _mean(self,gp):
        """
        Mass (or density) function
        """
        return self.gp_link.transf(gp)

    def _variance(self,gp):
        """
        Mass (or density) function
        """
        p = self.gp_link.transf(gp)
        return p*(1.-p)

    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        ns = np.ones_like(gp, dtype=int)
        Ysim = np.random.binomial(ns, self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)
