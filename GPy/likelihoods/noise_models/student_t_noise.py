# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
import scipy as sp
import gp_transformations
from noise_distributions import NoiseDistribution
from scipy import stats, integrate
from scipy.special import gammaln, gamma

class StudentT(NoiseDistribution):
    """
    Student T likelihood

    For nomanclature see Bayesian Data Analysis 2003 p576

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\Gamma\\left(\\frac{v+1}{2}\\right)}{\\Gamma\\left(\\frac{v}{2}\\right)\\sqrt{v\\pi\\sigma^{2}}}\\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - f_{i})^{2}}{\\sigma^{2}}\\right)\\right)^{\\frac{-v+1}{2}}

    """
    def __init__(self,gp_link=None,analytical_mean=True,analytical_variance=True, deg_free=5, sigma2=2):
        self.v = deg_free
        self.sigma2 = sigma2

        self._set_params(np.asarray(sigma2))
        super(StudentT, self).__init__(gp_link,analytical_mean,analytical_variance)
        self.log_concave = False

    def _get_params(self):
        return np.asarray(self.sigma2)

    def _get_param_names(self):
        return ["t_noise_std2"]

    def _set_params(self, x):
        self.sigma2 = float(x)

    @property
    def variance(self, extra_data=None):
        return (self.v / float(self.v - 2)) * self.sigma2

    def pdf_link(self, link_f, y, extra_data=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\Gamma\\left(\\frac{v+1}{2}\\right)}{\\Gamma\\left(\\frac{v}{2}\\right)\\sqrt{v\\pi\\sigma^{2}}}\\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - \\lambda(f_{i}))^{2}}{\\sigma^{2}}\\right)\\right)^{\\frac{-v+1}{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        #Careful gamma(big_number) is infinity!
        objective = ((np.exp(gammaln((self.v + 1)*0.5) - gammaln(self.v * 0.5))
                     / (np.sqrt(self.v * np.pi * self.sigma2)))
                     * ((1 + (1./float(self.v))*((e**2)/float(self.sigma2)))**(-0.5*(self.v + 1)))
                    )
        return np.prod(objective)

    def logpdf_link(self, link_f, y, extra_data=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|\lambda(f_{i})) = \\ln \\Gamma\\left(\\frac{v+1}{2}\\right) - \\ln \\Gamma\\left(\\frac{v}{2}\\right) - \\ln \\sqrt{v \\pi\\sigma^{2}} - \\frac{v+1}{2}\\ln \\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - \lambda(f_{i}))^{2}}{\\sigma^{2}}\\right)\\right)

        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        objective = (+ gammaln((self.v + 1) * 0.5)
                     - gammaln(self.v * 0.5)
                     - 0.5*np.log(self.sigma2 * self.v * np.pi)
                     - 0.5*(self.v + 1)*np.log(1 + (1/np.float(self.v))*((e**2)/self.sigma2))
                    )
        return np.sum(objective)

    def dlogpdf_dlink(self, link_f, y, extra_data=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{d\\lambda(f)} = \\frac{(v+1)(y_{i}-\lambda(f_{i}))}{(y_{i}-\lambda(f_{i}))^{2} + \\sigma^{2}v}

        :param link_f: latent variables (f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: gradient of likelihood evaluated at points
        :rtype: Nx1 array

        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        grad = ((self.v + 1) * e) / (self.v * self.sigma2 + (e**2))
        return grad

    def d2logpdf_dlink2(self, link_f, y, extra_data=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}\\lambda(f)} = \\frac{(v+1)((y_{i}-\lambda(f_{i}))^{2} - \\sigma^{2}v)}{((y_{i}-\lambda(f_{i}))^{2} + \\sigma^{2}v)^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        hess = ((self.v + 1)*(e**2 - self.v*self.sigma2)) / ((self.sigma2*self.v + e**2)**2)
        return hess

    def d3logpdf_dlink3(self, link_f, y, extra_data=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{-2(v+1)((y_{i} - \lambda(f_{i}))^3 - 3(y_{i} - \lambda(f_{i})) \\sigma^{2} v))}{((y_{i} - \lambda(f_{i})) + \\sigma^{2} v)^3}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        d3lik_dlink3 = ( -(2*(self.v + 1)*(-e)*(e**2 - 3*self.v*self.sigma2)) /
                       ((e**2 + self.sigma2*self.v)**3)
                    )
        return d3lik_dlink3

    def dlogpdf_link_dvar(self, link_f, y, extra_data=None):
        """
        Gradient of the log-likelihood function at y given f, w.r.t variance parameter (t_noise)

        .. math::
            \\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{d\\sigma^{2}} = \\frac{v((y_{i} - \lambda(f_{i}))^{2} - \\sigma^{2})}{2\\sigma^{2}(\\sigma^{2}v + (y_{i} - \lambda(f_{i}))^{2})}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        dlogpdf_dvar = self.v*(e**2 - self.sigma2)/(2*self.sigma2*(self.sigma2*self.v + e**2))
        return np.sum(dlogpdf_dvar)

    def dlogpdf_dlink_dvar(self, link_f, y, extra_data=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (t_noise)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{df}) = \\frac{-2\\sigma v(v + 1)(y_{i}-\lambda(f_{i}))}{(y_{i}-\lambda(f_{i}))^2 + \\sigma^2 v)^2}

        :param link_f: latent variables link_f
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        dlogpdf_dlink_dvar = (self.v*(self.v+1)*(-e))/((self.sigma2*self.v + e**2)**2)
        return dlogpdf_dlink_dvar

    def d2logpdf_dlink2_dvar(self, link_f, y, extra_data=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (t_noise)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}f}) = \\frac{v(v+1)(\\sigma^{2}v - 3(y_{i} - \lambda(f_{i}))^{2})}{(\\sigma^{2}v + (y_{i} - \lambda(f_{i}))^{2})^{3}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution
        :returns: derivative of hessian evaluated at points f and f_j w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        e = y - link_f
        d2logpdf_dlink2_dvar = ( (self.v*(self.v+1)*(self.sigma2*self.v - 3*(e**2)))
                              / ((self.sigma2*self.v + (e**2))**3)
                           )
        return d2logpdf_dlink2_dvar

    def dlogpdf_link_dtheta(self, f, y, extra_data=None):
        dlogpdf_dvar = self.dlogpdf_link_dvar(f, y, extra_data=extra_data)
        return np.asarray([[dlogpdf_dvar]])

    def dlogpdf_dlink_dtheta(self, f, y, extra_data=None):
        dlogpdf_dlink_dvar = self.dlogpdf_dlink_dvar(f, y, extra_data=extra_data)
        return dlogpdf_dlink_dvar

    def d2logpdf_dlink2_dtheta(self, f, y, extra_data=None):
        d2logpdf_dlink2_dvar = self.d2logpdf_dlink2_dvar(f, y, extra_data=extra_data)
        return d2logpdf_dlink2_dvar

    def _predictive_variance_analytical(self, mu, sigma, predictive_mean=None):
        """
        Compute predictive variance of student_t*normal p(y*|f*)p(f*)

        Need to find what the variance is at the latent points for a student t*normal p(y*|f*)p(f*)
        (((g((v+1)/2))/(g(v/2)*s*sqrt(v*pi)))*(1+(1/v)*((y-f)/s)^2)^(-(v+1)/2))
        *((1/(s*sqrt(2*pi)))*exp(-(1/(2*(s^2)))*((y-f)^2)))
        """

        #FIXME: Not correct
        #We want the variance around test points y which comes from int p(y*|f*)p(f*) df*
        #Var(y*) = Var(E[y*|f*]) + E[Var(y*|f*)]
        #Since we are given f* (mu) which is our mean (expected) value of y*|f* then the variance is the variance around this
        #Which was also given to us as (var)
        #We also need to know the expected variance of y* around samples f*, this is the variance of the student t distribution
        #However the variance of the student t distribution is not dependent on f, only on sigma and the degrees of freedom
        true_var = 1/(1/sigma**2 + 1/self.variance)

        return true_var

    def _predictive_mean_analytical(self, mu, sigma):
        """
        Compute mean of the prediction
        """
        #FIXME: Not correct
        return mu

    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #FIXME: Very slow as we are computing a new random variable per input!
        #Can't get it to sample all at the same time
        #student_t_samples = np.array([stats.t.rvs(self.v, self.gp_link.transf(gpj),scale=np.sqrt(self.sigma2), size=1) for gpj in gp])
        dfs = np.ones_like(gp)*self.v
        scales = np.ones_like(gp)*np.sqrt(self.sigma2)
        student_t_samples = stats.t.rvs(dfs, loc=self.gp_link.transf(gp),
                                        scale=scales)
        return student_t_samples.reshape(orig_shape)
