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
        \\ln p(y_{i}|f_{i}) = \\ln \\Gamma(\\frac{v+1}{2}) - \\ln \\Gamma(\\frac{v}{2})\\sqrt{v \\pi}\\sigma - \\frac{v+1}{2}\\ln (1 + \\frac{1}{v}\\left(\\frac{y_{i} - f_{i}}{\\sigma}\\right)^2)

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

    def logpdf_link(self, link_f, y, extra_data=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|f_{i}) = \\ln \\Gamma(\\frac{v+1}{2}) - \\ln \\Gamma(\\frac{v}{2})\\sqrt{v \\pi}\sigma - \\frac{v+1}{2}\\ln (1 + \\frac{1}{v}\\left(\\frac{y_{i} - f_{i}}{\\sigma}\\right)^2

        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        assert link_f.shape == y.shape
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
            \\frac{d \\ln p(y_{i}|f_{i})}{df} = \\frac{(v+1)(y_{i}-f_{i})}{(y_{i}-f_{i})^{2} + \\sigma^{2}v}

        :param link_f: latent variables (f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: gradient of likelihood evaluated at points
        :rtype: Nx1 array

        """
        assert y.shape == link_f.shape
        e = y - link_f
        grad = ((self.v + 1) * e) / (self.v * self.sigma2 + (e**2))
        return grad

    def d2logpdf_dlink2(self, link_f, y, extra_data=None):
        """
        Hessian at y, given link(f), w.r.t link(f) the hessian will be 0 unless i == j
        i.e. second derivative lik_function at y given f_{i} f_{j}  w.r.t f_{i} and f_{j}

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|f_{i})}{d^{2}f} = \\frac{(v+1)((y_{i}-f_{i})^{2} - \\sigma^{2}v)}{((y_{i}-f_{i})^{2} + \\sigma^{2}v)^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_{i} depends only on f_{i} not on f_{j!=i}
        """
        assert y.shape == link_f.shape
        e = y - link_f
        hess = ((self.v + 1)*(e**2 - self.v*self.sigma2)) / ((self.sigma2*self.v + e**2)**2)
        return hess

    def d3logpdf_dlink3(self, link_f, y, extra_data=None):
        """
        Third order derivative log-likelihood function at y given f w.r.t f

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|f_{i})}{d^{3}f} = \\frac{-2(v+1)((y_{i} - f_{i})^3 - 3(y_{i} - f_{i}) \\sigma^{2} v))}{((y_{i} - f_{i}) + \\sigma^{2} v)^3}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        assert y.shape == link_f.shape
        e = y - link_f
        d3lik_dlink3 = ( -(2*(self.v + 1)*(-e)*(e**2 - 3*self.v*self.sigma2)) /
                       ((e**2 + self.sigma2*self.v)**3)
                    )
        return d3lik_dlink3

    def dlogpdf_link_dvar(self, link_f, y, extra_data=None):
        """
        Gradient of the log-likelihood function at y given f, w.r.t variance parameter (t_noise)

        .. math::
            \\frac{d \\ln p(y_{i}|f_{i})}{d\\sigma^{2}} = \\frac{v((y_{i} - f_{i})^{2} - \\sigma^{2})}{2\\sigma^{2}(\\sigma^{2}v + (y_{i} - f_{i})^{2})}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: float
        """
        assert y.shape == link_f.shape
        e = y - link_f
        dlogpdf_dvar = self.v*(e**2 - self.sigma2)/(2*self.sigma2*(self.sigma2*self.v + e**2))
        return np.sum(dlogpdf_dvar)

    def dlogpdf_dlink_dvar(self, link_f, y, extra_data=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (t_noise)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d \\ln p(y_{i}|f_{i})}{df}) = \\frac{-2\\sigma v(v + 1)(y_{i}-f_{i})}{(y_{i}-f_{i})^2 + \\sigma^2 v)^2}

        :param link_f: latent variables link_f
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert y.shape == link_f.shape
        e = y - link_f
        dlogpdf_dlink_dvar = (self.v*(self.v+1)*(-e))/((self.sigma2*self.v + e**2)**2)
        return dlogpdf_dlink_dvar

    def d2logpdf_dlink2_dvar(self, link_f, y, extra_data=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (t_noise)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d^{2} \\ln p(y_{i}|f_{i})}{d^{2}f}) = \\frac{v(v+1)(\\sigma^{2}v - 3(y_{i} - f_{i})^{2})}{(\\sigma^{2}v + (y_{i} - f_{i})^{2})^{3}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of hessian evaluated at points f and f_j w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert y.shape == link_f.shape
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
        Compute  mean, and conficence interval (percentiles 5 and 95) of the prediction

        Need to find what the variance is at the latent points for a student t*normal p(y*|f*)p(f*)
        (((g((v+1)/2))/(g(v/2)*s*sqrt(v*pi)))*(1+(1/v)*((y-f)/s)^2)^(-(v+1)/2))
        *((1/(s*sqrt(2*pi)))*exp(-(1/(2*(s^2)))*((y-f)^2)))
        """

        #We want the variance around test points y which comes from int p(y*|f*)p(f*) df*
        #Var(y*) = Var(E[y*|f*]) + E[Var(y*|f*)]
        #Since we are given f* (mu) which is our mean (expected) value of y*|f* then the variance is the variance around this
        #Which was also given to us as (var)
        #We also need to know the expected variance of y* around samples f*, this is the variance of the student t distribution
        #However the variance of the student t distribution is not dependent on f, only on sigma and the degrees of freedom
        true_var = sigma**2 + self.variance

        return true_var

    def _predictive_mean_analytical(self, mu, var):
        """
        Compute mean of the prediction
        """
        return mu

    def sample_predicted_values(self, mu, var):
        """ Experimental sample approches and numerical integration """
        raise NotImplementedError
        #p_025 = stats.t.ppf(.025, mu)
        #p_975 = stats.t.ppf(.975, mu)

        num_test_points = mu.shape[0]
        #Each mu is the latent point f* at the test point x*,
        #and the var is the gaussian variance at this point
        #Take lots of samples from this, so we have lots of possible values
        #for latent point f* for each test point x* weighted by how likely we were to pick it
        print "Taking %d samples of f*".format(num_test_points)
        num_f_samples = 10
        num_y_samples = 10
        student_t_means = np.random.normal(loc=mu, scale=np.sqrt(var), size=(num_test_points, num_f_samples))
        print "Student t means shape: ", student_t_means.shape

        #Now we have lots of f*, lets work out the likelihood of getting this by sampling
        #from a student t centred on this point, sample many points from this distribution
        #centred on f*
        #for test_point, f in enumerate(student_t_means):
            #print test_point
            #print f.shape
            #student_t_samples = stats.t.rvs(self.v, loc=f[:,None],
                                            #scale=self.sigma,
                                            #size=(num_f_samples, num_y_samples))
            #print student_t_samples.shape

        student_t_samples = stats.t.rvs(self.v, loc=student_t_means[:, None],
                                        scale=self.sigma,
                                        size=(num_test_points, num_y_samples, num_f_samples))
        student_t_samples = np.reshape(student_t_samples,
                                       (num_test_points, num_y_samples*num_f_samples))

        #Now take the 97.5 and 0.25 percentile of these points
        p_025 = stats.scoreatpercentile(student_t_samples, .025, axis=1)[:, None]
        p_975 = stats.scoreatpercentile(student_t_samples, .975, axis=1)[:, None]

        ##Alernenately we could sample from int p(y|f*)p(f*|x*) df*
        def t_gaussian(f, mu, var):
            return (((gamma((self.v+1)*0.5)) / (gamma(self.v*0.5)*self.sigma*np.sqrt(self.v*np.pi))) * ((1+(1/self.v)*(((mu-f)/self.sigma)**2))**(-(self.v+1)*0.5))
                    * ((1/(np.sqrt(2*np.pi*var)))*np.exp(-(1/(2*var)) *((mu-f)**2)))
                    )

        def t_gauss_int(mu, var):
            print "Mu: ", mu
            print "var: ", var
            result = integrate.quad(t_gaussian, 0.025, 0.975, args=(mu, var))
            print "Result: ", result
            return result[0]

        vec_t_gauss_int = np.vectorize(t_gauss_int)

        p = vec_t_gauss_int(mu, var)
        p_025 = mu - p
        p_975 = mu + p
        return mu, np.nan*mu, p_025, p_975

