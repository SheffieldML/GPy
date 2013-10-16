# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations
from noise_distributions import NoiseDistribution

class Gaussian(NoiseDistribution):
    """
    Gaussian likelihood

    :param mean: mean value of the Gaussian distribution
    :param variance: mean value of the Gaussian distribution
    """
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False,variance=1., D=None, N=None):
        self.variance = variance
        self.D = D
        self.N = N
        self._set_params(np.asarray(variance))
        super(Gaussian, self).__init__(gp_link,analytical_mean,analytical_variance)

    def _get_params(self):
        return np.array([self.variance])

    def _get_param_names(self):
        return ['noise_model_variance']

    def _set_params(self, p):
        self.variance = float(p)
        self.I = np.eye(self.N)
        self.covariance_matrix = self.I * self.variance
        self.Ki = self.I*(1.0 / self.variance)
        #self.ln_det_K = np.sum(np.log(np.diag(self.covariance_matrix)))
        self.ln_det_K = self.N*np.log(self.variance)

    def _gradients(self,partial):
        return np.zeros(1)
        #return np.sum(partial)

    def _preprocess_values(self,Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.
        """
        return Y

    def _moments_match_analytical(self,data_i,tau_i,v_i):
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

    def _predictive_mean_analytical(self,mu,sigma):
        new_sigma2 = self.predictive_variance(mu,sigma)
        return new_sigma2*(mu/sigma**2 + self.gp_link.transf(mu)/self.variance)

    def _predictive_variance_analytical(self,mu,sigma,predictive_mean=None):
        return 1./(1./self.variance + 1./sigma**2)

    def pdf_link(self, link_f, y, extra_data=None):
        #FIXME: Careful now passing link_f in not gp (f)!
        #return std_norm_pdf( (self.gp_link.transf(gp)-obs)/np.sqrt(self.variance) )
        #Assumes no covariance, exp, sum, log for numerical stability
        #return np.exp(np.sum(np.log(stats.norm.pdf(obs,self.gp_link.transf(gp),np.sqrt(self.variance)))))
        #return np.exp(np.sum(np.log(stats.norm.pdf(y, link_f, np.sqrt(self.variance)))))
        return np.exp(np.sum(np.log(stats.norm.pdf(y, link_f, np.sqrt(self.variance)))))

    def _mass(self, link_f, y, extra_data=None):
        NotImplementedError("Deprecated, now doing chain in noise_model.py for link function evaluation\
                            Please negate your function and use pdf in noise_model.py, if implementing a likelihood\
                            rederivate the derivative without doing the chain and put in logpdf, dlogpdf_dlink or\
                            its derivatives")
    def _nlog_mass(self, link_f, y, extra_data=None):
        NotImplementedError("Deprecated, now doing chain in noise_model.py for link function evaluation\
                            Please negate your function and use logpdf in noise_model.py, if implementing a likelihood\
                            rederivate the derivative without doing the chain and put in logpdf, dlogpdf_dlink or\
                            its derivatives")

    def _dnlog_mass_dgp(self, link_f, y, extra_data=None):
        NotImplementedError("Deprecated, now doing chain in noise_model.py for link function evaluation\
                            Please negate your function and use dlogpdf_df in noise_model.py, if implementing a likelihood\
                            rederivate the derivative without doing the chain and put in logpdf, dlogpdf_dlink or\
                            its derivatives")

    def _d2nlog_mass_dgp2(self, link_f, y, extra_data=None):
        NotImplementedError("Deprecated, now doing chain in noise_model.py for link function evaluation\
                            Please negate your function and use d2logpdf_df2 in noise_model.py, if implementing a likelihood\
                            rederivate the derivative without doing the chain and put in logpdf, dlogpdf_dlink or\
                            its derivatives")

    def logpdf_link(self, link_f, y, extra_data=None):
        """
        Log likelihood function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert link_f.shape == y.shape
        return -0.5*(np.sum((y-link_f)**2/self.variance) + self.ln_det_K + self.N*np.log(2.*np.pi))

    def dlogpdf_dlink(self, link_f, y, extra_data=None):
        """
        Gradient of the pdf at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|f_{i})}{df} = \\frac{1}{\\sigma^{2}}(y_{i} - f_{i})

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: gradient of negative likelihood evaluated at points
        :rtype: Nx1 array
        """
        assert link_f.shape == y.shape
        s2_i = (1.0/self.variance)
        grad = s2_i*y - s2_i*link_f
        return grad

    def d2logpdf_dlink2(self, link_f, y, extra_data=None):
        """
        Hessian at y, given link_f, w.r.t link_f the hessian will be 0 unless i == j
        i.e. second derivative _nlog_mass at y given f_{i} f_{j}  w.r.t f_{i} and f_{j}

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|f_{i})}{d^{2}f} = -\\frac{1}{\\sigma^{2}}

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
        assert link_f.shape == y.shape
        hess = -(1.0/self.variance)*np.ones((self.N, 1))
        return hess

    def d3logpdf_dlink3(self, link_f, y, extra_data=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|f_{i})}{d^{3}f} = 0

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        assert link_f.shape == y.shape
        d3logpdf_dlink3 = np.diagonal(0*self.I)[:, None] # FIXME: CAREFUL THIS MAY NOT WORK WITH MULTIDIMENSIONS?
        return d3logpdf_dlink3

    def dlogpdf_link_dvar(self, link_f, y, extra_data=None):
        """
        Gradient of the negative log-likelihood function at y given link(f), w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d \\ln p(y_{i}|f_{i})}{d\\sigma^{2}} = \\frac{N}{2\\sigma^{2}} + \\frac{(y_{i} - f_{i})^{2}}{2\\sigma^{4}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: float
        """
        assert link_f.shape == y.shape
        e = y - link_f
        s_4 = 1.0/(self.variance**2)
        dlik_dsigma = -0.5*self.N/self.variance + 0.5*s_4*np.dot(e.T, e)
        return np.sum(dlik_dsigma) # Sure about this sum?

    def dlogpdf_dlink_dvar(self, link_f, y, extra_data=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d \\ln p(y_{i}|f_{i})}{df}) = \\frac{1}{\\sigma^{4}}(-y_{i} + f_{i})

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert link_f.shape == y.shape
        s_4 = 1.0/(self.variance**2)
        dlik_grad_dsigma = -np.dot(s_4*self.I, y) + np.dot(s_4*self.I, link_f)
        return dlik_grad_dsigma

    def d2logpdf_dlink2_dvar(self, link_f, y, extra_data=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d^{2} \\ln p(y_{i}|f_{i})}{d^{2}f}) = \\frac{1}{\\sigma^{4}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of hessian evaluated at points f and f_j w.r.t variance parameter
        :rtype: Nx1 array
        """
        assert link_f.shape == y.shape
        s_4 = 1.0/(self.variance**2)
        d2logpdf_dlink2_dvar = np.diag(s_4*self.I)[:, None]
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

    def _mean(self,gp):
        """
        Expected value of y under the Mass (or density) function p(y|f)

        .. math::
            E_{p(y|f)}[y]
        """
        return self.gp_link.transf(gp)

    def _dmean_dgp(self,gp):
        return self.gp_link.dtransf_df(gp)

    def _d2mean_dgp2(self,gp):
        return self.gp_link.d2transf_df2(gp)

    def _variance(self,gp):
        """
        Variance of y under the Mass (or density) function p(y|f)

        .. math::
            Var_{p(y|f)}[y]
        """
        return self.variance

    def _dvariance_dgp(self,gp):
        return 0

    def _d2variance_dgp2(self,gp):
        return 0
