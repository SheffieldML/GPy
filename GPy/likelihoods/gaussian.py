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
from . import link_functions
from .likelihood import Likelihood
from ..core.parameterization import Param
from paramz.transformations import Logexp
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

        if not isinstance(gp_link, link_functions.Identity):
            print("Warning, Exact inference is not implemeted for non-identity link functions,\
            if you are not already, ensure Laplace inference_method is used")

        super(Gaussian, self).__init__(gp_link, name=name)

        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)

        if isinstance(gp_link, link_functions.Identity):
            self.log_concave = True

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Gaussian, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.Gaussian"
        input_dict["variance"] = self.variance.values.tolist()
        return input_dict


    def betaY(self,Y,Y_metadata=None):
        #TODO: ~Ricardo this does not live here
        raise RuntimeError("Please notify the GPy developers, this should not happen")
        return Y/self.gaussian_variance(Y_metadata)

    def gaussian_variance(self, Y_metadata=None):
        return self.variance

    def update_gradients(self, grad):
        self.variance.gradient = grad

    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gk', boost_grad=1.):
        return self.exact_inference_gradients(dL_dKdiag)

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata=None):
        return dL_dKdiag.sum()

    def _preprocess_values(self, Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.
        """
        return Y

    def moments_match_ep(self, data_i, tau_i, v_i, Y_metadata_i=None):
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
        ln_det_cov = np.log(self.variance)
        return -(1.0/(2*self.variance))*((y-link_f)**2) - 0.5*ln_det_cov - 0.5*np.log(2.*np.pi)

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
        s2_i = 1.0/self.variance
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
        N = y.shape[0]
        D = link_f.shape[1]
        hess = -(1.0/self.variance)*np.ones((N, D))
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
        N = y.shape[0]
        D = link_f.shape[1]
        d3logpdf_dlink3 = np.zeros((N,D))
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
        e = y - link_f
        s_4 = 1.0/(self.variance**2)
        dlik_dsigma = -0.5/self.variance + 0.5*s_4*np.square(e)
        return dlik_dsigma

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
        s_4 = 1.0/(self.variance**2)
        N = y.shape[0]
        D = link_f.shape[1]
        d2logpdf_dlink2_dvar = np.ones((N, D))*s_4
        return d2logpdf_dlink2_dvar

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0,:,:] = self.dlogpdf_link_dvar(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0, :, :]= self.dlogpdf_dlink_dvar(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        d2logpdf_dlink2_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink2_dtheta[0, :, :] = self.d2logpdf_dlink2_dvar(f, y, Y_metadata=Y_metadata)
        return d2logpdf_dlink2_dtheta

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

    def log_predictive_density(self, y_test, mu_star, var_star, Y_metadata=None):
        """
        assumes independence
        """
        v = var_star + self.variance
        return -0.5*np.log(2*np.pi) -0.5*np.log(v) - 0.5*np.square(y_test - mu_star)/v

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if not isinstance(self.gp_link, link_functions.Identity):
            return super(Gaussian, self).variational_expectations(Y=Y, m=m, v=v, gh_points=gh_points, Y_metadata=Y_metadata)

        lik_var = float(self.variance)
        F = -0.5*np.log(2*np.pi) -0.5*np.log(lik_var) - 0.5*(np.square(Y) + np.square(m) + v - 2*m*Y)/lik_var
        dF_dmu = (Y - m)/lik_var
        dF_dv = np.ones_like(v)*(-0.5/lik_var)
        dF_dtheta = -0.5/lik_var + 0.5*(np.square(Y) + np.square(m) + v - 2*m*Y)/(lik_var**2)
        return F, dF_dmu, dF_dv, dF_dtheta.reshape(1, Y.shape[0], Y.shape[1])

class HeteroscedasticGaussian(Gaussian):
    def __init__(self, Y_metadata, gp_link=None, variance=1., name='het_Gauss'):
        if gp_link is None:
            gp_link = link_functions.Identity()

        if not isinstance(gp_link, link_functions.Identity):
            print("Warning, Exact inference is not implemeted for non-identity link functions,\
            if you are not already, ensure Laplace inference_method is used")

        super(HeteroscedasticGaussian, self).__init__(gp_link, np.ones(Y_metadata['output_index'].shape)*variance, name)

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        return dL_dKdiag[Y_metadata['output_index']]

    def gaussian_variance(self, Y_metadata=None):
        return self.variance[Y_metadata['output_index'].flatten()]

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        _s = self.variance[Y_metadata['output_index'].flatten()]
        if full_cov:
            if var.ndim == 2:
                var += np.eye(var.shape[0])*_s
            if var.ndim == 3:
                var += np.atleast_3d(np.eye(var.shape[0])*_s)
        else:
            var += _s
        return mu, var

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        _s = self.variance[Y_metadata['output_index'].flatten()]
        return  [stats.norm.ppf(q/100.)*np.sqrt(var + _s) + mu for q in quantiles]
