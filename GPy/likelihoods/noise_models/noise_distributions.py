# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
import pylab as pb
from GPy.util.plot import gpplot
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations
from GPy.util.misc import chain_1, chain_2, chain_3
from scipy.integrate import quad

class NoiseDistribution(object):
    """
    Likelihood class for doing approximations
    """
    def __init__(self,gp_link,analytical_mean=False,analytical_variance=False):
        assert isinstance(gp_link,gp_transformations.GPTransformation), "gp_link is not a valid GPTransformation."
        self.gp_link = gp_link
        self.analytical_mean = analytical_mean
        self.analytical_variance = analytical_variance
        if self.analytical_mean:
            self.moments_match = self._moments_match_analytical
            self.predictive_mean = self._predictive_mean_analytical
        else:
            self.moments_match = self._moments_match_numerical
            self.predictive_mean = self._predictive_mean_numerical
        if self.analytical_variance:
            self.predictive_variance = self._predictive_variance_analytical
        else:
            self.predictive_variance = self._predictive_variance_numerical

        self.log_concave = True

    def _get_params(self):
        return np.zeros(0)

    def _get_param_names(self):
        return []

    def _set_params(self,p):
        pass

    def _gradients(self,partial):
        return np.zeros(0)

    def _preprocess_values(self,Y):
        """
        In case it is needed, this function assess the output values or makes any pertinent transformation on them.

        :param Y: observed output
        :type Y: Nx1 numpy.darray

        """
        return Y

    def _moments_match_analytical(self,obs,tau,v):
        """
        If available, this function computes the moments analytically.
        """
        raise NotImplementedError

    def _moments_match_numerical(self,obs,tau,v):
        """
        Calculation of moments using quadrature

        :param obs: observed output
        :param tau: cavity distribution 1st natural parameter (precision)
        :param v: cavity distribution 2nd natural paramenter (mu*precision)
        """
        #Compute first integral for zeroth moment
        mu = v/tau
        def int_1(f):
            return self.pdf(f, obs)*np.exp(-0.5*tau*np.square(mu-f))
        z, accuracy = quad(int_1, -np.inf, np.inf)
        z /= np.sqrt(2*np.pi/tau)

        #Compute second integral for first moment
        def int_2(f):
            return f*self.pdf(f, obs)*np.exp(-0.5*tau*np.square(mu-f))
        mean, accuracy = quad(int_2, -np.inf, np.inf)
        mean /= np.sqrt(2*np.pi/tau)
        mean /= z

        #Compute integral for variance
        def int_3(f):
            return (f**2)*self.pdf(f, obs)*np.exp(-0.5*tau*np.square(mu-f))
        Ef2, accuracy = quad(int_3, -np.inf, np.inf)
        Ef2 /= np.sqrt(2*np.pi/tau)
        Ef2 /= z
        variance = Ef2 - mean**2

        return z, mean, variance

    def _predictive_mean_analytical(self,mu,sigma):
        """
        Predictive mean
        .. math::
            E(Y^{*}|Y) = E( E(Y^{*}|f^{*}, Y) )

        If available, this function computes the predictive mean analytically.
        """
        raise NotImplementedError

    def _predictive_variance_analytical(self,mu,sigma):
        """
        Predictive variance
        .. math::
            V(Y^{*}| Y) = E( V(Y^{*}|f^{*}, Y) ) + V( E(Y^{*}|f^{*}, Y) )

        If available, this function computes the predictive variance analytically.
        """
        raise NotImplementedError

    def _predictive_mean_numerical(self,mu,sigma):
        """
        Quadrature calculation of the predictive mean: E(Y_star|Y) = E( E(Y_star|f_star, Y) )

        :param mu: mean of posterior
        :param sigma: standard deviation of posterior

        """
        sigma2 = sigma**2
        #Compute first moment
        def int_mean(f):
            return self._mean(f)*np.exp(-(0.5/sigma2)*np.square(f - mu))
        scaled_mean, accuracy = quad(int_mean, -np.inf, np.inf)
        mean = scaled_mean / np.sqrt(2*np.pi*(sigma2))

        return mean

    def _predictive_variance_numerical(self,mu,sigma,predictive_mean=None):
        """
        Laplace approximation to the predictive variance: V(Y_star) = E( V(Y_star|f_star) ) + V( E(Y_star|f_star) )

        :param mu: mean of posterior
        :param sigma: standard deviation of posterior
        :predictive_mean: output's predictive mean, if None _predictive_mean function will be called.

        """
        sigma2 = sigma**2
        normalizer = np.sqrt(2*np.pi*sigma2)

        # E( V(Y_star|f_star) )
        #Compute expected value of variance
        def int_var(f):
            return self._variance(f)*np.exp(-(0.5/sigma2)*np.square(f - mu))
        scaled_exp_variance, accuracy = quad(int_var, -np.inf, np.inf)
        exp_var = scaled_exp_variance / normalizer

        #V( E(Y_star|f_star) ) =  E( E(Y_star|f_star)**2 ) - E( E(Y_star|f_star) )**2
        if predictive_mean is None:
            predictive_mean = self.predictive_mean(mu,sigma)

        predictive_mean_sq = predictive_mean**2
        def int_pred_mean_sq(f):
            return predictive_mean_sq*np.exp(-(0.5/(sigma2))*np.square(f - mu))

        scaled_exp_exp2, accuracy = quad(int_pred_mean_sq, -np.inf, np.inf)
        exp_exp2 = scaled_exp_exp2 / normalizer

        var_exp = exp_exp2 - predictive_mean**2
        # V(Y_star | f_star) = E( V(Y_star|f_star) ) + V( E(Y_star|f_star) )
        return exp_var + var_exp

    def _predictive_percentiles(self,p,mu,sigma):
        """
        Percentiles of the predictive distribution

        :parm p: lower tail probability
        :param mu: cavity distribution mean
        :param sigma: cavity distribution standard deviation
        :predictive_mean: output's predictive mean, if None _predictive_mean function will be called.

        """
        qf = stats.norm.ppf(p,mu,sigma)
        return self.gp_link.transf(qf)

    def pdf_link(self, link_f, y, extra_data=None):
        raise NotImplementedError

    def logpdf_link(self, link_f, y, extra_data=None):
        raise NotImplementedError

    def dlogpdf_dlink(self, link_f, y, extra_data=None):
        raise NotImplementedError

    def d2logpdf_dlink2(self, link_f, y, extra_data=None):
        raise NotImplementedError

    def d3logpdf_dlink3(self, link_f, y, extra_data=None):
        raise NotImplementedError

    def dlogpdf_link_dtheta(self, link_f, y, extra_data=None):
        """
        Need to check if it should even exist by checking length of getparams
        """
        raise NotImplementedError

    def dlogpdf_dlink_dtheta(self, link_f, y, extra_data=None):
        """
        Need to check if it should even exist by checking length of getparams
        """
        raise NotImplementedError

    def d2logpdf_dlink2_dtheta(self, link_f, y, extra_data=None):
        """
        Need to check if it should even exist by checking length of getparams
        """
        raise NotImplementedError

    def pdf(self, f, y, extra_data=None):
        """
        Evaluates the link function link(f) then computes the likelihood (pdf) using it

        .. math:
            p(y|\\lambda(f))

        :param f: latent variables f
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        link_f = self.gp_link.transf(f)
        return self.pdf_link(link_f, y, extra_data=extra_data)

    def logpdf(self, f, y, extra_data=None):
        """
        Evaluates the link function link(f) then computes the log likelihood (log pdf) using it

        .. math:
            \\log p(y|\\lambda(f))

        :param f: latent variables f
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: log likelihood evaluated for this point
        :rtype: float
        """
        link_f = self.gp_link.transf(f)
        return self.logpdf_link(link_f, y, extra_data=extra_data)

    def dlogpdf_df(self, f, y, extra_data=None):
        """
        Evaluates the link function link(f) then computes the derivative of log likelihood using it
        Uses the Faa di Bruno's formula for the chain rule

        .. math::
            \\frac{d\\log p(y|\\lambda(f))}{df} = \\frac{d\\log p(y|\\lambda(f))}{d\\lambda(f)}\\frac{d\\lambda(f)}{df}

        :param f: latent variables f
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: derivative of log likelihood evaluated for this point
        :rtype: 1xN array
        """
        link_f = self.gp_link.transf(f)
        dlogpdf_dlink = self.dlogpdf_dlink(link_f, y, extra_data=extra_data)
        dlink_df = self.gp_link.dtransf_df(f)
        return chain_1(dlogpdf_dlink, dlink_df)

    def d2logpdf_df2(self, f, y, extra_data=None):
        """
        Evaluates the link function link(f) then computes the second derivative of log likelihood using it
        Uses the Faa di Bruno's formula for the chain rule

        .. math::
            \\frac{d^{2}\\log p(y|\\lambda(f))}{df^{2}} = \\frac{d^{2}\\log p(y|\\lambda(f))}{d^{2}\\lambda(f)}\\left(\\frac{d\\lambda(f)}{df}\\right)^{2} + \\frac{d\\log p(y|\\lambda(f))}{d\\lambda(f)}\\frac{d^{2}\\lambda(f)}{df^{2}}

        :param f: latent variables f
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: second derivative of log likelihood evaluated for this point (diagonal only)
        :rtype: 1xN array
        """
        link_f = self.gp_link.transf(f)
        d2logpdf_dlink2 = self.d2logpdf_dlink2(link_f, y, extra_data=extra_data)
        dlink_df = self.gp_link.dtransf_df(f)
        dlogpdf_dlink = self.dlogpdf_dlink(link_f, y, extra_data=extra_data)
        d2link_df2 = self.gp_link.d2transf_df2(f)
        return chain_2(d2logpdf_dlink2, dlink_df, dlogpdf_dlink, d2link_df2)

    def d3logpdf_df3(self, f, y, extra_data=None):
        """
        Evaluates the link function link(f) then computes the third derivative of log likelihood using it
        Uses the Faa di Bruno's formula for the chain rule

        .. math::
            \\frac{d^{3}\\log p(y|\\lambda(f))}{df^{3}} = \\frac{d^{3}\\log p(y|\\lambda(f)}{d\\lambda(f)^{3}}\\left(\\frac{d\\lambda(f)}{df}\\right)^{3} + 3\\frac{d^{2}\\log p(y|\\lambda(f)}{d\\lambda(f)^{2}}\\frac{d\\lambda(f)}{df}\\frac{d^{2}\\lambda(f)}{df^{2}} + \\frac{d\\log p(y|\\lambda(f)}{d\\lambda(f)}\\frac{d^{3}\\lambda(f)}{df^{3}}

        :param f: latent variables f
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param extra_data: extra_data which is not used in student t distribution - not used
        :returns: third derivative of log likelihood evaluated for this point
        :rtype: float
        """
        link_f = self.gp_link.transf(f)
        d3logpdf_dlink3 = self.d3logpdf_dlink3(link_f, y, extra_data=extra_data)
        dlink_df = self.gp_link.dtransf_df(f)
        d2logpdf_dlink2 = self.d2logpdf_dlink2(link_f, y, extra_data=extra_data)
        d2link_df2 = self.gp_link.d2transf_df2(f)
        dlogpdf_dlink = self.dlogpdf_dlink(link_f, y, extra_data=extra_data)
        d3link_df3 = self.gp_link.d3transf_df3(f)
        return chain_3(d3logpdf_dlink3, dlink_df, d2logpdf_dlink2, d2link_df2, dlogpdf_dlink, d3link_df3)

    def dlogpdf_dtheta(self, f, y, extra_data=None):
        """
        TODO: Doc strings
        """
        if len(self._get_param_names()) > 0:
            link_f = self.gp_link.transf(f)
            return self.dlogpdf_link_dtheta(link_f, y, extra_data=extra_data)
        else:
            #Is no parameters so return an empty array for its derivatives
            return np.empty([1, 0])

    def dlogpdf_df_dtheta(self, f, y, extra_data=None):
        """
        TODO: Doc strings
        """
        if len(self._get_param_names()) > 0:
            link_f = self.gp_link.transf(f)
            dlink_df = self.gp_link.dtransf_df(f)
            dlogpdf_dlink_dtheta = self.dlogpdf_dlink_dtheta(link_f, y, extra_data=extra_data)
            return chain_1(dlogpdf_dlink_dtheta, dlink_df)
        else:
            #Is no parameters so return an empty array for its derivatives
            return np.empty([f.shape[0], 0])

    def d2logpdf_df2_dtheta(self, f, y, extra_data=None):
        """
        TODO: Doc strings
        """
        if len(self._get_param_names()) > 0:
            link_f = self.gp_link.transf(f)
            dlink_df = self.gp_link.dtransf_df(f)
            d2link_df2 = self.gp_link.d2transf_df2(f)
            d2logpdf_dlink2_dtheta = self.d2logpdf_dlink2_dtheta(link_f, y, extra_data=extra_data)
            dlogpdf_dlink_dtheta = self.dlogpdf_dlink_dtheta(link_f, y, extra_data=extra_data)
            return chain_2(d2logpdf_dlink2_dtheta, dlink_df, dlogpdf_dlink_dtheta, d2link_df2)
        else:
            #Is no parameters so return an empty array for its derivatives
            return np.empty([f.shape[0], 0])

    def _laplace_gradients(self, f, y, extra_data=None):
        dlogpdf_dtheta = self.dlogpdf_dtheta(f, y, extra_data=extra_data)
        dlogpdf_df_dtheta = self.dlogpdf_df_dtheta(f, y, extra_data=extra_data)
        d2logpdf_df2_dtheta = self.d2logpdf_df2_dtheta(f, y, extra_data=extra_data)

        #Parameters are stacked vertically. Must be listed in same order as 'get_param_names'
        # ensure we have gradients for every parameter we want to optimize
        assert dlogpdf_dtheta.shape[1] == len(self._get_param_names())
        assert dlogpdf_df_dtheta.shape[1] == len(self._get_param_names())
        assert d2logpdf_df2_dtheta.shape[1] == len(self._get_param_names())
        return dlogpdf_dtheta, dlogpdf_df_dtheta, d2logpdf_df2_dtheta

    def predictive_values(self,mu,var):
        """
        Compute  mean, variance and conficence interval (percentiles 5 and 95) of the  prediction.

        :param mu: mean of the latent variable, f
        :param var: variance of the latent variable, f

        """
        if isinstance(mu,float) or isinstance(mu,int):
            mu = [mu]
            var = [var]
        pred_mean = []
        pred_var = []
        q1 = []
        q3 = []
        for m,s in zip(mu,np.sqrt(var)):
            pred_mean.append(self.predictive_mean(m,s))
            pred_var.append(self.predictive_variance(m,s,pred_mean[-1]))
            q1.append(self._predictive_percentiles(.025,m,s))
            q3.append(self._predictive_percentiles(.975,m,s))
        pred_mean = np.vstack(pred_mean)
        pred_var = np.vstack(pred_var)
        q1 = np.vstack(q1)
        q3 = np.vstack(q3)
        return pred_mean, pred_var, q1, q3

    def samples(self, gp):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        raise NotImplementedError
