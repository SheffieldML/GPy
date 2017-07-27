# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats, special
import scipy as sp
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from . import link_functions
from .likelihood import Likelihood


class Weibull(Likelihood):
    """
    Implementing Weibull likelihood function ...

    """

    def __init__(self, gp_link=None, beta=1.):
        if gp_link is None:
            #Parameterised not as link_f but as f
            # gp_link = link_functions.Identity()
            #Parameterised as link_f
            gp_link = link_functions.Log()
        super(Weibull, self).__init__(gp_link, name='Weibull')

        self.r = Param('r_weibull_shape', float(beta), Logexp())
        self.link_parameter(self.r)

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)


        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in weibull distribution
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros((link_f.shape[0],))

        # log_objective = np.log(self.r) + (self.r - 1) * np.log(y) - link_f - (np.exp(-link_f) * (y ** self.r))
        # log_objective = stats.weibull_min.pdf(y,c=self.beta,loc=link_f,scale=1.)
        log_objective = self.logpdf_link(link_f, y, Y_metadata)
        return np.exp(log_objective)

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|\lambda(f_{i})) = \\alpha_{i}\\log \\beta - \\log \\Gamma(\\alpha_{i}) + (\\alpha_{i} - 1)\\log y_{i} - \\beta y_{i}\\\\
            \\alpha_{i} = \\beta y_{i}

        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        # alpha = self.gp_link.transf(gp)*self.beta    sum(log(a) + (a-1).*log(y)- f - exp(-f).*y.^a)
        # return (1. - alpha)*np.log(obs) + self.beta*obs - alpha * np.log(self.beta) + np.log(special.gamma(alpha))
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)* (np.log(self.r) + (self.r - 1) * np.log(y) - link_f - (np.exp(-link_f) * (y ** self.r)))
        # censored = (-c)*np.exp(-link_f)*(y**self.r)
        uncensored = (1-c)*( np.log(self.r)-np.log(link_f)+(self.r-1)*np.log(y) - y**self.r/link_f)
        censored = -c*y**self.r/link_f

        log_objective = uncensored + censored
        return log_objective

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\beta (\\log \\beta y_{i}) - \\Psi(\\alpha_{i})\\beta\\\\
            \\alpha_{i} = \\beta y_{i}

        :param link_f: latent variables (f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in gamma distribution
        :returns: gradient of likelihood evaluated at points
        :rtype: Nx1 array

        """
        # grad =  (1. - self.beta) / (y - link_f)
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)* ( -1 + np.exp(-link_f)*(y ** self.r))
        # censored = c*np.exp(-link_f)*(y**self.r)
        uncensored = (1-c)*(-1/link_f + y**self.r/link_f**2)
        censored = c*y**self.r/link_f**2
        grad = uncensored + censored
        return grad

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}\\lambda(f)} = -\\beta^{2}\\frac{d\\Psi(\\alpha_{i})}{d\\alpha_{i}}\\\\
            \\alpha_{i} = \\beta y_{i}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in gamma distribution
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        # hess = (self.beta - 1.) / (y - link_f)**2
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)* (-(y ** self.r) * np.exp(-link_f))
        # censored = -c*np.exp(-link_f)*y**self.r
        uncensored = (1-c)*(1/link_f**2 -2*y**self.r/link_f**3)
        censored = -c*2*y**self.r/link_f**3
        hess = uncensored + censored
        # hess = -(y ** self.r) * np.exp(-link_f)
        return hess

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\lambda(f_{i}))}{d^{3}\\lambda(f)} = -\\beta^{3}\\frac{d^{2}\\Psi(\\alpha_{i})}{d\\alpha_{i}}\\\\
            \\alpha_{i} = \\beta y_{i}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in gamma distribution
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        # d3lik_dlink3 = (1. - self.beta) / (y - link_f)**3

        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        # uncensored = (1-c)* ((y ** self.r) * np.exp(-link_f))
        # censored = c*np.exp(-link_f)*y**self.r
        uncensored = (1-c)*(-2/link_f**3+ 6*y**self.r/link_f**4)
        censored = c*6*y**self.r/link_f**4

        d3lik_dlink3 = uncensored + censored
        # d3lik_dlink3 = (y ** self.r) * np.exp(-link_f)
        return d3lik_dlink3

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata=None):
        return np.zeros(self.size)

    def dlogpdf_link_dr(self, inv_link_f, y, Y_metadata=None):
        """

        Gradient of the log-likelihood function at y given f, w.r.t shape parameter

        .. math::

        :param inv_link_f: latent variables link(f)
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: float
        """
        c = np.zeros_like(y)
        link_f = inv_link_f
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        uncensored = (1-c)* (1./self.r + np.log(y) - y**self.r*np.log(y)/link_f)
        censored = (-c*y**self.r*np.log(y)/link_f)
        dlogpdf_dr = uncensored + censored
        return dlogpdf_dr

    def dlogpdf_dlink_dr(self, inv_link_f, y, Y_metadata=None):
        """
        First order derivative derivative of loglikelihood wrt r:shape parameter

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in gamma distribution
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        # dlogpdf_dlink_dr = self.beta * y**(self.beta - 1) * np.exp(-link_f)
        # dlogpdf_dlink_dr = np.exp(-link_f) * (y ** self.r) * np.log(y)
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        link_f = inv_link_f
        # uncensored = (1-c)*(np.exp(-link_f)* (y ** self.r) * np.log(y))
        # censored = c*np.exp(-link_f)*(y**self.r)*np.log(y)
        uncensored = (1-c)*(y**self.r*np.log(y)/link_f**2)
        censored = c*(y**self.r*np.log(y)/link_f**2)
        dlogpdf_dlink_dr = uncensored + censored
        return dlogpdf_dlink_dr

    def d2logpdf_dlink2_dr(self, link_f, y, Y_metadata=None):
        """

        Derivative of hessian of loglikelihood wrt r-shape parameter.
        :param link_f:
        :param y:
        :param Y_metadata:
        :return:
        """

        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        # uncensored = (1-c)*( -np.exp(-link_f)* (y ** self.r) * np.log(y))
        # censored = -c*np.exp(-link_f)*(y**self.r)*np.log(y)
        uncensored = (1-c)*-2*y**self.r*np.log(y)/link_f**3
        censored = c*-2*y**self.r*np.log(y)/link_f**3
        d2logpdf_dlink_dr = uncensored + censored

        return d2logpdf_dlink_dr

    def d3logpdf_dlink3_dr(self, link_f, y, Y_metadata=None):
        """

        :param link_f:
        :param y:
        :param Y_metadata:
        :return:
        """
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        uncensored = (1-c)* ((y**self.r)*np.exp(-link_f)*np.log1p(y))
        censored = c*np.exp(-link_f)*(y**self.r)*np.log(y)
        d3logpdf_dlink3_dr = uncensored + censored
        return d3logpdf_dlink3_dr

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        """

        :param f:
        :param y:
        :param Y_metadata:
        :return:
        """
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0, :, :] = self.dlogpdf_link_dr(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        """

        :param f:
        :param y:
        :param Y_metadata:
        :return:
        """
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0, :, :] = self.dlogpdf_dlink_dr(f, y, Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        """

        :param f:
        :param y:
        :param Y_metadata:
        :return:
        """
        d2logpdf_dlink_dtheta2 = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink_dtheta2[0, :, :] = self.d2logpdf_dlink2_dr(f, y, Y_metadata)
        return d2logpdf_dlink_dtheta2

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        self.r.gradient = grads[0]

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations conditioned on a given value of latent variable f.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        weibull_samples = np.array([sp.stats.weibull_min.rvs(self.r, loc=0, scale=self.gp_link.transf(f)) for f in gp])
        return weibull_samples.reshape(orig_shape)