# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from . import link_functions
from .likelihood import Likelihood

class Weibull(Likelihood):
    """
    .. math::
        $$ p(y_{i}|f_{i}, z_{i}) = \\prod_{i=1}^{n} [ r^{1-z_{i}}\\exp(-(1-z_{i})f(x_{i}))y_{i}^{(1-z_{i})(r-1)}\\exp(-y_{i}^{r}/\\exp(f(x_{i})))] $$

    .. note:
        where z_{i} is the censoring indicator- 0 for non-censored data, and 1 for censored data and r is the shape parameter.
    """
    def __init__(self,gp_link=None,beta=1.):
        if gp_link is None:
            gp_link = link_functions.Log()
            # gp_link = link_functions.Identity()
        super(Weibull, self).__init__(gp_link, name='Weibull')

        self.r = Param('r_weibull_shape', float(beta), Logexp())
        self.link_parameter(self.r)

        # self.r.fix()

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\beta^{\\alpha_{i}}}{\\Gamma(\\alpha_{i})}y_{i}^{\\alpha_{i}-1}e^{-\\beta y_{i}}\\\\
            \\alpha_{i} = \\beta y_{i}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape

        c = np.zeros((link_f.shape[0],))

        log_objective = np.log(self.r) + (self.r - 1) * np.log(y) - link_f - (np.exp(-link_f)*(y ** self.r))
        # log_objective = stats.weibull_min.pdf(y,c=self.beta,loc=link_f,scale=1.)
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
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        #alpha = self.gp_link.transf(gp)*self.beta    sum(log(a) + (a-1).*log(y)- f - exp(-f).*y.^a)
        #return (1. - alpha)*np.log(obs) + self.beta*obs - alpha * np.log(self.beta) + np.log(special.gamma(alpha))
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        log_objective = np.log(self.r) + (self.r - 1) * np.log(y) - link_f - (np.exp(-link_f) * (y ** self.r))
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
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: likelihood evaluated for this point
        :rtype: Nx1 array

        """
        # grad =  (1. - self.beta) / (y - link_f)

        grad = -1 + np.exp(-link_f)*(y ** self.r)
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
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: likelihood evaluated for this point
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        # hess = (self.beta - 1.) / (y - link_f)**2
        hess = -(y ** self.r) * np.exp(-link_f)
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
        d3lik_dlink3 = (y ** self.r) * np.exp(-link_f)
        return d3lik_dlink3

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
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
        dlogpdf_dr = 1./self.r + np.log(y) - np.exp(-inv_link_f)*(y**self.r)*np.log(y)
        return dlogpdf_dr

    def dlogpdf_dlink_dr(self, link_f, y, Y_metadata=None):
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
        dlogpdf_dlink_dr = np.exp(-link_f)* (y ** self.r) * np.log(y)
        return dlogpdf_dlink_dr

    def d2logpdf_dlink2_dr(self, link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t shape parameter

        .. math::

        :param inv_link_f: latent variables link(f)
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: derivative of hessian evaluated at points f and f_j w.r.t variance parameter
        :rtype: Nx1 array
        """
        d2logpdf_dlink_dr =  -np.exp(-link_f)* (y ** self.r) * np.log(y)
        return d2logpdf_dlink_dr

    def d3logpdf_dlink3_dr(self, link_f, y, Y_metadata=None):
        d3logpdf_dlink_dr = np.exp(-link_f)* (y ** self.r) * np.log(y)
        return d3logpdf_dlink_dr

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0, :, :] = self.dlogpdf_link_dr(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0,:,:] = self.dlogpdf_dlink_dr(f,y,Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        d2logpdf_dlink_dtheta2 = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink_dtheta2[0,:,:] = self.d2logpdf_dlink2_dr(f,y,Y_metadata)
        return d2logpdf_dlink_dtheta2

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        self.r.gradient = grads[0]