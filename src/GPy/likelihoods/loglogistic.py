from __future__ import division
# Copyright (c) 2015 Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats,special
import scipy as sp
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from . import link_functions
from .likelihood import Likelihood
from .link_functions import Log

class LogLogistic(Likelihood):
    """
    .. math::
        $$ p(y_{i}|f_{i}, z_{i}) = \\prod_{i=1}^{n} (\\frac{ry^{r-1}}{\\exp{f(x_{i})}})^{1-z_i} (1 + (\\frac{y}{\\exp(f(x_{i}))})^{r})^{z_i-2}  $$

    .. note:
        where z_{i} is the censoring indicator- 0 for non-censored data, and 1 for censored data.
    """

    def __init__(self, gp_link=None, r=1.0):
        if gp_link is None:
            #Parameterised not as link_f but as f
            gp_link = Log()

        super(LogLogistic, self).__init__(gp_link, name='LogLogistic')
        self.r = Param('r_log_shape', float(r), Logexp())
        self.link_parameter(self.r)
        # self.censored = 'censored'



    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        return np.exp(self.logpdf_link(link_f, y, Y_metadata=Y_metadata))


    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        .. math::


        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(link_f)
        if Y_metadata is not None and  'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        link_f = np.clip(link_f, 1e-150, 1e100)
        # y_link_f = y/link_f
        # y_link_f_r = y_link_f**self.r
        # y_link_f_r = np.clip(y**self.r, 1e-150, 1e200) / np.clip(link_f**self.r, 1e-150, 1e200)
        # y_link_f_r = np.clip((y/link_f)**self.r, 1e-150, 1e200)
        y_r = np.clip(y**self.r, 1e-150, 1e200)
        link_f_r = np.clip(link_f**self.r, 1e-150, 1e200)
        y_link_f_r = np.clip(y_r / link_f_r, 1e-150, 1e200)
        #uncensored = (1-c)*(np.log(self.r) + (self.r+1)*np.log(y) - self.r*np.log(link_f) - 2*np.log1p(y_link_f_r))
        #uncensored = (1-c)*(np.log((self.r/link_f)*y_link_f**(self.r-1)) - 2*np.log1p(y_link_f_r))

        # clever way tp break it into censored and uncensored-parts ..
        uncensored = (1-c)*(np.log(self.r) + (self.r-1)*np.log(y) - self.r*np.log(link_f) - 2*np.log1p(y_link_f_r))
        censored = (c)*(-np.log1p(y_link_f_r))
        #
        return uncensored + censored
        # return uncensored

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::

        :param link_f: latent variables (f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: gradient of likelihood evaluated at points
        :rtype: Nx1 array

        """
        # c = Y_metadata['censored']
        # for debugging
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(link_f)

        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        #y_link_f = y/link_f
        #y_link_f_r = y_link_f**self.r
        y_link_f_r = np.clip(y**self.r, 1e-150, 1e200) / np.clip(link_f**self.r, 1e-150, 1e200)

        #In terms of link_f
        # uncensored = (1-c)*( (2*self.r*y**r)/(link_f**self.r + y**self.r) - link_f*self.r)
        uncensored = (1-c)*self.r*(y_link_f_r - 1)/(link_f*(1 + y_link_f_r))
        censored = c*(self.r*y_link_f_r/(link_f*y_link_f_r + link_f))
        return uncensored + censored
        # return uncensored

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::


        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        # c = Y_metadata['censored']
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(link_f)

        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        y_link_f = y/link_f
        y_link_f_r = y_link_f**self.r

        #In terms of link_f
        censored = c*(-self.r*y_link_f_r*(y_link_f_r + self.r + 1)/((link_f**2)*(y_link_f_r + 1)**2))
        uncensored = (1-c)*(-self.r*(2*self.r*y_link_f_r + y_link_f**(2*self.r) - 1) / ((link_f**2)*(1+ y_link_f_r)**2))
        hess = censored + uncensored
        return hess

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::


        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        # c = Y_metadata['censored']
        #  for debugging
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(link_f)

        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        y_link_f = y/link_f
        y_link_f_r = y_link_f**self.r

        #In terms of link_f
        censored = c*(self.r*y_link_f_r*(((self.r**2)*(-(y_link_f_r - 1))) + 3*self.r*(y_link_f_r + 1) + 2*(y_link_f_r + 1)**2)
                      / ((link_f**3)*(y_link_f_r + 1)**3))
        uncensored = (1-c)*(2*self.r*(-(self.r**2)*(y_link_f_r -1)*y_link_f_r + 3*self.r*(y_link_f_r + 1)*y_link_f_r + (y_link_f_r - 1)*(y_link_f_r + 1)**2)
                            / ((link_f**3)*(y_link_f_r + 1)**3))

        d3lik_dlink3 = censored + uncensored
        return d3lik_dlink3

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
        # c = Y_metadata['censored']
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']

        link_f = inv_link_f #FIXME: Change names consistently...
        y_link_f = y/link_f
        log_y_link_f = np.log(y) - np.log(link_f)
        y_link_f_r = y_link_f**self.r

        #In terms of link_f
        censored = c*(-y_link_f_r*log_y_link_f/(1 + y_link_f_r))
        uncensored = (1-c)*(1./self.r + np.log(y) - np.log(link_f) - (2*y_link_f_r*log_y_link_f) / (1 + y_link_f_r))

        dlogpdf_dr = censored + uncensored
        return dlogpdf_dr

    def dlogpdf_dlink_dr(self, inv_link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t shape parameter

        .. math::

        :param inv_link_f: latent variables inv_link_f
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: includes censoring information in dictionary key 'censored'
        :returns: derivative of likelihood evaluated at points f w.r.t variance parameter
        :rtype: Nx1 array
        """
        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        link_f = inv_link_f
        y_link_f = y/link_f
        y_link_f_r = y_link_f**self.r
        log_y_link_f = np.log(y) - np.log(link_f)

        #In terms of link_f
        censored = c*(y_link_f_r*(y_link_f_r + self.r*log_y_link_f + 1)/(link_f*(y_link_f_r + 1)**2))
        uncensored = (1-c)*(y_link_f**(2*self.r) + 2*self.r*y_link_f_r*log_y_link_f - 1) / (link_f*(1 + y_link_f_r)**2)

        # dlogpdf_dlink_dr = uncensored
        dlogpdf_dlink_dr = censored + uncensored
        return dlogpdf_dlink_dr

    def d2logpdf_dlink2_dr(self, inv_link_f, y, Y_metadata=None):
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
        # c = Y_metadata['censored']

        # c = np.zeros((y.shape[0],))
        c = np.zeros_like(y)
        if Y_metadata is not None and 'censored' in Y_metadata.keys():
            c = Y_metadata['censored']
        link_f = inv_link_f
        y_link_f = y/link_f
        y_link_f_r = y_link_f**self.r
        log_y_link_f = np.log(y) - np.log(link_f)

        #In terms of link_f
        y_link_f_2r = y_link_f**(2*self.r)
        denom2 = (link_f**2)*(1 + y_link_f_r)**2
        denom3 = (link_f**2)*(1 + y_link_f_r)**3

        censored = c*(-((y_link_f_r + self.r + 1)*y_link_f_r)/denom2
                      -(self.r*(y_link_f_r + self.r + 1)*y_link_f_r*log_y_link_f)/denom2
                      -(self.r*y_link_f_r*(y_link_f_r*log_y_link_f + 1))/denom2
                      +(2*self.r*(y_link_f_r + self.r + 1)*y_link_f_2r*log_y_link_f)/denom3
                      )

        uncensored = (1-c)*(-(2*self.r*y_link_f_r + y_link_f_2r - 1)/denom2
                            -(self.r*(2*y_link_f_r + 2*self.r*y_link_f_r*log_y_link_f + 2*y_link_f_2r*log_y_link_f)/denom2)
                            +(2*self.r*(2*self.r*y_link_f_r + y_link_f_2r - 1)*y_link_f_r*log_y_link_f)/denom3
                            )
        d2logpdf_dlink2_dr = censored + uncensored

        return d2logpdf_dlink2_dr

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dtheta[0, :, :] = self.dlogpdf_link_dr(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dtheta

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        dlogpdf_dlink_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        dlogpdf_dlink_dtheta[0, :, :] = self.dlogpdf_dlink_dr(f, y, Y_metadata=Y_metadata)
        return dlogpdf_dlink_dtheta

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        d2logpdf_dlink2_dtheta = np.zeros((self.size, f.shape[0], f.shape[1]))
        d2logpdf_dlink2_dtheta[0,:, :] = self.d2logpdf_dlink2_dr(f, y, Y_metadata=Y_metadata)
        return d2logpdf_dlink2_dtheta

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        self.r.gradient = grads[0]

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #rs = np.ones_like(gp)*self.r
        #scales = np.ones_like(gp)*np.sqrt(self.sigma2)
        #Ysim = sp.stats.fisk.rvs(rs, scale=self.gp_link.transf(gp))
        Ysim = np.array([sp.stats.fisk.rvs(self.r, loc=0, scale=self.gp_link.transf(f)) for f in gp])
        #np.random.fisk(self.gp_link.transf(gp), c=self.r)
        return Ysim.reshape(orig_shape)

