# Copyright (c) 2012-2014 Ricardo Andrade, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import special
from . import link_functions
from .likelihood import Likelihood


class Poisson(Likelihood):
    """
    Poisson likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\lambda(f_{i})^{y_{i}}}{y_{i}!}e^{-\\lambda(f_{i})}

    .. Note::
        Y is expected to take values in {0,1,2,...}
    """

    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Log()

        super(Poisson, self).__init__(gp_link, name="Poisson")

    def _conditional_mean(self, f):
        """
        the expected value of y given a value of f
        """
        return self.gp_link.transf(f)

    def logpdf(self, f, y, Y_metadata=None):
        # With the default Log link, λ=e^f. Work in f so deep-negative latents stay
        # finite: -e^f + y f - gammaln(y+1) rather than y*log(e^f) after underflow.
        if isinstance(self.gp_link, link_functions.Log):
            f = np.asarray(f, dtype=float)
            y = np.asarray(y, dtype=float)
            with np.errstate(over="ignore"):
                rate = np.exp(f)
            return -rate + y * f - special.gammaln(y + 1)
        return super(Poisson, self).logpdf(f, y, Y_metadata=Y_metadata)

    def dlogpdf_df(self, f, y, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Log):
            f = np.asarray(f, dtype=float)
            y = np.asarray(y, dtype=float)
            with np.errstate(over="ignore"):
                return y - np.exp(f)
        return super(Poisson, self).dlogpdf_df(f, y, Y_metadata=Y_metadata)

    def d2logpdf_df2(self, f, y, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Log):
            f = np.asarray(f, dtype=float)
            with np.errstate(over="ignore"):
                return -np.exp(f)
        return super(Poisson, self).d2logpdf_df2(f, y, Y_metadata=Y_metadata)

    def d3logpdf_df3(self, f, y, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Log):
            f = np.asarray(f, dtype=float)
            with np.errstate(over="ignore"):
                return -np.exp(f)
        return super(Poisson, self).d3logpdf_df3(f, y, Y_metadata=Y_metadata)

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\lambda(f_{i})^{y_{i}}}{y_{i}!}e^{-\\lambda(f_{i})}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        return np.exp(self.logpdf_link(link_f, y, Y_metadata))
        # return np.prod(stats.poisson.pmf(y,link_f))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = -\\lambda(f_{i}) + y_{i}\\log \\lambda(f_{i}) - \\log y_{i}!

        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        # y*log(λ) is 0*(-inf)=NaN when λ underflows and y=0; keep that case finite.
        link_f = np.asarray(link_f, dtype=float)
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rate = np.log(link_f)
        y_log_rate = np.where(y == 0, 0.0, y * log_rate)
        return -link_f + y_log_rate - special.gammaln(y + 1)

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\frac{y_{i}}{\\lambda(f_{i})} - 1

        :param link_f: latent variables (f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: gradient of likelihood evaluated at points
        :rtype: Nx1 array

        """
        link_f = np.asarray(link_f, dtype=float)
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = y / link_f - 1.0
        # lim_{λ→0} (0/λ - 1) = -1
        return np.where((y == 0) & (link_f == 0), -1.0, grad)

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{2}\\lambda(f)} = \\frac{-y_{i}}{\\lambda(f_{i})^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        return -y / (link_f**2)

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{2y_{i}}{\\lambda(f_{i})^{3}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        d3lik_dlink3 = 2 * y / (link_f) ** 3
        return d3lik_dlink3

    def conditional_mean(self, gp):
        """
        The mean of the random variable conditioned on one value of the GP
        """
        return self.gp_link.transf(gp)

    def conditional_variance(self, gp):
        """
        The variance of the random variable conditioned on one value of the GP
        """
        return self.gp_link.transf(gp)

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        # Ysim = np.random.poisson(self.gp_link.transf(gp), [samples, gp.size]).T
        # return Ysim.reshape(orig_shape+(samples,))
        Ysim = np.random.poisson(self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)
