# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from ..core.parameterization import Param
from . import link_functions
from .likelihood import Likelihood

class Gamma(Likelihood):
    """
    Gamma likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\beta^{\\alpha_{i}}}{\\Gamma(\\alpha_{i})}y_{i}^{\\alpha_{i}-1}e^{-\\beta y_{i}}\\\\
        \\alpha_{i} = \\beta y_{i}

    """
    def __init__(self,gp_link=None,beta=1.):
        if gp_link is None:
            gp_link = link_functions.Log()
        super(Gamma, self).__init__(gp_link, 'Gamma')

        self.beta = Param('beta', beta)
        self.link_parameter(self.beta)
        self.beta.fix()#TODO: gradients!

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
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        #return stats.gamma.pdf(obs,a = self.gp_link.transf(gp)/self.variance,scale=self.variance)
        alpha = link_f*self.beta
        objective = (y**(alpha - 1.) * np.exp(-self.beta*y) * self.beta**alpha)/ special.gamma(alpha)
        return np.exp(np.sum(np.log(objective)))

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
        #alpha = self.gp_link.transf(gp)*self.beta
        #return (1. - alpha)*np.log(obs) + self.beta*obs - alpha * np.log(self.beta) + np.log(special.gamma(alpha))
        alpha = link_f*self.beta
        log_objective = alpha*np.log(self.beta) - np.log(special.gamma(alpha)) + (alpha - 1)*np.log(y) - self.beta*y
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
        grad = self.beta*np.log(self.beta*y) - special.psi(self.beta*link_f)*self.beta
        #old
        #return -self.gp_link.dtransf_df(gp)*self.beta*np.log(obs) + special.psi(self.gp_link.transf(gp)*self.beta) * self.gp_link.dtransf_df(gp)*self.beta
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
        hess = -special.polygamma(1, self.beta*link_f)*(self.beta**2)
        #old
        #return -self.gp_link.d2transf_df2(gp)*self.beta*np.log(obs) + special.polygamma(1,self.gp_link.transf(gp)*self.beta)*(self.gp_link.dtransf_df(gp)*self.beta)**2 + special.psi(self.gp_link.transf(gp)*self.beta)*self.gp_link.d2transf_df2(gp)*self.beta
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
        d3lik_dlink3 = -special.polygamma(2, self.beta*link_f)*(self.beta**3)
        return d3lik_dlink3
