# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
from ..util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import link_functions
from likelihood_functions import LikelihoodFunction

class Binomial(LikelihoodFunction):
    """
    Probit likelihood
    Y is expected to take values in {-1,1}
    -----
    $$
    L(x) = \\Phi (Y_i*f_i)
    $$
    """
    def __init__(self,link=None):
        self.discrete = True
        self.support_limits = (0,1)
        self._analytical = link_functions.Probit
        if not link:
            link = self._analytical
        super(Binomial, self).__init__(link)

    def _mass(self,gp,obs):
        pass

    def _nlog_mass(self,gp,obs):
        pass

    def _preprocess_values(self,Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.

        ..Note:: Binary classification algorithm works better with classes {-1,1}
        """
        Y_prep = Y.copy()
        Y1 = Y[Y.flatten()==1].size
        Y2 = Y[Y.flatten()==0].size
        assert Y1 + Y2 == Y.size, 'Binomial likelihood is meant to be used only with outputs in {0,1}.'
        Y_prep[Y.flatten() == 0] = -1
        return Y_prep

    def _moments_match_analytical(self,data_i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        z = data_i*v_i/np.sqrt(tau_i**2 + tau_i)
        Z_hat = std_norm_cdf(z)
        phi = std_norm_pdf(z)
        mu_hat = v_i/tau_i + data_i*phi/(Z_hat*np.sqrt(tau_i**2 + tau_i))
        sigma2_hat = 1./tau_i - (phi/((tau_i**2+tau_i)*Z_hat))*(z+phi/Z_hat)
        return Z_hat, mu_hat, sigma2_hat

    def _predictive_mean_analytical(self,mu,sigma):
        return stats.norm.cdf(mu/np.sqrt(1+sigma**2))

    def predictive_values(self,mu,var):
        """
        Compute  mean, variance and conficence interval (percentiles 5 and 95) of the  prediction
        :param mu: mean of the latent variable
        :param var: variance of the latent variable
        """
        mu = mu.flatten()
        var = var.flatten()
        #mean = stats.norm.cdf(mu/np.sqrt(1+var))
        mean = self._predictive_mean_analytical(mu,np.sqrt(var))
        norm_025 = [stats.norm.ppf(.025,m,v) for m,v in zip(mu,var)]
        norm_975 = [stats.norm.ppf(.975,m,v) for m,v in zip(mu,var)]
        #p_025 = stats.norm.cdf(norm_025/np.sqrt(1+var))
        #p_975 = stats.norm.cdf(norm_975/np.sqrt(1+var))
        p_025 = self._predictive_mean_analytical(norm_025,np.sqrt(var))
        p_975 = self._predictive_mean_analytical(norm_975,np.sqrt(var))
        return mean[:,None], np.nan*var, p_025[:,None], p_975[:,None] # TODO: var
