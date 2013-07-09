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
    def __init__(self,gp_link=None,analytical_moments=False,mean=0,variance=1.):
        self.mean = mean
        self.variance = variance
        super(Gaussian, self).__init__(gp_link,analytical_moments)

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
        mu_hat = sigma2_hat*(self.mean/self.variance + v_i)
        Z_hat = np.sqrt(2*np.pi*sigma2_hat)*np.exp(-.5*(self.mean - v_i/tau_i)**2/(self.variance + 1./tau_i)) #TODO check
        return Z_hat, mu_hat, sigma2_hat

    def _predictive_mean_analytical(self,mu,sigma):
        new_sigma2 = 1./(1./self.variance + 1./sigma**2)
        return new_sigma2*(mu/sigma + self.mean/self.variance)

    def _mass(self,gp,obs):
        p = (self.gp_link.transf(gp)-self.mean)/np.sqrt(self.variance)
        return std_norm_pdf(p)

    def _nlog_mass(self,gp,obs):
        p = (self.gp_link.transf(gp)-self.mean)/np.sqrt(self.variance)
        return .5*np.log(2*np.pi*self.variance) + .5*(p-self.mean)**2/self.variance

    def _dnlog_mass_dgp(self,gp,obs):
        p = (self.gp_link.transf(gp)-self.mean)/np.sqrt(self.variance)
        dp = self.gp_link.dtransf_df(gp)
        return (p - self.mean)/self.variance * dp

    def _d2nlog_mass_dgp2(self,gp,obs):
        p = (self.gp_link.transf(gp)-self.mean)/np.sqrt(self.variance)
        dp = self.gp_link.dtransf_df(gp)
        d2p = self.gp_link.d2transf_df2(gp)
        return dp**2/self.variance + (p - self.mean)/self.variance * d2p

    def _mean(self,gp):
        """
        Mass (or density) function
        """
        return self.gp_link.transf(gp)

    def _dmean_dgp(self,gp):
        return self.gp_link.dtransf_df(gp)

    def _d2mean_dgp2(self,gp):
        return self.gp_link.d2transf_df2(gp)

    def _variance(self,gp):
        """
        Mass (or density) function
        """
        p = self.gp_link.transf(gp)
        return p*(1-p)

    def _dvariance_dgp(self,gp):
        return self.gp_link.dtransf_df(gp)*(1. - 2.*self.gp_link.transf(gp))

    def _d2variance_dgp2(self,gp):
        return self.gp_link.d2transf_df2(gp)*(1. - 2.*self.gp_link.transf(gp)) - 2*self.gp_link.dtransf_df(gp)**2
