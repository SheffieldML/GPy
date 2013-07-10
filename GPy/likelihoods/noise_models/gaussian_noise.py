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
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False,variance=1.):
        self.variance = variance
        super(Gaussian, self).__init__(gp_link,analytical_mean,analytical_variance)

    def _get_params(self):
        return self.variance

    def _get_param_names(self):
        return ['noise_model_variance']

    def _set_params(self,p):
        self.variance = p

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

    def _predictive_variance_analytical(self,mu,sigma,*args): #TODO *args?
        return 1./(1./self.variance + 1./sigma**2)

    def _mass(self,gp,obs):
        return std_norm_pdf( (self.gp_link.transf(gp)-obs)/np.sqrt(self.variance) )

    def _nlog_mass(self,gp,obs):
        return .5*((self.gp_link.transf(gp)-obs)**2/np.sqrt(self.variance) + np.log(2*np.pi*self.variance))

    def _dnlog_mass_dgp(self,gp,obs):
        return (self.gp_link.transf(gp)-obs)/np.sqrt(self.variance) * self.gp_link.dtransf_df(gp)

    def _d2nlog_mass_dgp2(self,gp,obs):
        return ((self.gp_link.transf(gp)-obs)*self.gp_link.d2transf_df2(gp) + self.gp_link.dtransf_df(gp)**2)/self.variance

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
        return self.variance

    def _dvariance_dgp(self,gp):
        return 0

    def _d2variance_dgp2(self,gp):
        return 0
