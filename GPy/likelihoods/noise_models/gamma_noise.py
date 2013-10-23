# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations
from noise_distributions import NoiseDistribution

class Gamma(NoiseDistribution):
    """
    Gamma likelihood
    Y is expected to take values in {0,1,2,...}
    -----
    $$
    L(x) = \exp(\lambda) * \lambda**Y_i / Y_i!
    $$
    """
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False,beta=1.):
        self.beta = beta
        super(Gamma, self).__init__(gp_link,analytical_mean,analytical_variance)

    def _preprocess_values(self,Y):
        return Y

    def _mass(self,gp,obs):
        """
        Mass (or density) function
        """
        #return stats.gamma.pdf(obs,a = self.gp_link.transf(gp)/self.variance,scale=self.variance)
        alpha = self.gp_link.transf(gp)*self.beta
        return obs**(alpha - 1.) * np.exp(-self.beta*obs) * self.beta**alpha / special.gamma(alpha)

    def _nlog_mass(self,gp,obs):
        """
        Negative logarithm of the un-normalized distribution: factors that are not a function of gp are omitted
        """
        alpha = self.gp_link.transf(gp)*self.beta
        return (1. - alpha)*np.log(obs) + self.beta*obs - alpha * np.log(self.beta) + np.log(special.gamma(alpha))

    def _dnlog_mass_dgp(self,gp,obs):
        return -self.gp_link.dtransf_df(gp)*self.beta*np.log(obs) + special.psi(self.gp_link.transf(gp)*self.beta) * self.gp_link.dtransf_df(gp)*self.beta

    def _d2nlog_mass_dgp2(self,gp,obs):
        return -self.gp_link.d2transf_df2(gp)*self.beta*np.log(obs) + special.polygamma(1,self.gp_link.transf(gp)*self.beta)*(self.gp_link.dtransf_df(gp)*self.beta)**2 + special.psi(self.gp_link.transf(gp)*self.beta)*self.gp_link.d2transf_df2(gp)*self.beta

    def _mean(self,gp):
        """
        Mass (or density) function
        """
        return self.gp_link.transf(gp)

    def _variance(self,gp):
        """
        Mass (or density) function
        """
        return self.gp_link.transf(gp)/self.beta
