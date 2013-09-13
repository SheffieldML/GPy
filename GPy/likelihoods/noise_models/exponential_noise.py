# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations
from noise_distributions import NoiseDistribution

class Exponential(NoiseDistribution):
    """
    Gamma likelihood
    Y is expected to take values in {0,1,2,...}
    -----
    $$
    L(x) = \exp(\lambda) * \lambda**Y_i / Y_i!
    $$
    """
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False):
        super(Exponential, self).__init__(gp_link,analytical_mean,analytical_variance)

    def _preprocess_values(self,Y):
        return Y

    def _mass(self,gp,obs):
        """
        Mass (or density) function
        """
        return np.exp(-obs/self.gp_link.transf(gp))/self.gp_link.transf(gp)

    def _nlog_mass(self,gp,obs):
        """
        Negative logarithm of the un-normalized distribution: factors that are not a function of gp are omitted
        """
        return obs/self.gp_link.transf(gp) + np.log(self.gp_link.transf(gp))

    def _dnlog_mass_dgp(self,gp,obs):
        return ( 1./self.gp_link.transf(gp) - obs/self.gp_link.transf(gp)**2) * self.gp_link.dtransf_df(gp)

    def _d2nlog_mass_dgp2(self,gp,obs):
        fgp = self.gp_link.transf(gp)
        return (2*obs/fgp**3 - 1./fgp**2) * self.gp_link.dtransf_df(gp)**2 + ( 1./fgp - obs/fgp**2) * self.gp_link.d2transf_df2(gp)

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
        return self.gp_link.transf(gp)**2

    def _dvariance_dgp(self,gp):
        return 2*self.gp_link.transf(gp)*self.gp_link.dtransf_df(gp)

    def _d2variance_dgp2(self,gp):
        return 2 * (self.gp_link.dtransf_df(gp)**2 + self.gp_link.transf(gp)*self.gp_link.d2transf_df2(gp))
