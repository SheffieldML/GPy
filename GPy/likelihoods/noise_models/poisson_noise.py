# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import gp_transformations
from noise_distributions import NoiseDistribution

class Poisson(NoiseDistribution):
    """
    Poisson likelihood

    .. math::
        L(x) = \\exp(\\lambda) * \\frac{\\lambda^Y_i}{Y_i!}

    ..Note: Y is expected to take values in {0,1,2,...}
    """
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False):
        super(Poisson, self).__init__(gp_link,analytical_mean,analytical_variance)

    def _preprocess_values(self,Y): #TODO
        return Y

    def _mass(self,gp,obs):
        """
        Mass (or density) function
        """
        return stats.poisson.pmf(obs,self.gp_link.transf(gp))

    def _nlog_mass(self,gp,obs):
        """
        Negative logarithm of the un-normalized distribution: factors that are not a function of gp are omitted
        """
        return self.gp_link.transf(gp) - obs * np.log(self.gp_link.transf(gp)) + np.log(special.gamma(obs+1))

    def _dnlog_mass_dgp(self,gp,obs):
        return self.gp_link.dtransf_df(gp) * (1. - obs/self.gp_link.transf(gp))

    def _d2nlog_mass_dgp2(self,gp,obs):
        d2_df = self.gp_link.d2transf_df2(gp)
        transf = self.gp_link.transf(gp)
        return obs * ((self.gp_link.dtransf_df(gp)/transf)**2 - d2_df/transf) + d2_df

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
        return self.gp_link.transf(gp)

    def _dvariance_dgp(self,gp):
        return self.gp_link.dtransf_df(gp)

    def _d2variance_dgp2(self,gp):
        return self.gp_link.d2transf_df2(gp)
