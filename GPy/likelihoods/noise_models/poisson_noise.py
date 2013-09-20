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
    Y is expected to take values in {0,1,2,...}

    .. math::
        L(x) = \\exp(\\lambda) * \\frac{\\lambda^Y_i}{Y_i!}

    """
    def __init__(self,gp_link=None,analytical_mean=False,analytical_variance=False):
        #self.discrete = True
        #self.support_limits = (0,np.inf)

        #self.analytical_mean = False
        super(Poisson, self).__init__(gp_link,analytical_mean,analytical_variance)

    def _preprocess_values(self,Y): #TODO
        #self.scale = .5*Y.max()
        #self.shift = Y.mean()
        return Y #(Y - self.shift)/self.scale

    def _mass(self,gp,obs):
        """
        Mass (or density) function
        """
        #obs = obs*self.scale + self.shift
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

    def _dnlog_mass_dobs(self,obs,gp): #TODO not needed
        return special.psi(obs+1) -  np.log(self.gp_link.transf(gp))

    def _d2nlog_mass_dobs2(self,obs,gp=None): #TODO not needed
        return special.polygamma(1,obs)

    def _d2nlog_mass_dcross(self,obs,gp): #TODO not needed
        return -self.gp_link.dtransf_df(gp)/self.gp_link.transf(gp)

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
