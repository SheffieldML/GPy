# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
#import pylab as pb
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import link_functions
from likelihood_functions import NoiseModel

class Poisson(NoiseModel):
    """
    Poisson likelihood
    Y is expected to take values in {0,1,2,...}
    -----
    $$
    L(x) = \exp(\lambda) * \lambda**Y_i / Y_i!
    $$
    """
    def __init__(self,link=None,analytical_moments=False):
        #self.discrete = True
        #self.support_limits = (0,np.inf)

        #self.analytical_moments = False
        super(Poisson, self).__init__(link,analytical_moments)

    def _preprocess_values(self,Y): #TODO
        #self.scale = .5*Y.max()
        #self.shift = Y.mean()
        return Y #(Y - self.shift)/self.scale

    def _mass(self,gp,obs):
        """
        Mass (or density) function
        """
        #obs = obs*self.scale + self.shift
        return stats.poisson.pmf(obs,self.link.inv_transf(gp))

    def _nlog_mass(self,gp,obs):
        """
        Negative logarithm of the un-normalized distribution: factors that are not a function of gp are omitted
        """
        return self.link.inv_transf(gp) - obs * np.log(self.link.inv_transf(gp)) + np.log(special.gamma(obs+1))

    def _dnlog_mass_dgp(self,gp,obs):
        return self.link.dinv_transf_df(gp) * (1. - obs/self.link.inv_transf(gp))

    def _d2nlog_mass_dgp2(self,gp,obs):
        d2_df = self.link.d2inv_transf_df2(gp)
        inv_transf = self.link.inv_transf(gp)
        return obs * ((self.link.dinv_transf_df(gp)/inv_transf)**2 - d2_df/inv_transf) + d2_df

    def _dnlog_mass_dobs(self,obs,gp): #TODO not needed
        return special.psi(obs+1) -  np.log(self.link.inv_transf(gp))

    def _d2nlog_mass_dobs2(self,obs,gp=None): #TODO not needed
        return special.polygamma(1,obs)

    def _d2nlog_mass_dcross(self,obs,gp): #TODO not needed
        return -self.link.dinv_transf_df(gp)/self.link.inv_transf(gp)

    def _mean(self,gp):
        """
        Mass (or density) function
        """
        return self.link.inv_transf(gp)

    #def _variance(self,gp):
    #    return self.link.inv_transf(gp)

    def _dmean_dgp(self,gp):
        return self.link.dinv_transf_df(gp)

    def _d2mean_dgp2(self,gp):
        return self.link.d2inv_transf_df2(gp)

    def _variance(self,gp):
        """
        Mass (or density) function
        """
        return self.link.inv_transf(gp)

    #def _variance(self,gp):
    #    return self.link.inv_transf(gp)

    def _dvariance_dgp(self,gp):
        return self.link.dinv_transf_df(gp)

    def _d2variance_dgp2(self,gp):
        return self.link.d2inv_transf_df2(gp)
