# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats,special
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
from ..util.univariate_Gaussian import std_norm_pdf,std_norm_cdf
import link_functions

class LikelihoodFunction(object):
    """
    Likelihood class for doing Expectation propagation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the LikelihoodFunction used
    """
    def __init__(self,link):
        if link == self._analytical:
            self.moments_match = self._moments_match_analytical
        else:
            assert isinstance(link,link_functions.LinkFunction)
            self.link = link
            self.moments_match = self._moments_match_numerical

    def _preprocess_values(self,Y):
        return Y

    def _product(self,gp,obs,mu,sigma):
        """
        Product between the cavity distribution and a likelihood factor
        """
        return stats.norm.pdf(gp,loc=mu,scale=sigma) * self._mass(gp,obs)

    def _nlog_product_scaled(self,gp,obs,mu,sigma):
        """
        Negative log-product between the cavity distribution and a likelihood factor
        """
        return .5*(gp-mu)**2/sigma**2 + self._nlog_mass_scaled(gp,obs)

    def _dlog_product_dgp(self,gp,obs,mu,sigma):
        """
        Derivative wrt gp of the log-product between the cavity distribution and a likelihood factor
        """
        return -(gp - mu)/sigma**2 + self._dlog_mass_dgp(gp,obs)

    def _d2log_product_dgp2(self,gp,obs,mu,sigma):
        """
        Second derivative wrt gp of the log-product between the cavity distribution and a likelihood factor
        """
        return -1./sigma**2 + self._d2log_mass_dgp2(gp,obs)

    #def _dlog_product_dobs(self,obs,gp):
    #    return self._dlog_mass_dobs(obs,gp)

    #def _d2log_product_dobs2(self,obs,gp):
    #    return self._d2log_mass_dobs2(obs,gp)

    #def _d2log_product_dcross(self,gp,obs):

    def _gradient_log_product(self,x,mu,sigma):
        return np.array((self._dlog_product_dgp(gp=x[0],obs=x[1],mu=mu,sigma=sigma),self._dlog_mass_dobs(obs=x[1],gp=x[0])))

    def _hessian_log_product(self,x,mu,sigma):
        cross_derivative = self._d2log_mass_dcross(gp=x[0],obs=x[1])
        return np.array((self._d2log_product_dgp2(gp=x[0],obs=x[1],mu=mu,sigma=sigma),cross_derivative,cross_derivative,self._d2log_mass_dobs2(obs=x[1],gp=x[0]))).reshape(2,2)


    def _product_mode(self,obs,mu,sigma):
        """
        Brent's method to find the mode in the _product function (cavity x likelihood factor)
        """
        lower = -1 if obs == 0 else np.array([np.log(obs),mu]).min() #Lower limit #FIXME
        upper = 2*np.array([np.log(obs),mu]).max() #Upper limit #FIXME
        print lower,upper
        return sp.optimize.brent(self._nlog_product_scaled, args=(obs,mu,sigma), brack=(lower,upper)) #Better to work with _nlog_product than with _product

    def _moments_match_numerical(self,obs,tau,v):
        """
        Lapace approximation to calculate the moments mumerically.
        """
        mu = v/tau
        mu_hat = self._product_mode(obs,mu,np.sqrt(1./tau))
        sigma2_hat = 1./(tau - self._d2log_mass_dgp2(mu_hat,obs))
        Z_hat = np.exp(-.5*tau*(mu_hat-mu)**2) * self._mass(mu_hat,obs)*np.sqrt(tau*sigma2_hat)
        return Z_hat,mu_hat,sigma2_hat

    def _nlog_joint_posterior_scaled(x,mu,sigma):
        """
        x = np.array([gp,obs])
        """
        return self._product(x[0],x[1],mu,sigma)

    def _gradient_log_joint_posterior(x,mu,sigma):
        return self._dlog_product_dgp(x[0],x[1],mu,sigma) + self._dlog_mass_dgp(gp,obs), 

    def _predictive_values_numerical(self,mu,var):
        """
        Lapace approximation to calculate the predictive values.
        """
        mu = mu.flatten()
        var = var.flatten()
        tranf_mu = self.link.transf(mu)
        mu_hat = [self._product_mode(t_i,m_i,np.sqrt(v_i)) for t_i,mu_i,v_i in zip(transf_mu,mu,var)]
        sigma2_hat = [1./(1./var - self._d2log_mass_dgp2(m_i,t_i)) for m_i,t_i in zip(mu_hat,transf_mu)]


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
        self._analytical = link_functions.Probit
        if not link:
            link = self._analytical
        super(Binomial, self).__init__(link)

    def _mass(self,gp,obs):
        pass

    def _nlog_mass_scaled(self,gp,obs):
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

    def _predictive_values_analytical(self,mu,var):
        """
        Compute  mean, variance and conficence interval (percentiles 5 and 95) of the  prediction
        :param mu: mean of the latent variable
        :param var: variance of the latent variable
        """
        mu = mu.flatten()
        var = var.flatten()
        mean = stats.norm.cdf(mu/np.sqrt(1+var))
        norm_025 = [stats.norm.ppf(.025,m,v) for m,v in zip(mu,var)]
        norm_975 = [stats.norm.ppf(.975,m,v) for m,v in zip(mu,var)]
        p_025 = stats.norm.cdf(norm_025/np.sqrt(1+var))
        p_975 = stats.norm.cdf(norm_975/np.sqrt(1+var))
        return mean[:,None], np.nan*var, p_025[:,None], p_975[:,None] # TODO: var

class Poisson(LikelihoodFunction):
    """
    Poisson likelihood
    Y is expected to take values in {0,1,2,...}
    -----
    $$
    L(x) = \exp(\lambda) * \lambda**Y_i / Y_i!
    $$
    """
    def __init__(self,link=None):
        self._analytical = None
        if not link:
            link = link_functions.Log()
        super(Poisson, self).__init__(link)

    def _mass(self,gp,obs):
        """
        Mass (or density) function
        """
        return stats.poisson.pmf(obs,self.link.inv_transf(gp))

    def _nlog_mass_scaled(self,gp,obs):
        """
        Negative logarithm of the un-normalized distribution: factors that are not a function of gp are omitted
        """
        return self.link.inv_transf(gp) - obs * np.log(self.link.inv_transf(gp))

    def _dlog_mass_dgp(self,gp,obs):
        return self.link.dinv_transf_df(gp) * (obs/self.link.inv_transf(gp) - 1)

    def _d2log_mass_dgp2(self,gp,obs):
        d2_df = self.link.d2inv_transf_df2(gp)
        inv_transf = self.link.inv_transf(gp)
        return obs * ( d2_df/inv_transf - (self.link.dinv_transf_df(gp)/inv_transf)**2 ) - d2_df

    def _dlog_mass_dobs(self,obs,gp):
        return np.log(self.link.inv_transf(gp)) - special.psi(obs+1)

    def _d2log_mass_dobs2(self,obs,gp=None):
        return -special.polygamma(1,obs)

    def _d2log_mass_dcross(self,obs,gp):
        return self.link.dinv_transf_df(gp)/self.link.inv_transf(gp)

    def predictive_values(self,mu,var):
        """
        Compute  mean, and conficence interval (percentiles 5 and 95) of the  prediction
        """
        mean = self.link.transf(mu)
        tmp = stats.poisson.ppf(np.array([.025,.975]),mean)
        p_025 = tmp[:,0]
        p_975 = tmp[:,1]
        return mean,np.nan*mean,p_025,p_975 # better variance here TODO

        """
        simpson approximation
        width = 3./np.log(max(obs,2))
        A = opt - width #Grid's lower limit
        B = opt + width #Grid's Upper limit
        K =  10*int(np.log(max(obs,150))) #Number of points in the grid
        h = (B-A)/K # length of the intervals
        grid_x = np.hstack([np.linspace(opt-width,opt,K/2+1)[1:-1], np.linspace(opt,opt+width,K/2+1)]) # grid of points (X axis)
        x = np.hstack([A,B,grid_x[range(1,K,2)],grid_x[range(2,K-1,2)]]) # grid_x rearranged, just to make Simpson's algorithm easier
        _aux1 = self._product(A,obs,mu,sigma)
        _aux2 = self._product(B,obs,mu,sigma)
        _aux3 = 4*self._product(grid_x[range(1,K,2)],obs,mu,sigma)
        _aux4 = 2*self._product(grid_x[range(2,K-1,2)],obs,mu,sigma)
        zeroth = np.hstack((_aux1,_aux2,_aux3,_aux4)) # grid of points (Y axis) rearranged
        first = zeroth*x
        second = first*x
        Z_hat = sum(zeroth)*h/3 # Zero-th moment
        mu_hat = sum(first)*h/(3*Z_hat) # First moment
        m2 = sum(second)*h/(3*Z_hat) # Second moment
        sigma2_hat = m2 - mu_hat**2 # Second central moment
        return float(Z_hat), float(mu_hat), float(sigma2_hat)
        """

