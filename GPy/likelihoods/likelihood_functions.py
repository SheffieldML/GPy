# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
#from . import EP

class likelihood_function:
    """
    Likelihood class for doing Expectation propagation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the likelihood_function used
    """
    def __init__(self,location=0,scale=1):
        self.location = location
        self.scale = scale

class probit(likelihood_function):
    """
    Probit likelihood
    Y is expected to take values in {-1,1}
    -----
    $$
    L(x) = \\Phi (Y_i*f_i)
    $$
    """

    def moments_match(self,data_i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        # TODO: some version of assert np.sum(np.abs(Y)-1) == 0, "Output values must be either -1 or 1"
        z = data_i*v_i/np.sqrt(tau_i**2 + tau_i)
        Z_hat = stats.norm.cdf(z)
        phi = stats.norm.pdf(z)
        mu_hat = v_i/tau_i + data_i*phi/(Z_hat*np.sqrt(tau_i**2 + tau_i))
        sigma2_hat = 1./tau_i - (phi/((tau_i**2+tau_i)*Z_hat))*(z+phi/Z_hat)
        return Z_hat, mu_hat, sigma2_hat

    def predictive_values(self,mu,var):
        """
        Compute  mean, and conficence interval (percentiles 5 and 95) of the  prediction
        """
        mu = mu.flatten()
        var = var.flatten()
        mean = stats.norm.cdf(mu/np.sqrt(1+var))
        p_05 = np.zeros([mu.size])
        p_95 = np.ones([mu.size])
        return mean, p_05, p_95

class Poisson(likelihood_function):
    """
    Poisson likelihood
    Y is expected to take values in {0,1,2,...}
    -----
    $$
    L(x) = \exp(\lambda) * \lambda**Y_i / Y_i!
    $$
    """
    def moments_match(self,i,tau_i,v_i):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        mu = v_i/tau_i
        sigma = np.sqrt(1./tau_i)
        def poisson_norm(f):
            """
            Product of the likelihood and the cavity distribution
            """
            pdf_norm_f = stats.norm.pdf(f,loc=mu,scale=sigma)
            rate = np.exp( (f*self.scale)+self.location)
            poisson = stats.poisson.pmf(float(self.Y[i]),rate)
            return pdf_norm_f*poisson

        def log_pnm(f):
            """
            Log of poisson_norm
            """
            return -(-.5*(f-mu)**2/sigma**2 - np.exp( (f*self.scale)+self.location) + ( (f*self.scale)+self.location)*self.Y[i])

        """
        Golden Search and Simpson's Rule
        --------------------------------
        Simpson's Rule is used to calculate the moments mumerically, it needs a grid of points as input.
        Golden Search is used to find the mode in the poisson_norm distribution and define around it the grid for Simpson's Rule
        """
        #TODO golden search & simpson's rule can be defined in the general likelihood class, rather than in each specific case.

        #Golden search
        golden_A = -1 if self.Y[i] == 0 else np.array([np.log(self.Y[i]),mu]).min() #Lower limit
        golden_B = np.array([np.log(self.Y[i]),mu]).max() #Upper limit
        golden_A = (golden_A - self.location)/self.scale
        golden_B = (golden_B - self.location)/self.scale
        opt = sp.optimize.golden(log_pnm,brack=(golden_A,golden_B)) #Better to work with log_pnm than with poisson_norm

        # Simpson's approximation
        width = 3./np.log(max(self.Y[i],2))
        A = opt - width #Lower limit
        B = opt + width #Upper limit
        K =  10*int(np.log(max(self.Y[i],150))) #Number of points in the grid, we DON'T want K to be the same number for every case
        h = (B-A)/K # length of the intervals
        grid_x = np.hstack([np.linspace(opt-width,opt,K/2+1)[1:-1], np.linspace(opt,opt+width,K/2+1)]) # grid of points (X axis)
        x = np.hstack([A,B,grid_x[range(1,K,2)],grid_x[range(2,K-1,2)]]) # grid_x rearranged, just to make Simpson's algorithm easier
        zeroth = np.hstack([poisson_norm(A),poisson_norm(B),[4*poisson_norm(f) for f in grid_x[range(1,K,2)]],[2*poisson_norm(f) for f in grid_x[range(2,K-1,2)]]]) # grid of points (Y axis) rearranged like x
        first = zeroth*x
        second = first*x
        Z_hat = sum(zeroth)*h/3 # Zero-th moment
        mu_hat = sum(first)*h/(3*Z_hat) # First moment
        m2 = sum(second)*h/(3*Z_hat) # Second moment
        sigma2_hat = m2 - mu_hat**2 # Second central moment
        return float(Z_hat), float(mu_hat), float(sigma2_hat)

    def predictive_values(self,mu,var):
        """
        Compute  mean, and conficence interval (percentiles 5 and 95) of the  prediction
        """
        mean = np.exp(mu*self.scale + self.location)
        tmp = stats.poisson.ppf(np.array([.05,.95]),mu)
        p_05 = tmp[:,0]
        p_95 = tmp[:,1]
        return mean,p_05,p_95
