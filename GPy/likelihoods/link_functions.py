# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
from ..util.univariate_Gaussian import std_norm_pdf,std_norm_cdf,inv_std_norm_cdf

class LinkFunction(object):
    """
    Link function class for doing non-Gaussian likelihoods approximation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the likelihood_function used
    """
    def __init__(self):
        pass

class Identity(LinkFunction):
    """
    $$
    g(f) = f
    $$
    """
    def transf(self,mu):
        return mu

    def inv_transf(self,f):
        return f

    def dinv_transf_df(self,f):
        return 1.

    def d2inv_transf_df2(self,f):
        return 0


class Probit(LinkFunction):
    """
    $$
    g(f) = \\Phi^{-1} (mu)
    $$
    """
    def transf(self,mu):
        return inv_std_norm_cdf(mu)

    def inv_transf(self,f):
        return std_norm_cdf(f)

    def dinv_transf_df(self,f):
        return std_norm_pdf(f)

    def d2inv_transf_df2(self,f):
        return -f * std_norm_pdf(f)

class Log(LinkFunction):
    """
    $$
    g(f) = \log(\mu)
    $$
    """
    def transf(self,mu):
        return np.log(mu)

    def inv_transf(self,f):
        return np.exp(f)

    def dinv_transf_df(self,f):
        return np.exp(f)

    def d2inv_transf_df2(self,f):
        return np.exp(f)

class Log_ex_1(LinkFunction):
    """
    $$
    g(f) = \log(\exp(\mu) - 1)
    $$
    """
    def transf(self,mu):
        """
        function: output space -> latent space
        """
        return np.log(np.exp(mu) - 1)

    def inv_transf(self,f):
        """
        function: latent space -> output space
        """
        return np.log(np.exp(f)+1)

    def dinv_transf_df(self,f):
        return np.exp(f)/(1.+np.exp(f))

    def d2inv_transf_df2(self,f):
        aux = np.exp(f)/(1.+np.exp(f))
        return aux*(1.-aux)
