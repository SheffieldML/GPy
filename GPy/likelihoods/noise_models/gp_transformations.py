# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
import scipy as sp
import pylab as pb
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf,inv_std_norm_cdf

class GPTransformation(object):
    """
    Link function class for doing non-Gaussian likelihoods approximation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the likelihood_function used
    """
    def __init__(self):
        pass

class Identity(GPTransformation):
    """
    $$
    g(f) = f
    $$
    """
    #def transf(self,mu):
    #    return mu

    def transf(self,f):
        return f

    def dtransf_df(self,f):
        return 1.

    def d2transf_df2(self,f):
        return 0


class Probit(GPTransformation):
    """
    $$
    g(f) = \\Phi^{-1} (mu)
    $$
    """
    #def transf(self,mu):
    #    return inv_std_norm_cdf(mu)

    def transf(self,f):
        return std_norm_cdf(f)

    def dtransf_df(self,f):
        return std_norm_pdf(f)

    def d2transf_df2(self,f):
        return -f * std_norm_pdf(f)

class Log(GPTransformation):
    """
    $$
    g(f) = \log(\mu)
    $$
    """
    #def transf(self,mu):
    #    return np.log(mu)

    def transf(self,f):
        return np.exp(f)

    def dtransf_df(self,f):
        return np.exp(f)

    def d2transf_df2(self,f):
        return np.exp(f)

class Log_ex_1(GPTransformation):
    """
    $$
    g(f) = \log(\exp(\mu) - 1)
    $$
    """
    #def transf(self,mu):
    #    """
    #    function: output space -> latent space
    #    """
    #    return np.log(np.exp(mu) - 1)

    def transf(self,f):
        """
        function: latent space -> output space
        """
        return np.log(1.+np.exp(f))

    def dtransf_df(self,f):
        return np.exp(f)/(1.+np.exp(f))

    def d2transf_df2(self,f):
        aux = np.exp(f)/(1.+np.exp(f))
        return aux*(1.-aux)

class Reciprocal(GPTransformation):
    def transf(sefl,f):
        return 1./f

    def dtransf_df(self,f):
        return -1./f**2

    def d2transf_df2(self,f):
        return 2./f**3

class Heaviside(GPTransformation):
    """
    $$
    g(f) = I_{x \in A}
    $$
    """
    def transf(self,f):
        #transformation goes here
        return np.where(f>0, 1, -1)

    def dtransf_df(self,f):
        raise NotImplementedError, "this function is not differentiable!"

    def d2transf_df2(self,f):
        raise NotImplementedError, "this function is not differentiable!"
