# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats
import scipy as sp
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf,inv_std_norm_cdf

_exp_lim_val = np.finfo(np.float64).max
_lim_val = np.log(_exp_lim_val)

class GPTransformation(object):
    """
    Link function class for doing non-Gaussian likelihoods approximation

    :param Y: observed output (Nx1 numpy.darray)

    .. note:: Y values allowed depend on the likelihood_function used

    """
    def __init__(self):
        pass

    def transf(self,f):
        """
        Gaussian process tranformation function, latent space -> output space
        """
        raise NotImplementedError

    def dtransf_df(self,f):
        """
        derivative of transf(f) w.r.t. f
        """
        raise NotImplementedError

    def d2transf_df2(self,f):
        """
        second derivative of transf(f) w.r.t. f
        """
        raise NotImplementedError

    def d3transf_df3(self,f):
        """
        third derivative of transf(f) w.r.t. f
        """
        raise NotImplementedError

class Identity(GPTransformation):
    """
    .. math::

        g(f) = f

    """
    def transf(self,f):
        return f

    def dtransf_df(self,f):
        return np.ones_like(f)

    def d2transf_df2(self,f):
        return np.zeros_like(f)

    def d3transf_df3(self,f):
        return np.zeros_like(f)


class Probit(GPTransformation):
    """
    .. math::

        g(f) = \\Phi^{-1} (mu)
    
    """
    def transf(self,f):
        return std_norm_cdf(f)

    def dtransf_df(self,f):
        return std_norm_pdf(f)

    def d2transf_df2(self,f):
        #FIXME
        return -f * std_norm_pdf(f)

    def d3transf_df3(self,f):
        #FIXME
        f2 = f**2
        return -(1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(f2))*(1-f2)


class Cloglog(GPTransformation):
    """
    Complementary log-log link
    .. math::

        p(f) = 1 - e^{-e^f}

        or

        f = \log (-\log(1-p))
    
    """
    def transf(self,f):
        return 1-np.exp(-np.exp(f))

    def dtransf_df(self,f):
        return np.exp(f-np.exp(f))

    def d2transf_df2(self,f):
        ef = np.exp(f)
        return -np.exp(f-ef)*(ef-1.)

    def d3transf_df3(self,f):
        ef = np.exp(f)
        return np.exp(f-ef)*(1.-3*ef + ef**2)


class Log(GPTransformation):
    """
    .. math::

        g(f) = \\log(\\mu)

    """
    def transf(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

    def dtransf_df(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

    def d2transf_df2(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

    def d3transf_df3(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

class Log_ex_1(GPTransformation):
    """
    .. math::

        g(f) = \\log(\\exp(\\mu) - 1)

    """
    def transf(self,f):
        return np.log(1.+np.exp(f))

    def dtransf_df(self,f):
        return np.exp(f)/(1.+np.exp(f))

    def d2transf_df2(self,f):
        aux = np.exp(f)/(1.+np.exp(f))
        return aux*(1.-aux)

    def d3transf_df3(self,f):
        aux = np.exp(f)/(1.+np.exp(f))
        daux_df = aux*(1.-aux)
        return daux_df - (2.*aux*daux_df)

class Reciprocal(GPTransformation):
    def transf(self,f):
        return 1./f

    def dtransf_df(self,f):
        return -1./(f**2)

    def d2transf_df2(self,f):
        return 2./(f**3)

    def d3transf_df3(self,f):
        return -6./(f**4)

class Heaviside(GPTransformation):
    """

    .. math::

        g(f) = I_{x \\in A}

    """
    def transf(self,f):
        #transformation goes here
        return np.where(f>0, 1, 0)

    def dtransf_df(self,f):
        raise NotImplementedError, "This function is not differentiable!"

    def d2transf_df2(self,f):
        raise NotImplementedError, "This function is not differentiable!"
