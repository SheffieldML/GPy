# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
from ..util.univariate_Gaussian import std_norm_pdf,std_norm_cdf

class LinkFunction(object):
    """
    Link function class for doing non-Gaussian likelihoods approximation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the likelihood_function used
    """
    def __init__(self):
        pass

class Identity(LinkFunction):
    def transf(self,mu):
        return mu

    def inv_transf(self,f):
        return f

    def log_inv_transf(self,f):
        return np.log(f)

class Log(LinkFunction):

    def transf(self,mu):
        return np.log(mu)

    def inv_transf(self,f):
        return np.exp(f)

    def log_inv_transf(self,f):
        return f

class Log_ex_1(LinkFunction):
    def transf(self,mu):
        return np.log(np.exp(mu) - 1)

    def inv_transf(self,f):
        return np.log(np.exp(f)+1)

    def log_inv_tranf(self,f):
        return np.log(np.log(np.exp(f)+1))

class Probit(LinkFunction):

    def inv_transf(self,f):
        return std_norm_cdf(f)

    def log_inv_transf(self,f):
        return np.log(std_norm_cdf(f))
