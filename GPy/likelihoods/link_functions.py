# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
import scipy as sp
import pylab as pb
from ..util.plot import gpplot
from ..util.univariate_Gaussian import std_norm_pdf,std_norm_cdf

class link_function(object):
    """
    Link function class for doing non-Gaussian likelihoods approximation

    :param Y: observed output (Nx1 numpy.darray)
    ..Note:: Y values allowed depend on the likelihood_function used
    """
    def __init__(self):
        pass

class identity(link_function):
    def transf(self,mu):
        return mu

    def inv_transf(self,f):
        return f

    def log_inv_transf(self,f):
        return np.log(f)

class log(link_function):

    def transf(self,mu):
        return np.log(mu)

    def inv_transf(self,f):
        return np.exp(f)

    def log_inv_transf(self,f):
        return f

class log_ex_1(link_function):
    def transf(self,mu):
        return np.log(np.exp(mu) - 1)

    def inv_transf(self,f):
        return np.log(np.exp(f)+1)

    def log_inv_tranf(self,f):
        return np.log(np.log(np.exp(f)+1))

class probit(link_function):

    def inv_transf(self,f):
        return std_norm_cdf(f)

    def log_inv_transf(self,f):
        return np.log(std_norm_cdf(f))

