import numpy as np
import scipy as sp
import GPy
from GPy.util.linalg import jitchol
from functools import partial
from GPy.likelihoods.likelihood import likelihood
from GPy.util.linalg import pdinv,mdot



class Laplace(likelihood):
    """Laplace approximation to a posterior"""

    def __init__(self,data,likelihood_function):
        """
        Laplace Approximation

        First find the moments \hat{f} and the hessian at this point (using Newton-Raphson)
        then find the z^{prime} which allows this to be a normalised gaussian instead of a
        non-normalized gaussian

        Finally we must compute the GP variables (i.e. generate some Y^{squiggle} and z^{squiggle}
        which makes a gaussian the same as the laplace approximation

        Arguments
        ---------

        :data: @todo
        :likelihood_function: @todo

        """
        self.data = data
        self.likelihood_function = likelihood_function

        #Inital values
        self.N, self.D = self.data.shape

    def _compute_GP_variables(self):
        """
        Generates data Y which would give the normal distribution identical to the laplace approximation

        GPy expects a likelihood to be gaussian, so need to caluclate the points Y^{squiggle} and Z^{squiggle}
        that makes the posterior match that found by a laplace approximation to a non-gaussian likelihood
        """
        z_hat = N(f_hat|f_hat, hess_hat) / self.height_unnormalised

    def fit_full(self, K):
        """
        The laplace approximation algorithm
        For nomenclature see Rasmussen & Williams 2006
        :K: Covariance matrix
        """
        f = np.zeros((self.N, 1))
        print K.shape
        print f.shape
        print self.data.shape
        (Ki, _, _, log_Kdet) = pdinv(K)
        obj_constant = (0.5 * log_Kdet) - ((0.5 * self.N) * np.log(2*np.pi))

        #Find \hat(f) using a newton raphson optimizer for example
        #TODO: Add newton-raphson as subclass of optimizer class

        #FIXME: Can we get rid of this horrible reshaping?
        def obj(f):
            f = f[:, None]
            res = -1 * (self.likelihood_function.link_function(self.data, f) - 0.5 * mdot(f.T, (Ki, f)) + obj_constant)
            return float(res)

        def obj_grad(f):
            f = f[:, None]
            res = -1 * (self.likelihood_function.link_grad(self.data, f) - mdot(Ki, f))
            return np.squeeze(res)

        def obj_hess(f):
            f = f[:, None]
            res = -1 * (np.diag(self.likelihood_function.link_hess(self.data, f)) - Ki)
            return np.squeeze(res)

        self.f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess)

        #At this point get the hessian matrix
        self.hess_hat = obj_hess(f_hat)

        #Need to add the constant as we previously were trying to avoid computing it (seems like a small overhead though...)
        self.height_unnormalised = obj(f_hat) #FIXME: Is it -1?

        return _compute_GP_variables()
