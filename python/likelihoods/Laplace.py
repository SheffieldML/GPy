import numpy as np
import scipy as sp
import GPy
from GPy.util.linalg import jitchol
from functools import partial
from GPy.likelihoods.likelihood import likelihood
from GPy.util.linalg import pdinv,mdot
from scipy.stats import norm


class Laplace(likelihood):
    """Laplace approximation to a posterior"""

    def __init__(self, data, likelihood_function):
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

        Given we are approximating $p(y|f)p(f)$ with a normal distribution (given $p(y|f)$ is not normal)
        then we have a rescaled normal distibution z*N(f|f_hat,hess_hat^-1) with the same area as p(y|f)p(f)
        due to the z rescaling.

        at the moment the data Y correspond to the normal approximation z*N(f|f_hat,hess_hat^1)

        This function finds the data D=(Y_tilde,X) that would produce z*N(f|f_hat,hess_hat^1)
        giving a normal approximation of z_tilde*p(Y_tilde|f,X)p(f)

        $$\tilde{Y} = \tilde{\Sigma} Hf$$
        where
        $$\tilde{\Sigma}^{-1} = H - K^{-1}$$
        i.e. $$\tilde{\Sigma}^{-1} = diag(\nabla\nabla \log(y|f))$$
        since $diag(\nabla\nabla \log(y|f)) = H - K^{-1}$
        and $$\ln \tilde{z} = \ln z + \frac{N}{2}\ln 2\pi + \frac{1}{2}\tilde{Y}\tilde{\Sigma}^{-1}\tilde{Y}$$

        """
        self.Sigma_tilde = self.hess_hat -
        self.Z =
        #self.Y =
        #self.YYT =
        #self.covariance_matrix =
        #self.precision =

    def fit_full(self, K):
        """
        The laplace approximation algorithm
        For nomenclature see Rasmussen & Williams 2006
        :K: Covariance matrix
        """
        f = np.zeros((self.N, 1))
        #K = np.diag(np.ones(self.N))
        (self.Ki, _, _, self.log_Kdet) = pdinv(K)
        obj_constant = (0.5 * self.log_Kdet) - ((0.5 * self.N) * np.log(2 * np.pi))

        #Find \hat(f) using a newton raphson optimizer for example
        #TODO: Add newton-raphson as subclass of optimizer class

        #FIXME: Can we get rid of this horrible reshaping?
        def obj(f):
            f = f[:, None]
            res = -1 * (self.likelihood_function.link_function(self.data, f) - 0.5 * mdot(f.T, (self.Ki, f)) + obj_constant)
            return float(res)

        def obj_grad(f):
            f = f[:, None]
            res = -1 * (self.likelihood_function.link_grad(self.data, f) - mdot(self.Ki, f))
            return np.squeeze(res)

        def obj_hess(f):
            f = f[:, None]
            res = -1 * (np.diag(self.likelihood_function.link_hess(self.data, f)) - self.Ki)
            return np.squeeze(res)

        self.f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess)
        print self.f_hat

        #At this point get the hessian matrix
        self.hess_hat = obj_hess(self.f_hat)

        #Need to add the constant as we previously were trying to avoid computing it (seems like a small overhead though...)
        self.height_unnormalised = -1*obj(self.f_hat) #FIXME: Is it - obj constant and *-1?
        #z_hat is how much we need to scale the normal distribution by to get the area of our approximation close to
        #the area of p(f)p(y|f) we do this by matching the height of the distributions at the mode
        #z_hat = -0.5*ln|H| - 0.5*ln|K| - 0.5*f_hat*K^{-1}*f_hat \sum_{n} ln p(y_n|f_n)
        self.z_hat = np.exp(-0.5*np.log(np.linalg.det(hess_hat)) + self.height_unnormalised)

        return self._compute_GP_variables()
