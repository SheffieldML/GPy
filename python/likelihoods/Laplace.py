import numpy as np
import scipy as sp
import GPy
from scipy.linalg import cholesky, eig, inv
from functools import partial
from GPy.likelihoods.likelihood import likelihood
from GPy.util.linalg import pdinv,mdot
#import numpy.testing.assert_array_equal

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

        self.NORMAL_CONST = -((0.5 * self.N) * np.log(2 * np.pi))

        #Initial values for the GP variables
        self.Y = np.zeros((self.N,1))
        self.covariance_matrix = np.eye(self.N)
        self.precision = np.ones(self.N)[:,None]
        self.Z = 0
        self.YYT = None

    def predictive_values(self,mu,var):
        return self.likelihood_function.predictive_values(mu,var)

    def _get_params(self):
        return np.zeros(0)

    def _get_param_names(self):
        return []

    def _set_params(self,p):
        pass # TODO: Laplace likelihood might want to take some parameters...

    def _gradients(self,partial):
        return np.zeros(0) # TODO: Laplace likelihood might want to take some parameters...
        raise NotImplementedError

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
        self.Sigma_tilde_i = self.hess_hat_i #self.W #self.hess_hat_i - self.Ki
        #Do we really need to inverse Sigma_tilde_i? :(
        if self.likelihood_function.log_concave:
            (self.Sigma_tilde, _, _, _) = pdinv(self.Sigma_tilde_i)
        else:
            self.Sigma_tilde = inv(self.Sigma_tilde_i)
        #f_hat? should be f but we must have optimized for them I guess?
        Y_tilde = mdot(self.Sigma_tilde, self.hess_hat, self.f_hat)
        self.Z_tilde = np.exp(self.ln_z_hat - self.NORMAL_CONST
                              - 0.5*mdot(self.f_hat, self.hess_hat, self.f_hat)
                              + 0.5*mdot(Y_tilde.T, (self.Sigma_tilde_i, Y_tilde))
                              )

        self.Z = self.Z_tilde
        self.Y = Y_tilde
        self.covariance_matrix = self.Sigma_tilde
        self.precision = 1 / np.diag(self.Sigma_tilde)[:, None]
        self.YYT = np.dot(self.Y, self.Y.T)
        import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    def fit_full(self, K):
        """
        The laplace approximation algorithm
        For nomenclature see Rasmussen & Williams 2006
        :K: Covariance matrix
        """
        f = np.zeros((self.N, 1))
        (self.Ki, _, _, self.log_Kdet) = pdinv(K)
        LOG_K_CONST = -(0.5 * self.log_Kdet)
        OBJ_CONST = self.NORMAL_CONST + LOG_K_CONST
        #Find \hat(f) using a newton raphson optimizer for example
        #TODO: Add newton-raphson as subclass of optimizer class

        #FIXME: Can we get rid of this horrible reshaping?
        def obj(f):
            #f = f[:, None]
            res = -1 * (self.likelihood_function.link_function(self.data[:, 0], f) - 0.5 * mdot(f.T, (self.Ki, f)) + OBJ_CONST)
            return float(res)

        def obj_grad(f):
            #f = f[:, None]
            res = -1 * (self.likelihood_function.link_grad(self.data[:, 0], f) - mdot(self.Ki, f))
            return np.squeeze(res)

        def obj_hess(f):
            res = -1 * (-np.diag(self.likelihood_function.link_hess(self.data[:, 0], f)) - self.Ki)
            return np.squeeze(res)

        self.f_hat = sp.optimize.fmin_ncg(obj, f, fprime=obj_grad, fhess=obj_hess)

        #At this point get the hessian matrix
        self.W = -np.diag(self.likelihood_function.link_hess(self.data[:, 0], self.f_hat))
        self.hess_hat = self.Ki + self.W
        (self.hess_hat_i, _, _, self.log_hess_hat_det) = pdinv(self.hess_hat)

        #Check hess_hat is positive definite
        try:
            cholesky(self.hess_hat)
        except:
            raise ValueError("Must be positive definite")

        #Check its eigenvalues are positive
        eigenvalues = eig(self.hess_hat)
        if not np.all(eigenvalues > 0):
            raise ValueError("Eigen values not positive")

        #z_hat is how much we need to scale the normal distribution by to get the area of our approximation close to
        #the area of p(f)p(y|f) we do this by matching the height of the distributions at the mode
        #z_hat = -0.5*ln|H| - 0.5*ln|K| - 0.5*f_hat*K^{-1}*f_hat \sum_{n} ln p(y_n|f_n)
        #Unsure whether its log_hess or log_hess_i
        self.ln_z_hat = -0.5*np.log(self.log_hess_hat_det) - 0.5*self.log_Kdet + -1*self.likelihood_function.link_function(self.data[:,0], self.f_hat) - mdot(self.f_hat.T, (self.Ki, self.f_hat))
        import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

        return self._compute_GP_variables()
