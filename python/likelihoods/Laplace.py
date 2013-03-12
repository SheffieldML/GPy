import nump as np
import GPy
from GPy.util.linalg import jitchol

class Laplace(GPy.likelihoods.likelihood):
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
        GPy.likelihoods.likelihood.__init__(self)

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
        raise NotImplementedError

    def fit_full(self, K):
        """
        The laplace approximation algorithm
        For nomenclature see Rasmussen & Williams 2006
        :K: Covariance matrix
        """
        self.f = np.zeros(self.N)

        #Find \hat(f) using a newton raphson optimizer for example

        #At this point get the hessian matrix

