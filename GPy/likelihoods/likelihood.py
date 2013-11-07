import numpy as np
import copy
from ..core.parameterized import Parameterized

class likelihood(Parameterized):
    """
    The atom for a likelihood class

    This object interfaces the GP and the data. The most basic likelihood
    (Gaussian) inherits directly from this, as does the EP algorithm

    Some things must be defined for this to work properly:

        - self.Y : the effective Gaussian target of the GP
        - self.N, self.D : Y.shape
        - self.covariance_matrix : the effective (noise) covariance of the GP targets
        - self.Z : a factor which gets added to the likelihood (0 for a Gaussian, Z_EP for EP)
        - self.is_heteroscedastic : enables significant computational savings in GP
        - self.precision : a scalar or vector representation of the effective target precision
        - self.YYT : (optional) = np.dot(self.Y, self.Y.T) enables computational savings for D>N
        - self.V : self.precision * self.Y

    """
    def __init__(self):
        Parameterized.__init__(self)
        self.dZ_dK = 0

    def _get_params(self):
        raise NotImplementedError

    def _get_param_names(self):
        raise NotImplementedError

    def _set_params(self, x):
        raise NotImplementedError

    def fit_full(self, K):
        """
        No approximations needed by default
        """
        pass

    def restart(self):
        """
        No need to restart if not an approximation
        """
        pass

    def _gradients(self, partial):
        raise NotImplementedError

    def predictive_values(self, mu, var):
        raise NotImplementedError

    def log_predictive_density(self, y_test, mu_star, var_star):
        """
        Calculation of the predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param mu_star: predictive mean of gaussian p(f_{*}|mu_{*}, var_{*})
        :type mu_star: (Nx1) array
        :param var_star: predictive variance of gaussian p(f_{*}|mu_{*}, var_{*})
        :type var_star: (Nx1) array
        """
        raise NotImplementedError
