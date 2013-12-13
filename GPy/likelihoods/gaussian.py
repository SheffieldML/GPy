import numpy as np
from likelihood import likelihood
from ..util.linalg import jitchol


class Gaussian(likelihood):
    """
    Likelihood class for doing Expectation propagation

    :param data: observed output
    :type data: Nx1 numpy.darray
    :param variance: noise parameter
    :param normalize:  whether to normalize the data before computing (predictions will be in original scales)
    :type normalize: False|True
    """
    def __init__(self, data, variance=1., normalize=False):
        self.is_heteroscedastic = False
        self.num_params = 1
        self.Z = 0. # a correction factor which accounts for the approximation made
        N, self.output_dim = data.shape

        # normalization
        if normalize:
            self._offset = data.mean(0)[None, :]
            self._scale = data.std(0)[None, :]
            # Don't scale outputs which have zero variance to zero.
            self._scale[np.nonzero(self._scale == 0.)] = 1.0e-3
        else:
            self._offset = np.zeros((1, self.output_dim))
            self._scale = np.ones((1, self.output_dim))

        self.set_data(data)

        self._variance = np.asarray(variance) + 1.
        self._set_params(np.asarray(variance))

        super(Gaussian, self).__init__()

    def set_data(self, data):
        self.data = data
        self.N, D = data.shape
        assert D == self.output_dim
        self.Y = (self.data - self._offset) / self._scale
        if D > self.N:
            self.YYT = np.dot(self.Y, self.Y.T)
            self.trYYT = np.trace(self.YYT)
            self.YYT_factor = jitchol(self.YYT)
        else:
            self.YYT = None
            self.trYYT = np.sum(np.square(self.Y))
            self.YYT_factor = self.Y

    def _get_params(self):
        return np.asarray(self._variance)

    def _get_param_names(self):
        return ["noise_variance"]

    def _set_params(self, x):
        x = np.float64(x)
        if np.all(self._variance != x):
            if x == 0.:#special case of zero noise
                self.precision = np.inf
                self.V = None
            else:
                self.precision = 1. / x
                self.V = (self.precision) * self.Y
                self.VVT_factor = self.precision * self.YYT_factor
            self.covariance_matrix = np.eye(self.N) * x
            self._variance = x

    def predictive_values(self, mu, var, full_cov, **likelihood_args):
        """
        Un-normalize the prediction and add the likelihood variance, then return the 5%, 95% interval
        """
        mean = mu * self._scale + self._offset
        if full_cov:
            if self.output_dim > 1:
                raise NotImplementedError, "TODO"
                # Note. for output_dim>1, we need to re-normalise all the outputs independently.
                # This will mess up computations of diag(true_var), below.
                # note that the upper, lower quantiles should be the same shape as mean
            # Augment the output variance with the likelihood variance and rescale.
            true_var = (var + np.eye(var.shape[0]) * self._variance) * self._scale ** 2
            _5pc = mean - 2.*np.sqrt(np.diag(true_var))
            _95pc = mean + 2.*np.sqrt(np.diag(true_var))
        else:
            true_var = (var + self._variance) * self._scale ** 2
            _5pc = mean - 2.*np.sqrt(true_var)
            _95pc = mean + 2.*np.sqrt(true_var)
        return mean, true_var, _5pc, _95pc

    def log_predictive_density(self, y_test, mu_star, var_star):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param mu_star: predictive mean of gaussian p(f_{*}|mu_{*}, var_{*})
        :type mu_star: (Nx1) array
        :param var_star: predictive variance of gaussian p(f_{*}|mu_{*}, var_{*})
        :type var_star: (Nx1) array

        .. Note:
            Works as if each test point was provided individually, i.e. not full_cov
        """
        y_rescaled = (y_test - self._offset)/self._scale
        return -0.5*np.log(2*np.pi) -0.5*np.log(var_star + self._variance) -0.5*(np.square(y_rescaled - mu_star))/(var_star + self._variance)

    def _gradients(self, partial):
        return np.sum(partial)
