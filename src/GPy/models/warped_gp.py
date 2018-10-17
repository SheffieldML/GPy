# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
#from ..util.warping_functions import *
from ..core import GP
from .. import likelihoods
from paramz import ObsAr
#from GPy.util.warping_functions import TanhFunction
from ..util.warping_functions import TanhFunction
from GPy import kern

class WarpedGP(GP):
    """
    This defines a GP Regression model that applies a 
    warping function to the output.
    """
    def __init__(self, X, Y, kernel=None, warping_function=None, warping_terms=3, normalizer=False):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])
        if warping_function == None:
            self.warping_function = TanhFunction(warping_terms)
            self.warping_params = (np.random.randn(self.warping_function.n_terms * 3 + 1) * 1)
        else:
            self.warping_function = warping_function
        likelihood = likelihoods.Gaussian()
        super(WarpedGP, self).__init__(X, Y.copy(), likelihood=likelihood, kernel=kernel, normalizer=normalizer)
        self.Y_normalized = self.Y_normalized.copy()
        self.Y_untransformed = self.Y_normalized.copy()
        self.predict_in_warped_space = True
        self.link_parameter(self.warping_function)

    def set_XY(self, X=None, Y=None):
        super(WarpedGP, self).set_XY(X, Y)
        self.Y_untransformed = self.Y_normalized.copy()
        self.update_model(True)

    def parameters_changed(self):
        """
        Notice that we update the warping function gradients here.
        """
        self.Y_normalized[:] = self.transform_data()
        super(WarpedGP, self).parameters_changed()
        Kiy = self.posterior.woodbury_vector.flatten()
        self.warping_function.update_grads(self.Y_untransformed, Kiy)

    def transform_data(self):
        Y = self.warping_function.f(self.Y_untransformed.copy()).copy()
        return Y

    def log_likelihood(self):
        """
        Notice we add the jacobian of the warping function here.
        """
        ll = GP.log_likelihood(self)
        jacobian = self.warping_function.fgrad_y(self.Y_untransformed)
        return ll + np.log(jacobian).sum()

    def plot_warping(self):
        self.warping_function.plot(self.Y_untransformed.min(), self.Y_untransformed.max())

    def _get_warped_term(self, mean, std, gh_samples, pred_init=None):
        arg1 = gh_samples.dot(std.T) * np.sqrt(2)
        arg2 = np.ones(shape=gh_samples.shape).dot(mean.T)
        return self.warping_function.f_inv(arg1 + arg2, y=pred_init)

    def _get_warped_mean(self, mean, std, pred_init=None, deg_gauss_hermite=20):
        """
        Calculate the warped mean by using Gauss-Hermite quadrature.
        """
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:, None]
        gh_weights = gh_weights[None, :]
        return gh_weights.dot(self._get_warped_term(mean, std, gh_samples)) / np.sqrt(np.pi)

    def _get_warped_variance(self, mean, std, pred_init=None, deg_gauss_hermite=20):
        """
        Calculate the warped variance by using Gauss-Hermite quadrature.
        """
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:, None]
        gh_weights = gh_weights[None, :]
        arg1 = gh_weights.dot(self._get_warped_term(mean, std, gh_samples, 
                                                    pred_init=pred_init) ** 2) / np.sqrt(np.pi)
        arg2 = self._get_warped_mean(mean, std, pred_init=pred_init,
                                     deg_gauss_hermite=deg_gauss_hermite)
        return arg1 - (arg2 ** 2)

    def predict(self, Xnew, kern=None, pred_init=None, Y_metadata=None,
                median=False, deg_gauss_hermite=20, likelihood=None):
        """
        Prediction results depend on:
        - The value of the self.predict_in_warped_space flag
        - The median flag passed as argument
        The likelihood keyword is never used, it is just to follow the plotting API.
        """
        #mu, var = GP._raw_predict(self, Xnew)
        # now push through likelihood
        #mean, var = self.likelihood.predictive_values(mu, var)
        
        mean, var = super(WarpedGP, self).predict(Xnew, kern=kern, full_cov=False, likelihood=likelihood)


        if self.predict_in_warped_space:
            std = np.sqrt(var)
            if median:
                wmean = self.warping_function.f_inv(mean, y=pred_init)
            else:
                wmean = self._get_warped_mean(mean, std, pred_init=pred_init,
                                              deg_gauss_hermite=deg_gauss_hermite).T
            wvar = self._get_warped_variance(mean, std, pred_init=pred_init,
                                             deg_gauss_hermite=deg_gauss_hermite).T
        else:
            wmean = mean
            wvar = var
        return wmean, wvar

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, likelihood=None, kern=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.input_dim), np.ndarray (Xnew x self.input_dim)]
        """
        qs = super(WarpedGP, self).predict_quantiles(X, quantiles, Y_metadata=Y_metadata, likelihood=likelihood, kern=kern)
        if self.predict_in_warped_space:
            return [self.warping_function.f_inv(q) for q in qs]
        return qs
        #m, v = self._raw_predict(X,  full_cov=False)
        #if self.normalizer is not None:
        #    m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)
        #a, b = self.likelihood.predictive_quantiles(m, v, quantiles, Y_metadata)
        #if not self.predict_in_warped_space:
        #    return [a, b]
        #new_a = self.warping_function.f_inv(a)
        #new_b = self.warping_function.f_inv(b)
        #return [new_a, new_b]

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        """
        Calculation of the log predictive density. Notice we add
        the jacobian of the warping function here.

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        mu_star, var_star = self._raw_predict(x_test)
        fy = self.warping_function.f(y_test)
        ll_lpd = self.likelihood.log_predictive_density(fy, mu_star, var_star, Y_metadata=Y_metadata)
        return ll_lpd + np.log(self.warping_function.fgrad_y(y_test))


if __name__ == '__main__':
    X = np.random.randn(100, 1)
    Y = np.sin(X) + np.random.randn(100, 1)*0.05

    m = WarpedGP(X, Y)
