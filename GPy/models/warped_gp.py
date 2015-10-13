# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.warping_functions import *
from ..core import GP
from .. import likelihoods
from GPy.util.warping_functions import TanhWarpingFunction_d
from GPy import kern

class WarpedGP(GP):
    def __init__(self, X, Y, kernel=None, warping_function=None, warping_terms=3):

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        if warping_function == None:
            self.warping_function = TanhWarpingFunction_d(warping_terms)
            self.warping_params = (np.random.randn(self.warping_function.n_terms * 3 + 1) * 1)
        else:
            self.warping_function = warping_function

        self.scale_data = False
        if self.scale_data:
            Y = self._scale_data(Y)
        #self.has_uncertain_inputs = False
        self.Y_untransformed = Y.copy()
        self.predict_in_warped_space = True
        likelihood = likelihoods.Gaussian()

        GP.__init__(self, X, self.transform_data(), likelihood=likelihood, kernel=kernel)
        self.link_parameter(self.warping_function)

    def _scale_data(self, Y):
        self._Ymax = Y.max()
        self._Ymin = Y.min()
        return (Y - self._Ymin) / (self._Ymax - self._Ymin) - 0.5

    def _unscale_data(self, Y):
        return (Y + 0.5) * (self._Ymax - self._Ymin) + self._Ymin

    def parameters_changed(self):
        self.Y[:] = self.transform_data()
        super(WarpedGP, self).parameters_changed()

        Kiy = self.posterior.woodbury_vector.flatten()

        grad_y = self.warping_function.fgrad_y(self.Y_untransformed)
        grad_y_psi, grad_psi = self.warping_function.fgrad_y_psi(self.Y_untransformed,
                                                                 return_covar_chain=True)
        djac_dpsi = ((1.0 / grad_y[:, :, None, None]) * grad_y_psi).sum(axis=0).sum(axis=0)
        dquad_dpsi = (Kiy[:, None, None, None] * grad_psi).sum(axis=0).sum(axis=0)

        warping_grads = -dquad_dpsi + djac_dpsi

        self.warping_function.psi.gradient[:] = warping_grads[:, :-1]
        self.warping_function.d.gradient[:] = warping_grads[0, -1]

    def transform_data(self):
        Y = self.warping_function.f(self.Y_untransformed.copy()).copy()
        return Y

    def log_likelihood(self):
        ll = GP.log_likelihood(self)
        jacobian = self.warping_function.fgrad_y(self.Y_untransformed)
        return ll + np.log(jacobian).sum()

    def plot_warping(self):
        self.warping_function.plot(self.Y_untransformed.min(), self.Y_untransformed.max())

    def _get_warped_term(self, mean, std, gh_samples, pred_init=None):
        arg1 = gh_samples.dot(std.T) * np.sqrt(2)
        arg2 = np.ones(shape=gh_samples.shape).dot(mean.T)
        return self.warping_function.f_inv(arg1 + arg2, y=pred_init)

    def _get_warped_mean(self, mean, std, pred_init=None, deg_gauss_hermite=100):
        """
        Calculate the warped mean by using Gauss-Hermite quadrature.
        """
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:,None]
        gh_weights = gh_weights[None,:]
        return gh_weights.dot(self._get_warped_term(mean, std, gh_samples)) / np.sqrt(np.pi)

    def _get_warped_variance(self, mean, std, pred_init=None, deg_gauss_hermite=100):
        """
        Calculate the warped variance by using Gauss-Hermite quadrature.
        """
        gh_samples, gh_weights = np.polynomial.hermite.hermgauss(deg_gauss_hermite)
        gh_samples = gh_samples[:,None]
        gh_weights = gh_weights[None,:]
        arg1 = gh_weights.dot(self._get_warped_term(mean, std, gh_samples, 
                                                    pred_init=pred_init) ** 2) / np.sqrt(np.pi)
        arg2 = self._get_warped_mean(mean, std, pred_init=pred_init,
                                     deg_gauss_hermite=deg_gauss_hermite)
        return arg1 - (arg2 ** 2)

    def predict(self, Xnew, which_parts='all', pred_init=None, full_cov=False, Y_metadata=None,
                median=False, deg_gauss_hermite=100):
        # normalize X values
        # Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        mu, var = GP._raw_predict(self, Xnew)

        # now push through likelihood
        mean, var = self.likelihood.predictive_values(mu, var)

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

        if self.scale_data:
            pred = self._unscale_data(pred)

        return wmean, wvar

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.input_dim), np.ndarray (Xnew x self.input_dim)]
        """
        m, v = self._raw_predict(X,  full_cov=False)
        if self.normalizer is not None:
            m, v = self.normalizer.inverse_mean(m), self.normalizer.inverse_variance(v)
        a, b = self.likelihood.predictive_quantiles(m, v, quantiles, Y_metadata)
        #return [a, b]
        if not self.predict_in_warped_space:
            return [a, b]
        #print a.shape
        new_a = self.warping_function.f_inv(a)
        new_b = self.warping_function.f_inv(b)

        return [new_a, new_b]
        #return self.likelihood.predictive_quantiles(m, v, quantiles, Y_metadata)


if __name__ == '__main__':
    X = np.random.randn(100, 1)
    Y = np.sin(X) + np.random.randn(100, 1)*0.05

    m = WarpedGP(X, Y)
