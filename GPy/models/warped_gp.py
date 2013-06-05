# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..util.warping_functions import *
from ..core import GP
from .. import likelihoods
from GPy.util.warping_functions import TanhWarpingFunction_d
from GPy import kern

class WarpedGP(GP):
    def __init__(self, X, Y, kernel=None, warping_function=None, warping_terms=3, normalize_X=False, normalize_Y=False):

        if kernel is None:
            kernel = kern.rbf(X.shape[1])

        if warping_function == None:
            self.warping_function = TanhWarpingFunction_d(warping_terms)
            self.warping_params = (np.random.randn(self.warping_function.n_terms * 3 + 1,) * 1)

        Y = self._scale_data(Y)
        self.has_uncertain_inputs = False
        self.Y_untransformed = Y.copy()
        self.predict_in_warped_space = False
        likelihood = likelihoods.Gaussian(self.transform_data(), normalize=normalize_Y)

        GP.__init__(self, X, likelihood, kernel, normalize_X=normalize_X)
        self._set_params(self._get_params())

    def _scale_data(self, Y):
        self._Ymax = Y.max()
        self._Ymin = Y.min()
        return (Y - self._Ymin) / (self._Ymax - self._Ymin) - 0.5

    def _unscale_data(self, Y):
        return (Y + 0.5) * (self._Ymax - self._Ymin) + self._Ymin

    def _set_params(self, x):
        self.warping_params = x[:self.warping_function.num_parameters]
        Y = self.transform_data()
        self.likelihood.set_data(Y)
        GP._set_params(self, x[self.warping_function.num_parameters:].copy())

    def _get_params(self):
        return np.hstack((self.warping_params.flatten().copy(), GP._get_params(self).copy()))

    def _get_param_names(self):
        warping_names = self.warping_function._get_param_names()
        param_names = GP._get_param_names(self)
        return warping_names + param_names

    def transform_data(self):
        Y = self.warping_function.f(self.Y_untransformed.copy(), self.warping_params).copy()
        return Y

    def log_likelihood(self):
        ll = GP.log_likelihood(self)
        jacobian = self.warping_function.fgrad_y(self.Y_untransformed, self.warping_params)
        return ll + np.log(jacobian).sum()

    def _log_likelihood_gradients(self):
        ll_grads = GP._log_likelihood_gradients(self)
        alpha = np.dot(self.Ki, self.likelihood.Y.flatten())
        warping_grads = self.warping_function_gradients(alpha)

        warping_grads = np.append(warping_grads[:, :-1].flatten(), warping_grads[0, -1])
        return np.hstack((warping_grads.flatten(), ll_grads.flatten()))

    def warping_function_gradients(self, Kiy):
        grad_y = self.warping_function.fgrad_y(self.Y_untransformed, self.warping_params)
        grad_y_psi, grad_psi = self.warping_function.fgrad_y_psi(self.Y_untransformed, self.warping_params,
                                                                 return_covar_chain=True)
        djac_dpsi = ((1.0 / grad_y[:, :, None, None]) * grad_y_psi).sum(axis=0).sum(axis=0)
        dquad_dpsi = (Kiy[:, None, None, None] * grad_psi).sum(axis=0).sum(axis=0)

        return -dquad_dpsi + djac_dpsi

    def plot_warping(self):
        self.warping_function.plot(self.warping_params, self.Y_untransformed.min(), self.Y_untransformed.max())

    def _raw_predict(self, *args, **kwargs):
        mu, var = GP._raw_predict(self, *args, **kwargs)

        if self.predict_in_warped_space:
            mu = self.warping_function.f_inv(mu, self.warping_params)
            var = self.warping_function.f_inv(var, self.warping_params)
            mu = self._unscale_data(mu)
        return mu, var
