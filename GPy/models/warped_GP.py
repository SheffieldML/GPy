# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .. import kern
from ..core import model
from ..util.linalg import pdinv
from ..util.plot import gpplot
from ..util.warping_functions import *
from GP_regression import GP_regression


class warpedGP(GP_regression):
    def __init__(self, X, Y, warping_function = None, warping_terms = 3, **kwargs):

        if warping_function == None:
            self.warping_function = TanhWarpingFunction_d(warping_terms)
            self.warping_params = (np.random.randn(self.warping_function.n_terms*3+1,) * 1)

        self.Z = Y.copy()
        self.N, self.D = Y.shape
        GP_regression.__init__(self, X, self.transform_data(), **kwargs)

    def _set_params(self, x):
        self.warping_params = x[:self.warping_function.num_parameters]
        Y = self.transform_data()
        self.likelihood.set_data(Y)
        GP_regression._set_params(self, x[self.warping_function.num_parameters:].copy())

    def _get_params(self):
        return np.hstack((self.warping_params.flatten().copy(), GP_regression._get_params(self).copy()))

    def _get_param_names(self):
        warping_names = self.warping_function._get_param_names()
        param_names = GP_regression._get_param_names(self)
        return warping_names + param_names

    def transform_data(self):
        Y = self.warping_function.f(self.Z.copy(), self.warping_params).copy()
        return Y

    def log_likelihood(self):
        ll = GP_regression.log_likelihood(self)
        jacobian = self.warping_function.fgrad_y(self.Z, self.warping_params)
        return ll + np.log(jacobian).sum()

    def _log_likelihood_gradients(self):
        ll_grads = GP_regression._log_likelihood_gradients(self)
        alpha = np.dot(self.Ki, self.likelihood.Y.flatten())
        warping_grads = self.warping_function_gradients(alpha)

        warping_grads = np.append(warping_grads[:,:-1].flatten(), warping_grads[0,-1])
        return np.hstack((warping_grads.flatten(), ll_grads.flatten()))

    def warping_function_gradients(self, Kiy):
        grad_y = self.warping_function.fgrad_y(self.Z, self.warping_params)
        grad_y_psi, grad_psi = self.warping_function.fgrad_y_psi(self.Z, self.warping_params,
                                                                 return_covar_chain = True)
        djac_dpsi = ((1.0/grad_y[:,:, None, None])*grad_y_psi).sum(axis=0).sum(axis=0)
        dquad_dpsi = (Kiy[:,None,None,None] * grad_psi).sum(axis=0).sum(axis=0)


        return -dquad_dpsi + djac_dpsi

    def plot_warping(self):
        self.warping_function.plot(self.warping_params, self.Z.min(), self.Z.max())

    def predict(self, X, in_unwarped_space = False, **kwargs):
        mu, var, _025pm, _975pm = GP_regression.predict(self, X, **kwargs)

        # The plot() function calls _set_params() before calling predict()
        # this is causing the observations to be plotted in the transformed
        # space (where Y lives), making the plot looks very wrong
        # if the predictions are made in the untransformed space
        # (where Z lives). To fix this I included the option below. It's
        # just a quick fix until I figure out something smarter.
        if in_unwarped_space:
            mu = self.warping_function.f_inv(mu, self.warping_params)
            var = self.warping_function.f_inv(var[:, None], self.warping_params)

        return mu, var, _025pm, _975pm
