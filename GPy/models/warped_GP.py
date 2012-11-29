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
    """
    TODO: fucking docstrings!

    @nfusi: I'#ve hacked a little on this, but no guarantees. J.
    """
    def __init__(self, X, Y, warping_function = None, warping_terms = 3, **kwargs):

        if warping_function == None:
            self.warping_function = TanhWarpingFunction(warping_terms)
            # self.warping_params = np.random.randn(self.warping_function.n_terms, 3)
            self.warping_params = np.ones((self.warping_function.n_terms, 3))*1.0 # TODO better init
            self.warp_params_shape = (self.warping_function.n_terms, 3) # todo get this from the subclass

        self.Z = Y.copy()
        self.N, self.D = Y.shape
        self.transform_data()
        GP_regression.__init__(self, X, self.Y, **kwargs)

    def set_param(self, x):
        self.warping_params = x[:self.warping_function.num_parameters].reshape(self.warp_params_shape).copy()
        self.transform_data()
        GP_regression.set_param(self, x[self.warping_function.num_parameters:].copy())

    def get_param(self):
        return np.hstack((self.warping_params.flatten().copy(), GP_regression.get_param(self).copy()))

    def get_param_names(self):
        warping_names = self.warping_function.get_param_names()
        param_names = GP_regression.get_param_names(self)
        return warping_names + param_names

    def transform_data(self):
        self.Y = self.warping_function.f(self.Z.copy(), self.warping_params).copy()

        # this supports the 'smart' behaviour in GP_regression
        if self.D > self.N:
            self.Youter = np.dot(self.Y, self.Y.T)
        else:
            self.Youter = None

        return self.Y

    def log_likelihood(self):
        ll = GP_regression.log_likelihood(self)
        jacobian = self.warping_function.fgrad_y(self.Z, self.warping_params)
        return ll + np.log(jacobian).sum()

    def log_likelihood_gradients(self):
        ll_grads = GP_regression.log_likelihood_gradients(self)
        alpha = np.dot(self.Ki, self.Y.flatten())
        warping_grads = self.warping_function_gradients(alpha)
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
        mu, var = GP_regression.predict(self, X, **kwargs)

        # The plot() function calls set_param() before calling predict()
        # this is causing the observations to be plotted in the transformed
        # space (where Y lives), making the plot looks very wrong
        # if the predictions are made in the untransformed space
        # (where Z lives). To fix this I included the option below. It's
        # just a quick fix until I figure out something smarter.
        if in_unwarped_space:
            mu = self.warping_function.f_inv(mu, self.warping_params)
            var = self.warping_function.f_inv(var, self.warping_params)

        return mu, var
