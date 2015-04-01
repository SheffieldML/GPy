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
            self.warping_params = (np.random.randn(self.warping_function.n_terms * 3 + 1,) * 1)
        else:
            self.warping_function = warping_function

        self.scale_data = False
        if self.scale_data:
            Y = self._scale_data(Y)
        self.has_uncertain_inputs = False
        self.Y_untransformed = Y.copy()
        self.predict_in_warped_space = False
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

    def predict(self, Xnew, which_parts='all', pred_init=None):
        # normalize X values
        # Xnew = (Xnew.copy() - self._Xoffset) / self._Xscale
        mu, var = GP._raw_predict(self, Xnew)

        # now push through likelihood
        mean, var = self.likelihood.predictive_values(mu, var)

        if self.predict_in_warped_space:
            mean = self.warping_function.f_inv(mean,  y=pred_init)
            var = self.warping_function.f_inv(var)

        if self.scale_data:
            mean = self._unscale_data(mean)

        return mean, var

if __name__ == '__main__':
    X = np.random.randn(100, 1)
    Y = np.sin(X) + np.random.randn(100, 1)*0.05

    m = WarpedGP(X, Y)
