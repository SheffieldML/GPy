# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from ..core import GP
from .. import likelihoods
from ..util.input_warping_functions import KumarWarping
from .. import kern


class InputWarpedGP(GP):
    """Input Warped GP

    This defines a GP model that applies a warping function to the Input.
    By default, it uses Kumar Warping (CDF of Kumaraswamy distribution)

    Parameters
    ----------
    X : array_like, shape = (n_samples, n_features) for input data

    Y : array_like, shape = (n_samples, 1) for output data

    kernel : object, optional
        An instance of kernel function defined in GPy.kern
        Default to Matern 32

    warping_function : object, optional
        An instance of warping function defined in GPy.util.input_warping_functions
        Default to KumarWarping

    warping_indices : list of int, optional
        An list of indices of which features in X should be warped.
        It is used in the Kumar warping function

    normalizer : bool, optional
        A bool variable indicates whether to normalize the output

    Xmin : list of float, optional
        The min values for every feature in X
        It is used in the Kumar warping function

    Xmax : list of float, optional
        The max values for every feature in X
        It is used in the Kumar warping function

    epsilon : float, optional
        We normalize X to [0+e, 1-e]. If not given, using the default value defined in KumarWarping function

    Attributes
    ----------
    X_untransformed : array_like, shape = (n_samples, n_features)
        A copy of original input X

    X_warped : array_like, shape = (n_samples, n_features)
        Input data after warping

    warping_function : object, optional
        An instance of warping function defined in GPy.util.input_warping_functions
        Default to KumarWarping

    Notes
    -----
    Kumar warping uses the CDF of Kumaraswamy distribution. More on the Kumaraswamy distribution can be found at the
    wiki page: https://en.wikipedia.org/wiki/Kumaraswamy_distribution

    References
    ----------
    Snoek, J.; Swersky, K.; Zemel, R. S. & Adams, R. P.
    Input Warping for Bayesian Optimization of Non-stationary Functions
    preprint arXiv:1402.0929, 2014
    """
    def __init__(self, X, Y, kernel=None, normalizer=False, warping_function=None, warping_indices=None, Xmin=None, Xmax=None, epsilon=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_untransformed = X.copy()

        if kernel is None:
            kernel = kern.sde_Matern32(X.shape[1], variance=1.)
        self.kernel = kernel

        if warping_function is None:
            self.warping_function = KumarWarping(self.X_untransformed, warping_indices, epsilon, Xmin, Xmax)
        else:
            self.warping_function = warping_function

        self.X_warped = self.transform_data(self.X_untransformed)
        likelihood = likelihoods.Gaussian()
        super(InputWarpedGP, self).__init__(self.X_warped, Y, likelihood=likelihood, kernel=kernel, normalizer=normalizer)

        # Add the parameters in the warping function to the model parameters hierarchy
        self.link_parameter(self.warping_function)

    def parameters_changed(self):
        """Update the gradients of parameters for warping function

        This method is called when having new values of parameters for warping function, kernels
        and other parameters in a normal GP
        """
        # using the warped X to update
        self.X = self.transform_data(self.X_untransformed)
        super(InputWarpedGP, self).parameters_changed()
        # the gradient of log likelihood w.r.t. input AFTER warping is a product of dL_dK and dK_dX
        dL_dX = self.kern.gradients_X(self.grad_dict['dL_dK'], self.X)
        self.warping_function.update_grads(self.X_untransformed, dL_dX)

    def transform_data(self, X, test_data=False):
        """Apply warping_function to some Input data

        Parameters
        ----------
        X : array_like, shape = (n_samples, n_features)

        test_data: bool, optional
            Default to False, should set to True when transforming test data
        """
        return self.warping_function.f(X, test_data)

    def log_likelihood(self):
        """Compute the marginal log likelihood

        For input warping, just use the normal GP log likelihood
        """
        return GP.log_likelihood(self)

    def predict(self, Xnew):
        """Prediction on the new data

        Parameters
        ----------
        Xnew : array_like, shape = (n_samples, n_features)
            The test data.

        Returns
        -------
        mean : array_like, shape = (n_samples, output.dim)
            Posterior mean at the location of Xnew

        var : array_like, shape = (n_samples, 1)
            Posterior variance at the location of Xnew
        """
        Xnew_warped = self.transform_data(Xnew, test_data=True)
        mean, var = super(InputWarpedGP, self).predict(Xnew_warped, kern=self.kernel, full_cov=False)
        return mean, var

if __name__ == '__main__':
    X = np.random.randn(100, 1)
    Y = np.sin(X) + np.random.randn(100, 1)*0.05
    m = InputWarpedGP(X, Y)
