# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.parameterization import Parameterized, Param
from ..core.parameterization.priors import LogGaussian


class InputWarpingFunction(Parameterized):
    """Abstract class for input warping functions
    """

    def __init__(self, name):
        super(InputWarpingFunction, self).__init__(name=name)

    def f(self, X, test=False):

        raise NotImplementedError

    def fgrad_x(self, X):
        raise NotImplementedError

    def update_grads(self, X, dL_dW):
        raise NotImplementedError


class IdentifyWarping(InputWarpingFunction):
    """The identity warping function, for testing"""
    def __init__(self):
        super(IdentifyWarping, self).__init__(name='input_warp_identity')

    def f(self, X, test_data=False):
        return X

    def fgrad_X(self, X):
        return np.zeros(X.shape)

    def update_grads(self, X, dL_dW):
        pass


class InputWarpingTest(InputWarpingFunction):
    """The identity warping function, for testing"""
    def __init__(self):
        super(InputWarpingTest, self).__init__(name='input_warp_test')
        self.a = Param('a', 1.0)
        self.set_prior(LogGaussian(0.0, 0.75))
        self.link_parameter(self.a)

    def f(self, X, test_data=False):
        return X * self.a

    def fgrad_X(self, X):
        return self.ones(X.shape) * self.a

    def update_grads(self, X, dL_dW):
        self.a.gradient[:] = np.sum(dL_dW * X)


class KumarWarping(InputWarpingFunction):
    """Kumar Warping for input data

    Parameters
    ----------
    X : array_like, shape = (n_samples, n_features)
        The input data that is going to be warped

    warping_indices: list of int, optional
        The features that are going to be warped
        Default to warp all the features

    epsilon: float, optional
        Used to normalized input data to [0+e, 1-e]
        Default to 1e-6

    Xmin : list of float, Optional
        The min values for each feature defined by users
        Default to the train minimum

    Xmax : list of float, Optional
        The max values for each feature defined by users
        Default to the train maximum

    Attributes
    ----------
    warping_indices: list of int
        The features that are going to be warped
        Default to warp all the features

    warping_dim: int
        The number of features to be warped

    Xmin : list of float
        The min values for each feature defined by users
        Default to the train minimum

    Xmax : list of float
        The max values for each feature defined by users
        Default to the train maximum

    epsilon: float
        Used to normalized input data to [0+e, 1-e]
        Default to 1e-6

    X_normalized : array_like, shape = (n_samples, n_features)
        The normalized training X

    scaling : list of float, length = n_features in X
        Defined as 1.0 / (self.Xmax - self.Xmin)

    params : list of Param
        The list of all the parameters used in Kumar Warping

    num_parameters: int
        The number of parameters used in Kumar Warping
    """

    def __init__(self, X, warping_indices=None, epsilon=None, Xmin=None, Xmax=None):

        super(KumarWarping, self).__init__(name='input_warp_kumar')

        if warping_indices is not None and np.max(warping_indices) > X.shape[1] -1:
            raise ValueError("Kumar warping indices exceed feature dimension")

        if warping_indices is not None and np.min(warping_indices) < 0:
            raise ValueError("Kumar warping indices should be larger than 0")

        if warping_indices is not None and np.any(list(map(lambda x: not isinstance(x, int), warping_indices))):
            raise ValueError("Kumar warping indices should be integer")

        if Xmin is None and Xmax is None:
            Xmin = X.min(axis=0)
            Xmax = X.max(axis=0)
        else:
            if Xmin is None or Xmax is None:
                raise ValueError("Xmin and Xmax need to be provide at the same time!")
            if len(Xmin) != X.shape[1] or len(Xmax) != X.shape[1]:
                raise ValueError("Xmin and Xmax should have n_feature values!")

        if epsilon is None:
            epsilon = 1e-6
        self.epsilon = epsilon

        self.Xmin = Xmin - self.epsilon
        self.Xmax = Xmax + self.epsilon
        self.scaling = 1.0 / (self.Xmax - self.Xmin)
        self.X_normalized = (X - self.Xmin) / (self.Xmax - self.Xmin)

        if warping_indices is None:
            warping_indices = range(X.shape[1])

        self.warping_indices = warping_indices
        self.warping_dim = len(self.warping_indices)
        self.num_parameters = 2 * self.warping_dim

        # create parameters
        self.params = [[Param('a%d' % i, 1.0), Param('b%d' % i, 1.0)] for i in range(self.warping_dim)]

        # add constraints
        for i in range(self.warping_dim):
            self.params[i][0].constrain_bounded(0.0, 10.0)
            self.params[i][1].constrain_bounded(0.0, 10.0)

        # set priors and add them into handler
        for i in range(self.warping_dim):
            self.params[i][0].set_prior(LogGaussian(0.0, 0.75))
            self.params[i][1].set_prior(LogGaussian(0.0, 0.75))
            self.link_parameter(self.params[i][0])
            self.link_parameter(self.params[i][1])

    def f(self, X, test_data=False):
        """Apply warping_function to some Input data

        Parameters:
        -----------
        X : array_like, shape = (n_samples, n_features)

        test_data: bool, optional
            Default to False, should set to True when transforming test data

        Returns
        -------
        X_warped : array_like, shape = (n_samples, n_features)
            The warped input data

        Math
        ----
        f(x) = 1 - (1 - x^a)^b
        """
        X_warped = X.copy()
        if test_data:
            X_normalized = (X - self.Xmin) / (self.Xmax - self.Xmin)
        else:
            X_normalized = self.X_normalized

        for i_seq, i_fea in enumerate(self.warping_indices):
            a, b = self.params[i_seq][0], self.params[i_seq][1]
            X_warped[:, i_fea] = 1 - np.power(1 - np.power(X_normalized[:, i_fea], a), b)
        return X_warped

    def fgrad_X(self, X):
        """Compute the gradient of warping function with respect to X

        Parameters
        ----------
        X : array_like, shape = (n_samples, n_features)
            The location to compute gradient

        Returns
        -------
        grad : array_like, shape = (n_samples, n_features)
            The gradient for every location at X

        Math
        ----
        grad = a * b * x ^(a-1) * (1 - x^a)^(b-1)
        """
        grad = np.zeros(X.shape)
        for i_seq, i_fea in enumerate(self.warping_indices):
            a, b = self.params[i_seq][0], self.params[i_seq][1]
            grad[:, i_fea] = a * b * np.power(self.X_normalized[:, i_fea], a-1) *  \
                             np.power(1 - np.power(self.X_normalized[:, i_fea], a), b-1) * self.scaling[i_fea]
        return grad

    def update_grads(self, X, dL_dW):
        """Update the gradients of marginal log likelihood with respect to the parameters of warping function

        Parameters
        ----------
        X : array_like, shape = (n_samples, n_features)
            The input BEFORE warping

        dL_dW : array_like, shape = (n_samples, n_features)
            The gradient of marginal log likelihood with respect to the Warped input

        Math
        ----
        let w = f(x), the input after warping, then
        dW_da = b * (1 - x^a)^(b - 1) * x^a * ln(x)
        dW_db = - (1 - x^a)^b * ln(1 - x^a)
        dL_da = dL_dW * dW_da
        dL_db = dL_dW * dW_db
        """
        for i_seq, i_fea in enumerate(self.warping_indices):
            ai, bi = self.params[i_seq][0], self.params[i_seq][1]

            # cache some value for save some computation
            x_pow_a = np.power(self.X_normalized[:, i_fea], ai)

            # compute gradient for ai, bi on all X
            dz_dai = bi * np.power(1 - x_pow_a, bi-1) * x_pow_a * np.log(self.X_normalized[:, i_fea])
            dz_dbi = - np.power(1 - x_pow_a, bi) * np.log(1 - x_pow_a)

            # sum gradients on all the data
            dL_dai = np.sum(dL_dW[:, i_fea] * dz_dai)
            dL_dbi = np.sum(dL_dW[:, i_fea] * dz_dbi)
            self.params[i_seq][0].gradient[:] = dL_dai
            self.params[i_seq][1].gradient[:] = dL_dbi




