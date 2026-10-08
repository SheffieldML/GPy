# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import numpy as np
import GPy


class TestGridModel:
    def setup_method(self):
        ######################################
        # # 3 dimensional example

        # sample inputs and outputs
        self.X = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        self.Y = np.random.randn(8, 1) * 100
        self.dim = self.X.shape[1]

    def test_alpha_match(self):
        self.setup_method()
        kernel = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m = GPy.models.GPRegressionGrid(self.X, self.Y, kernel)

        kernel2 = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        np.testing.assert_almost_equal(m.posterior.alpha, m2.posterior.woodbury_vector)

    def test_gradient_match(self):
        self.setup_method()
        kernel = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m = GPy.models.GPRegressionGrid(self.X, self.Y, kernel)

        kernel2 = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        np.testing.assert_almost_equal(
            kernel.variance.gradient, kernel2.variance.gradient
        )
        np.testing.assert_almost_equal(
            kernel.lengthscale.gradient, kernel2.lengthscale.gradient
        )
        np.testing.assert_almost_equal(
            m.likelihood.variance.gradient, m2.likelihood.variance.gradient
        )

    def test_prediction_match(self):
        self.setup_method()
        kernel = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m = GPy.models.GPRegressionGrid(self.X, self.Y, kernel)

        kernel2 = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        test = np.array([[0, 0, 2], [-1, 3, -4]])

        np.testing.assert_almost_equal(m.predict(test), m2.predict(test))

    def test_match_on_grid_with_more_than_three_values(self):
        # the unique values of each dimension must be taken in sorted order;
        # a set does not keep that order for more than three floats
        x1 = np.linspace(0, 1, 5)
        x2 = np.linspace(0, 1, 4)
        X = np.array([[a, b] for a in x1 for b in x2])
        Y = np.sin(3 * X[:, :1]) + np.cos(2 * X[:, 1:])

        kernel = GPy.kern.RBF(input_dim=2, lengthscale=[0.5, 0.7], ARD=True)
        m = GPy.models.GPRegressionGrid(X, Y, kernel)
        kernel2 = GPy.kern.RBF(input_dim=2, lengthscale=[0.5, 0.7], ARD=True)
        m2 = GPy.models.GPRegression(X, Y, kernel2)

        np.testing.assert_allclose(np.ravel(m.log_likelihood())[0], m2.log_likelihood())
        np.testing.assert_allclose(m.gradient, m2.gradient)
        test = np.array([[0.1, 0.2], [0.75, 0.4]])
        np.testing.assert_allclose(m.predict(test), m2.predict(test))

