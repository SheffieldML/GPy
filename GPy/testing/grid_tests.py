# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import unittest
import numpy as np
import GPy


class GridModelTest(unittest.TestCase):
    def setUp(self):
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
        kernel = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m = GPy.models.GPRegressionGrid(self.X, self.Y, kernel)

        kernel2 = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        np.testing.assert_almost_equal(m.posterior.alpha, m2.posterior.woodbury_vector)

    def test_gradient_match(self):
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
        kernel = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m = GPy.models.GPRegressionGrid(self.X, self.Y, kernel)

        kernel2 = GPy.kern.RBF(input_dim=self.dim, variance=1, ARD=True)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        test = np.array([[0, 0, 2], [-1, 3, -4]])

        np.testing.assert_almost_equal(m.predict(test), m2.predict(test))
