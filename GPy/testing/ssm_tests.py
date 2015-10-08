# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import unittest
import random
import numpy as np
import GPy

class SsmModelTest(unittest.TestCase):
    def setUp(self):
        ######################################
        # sample inputs and outputs

        N = 5

        self.X = np.array([[2],[19],[20],[23],[34]])
        self.Y = np.random.randn(N, 1) * 100

    def test_likelihood_match(self):
        kernel = GPy.kern.Matern32_SSM(input_dim=1, variance=1, lengthscale=1)
        m = GPy.models.GPRegressionSSM(self.X, self.Y, kernel)

        kernel2 = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=1)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

    def test_prediction_match(self):
        kernel = GPy.kern.Matern32_SSM(input_dim=1, variance=1, lengthscale=1)
        m = GPy.models.GPRegressionSSM(self.X, self.Y, kernel)

        kernel2 = GPy.kern.Matern32(input_dim=1, variance=1, lengthscale=1)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel2)

        test = np.array([[-34], [10], [123]])

        np.testing.assert_almost_equal(m.predict(test), m2.predict(test))

