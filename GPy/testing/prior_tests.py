# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class PriorTests(unittest.TestCase):
    def test_lognormal(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        m.ensure_default_constraints()
        lognormal = GPy.priors.LogGaussian(1, 2)
        m.set_prior('rbf', lognormal)
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_Gamma(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        m.ensure_default_constraints()
        Gamma = GPy.priors.Gamma(1, 1)
        m.set_prior('rbf', Gamma)
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_incompatibility(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        m.ensure_default_constraints()
        gaussian = GPy.priors.Gaussian(1, 1)
        success = False

        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        try:
            m.set_prior('rbf', gaussian)
        except AssertionError:
            success = True

        self.assertTrue(success)


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
