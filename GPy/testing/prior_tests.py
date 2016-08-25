# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class PriorTests(unittest.TestCase):
    def test_studentT(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        studentT = GPy.priors.StudentT(1, 2, 4)
        # setting a StudentT prior on non-negative parameters
        # should raise an assertionerror.
        self.assertRaises(AssertionError, m.rbf.set_prior, studentT)
        
        m = GPy.models.SparseGPRegression(X, y)

        gaussian = GPy.priors.Gaussian(1, 1)
        m.Z.set_prior(studentT)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        #self.assertRaises(AssertionError, m.Z.set_prior, gaussian)
        self.assertTrue(m.checkgrad())
    
    def test_lognormal(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        lognormal = GPy.priors.LogGaussian(1, 2)
        m.rbf.set_prior(lognormal)
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
        Gamma = GPy.priors.Gamma(1, 1)
        m.rbf.set_prior(Gamma)
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
        gaussian = GPy.priors.Gaussian(1, 1)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        self.assertRaises(AssertionError, m.rbf.set_prior, gaussian)

    def test_set_prior(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)

        gaussian = GPy.priors.Gaussian(1, 1)
        #m.rbf.set_prior(gaussian)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        self.assertRaises(AssertionError, m.rbf.set_prior, gaussian)

    def test_set_gaussian_for_reals(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.SparseGPRegression(X, y)

        gaussian = GPy.priors.Gaussian(1, 1)
        m.Z.set_prior(gaussian)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        #self.assertRaises(AssertionError, m.Z.set_prior, gaussian)
        self.assertTrue(m.checkgrad())


    def test_fixed_domain_check(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)

        m.rbf.fix()
        gaussian = GPy.priors.Gaussian(1, 1)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        self.assertRaises(AssertionError, m.rbf.set_prior, gaussian)

    def test_fixed_domain_check1(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)

        m.kern.lengthscale.fix()
        gaussian = GPy.priors.Gaussian(1, 1)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        self.assertRaises(AssertionError, m.rbf.set_prior, gaussian)


if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()
