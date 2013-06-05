# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import unittest
import numpy as np
import GPy
from GPy.likelihoods.likelihood_functions import Binomial

class GradientTests(unittest.TestCase):
    def setUp(self):
        ######################################
        # # 1 dimensional example

        # sample inputs and outputs
        self.X1D = np.random.uniform(-3., 3., (20, 1))
        self.Y1D = np.sin(self.X1D) + np.random.randn(20, 1) * 0.05

        ######################################
        # # 2 dimensional example

        # sample inputs and outputs
        self.X2D = np.random.uniform(-3., 3., (40, 2))
        self.Y2D = np.sin(self.X2D[:, 0:1]) * np.sin(self.X2D[:, 1:2]) + np.random.randn(40, 1) * 0.05

    def check_model_with_white(self, kern, model_type='GPRegression', dimension=1):
        # Get the correct gradients
        if dimension == 1:
            X = self.X1D
            Y = self.Y1D
        else:
            X = self.X2D
            Y = self.Y2D
        # Get model type (GPRegression, SparseGPRegression, etc)
        model_fit = getattr(GPy.models, model_type)

        noise = GPy.kern.white(dimension)
        kern = kern + noise
        m = model_fit(X, Y, kernel=kern)
        m.ensure_default_constraints()
        m.randomize()
        # contrain all parameters to be positive
        self.assertTrue(m.checkgrad())

    def test_GPRegression_rbf_1d(self):
        ''' Testing the GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.rbf(1)
        self.check_model_with_white(rbf, model_type='GPRegression', dimension=1)

    def test_GPRegression_rbf_2D(self):
        ''' Testing the GP regression with rbf and white kernel on 2d data '''
        rbf = GPy.kern.rbf(2)
        self.check_model_with_white(rbf, model_type='GPRegression', dimension=2)

    def test_GPRegression_rbf_ARD_2D(self):
        ''' Testing the GP regression with rbf and white kernel on 2d data '''
        k = GPy.kern.rbf(2, ARD=True)
        self.check_model_with_white(k, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern52_1D(self):
        ''' Testing the GP regression with matern52 kernel on 1d data '''
        matern52 = GPy.kern.Matern52(1)
        self.check_model_with_white(matern52, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern52_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2)
        self.check_model_with_white(matern52, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern52_ARD_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2, ARD=True)
        self.check_model_with_white(matern52, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern32_1D(self):
        ''' Testing the GP regression with matern32 kernel on 1d data '''
        matern32 = GPy.kern.Matern32(1)
        self.check_model_with_white(matern32, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern32_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2)
        self.check_model_with_white(matern32, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern32_ARD_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2, ARD=True)
        self.check_model_with_white(matern32, model_type='GPRegression', dimension=2)

    def test_GPRegression_exponential_1D(self):
        ''' Testing the GP regression with exponential kernel on 1d data '''
        exponential = GPy.kern.exponential(1)
        self.check_model_with_white(exponential, model_type='GPRegression', dimension=1)

    def test_GPRegression_exponential_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.exponential(2)
        self.check_model_with_white(exponential, model_type='GPRegression', dimension=2)

    def test_GPRegression_exponential_ARD_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.exponential(2, ARD=True)
        self.check_model_with_white(exponential, model_type='GPRegression', dimension=2)

    def test_GPRegression_bias_kern_1D(self):
        ''' Testing the GP regression with bias kernel on 1d data '''
        bias = GPy.kern.bias(1)
        self.check_model_with_white(bias, model_type='GPRegression', dimension=1)

    def test_GPRegression_bias_kern_2D(self):
        ''' Testing the GP regression with bias kernel on 2d data '''
        bias = GPy.kern.bias(2)
        self.check_model_with_white(bias, model_type='GPRegression', dimension=2)

    def test_GPRegression_linear_kern_1D_ARD(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.linear(1, ARD=True)
        self.check_model_with_white(linear, model_type='GPRegression', dimension=1)

    def test_GPRegression_linear_kern_2D_ARD(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.linear(2, ARD=True)
        self.check_model_with_white(linear, model_type='GPRegression', dimension=2)

    def test_GPRegression_linear_kern_1D(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.linear(1)
        self.check_model_with_white(linear, model_type='GPRegression', dimension=1)

    def test_GPRegression_linear_kern_2D(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.linear(2)
        self.check_model_with_white(linear, model_type='GPRegression', dimension=2)

    def test_SparseGPRegression_rbf_white_kern_1d(self):
        ''' Testing the sparse GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.rbf(1)
        self.check_model_with_white(rbf, model_type='SparseGPRegression', dimension=1)

    def test_SparseGPRegression_rbf_white_kern_2D(self):
        ''' Testing the sparse GP regression with rbf and white kernel on 2d data '''
        rbf = GPy.kern.rbf(2)
        self.check_model_with_white(rbf, model_type='SparseGPRegression', dimension=2)

    def test_GPLVM_rbf_bias_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias and white kernel """
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = GPy.kern.rbf(input_dim, 0.5, 0.9 * np.ones((1,))) + GPy.kern.bias(input_dim, 0.1) + GPy.kern.white(input_dim, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, kernel=k)
        m.ensure_default_constraints()
        self.assertTrue(m.checkgrad())

    def test_GPLVM_rbf_linear_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias and white kernel """
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim, 0.1) + GPy.kern.white(input_dim, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, init='PCA', kernel=k)
        m.ensure_default_constraints()
        self.assertTrue(m.checkgrad())

    def test_GP_EP_probit(self):
        N = 20
        X = np.hstack([np.random.normal(5, 2, N / 2), np.random.normal(10, 2, N / 2)])[:, None]
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        kernel = GPy.kern.rbf(1)
        distribution = GPy.likelihoods.likelihood_functions.Binomial()
        likelihood = GPy.likelihoods.EP(Y, distribution)
        m = GPy.core.GP(X, likelihood, kernel)
        m.ensure_default_constraints()
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

    def test_sparse_EP_DTC_probit(self):
        N = 20
        X = np.hstack([np.random.normal(5, 2, N / 2), np.random.normal(10, 2, N / 2)])[:, None]
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        Z = np.linspace(0, 15, 4)[:, None]
        kernel = GPy.kern.rbf(1)
        distribution = GPy.likelihoods.likelihood_functions.Binomial()
        likelihood = GPy.likelihoods.EP(Y, distribution)
        m = GPy.core.SparseGP(X, likelihood, kernel, Z)
        m.ensure_default_constraints()
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

    def test_generalized_FITC(self):
        N = 20
        X = np.hstack([np.random.rand(N / 2) + 1, np.random.rand(N / 2) - 1])[:, None]
        k = GPy.kern.rbf(1) + GPy.kern.white(1)
        Y = np.hstack([np.ones(N/2),np.zeros(N/2)])[:,None]
        m = GPy.models.FITCClassification(X, Y=Y)
        m.ensure_default_constraints()
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
