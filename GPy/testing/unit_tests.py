# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import unittest
import numpy as np
import GPy

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

    def check_model(self, kern, model_type='GPRegression', dimension=1, uncertain_inputs=False):
        # Get the correct gradients
        if dimension == 1:
            X = self.X1D
            Y = self.Y1D
        else:
            X = self.X2D
            Y = self.Y2D
        # Get model type (GPRegression, SparseGPRegression, etc)
        model_fit = getattr(GPy.models, model_type)

        # noise = GPy.kern.white(dimension)
        kern = kern #  + noise
        if uncertain_inputs:
            m = model_fit(X, Y, kernel=kern, X_variance=np.random.rand(X.shape[0], X.shape[1]))
        else:
            m = model_fit(X, Y, kernel=kern)
        m.randomize()
        # contrain all parameters to be positive
        self.assertTrue(m.checkgrad())

    def test_GPRegression_rbf_1d(self):
        ''' Testing the GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.rbf(1)
        self.check_model(rbf, model_type='GPRegression', dimension=1)

    def test_GPRegression_rbf_2D(self):
        ''' Testing the GP regression with rbf kernel on 2d data '''
        rbf = GPy.kern.rbf(2)
        self.check_model(rbf, model_type='GPRegression', dimension=2)

    def test_GPRegression_rbf_ARD_2D(self):
        ''' Testing the GP regression with rbf kernel on 2d data '''
        k = GPy.kern.rbf(2, ARD=True)
        self.check_model(k, model_type='GPRegression', dimension=2)

    def test_GPRegression_mlp_1d(self):
        ''' Testing the GP regression with mlp kernel with white kernel on 1d data '''
        mlp = GPy.kern.mlp(1)
        self.check_model(mlp, model_type='GPRegression', dimension=1)

    def test_GPRegression_poly_1d(self):
        ''' Testing the GP regression with polynomial kernel with white kernel on 1d data '''
        mlp = GPy.kern.poly(1, degree=5)
        self.check_model(mlp, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern52_1D(self):
        ''' Testing the GP regression with matern52 kernel on 1d data '''
        matern52 = GPy.kern.Matern52(1)
        self.check_model(matern52, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern52_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2)
        self.check_model(matern52, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern52_ARD_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2, ARD=True)
        self.check_model(matern52, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern32_1D(self):
        ''' Testing the GP regression with matern32 kernel on 1d data '''
        matern32 = GPy.kern.Matern32(1)
        self.check_model(matern32, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern32_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2)
        self.check_model(matern32, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern32_ARD_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2, ARD=True)
        self.check_model(matern32, model_type='GPRegression', dimension=2)

    def test_GPRegression_exponential_1D(self):
        ''' Testing the GP regression with exponential kernel on 1d data '''
        exponential = GPy.kern.exponential(1)
        self.check_model(exponential, model_type='GPRegression', dimension=1)

    def test_GPRegression_exponential_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.exponential(2)
        self.check_model(exponential, model_type='GPRegression', dimension=2)

    def test_GPRegression_exponential_ARD_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.exponential(2, ARD=True)
        self.check_model(exponential, model_type='GPRegression', dimension=2)

    def test_GPRegression_bias_kern_1D(self):
        ''' Testing the GP regression with bias kernel on 1d data '''
        bias = GPy.kern.bias(1)
        self.check_model(bias, model_type='GPRegression', dimension=1)

    def test_GPRegression_bias_kern_2D(self):
        ''' Testing the GP regression with bias kernel on 2d data '''
        bias = GPy.kern.bias(2)
        self.check_model(bias, model_type='GPRegression', dimension=2)

    def test_GPRegression_linear_kern_1D_ARD(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.linear(1, ARD=True)
        self.check_model(linear, model_type='GPRegression', dimension=1)

    def test_GPRegression_linear_kern_2D_ARD(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.linear(2, ARD=True)
        self.check_model(linear, model_type='GPRegression', dimension=2)

    def test_GPRegression_linear_kern_1D(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.linear(1)
        self.check_model(linear, model_type='GPRegression', dimension=1)

    def test_GPRegression_linear_kern_2D(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.linear(2)
        self.check_model(linear, model_type='GPRegression', dimension=2)

    def test_SparseGPRegression_rbf_white_kern_1d(self):
        ''' Testing the sparse GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.rbf(1)
        self.check_model(rbf, model_type='SparseGPRegression', dimension=1)

    def test_SparseGPRegression_rbf_white_kern_2D(self):
        ''' Testing the sparse GP regression with rbf kernel on 2d data '''
        rbf = GPy.kern.rbf(2)
        self.check_model(rbf, model_type='SparseGPRegression', dimension=2)

    def test_SparseGPRegression_rbf_linear_white_kern_1D(self):
        ''' Testing the sparse GP regression with rbf kernel on 2d data '''
        rbflin = GPy.kern.rbf(1) + GPy.kern.linear(1)
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=1)

    def test_SparseGPRegression_rbf_linear_white_kern_2D(self):
        ''' Testing the sparse GP regression with rbf kernel on 2d data '''
        rbflin = GPy.kern.rbf(2) + GPy.kern.linear(2)
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=2)

    #@unittest.expectedFailure
    def test_SparseGPRegression_rbf_linear_white_kern_2D_uncertain_inputs(self):
        ''' Testing the sparse GP regression with rbf, linear kernel on 2d data with uncertain inputs'''
        rbflin = GPy.kern.rbf(2) + GPy.kern.linear(2)
        raise unittest.SkipTest("This is not implemented yet!")
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=2, uncertain_inputs=1)

    #@unittest.expectedFailure
    def test_SparseGPRegression_rbf_linear_white_kern_1D_uncertain_inputs(self):
        ''' Testing the sparse GP regression with rbf, linear kernel on 1d data with uncertain inputs'''
        rbflin = GPy.kern.rbf(1) + GPy.kern.linear(1)
        raise unittest.SkipTest("This is not implemented yet!")
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=1, uncertain_inputs=1)

    def test_GPLVM_rbf_bias_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias kernel """
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = GPy.kern.rbf(input_dim, 0.5, 0.9 * np.ones((1,))) + GPy.kern.bias(input_dim, 0.1) + GPy.kern.white(input_dim, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, kernel=k)
        self.assertTrue(m.checkgrad())

    def test_GPLVM_rbf_linear_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias kernel """
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = GPy.kern.linear(input_dim) + GPy.kern.bias(input_dim, 0.1) + GPy.kern.white(input_dim, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, init='PCA', kernel=k)
        self.assertTrue(m.checkgrad())

    def test_GP_EP_probit(self):
        N = 20
        X = np.hstack([np.random.normal(5, 2, N / 2), np.random.normal(10, 2, N / 2)])[:, None]
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        kernel = GPy.kern.rbf(1)
        m = GPy.models.GPClassification(X,Y,kernel=kernel)
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

    def test_sparse_EP_DTC_probit(self):
        N = 20
        X = np.hstack([np.random.normal(5, 2, N / 2), np.random.normal(10, 2, N / 2)])[:, None]
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        Z = np.linspace(0, 15, 4)[:, None]
        kernel = GPy.kern.rbf(1)
        m = GPy.models.SparseGPClassification(X,Y,kernel=kernel,Z=Z)
        #distribution = GPy.likelihoods.likelihood_functions.Bernoulli()
        #likelihood = GPy.likelihoods.EP(Y, distribution)
        #m = GPy.core.SparseGP(X, likelihood, kernel, Z)
        #m.ensure_default_constraints()
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

    def test_generalized_FITC(self):
        N = 20
        X = np.hstack([np.random.rand(N / 2) + 1, np.random.rand(N / 2) - 1])[:, None]
        k = GPy.kern.rbf(1) + GPy.kern.white(1)
        Y = np.hstack([np.ones(N/2),np.zeros(N/2)])[:,None]
        m = GPy.models.FITCClassification(X, Y, kernel = k)
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

    def multioutput_regression_1D(self):
        X1 = np.random.rand(50, 1) * 8
        X2 = np.random.rand(30, 1) * 5
        X = np.vstack((X1, X2))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
        Y2 = -np.sin(X2) + np.random.randn(*X2.shape) * 0.05
        Y = np.vstack((Y1, Y2))

        k1 = GPy.kern.rbf(1)
        m = GPy.models.GPMultioutputRegression(X_list=[X1,X2],Y_list=[Y1,Y2],kernel_list=[k1])
        m.constrain_fixed('.*rbf_var', 1.)
        self.assertTrue(m.checkgrad())

    def multioutput_sparse_regression_1D(self):
        X1 = np.random.rand(500, 1) * 8
        X2 = np.random.rand(300, 1) * 5
        X = np.vstack((X1, X2))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
        Y2 = -np.sin(X2) + np.random.randn(*X2.shape) * 0.05
        Y = np.vstack((Y1, Y2))

        k1 = GPy.kern.rbf(1)
        m = GPy.models.SparseGPMultioutputRegression(X_list=[X1,X2],Y_list=[Y1,Y2],kernel_list=[k1])
        m.constrain_fixed('.*rbf_var', 1.)
        self.assertTrue(m.checkgrad())

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
