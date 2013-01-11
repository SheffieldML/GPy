# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import unittest
import numpy as np
import GPy

class GradientTests(unittest.TestCase):
    def setUp(self):
        ######################################
        ## 1 dimensional example

        # sample inputs and outputs
        self.X1D = np.random.uniform(-3.,3.,(20,1))
        self.Y1D = np.sin(self.X1D)+np.random.randn(20,1)*0.05

        ######################################
        ## 2 dimensional example

        # sample inputs and outputs
        self.X2D = np.random.uniform(-3.,3.,(40,2))
        self.Y2D = np.sin(self.X2D[:,0:1]) * np.sin(self.X2D[:,1:2])+np.random.randn(40,1)*0.05

    def check_model_with_white(self, kern, model_type='GP_regression', dimension=1, constraint=''):
        #Get the correct gradients
        if dimension == 1:
            X = self.X1D
            Y = self.Y1D
        else:
            X = self.X2D
            Y = self.Y2D

        #Get model type (GP_regression, GP_sparse_regression, etc)
        model_fit = getattr(GPy.models, model_type)

        noise = GPy.kern.white(dimension)
        kern = kern + noise
        m = model_fit(X, Y, kernel=kern)
        m.constrain_positive(constraint)
        m.randomize()
        # contrain all parameters to be positive
        self.assertTrue(m.checkgrad())

    def test_gp_regression_rbf_white_kern_1d(self):
        ''' Testing the GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.rbf(1)
        self.check_model_with_white(rbf, model_type='GP_regression', dimension=1)

    def test_GP_regression_rbf_ARD_white_kern_2D(self):
        ''' Testing the GP regression with rbf and white kernel on 2d data '''
        k = GPy.kern.rbf_ARD(2)
        self.check_model_with_white(k, model_type='GP_regression', dimension=2)

    def test_GP_regression_rbf_white_kern_2D(self):
        ''' Testing the GP regression with rbf and white kernel on 2d data '''
        rbf = GPy.kern.rbf(2)
        self.check_model_with_white(rbf, model_type='GP_regression', dimension=2)

    def test_GP_regression_matern52_kern_1D(self):
        ''' Testing the GP regression with matern52 kernel on 1d data '''
        matern52 = GPy.kern.Matern52(1)
        self.check_model_with_white(matern52, model_type='GP_regression', dimension=1)

    def test_GP_regression_matern52_kern_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2)
        self.check_model_with_white(matern52, model_type='GP_regression', dimension=2)

    def test_GP_regression_matern32_kern_1D(self):
        ''' Testing the GP regression with matern32 kernel on 1d data '''
        matern32 = GPy.kern.Matern32(1)
        self.check_model_with_white(matern32, model_type='GP_regression', dimension=1)

    def test_GP_regression_matern32_kern_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2)
        self.check_model_with_white(matern32, model_type='GP_regression', dimension=2)

    def test_GP_regression_exponential_kern_1D(self):
        ''' Testing the GP regression with exponential kernel on 1d data '''
        exponential = GPy.kern.exponential(1)
        self.check_model_with_white(exponential, model_type='GP_regression', dimension=1)

    def test_GP_regression_exponential_kern_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.exponential(2)
        self.check_model_with_white(exponential, model_type='GP_regression', dimension=2)

    def test_GP_regression_bias_kern_1D(self):
        ''' Testing the GP regression with bias kernel on 1d data '''
        bias = GPy.kern.bias(1)
        self.check_model_with_white(bias, model_type='GP_regression', dimension=1)

    def test_GP_regression_bias_kern_2D(self):
        ''' Testing the GP regression with bias kernel on 2d data '''
        bias = GPy.kern.bias(2)
        self.check_model_with_white(bias, model_type='GP_regression', dimension=2)

    def test_GP_regression_linear_kern_1D(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.linear(1)
        self.check_model_with_white(linear, model_type='GP_regression', dimension=1)

    def test_GP_regression_linear_kern_2D(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.linear(2)
        self.check_model_with_white(linear, model_type='GP_regression', dimension=2)

    def test_sparse_GP_regression_rbf_white_kern_1d(self):
        ''' Testing the sparse GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.rbf(1)
        self.check_model_with_white(rbf, model_type='sparse_GP_regression', dimension=1, constraint='(variance|lengthscale|precision)')

    def test_sparse_GP_regression_rbf_white_kern_2D(self):
        ''' Testing the sparse GP regression with rbf and white kernel on 2d data '''
        rbf = GPy.kern.rbf(2)
        self.check_model_with_white(rbf, model_type='sparse_GP_regression', dimension=2, constraint='(variance|lengthscale|precision)')

    def test_GPLVM_rbf_bias_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias and white kernel """
        N, Q, D = 50, 1, 2
        X = np.random.rand(N, Q)
        k = GPy.kern.rbf(Q, 0.5, 0.9) + GPy.kern.bias(Q, 0.1) + GPy.kern.white(Q, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        m = GPy.models.GPLVM(Y, Q, kernel = k)
        m.constrain_positive('(rbf|bias|white)')
        self.assertTrue(m.checkgrad())

    def test_GPLVM_rbf_linear_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias and white kernel """
        N, Q, D = 50, 1, 2
        X = np.random.rand(N, Q)
        k = GPy.kern.linear(Q) + GPy.kern.bias(Q, 0.1) + GPy.kern.white(Q, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        m = GPy.models.GPLVM(Y, Q, init = 'PCA', kernel = k)
        m.constrain_positive('(linear|bias|white)')
        self.assertTrue(m.checkgrad())

    def test_GP_EP(self):
        return # Disabled TODO
        N = 20
        X = np.hstack([np.random.rand(N/2)+1,np.random.rand(N/2)-1])[:,None]
        k = GPy.kern.rbf(1) + GPy.kern.white(1)
        Y = np.hstack([np.ones(N/2),-np.ones(N/2)])[:,None]
        likelihood = GPy.inference.likelihoods.probit(Y)
        m = GPy.models.GP_EP(X,likelihood,k)
        m.constrain_positive('(var|len)')
        m.approximate_likelihood()
        self.assertTrue(m.checkgrad())

    def test_generalized_FITC(self):
        N = 20
        X = np.hstack([np.random.rand(N/2)+1,np.random.rand(N/2)-1])[:,None]
        k = GPy.kern.rbf(1) + GPy.kern.white(1)
        Y = np.hstack([np.ones(N/2),-np.ones(N/2)])[:,None]
        likelihood = GPy.inference.likelihoods.probit(Y)
        m = GPy.models.generalized_FITC(X,likelihood,k,inducing=4)
        m.constrain_positive('(var|len)')
        m.approximate_likelihood()
        self.assertTrue(m.checkgrad())

    def test_warped_GP(self):
        xmin, xmax = 1, 2.5*np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y  = b*X + C + 1*np.sin(X)
        y += 0.05*np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.warpedGP(X, y, warping_terms = 3)
        m.constrain_positive('(tanh_a|tanh_b|rbf|white|bias)')
        self.assertTrue(m.checkgrad())


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
