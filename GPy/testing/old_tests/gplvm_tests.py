# Copyright (c) 2012, Nicolo Fusi
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class GPLVMTests(unittest.TestCase):
    def test_bias_kern(self):
        num_data, num_inducing, input_dim, output_dim = 10, 3, 2, 4
        X = np.random.rand(num_data, input_dim)
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(num_data),K,output_dim).T
        k = GPy.kern.Bias(input_dim) + GPy.kern.White(input_dim, 0.00001)
        m = GPy.models.GPLVM(Y, input_dim, kernel = k)
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_linear_kern(self):
        num_data, num_inducing, input_dim, output_dim = 10, 3, 2, 4
        X = np.random.rand(num_data, input_dim)
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(num_data),K,output_dim).T
        k = GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim, 0.00001)
        m = GPy.models.GPLVM(Y, input_dim, kernel = k)
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_rbf_kern(self):
        num_data, num_inducing, input_dim, output_dim = 10, 3, 2, 4
        X = np.random.rand(num_data, input_dim)
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(num_data),K,output_dim).T
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        m = GPy.models.GPLVM(Y, input_dim, kernel = k)
        m.randomize()
        self.assertTrue(m.checkgrad())

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
