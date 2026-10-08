# Copyright (c) 2013, GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class BCGPLVMTests(unittest.TestCase):
    def test_kernel_backconstraint(self):
        num_data, num_inducing, input_dim, output_dim = 10, 3, 2, 4
        X = np.random.rand(num_data, input_dim)
        k = GPy.kern.rbf(input_dim) + GPy.kern.white(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(num_data),K,output_dim).T
        k = GPy.kern.mlp(input_dim) + GPy.kern.bias(input_dim)
        bk = GPy.kern.rbf(output_dim)
        mapping = GPy.mappings.Kernel(output_dim=input_dim, X=Y, kernel=bk)
        m = GPy.models.BCGPLVM(Y, input_dim, kernel = k, mapping=mapping)
        m.randomize()
        assert m.checkgrad()
        
    def test_linear_backconstraint(self):
        num_data, num_inducing, input_dim, output_dim = 10, 3, 2, 4
        X = np.random.rand(num_data, input_dim)
        k = GPy.kern.rbf(input_dim) + GPy.kern.white(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(num_data),K,output_dim).T
        k = GPy.kern.mlp(input_dim) + GPy.kern.bias(input_dim)
        bk = GPy.kern.rbf(output_dim)
        mapping = GPy.mappings.Linear(output_dim=input_dim, input_dim=output_dim)
        m = GPy.models.BCGPLVM(Y, input_dim, kernel = k, mapping=mapping)
        m.randomize()
        assert m.checkgrad()
        
    def test_mlp_backconstraint(self):
        num_data, num_inducing, input_dim, output_dim = 10, 3, 2, 4
        X = np.random.rand(num_data, input_dim)
        k = GPy.kern.rbf(input_dim) + GPy.kern.white(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(num_data),K,output_dim).T
        k = GPy.kern.mlp(input_dim) + GPy.kern.bias(input_dim)
        bk = GPy.kern.rbf(output_dim)
        mapping = GPy.mappings.MLP(output_dim=input_dim, input_dim=output_dim, hidden_dim=[5, 4, 7])
        m = GPy.models.BCGPLVM(Y, input_dim, kernel = k, mapping=mapping)
        m.randomize()
        assert m.checkgrad()

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
