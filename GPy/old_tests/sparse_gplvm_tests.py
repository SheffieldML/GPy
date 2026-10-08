# Copyright (c) 2012, Nicolo Fusi, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
from ..models import SparseGPLVM

class sparse_GPLVMTests(unittest.TestCase):
    def test_bias_kern(self):
        N, num_inducing, input_dim, D = 10, 3, 2, 4
        X = np.random.rand(N, input_dim)
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,input_dim).T
        k = GPy.kern.Bias(input_dim) + GPy.kern.White(input_dim, 0.00001)
        m = SparseGPLVM(Y, input_dim, kernel=k, num_inducing=num_inducing)
        m.randomize()
        assert m.checkgrad()

    def test_linear_kern(self):
        N, num_inducing, input_dim, D = 10, 3, 2, 4
        X = np.random.rand(N, input_dim)
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,input_dim).T
        k = GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim, 0.00001)
        m = SparseGPLVM(Y, input_dim, kernel=k, num_inducing=num_inducing)
        m.randomize()
        assert m.checkgrad()

    def test_rbf_kern(self):
        N, num_inducing, input_dim, D = 10, 3, 2, 4
        X = np.random.rand(N, input_dim)
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,input_dim).T
        k = GPy.kern.RBF(input_dim) + GPy.kern.White(input_dim, 0.00001)
        m = SparseGPLVM(Y, input_dim, kernel=k, num_inducing=num_inducing)
        m.randomize()
        assert m.checkgrad()

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
