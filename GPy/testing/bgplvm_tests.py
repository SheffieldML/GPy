# Copyright (c) 2012, Nicolo Fusi
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class BGPLVMTests(unittest.TestCase):
    def test_bias_kern(self):
        N, M, Q, D = 10, 3, 2, 4
        X = np.random.rand(N, Q)
        k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        Y -= Y.mean(axis=0)
        k = GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
        m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
        m.ensure_default_constraints()
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_linear_kern(self):
        N, M, Q, D = 10, 3, 2, 4
        X = np.random.rand(N, Q)
        k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        Y -= Y.mean(axis=0)
        k = GPy.kern.linear(Q) + GPy.kern.white(Q, 0.00001)
        m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
        m.ensure_default_constraints()
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_rbf_kern(self):
        N, M, Q, D = 10, 3, 2, 4
        X = np.random.rand(N, Q)
        k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        Y -= Y.mean(axis=0)
        k = GPy.kern.rbf(Q) + GPy.kern.white(Q, 0.00001)
        m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
        m.ensure_default_constraints()
        m.randomize()
        self.assertTrue(m.checkgrad())

    def test_rbf_bias_kern(self):
        N, M, Q, D = 10, 3, 2, 4
        X = np.random.rand(N, Q)
        k = GPy.kern.rbf(Q) +  GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        Y -= Y.mean(axis=0)
        k = GPy.kern.rbf(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
        m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
        m.ensure_default_constraints()
        m.randomize()
        self.assertTrue(m.checkgrad())

    #@unittest.skip('psi2 cross terms are NotImplemented for this combination')
    def test_linear_bias_kern(self):
        N, M, Q, D = 30, 5, 4, 30
        X = np.random.rand(N, Q)
        k = GPy.kern.linear(Q) +  GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N),K,D).T
        Y -= Y.mean(axis=0)
        k = GPy.kern.linear(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
        m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k,  M=M)
        m.ensure_default_constraints()
        m.randomize()
        self.assertTrue(m.checkgrad())


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
