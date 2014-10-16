# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class TieTests(unittest.TestCase):
    def test_tie_together(self):
        m = GPy.examples.regression.sparse_GP_regression_1D(optimize=False, plot=False, checkgrad=False)
        m.Z.constrain_positive(warning=False)
        m.Z.tie_together()
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.ties.checkTieTogether([m.Z]))
        self.assertTrue(m.checkgrad())

    def test_tie_together_two(self):
        m = GPy.examples.regression.sparse_GP_regression_1D(optimize=False, plot=False, checkgrad=False)
        m.Z.constrain_positive(warning=False)
        m.Z[:2].tie_together()
        m.Z[2:4].tie_together()
        self.assertTrue(m.ties.checkTieTogether([m.Z[:2]]))
        self.assertTrue(m.ties.checkTieTogether([m.Z[2:4]]))
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.checkgrad())

    def test_tie_together_merge(self):
        m = GPy.examples.regression.sparse_GP_regression_1D(optimize=False, plot=False, checkgrad=False)
        m.Z.constrain_positive(warning=False)
        m.Z[:2].tie_together()
        m.Z[1:3].tie_together()
        self.assertTrue(m.ties.checkTieTogether([m.Z[:3]]))
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.checkgrad())
        
    def test_tie_vector(self):
        m = GPy.examples.regression.sparse_GP_regression_1D(optimize=False, plot=False, checkgrad=False)
        m.Z.constrain_positive(warning=False)
        m.Z[:2].tie_vector(m.Z[2:4])
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.ties.checkTieVector([m.Z[:2],m.Z[2:4]]))
        self.assertTrue(m.checkgrad())
        
    def test_tie_multi_vector(self):
        m = GPy.examples.dimensionality_reduction.bgplvm_oil(N=100, optimize=False,plot=False)
        m.X.mean[:1].tie_vector(m.X.mean[1:2], m.X.mean[2:3])
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.ties.checkTieVector([m.X.mean[:1],m.X.mean[1:2], m.X.mean[2:3]]))
        self.assertTrue(m.checkgrad())
        
    def test_tie_vector_merge(self):
        m = GPy.examples.regression.sparse_GP_regression_2D(optimize=False, plot=False)
        m.Z.constrain_positive(warning=False)
        m.Z[:10].tie_vector(m.Z[10:20])
        m.Z[20:30].tie_vector(m.Z[30:40])
        m.Z[10:20].tie_vector(m.Z[20:30])
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.ties.checkTieVector([m.Z[:10],m.Z[10:20],m.Z[20:30],m.Z[30:40]]))
        self.assertTrue(m.checkgrad())
        
    def test_remove_tie(self):
        x = np.random.rand(100,1)
        y = np.random.rand(100,1)
        m = GPy.models.SparseGPRegression(x,y,kernel=GPy.kern.RBF(1)+GPy.kern.Matern32(1))
        m.kern.rbf.lengthscale.tie_together(m.kern.Mat32.lengthscale)
        m.Z[:1].tie_together(m.Z[1:2])
        m.kern.rbf.variance.tie_together(m.kern.Mat32.variance)
        m.kern.rbf.lengthscale.untie()
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.ties.checkTieTogether([m.kern.rbf.variance,m.kern.Mat32.variance]))
        self.assertTrue(m.ties.checkTieVector([m.Z[:1],m.Z[1:2]]))
        self.assertTrue(m.checkgrad())

    def test_tie_variational_posterior(self):
        m = GPy.examples.dimensionality_reduction.bgplvm_oil_100(plot=False,optimize=False)
        m.X[:10].tie_vector(m.X[10:20])
        self.assertTrue(m.ties.checkValueConsistency())
        self.assertTrue(m.ties.checkConstraintConsistency())
        self.assertTrue(m.ties.checkTieVector([m.X[:10],m.X[10:20]]))
        self.assertTrue(m.checkgrad())

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
