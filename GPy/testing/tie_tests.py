# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
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

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
