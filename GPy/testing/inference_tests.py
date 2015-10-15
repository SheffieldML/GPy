# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The test cases for various inference algorithms
"""

import unittest
import numpy as np
import GPy
#np.seterr(invalid='raise')

class InferenceXTestCase(unittest.TestCase):

    def genData(self):
        np.random.seed(1111)
        Ylist = GPy.examples.dimensionality_reduction._simulate_matern(5, 1, 1, 10, 3, False)[0]
        return Ylist[0]

    def test_inferenceX_BGPLVM_Linear(self):
        Ys = self.genData()
        m = GPy.models.BayesianGPLVM(Ys,3,kernel=GPy.kern.Linear(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X.mean, mi.X.mean, decimal=2)
        np.testing.assert_array_almost_equal(m.X.variance, mi.X.variance, decimal=2)

    def test_inferenceX_BGPLVM_RBF(self):
        Ys = self.genData()
        m = GPy.models.BayesianGPLVM(Ys,3,kernel=GPy.kern.RBF(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X.mean, mi.X.mean, decimal=2)
        np.testing.assert_array_almost_equal(m.X.variance, mi.X.variance, decimal=2)

    def test_inferenceX_GPLVM_Linear(self):
        Ys = self.genData()
        m = GPy.models.GPLVM(Ys,3,kernel=GPy.kern.Linear(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X, mi.X, decimal=2)

    def test_inferenceX_GPLVM_RBF(self):
        Ys = self.genData()
        m = GPy.models.GPLVM(Ys,3,kernel=GPy.kern.RBF(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X, mi.X, decimal=2)


if __name__ == "__main__":
    unittest.main()
