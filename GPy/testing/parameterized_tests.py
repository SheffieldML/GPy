'''
Created on Feb 13, 2014

@author: maxzwiessele
'''
import unittest
import GPy
import numpy as np
from GPy.core.parameterization.parameter_core import HierarchyError
from GPy.core.parameterization.observable_array import ObsAr

class ArrayCoreTest(unittest.TestCase):
    def setUp(self):
        self.X = np.random.normal(1,1, size=(100,10))
        self.obsX = ObsAr(self.X)

    def test_init(self):
        X = ObsAr(self.X)
        X2 = ObsAr(X)
        self.assertIs(X, X2, "no new Observable array, when Observable is given")

    def test_slice(self):
        t1 = self.X[2:78]
        t2 = self.obsX[2:78]
        self.assertListEqual(t1.tolist(), t2.tolist(), "Slicing should be the exact same, as in ndarray")

class ParameterizedTest(unittest.TestCase):

    def setUp(self):
        self.rbf = GPy.kern.RBF(1)
        self.white = GPy.kern.White(1)
        from GPy.core.parameterization import Param
        from GPy.core.parameterization.transformations import Logistic
        self.param = Param('param', np.random.rand(25,2), Logistic(0, 1))

        self.test1 = GPy.core.Parameterized("test model")
        self.test1.param = self.param
        self.test1.kern = self.rbf+self.white
        self.test1.add_parameter(self.test1.kern)
        self.test1.add_parameter(self.param, 0)

        x = np.linspace(-2,6,4)[:,None]
        y = np.sin(x)
        self.testmodel = GPy.models.GPRegression(x,y)

    def test_add_parameter(self):
        self.assertEquals(self.rbf._parent_index_, 0)
        self.assertEquals(self.white._parent_index_, 1)
        self.assertEquals(self.param._parent_index_, 0)
        pass

    def test_fixes(self):
        self.white.fix(warning=False)
        self.test1.remove_parameter(self.param)
        self.assertTrue(self.test1._has_fixes())
        from GPy.core.parameterization.transformations import FIXED, UNFIXED
        self.assertListEqual(self.test1._fixes_.tolist(),[UNFIXED,UNFIXED,FIXED])
        self.test1.kern.add_parameter(self.white, 0)
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED,UNFIXED,UNFIXED])
        self.test1.kern.rbf.fix()
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED]*3)
        self.test1.fix()
        self.assertTrue(self.test1.is_fixed)
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED]*self.test1.size)

    def test_remove_parameter(self):
        from GPy.core.parameterization.transformations import FIXED, UNFIXED, __fixed__, Logexp
        self.white.fix()
        self.test1.kern.remove_parameter(self.white)
        self.assertIs(self.test1._fixes_,None)

        self.assertListEqual(self.white._fixes_.tolist(), [FIXED])
        self.assertEquals(self.white.constraints._offset, 0)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)

        self.test1.add_parameter(self.white, 0)
        self.assertIs(self.test1.constraints, self.white.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)
        self.assertListEqual(self.test1.constraints[__fixed__].tolist(), [0])
        self.assertIs(self.white._fixes_,None)
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED] + [UNFIXED] * 52)

        self.test1.remove_parameter(self.white)
        self.assertIs(self.test1._fixes_,None)
        self.assertListEqual(self.white._fixes_.tolist(), [FIXED])
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)
        self.assertListEqual(self.test1.constraints[Logexp()].tolist(), range(self.param.size, self.param.size+self.rbf.size))

    def test_remove_parameter_param_array_grad_array(self):
        val = self.test1.kern.param_array.copy()
        self.test1.kern.remove_parameter(self.white)
        self.assertListEqual(self.test1.kern.param_array.tolist(), val[:2].tolist())

    def test_add_parameter_already_in_hirarchy(self):
        self.assertRaises(HierarchyError, self.test1.add_parameter, self.white._parameters_[0])

    def test_default_constraints(self):
        self.assertIs(self.rbf.variance.constraints._param_index_ops, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertListEqual(self.rbf.constraints.indices()[0].tolist(), range(2))
        from GPy.core.parameterization.transformations import Logexp
        kern = self.test1.kern
        self.test1.remove_parameter(kern)
        self.assertListEqual(kern.constraints[Logexp()].tolist(), range(3))

    def test_constraints(self):
        self.rbf.constrain(GPy.transformations.Square(), False)
        self.assertListEqual(self.test1.constraints[GPy.transformations.Square()].tolist(), range(self.param.size, self.param.size+self.rbf.size))
        self.assertListEqual(self.test1.constraints[GPy.transformations.Logexp()].tolist(), [self.param.size+self.rbf.size])

        self.test1.kern.remove_parameter(self.rbf)
        self.assertListEqual(self.test1.constraints[GPy.transformations.Square()].tolist(), [])

    def test_constraints_views(self):
        self.assertEqual(self.white.constraints._offset, self.param.size+self.rbf.size)
        self.assertEqual(self.rbf.constraints._offset, self.param.size)
        self.assertEqual(self.param.constraints._offset, 0)

    def test_fixing_randomize(self):
        self.white.fix(warning=True)
        val = float(self.white.variance)
        self.test1.randomize()
        self.assertEqual(val, self.white.variance)

    def test_randomize(self):
        ps = self.test1.param.view(np.ndarray).copy()
        self.test1.param.randomize()
        self.assertFalse(np.all(ps==self.test1.param))

    def test_fixing_randomize_parameter_handling(self):
        self.rbf.fix(warning=True)
        val = float(self.rbf.variance)
        self.test1.kern.randomize()
        self.assertEqual(val, self.rbf.variance)

    def test_fixing_optimize(self):
        self.testmodel.kern.lengthscale.fix()
        val = float(self.testmodel.kern.lengthscale)
        self.testmodel.randomize()
        self.assertEqual(val, self.testmodel.kern.lengthscale)

    def test_regular_expression_misc(self):
        self.testmodel.kern.lengthscale.fix()
        val = float(self.testmodel.kern.lengthscale)
        self.testmodel.randomize()
        self.assertEqual(val, self.testmodel.kern.lengthscale)

        variances = self.testmodel['.*var'].values()
        self.testmodel['.*var'].fix()
        self.testmodel.randomize()
        np.testing.assert_equal(variances, self.testmodel['.*var'].values())

    def test_fix_unfix(self):
        fixed = self.testmodel.kern.lengthscale.fix()
        self.assertListEqual(fixed.tolist(), [0])
        unfixed = self.testmodel.kern.lengthscale.unfix()
        self.testmodel.kern.lengthscale.constrain_positive()
        self.assertListEqual(unfixed.tolist(), [0])

        fixed = self.testmodel.kern.fix()
        self.assertListEqual(fixed.tolist(), [0,1])
        unfixed = self.testmodel.kern.unfix()
        self.assertListEqual(unfixed.tolist(), [0,1])

    def test_printing(self):
        print self.test1
        print self.param
        print self.test1['']

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_add_parameter']
    unittest.main()