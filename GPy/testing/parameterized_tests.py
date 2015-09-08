'''
Created on Feb 13, 2014

@author: maxzwiessele
'''
import unittest
import GPy
import numpy as np
from GPy.core.parameterization.parameter_core import HierarchyError
from GPy.core.parameterization.observable_array import ObsAr
from GPy.core.parameterization.transformations import NegativeLogexp, Logistic
from GPy.core.parameterization.parameterized import Parameterized
from GPy.core.parameterization.param import Param
from GPy.core.parameterization.index_operations import ParameterIndexOperations
from functools import reduce

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
        self.rbf = GPy.kern.RBF(20)
        self.white = GPy.kern.White(1)
        from GPy.core.parameterization import Param
        from GPy.core.parameterization.transformations import Logistic
        self.param = Param('param', np.random.uniform(0,1,(10,5)), Logistic(0, 1))

        self.test1 = GPy.core.Parameterized("test model")
        self.test1.param = self.param
        self.test1.kern = self.rbf+self.white
        self.test1.link_parameter(self.test1.kern)
        self.test1.link_parameter(self.param, 0)

        # print self.test1:
        #=============================================================================
        # test_model.          |    Value    |  Constraint   |  Prior  |  Tied to
        # param                |  (25L, 2L)  |   {0.0,1.0}   |         |
        # add.rbf.variance     |        1.0  |  0.0,1.0 +ve  |         |
        # add.rbf.lengthscale  |        1.0  |  0.0,1.0 +ve  |         |
        # add.white.variance   |        1.0  |  0.0,1.0 +ve  |         |
        #=============================================================================

        x = np.linspace(-2,6,4)[:,None]
        y = np.sin(x)
        self.testmodel = GPy.models.GPRegression(x,y)
        # print self.testmodel:
        #=============================================================================
        # GP_regression.           |  Value  |  Constraint  |  Prior  |  Tied to
        # rbf.variance             |    1.0  |     +ve      |         |
        # rbf.lengthscale          |    1.0  |     +ve      |         |
        # Gaussian_noise.variance  |    1.0  |     +ve      |         |
        #=============================================================================

    def test_add_parameter(self):
        self.assertEquals(self.rbf._parent_index_, 0)
        self.assertEquals(self.white._parent_index_, 1)
        self.assertEquals(self.param._parent_index_, 0)
        pass

    def test_fixes(self):
        self.white.fix(warning=False)
        self.test1.unlink_parameter(self.param)
        self.assertTrue(self.test1._has_fixes())
        from GPy.core.parameterization.transformations import FIXED, UNFIXED
        self.assertListEqual(self.test1._fixes_.tolist(),[UNFIXED,UNFIXED,FIXED])
        self.test1.kern.link_parameter(self.white, 0)
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED,UNFIXED,UNFIXED])
        self.test1.kern.rbf.fix()
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED]*3)
        self.test1.fix()
        self.assertTrue(self.test1.is_fixed)
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED]*self.test1.size)

    def test_remove_parameter(self):
        from GPy.core.parameterization.transformations import FIXED, UNFIXED, __fixed__, Logexp
        self.white.fix()
        self.test1.kern.unlink_parameter(self.white)
        self.assertIs(self.test1._fixes_,None)

        self.assertIsInstance(self.white.constraints, ParameterIndexOperations)
        self.assertListEqual(self.white._fixes_.tolist(), [FIXED])
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)

        self.test1.link_parameter(self.white, 0)
        self.assertIs(self.test1.constraints, self.white.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)
        self.assertListEqual(self.test1.constraints[__fixed__].tolist(), [0])
        self.assertIs(self.white._fixes_,None)
        self.assertListEqual(self.test1._fixes_.tolist(),[FIXED] + [UNFIXED] * 52)

        self.test1.unlink_parameter(self.white)
        self.assertIs(self.test1._fixes_,None)
        self.assertListEqual(self.white._fixes_.tolist(), [FIXED])
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.param.constraints._param_index_ops)
        self.assertListEqual(self.test1.constraints[Logexp()].tolist(), list(range(self.param.size, self.param.size+self.rbf.size)))

    def test_remove_parameter_param_array_grad_array(self):
        val = self.test1.kern.param_array.copy()
        self.test1.kern.unlink_parameter(self.white)
        self.assertListEqual(self.test1.kern.param_array.tolist(), val[:2].tolist())

    def test_add_parameter_already_in_hirarchy(self):
        self.assertRaises(HierarchyError, self.test1.link_parameter, self.white.parameters[0])

    def test_default_constraints(self):
        self.assertIs(self.rbf.variance.constraints._param_index_ops, self.rbf.constraints._param_index_ops)
        self.assertIs(self.test1.constraints, self.rbf.constraints._param_index_ops)
        self.assertListEqual(self.rbf.constraints.indices()[0].tolist(), list(range(2)))
        from GPy.core.parameterization.transformations import Logexp
        kern = self.test1.kern
        self.test1.unlink_parameter(kern)
        self.assertListEqual(kern.constraints[Logexp()].tolist(), list(range(3)))

    def test_constraints(self):
        self.rbf.constrain(GPy.transformations.Square(), False)
        self.assertListEqual(self.test1.constraints[GPy.transformations.Square()].tolist(), list(range(self.param.size, self.param.size+self.rbf.size)))
        self.assertListEqual(self.test1.constraints[GPy.transformations.Logexp()].tolist(), [self.param.size+self.rbf.size])

        self.test1.kern.unlink_parameter(self.rbf)
        self.assertListEqual(self.test1.constraints[GPy.transformations.Square()].tolist(), [])

    def test_constraints_link_unlink(self):
        self.test1.unlink_parameter(self.test1.kern)
        self.test1.kern.rbf.unlink_parameter(self.test1.kern.rbf.lengthscale)
        self.test1.kern.rbf.link_parameter(self.test1.kern.rbf.lengthscale)
        self.test1.kern.rbf.unlink_parameter(self.test1.kern.rbf.lengthscale)
        self.test1.link_parameter(self.test1.kern)

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
        self.test1.param[2:5].fix()
        self.test1.param.randomize()
        self.assertFalse(np.all(ps==self.test1.param),str(ps)+str(self.test1.param))

    def test_fixing_randomize_parameter_handling(self):
        self.rbf.fix(warning=True)
        val = float(self.rbf.variance)
        self.test1.kern.randomize()
        self.assertEqual(val, self.rbf.variance)

    def test_updates(self):
        val = float(self.testmodel.log_likelihood())
        self.testmodel.update_model(False)
        self.testmodel.kern.randomize()
        self.testmodel.likelihood.randomize()
        self.assertEqual(val, self.testmodel.log_likelihood())
        self.testmodel.update_model(True)
        self.assertNotEqual(val, self.testmodel.log_likelihood())

    def test_fixing_optimize(self):
        self.testmodel.kern.lengthscale.fix()
        val = float(self.testmodel.kern.lengthscale)
        self.testmodel.randomize()
        self.assertEqual(val, self.testmodel.kern.lengthscale)

    def test_add_parameter_in_hierarchy(self):
        self.test1.kern.rbf.link_parameter(Param("NEW", np.random.rand(2), NegativeLogexp()), 1)
        self.assertListEqual(self.test1.constraints[NegativeLogexp()].tolist(), list(range(self.param.size+1, self.param.size+1 + 2)))
        self.assertListEqual(self.test1.constraints[GPy.transformations.Logistic(0,1)].tolist(), list(range(self.param.size)))
        self.assertListEqual(self.test1.constraints[GPy.transformations.Logexp(0,1)].tolist(), np.r_[50, 53:55].tolist())

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

    def test_constraints_in_init(self):
        class Test(Parameterized):
            def __init__(self, name=None, parameters=[], *a, **kw):
                super(Test, self).__init__(name=name)
                self.x = Param('x', np.random.uniform(0,1,(3,4)))
                self.x[0].constrain_bounded(0,1)
                self.link_parameter(self.x)
                self.x[1].fix()
        t = Test()
        c = {Logistic(0,1): np.array([0, 1, 2, 3]), 'fixed': np.array([4, 5, 6, 7])}
        np.testing.assert_equal(t.x.constraints[Logistic(0,1)], c[Logistic(0,1)])
        np.testing.assert_equal(t.x.constraints['fixed'], c['fixed'])

    def test_parameter_modify_in_init(self):
        class TestLikelihood(Parameterized):
            def __init__(self, param1 = 2., param2 = 3.):
                super(TestLikelihood, self).__init__("TestLike")
                self.p1 = Param('param1', param1)
                self.p2 = Param('param2', param2)

                self.link_parameter(self.p1)
                self.link_parameter(self.p2)

                self.p1.fix()
                self.p1.unfix()
                self.p2.constrain_negative()
                self.p1.fix()
                self.p2.constrain_positive()
                self.p2.fix()
                self.p2.constrain_positive()

        m = TestLikelihood()
        print(m)
        val = m.p1.values.copy()
        self.assert_(m.p1.is_fixed)
        self.assert_(m.constraints[GPy.constraints.Logexp()].tolist(), [1])
        m.randomize()
        self.assertEqual(m.p1, val)

    def test_checkgrad(self):
        assert(self.testmodel.kern.checkgrad())
        assert(self.testmodel.kern.lengthscale.checkgrad())
        assert(self.testmodel.likelihood.checkgrad())

    def test_printing(self):
        print(self.test1)
        print(self.param)
        print(self.test1[''])
        print(self.testmodel.hierarchy_name(False))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_add_parameter']
    unittest.main()
