'''
Created on 13 Mar 2014

@author: maxz
'''
import unittest, itertools
#import cPickle as pickle
import pickle
import numpy as np
from GPy.core.parameterization.index_operations import ParameterIndexOperations,\
    ParameterIndexOperationsView
import tempfile
from GPy.core.parameterization.param import Param
from GPy.core.parameterization.observable_array import ObsAr
from GPy.core.parameterization.priors import Gaussian
from GPy.kern._src.rbf import RBF
from GPy.kern._src.linear import Linear
from GPy.kern._src.static import Bias, White
from GPy.examples.dimensionality_reduction import mrd_simulation
from GPy.core.parameterization.variational import NormalPosterior
from GPy.models.gp_regression import GPRegression
from functools import reduce
from GPy.util.caching import Cacher
from pickle import PicklingError

def toy_model():
    X = np.linspace(0,1,50)[:, None]
    Y = np.sin(X)
    m = GPRegression(X=X, Y=Y)
    return m

class ListDictTestCase(unittest.TestCase):
    def assertListDictEquals(self, d1, d2, msg=None):
        #py3 fix
        #for k,v in d1.iteritems():
        for k,v in d1.items():
            self.assertListEqual(list(v), list(d2[k]), msg)
    def assertArrayListEquals(self, l1, l2):
        for a1, a2 in zip(l1,l2):
            np.testing.assert_array_equal(a1, a2)

class Test(ListDictTestCase):
    def test_parameter_index_operations(self):
        pio = ParameterIndexOperations(dict(test1=np.array([4,3,1,6,4]), test2=np.r_[2:130]))
        piov = ParameterIndexOperationsView(pio, 20, 250)
        #py3 fix
        #self.assertListDictEquals(dict(piov.items()), dict(piov.copy().iteritems()))
        self.assertListDictEquals(dict(piov.items()), dict(piov.copy().items()))

        #py3 fix
        #self.assertListDictEquals(dict(pio.iteritems()), dict(pio.copy().items()))
        self.assertListDictEquals(dict(pio.items()), dict(pio.copy().items()))

        self.assertArrayListEquals(pio.copy().indices(), pio.indices())
        self.assertArrayListEquals(piov.copy().indices(), piov.indices())

        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(pio, f)
            f.seek(0)
            pio2 = pickle.load(f)
            self.assertListDictEquals(pio._properties, pio2._properties)

        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(piov, f)
            f.seek(0)
            pio2 = pickle.load(f)
            #py3 fix
            #self.assertListDictEquals(dict(piov.items()), dict(pio2.iteritems()))
            self.assertListDictEquals(dict(piov.items()), dict(pio2.items()))

    def test_param(self):
        param = Param('test', np.arange(4*2).reshape(4,2))
        param[0].constrain_positive()
        param[1].fix()
        param[2].set_prior(Gaussian(0,1))
        pcopy = param.copy()
        self.assertListEqual(param.tolist(), pcopy.tolist())
        self.assertListEqual(str(param).split('\n'), str(pcopy).split('\n'))
        self.assertIsNot(param, pcopy)
        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(param, f)
            f.seek(0)
            pcopy = pickle.load(f)
        self.assertListEqual(param.tolist(), pcopy.tolist())
        self.assertSequenceEqual(str(param), str(pcopy))

    def test_observable_array(self):
        obs = ObsAr(np.arange(4*2).reshape(4,2))
        pcopy = obs.copy()
        self.assertListEqual(obs.tolist(), pcopy.tolist())
        with tempfile.TemporaryFile('w+b') as f:
            pickle.dump(obs, f)
            f.seek(0)
            pcopy = pickle.load(f)
        self.assertListEqual(obs.tolist(), pcopy.tolist())
        self.assertSequenceEqual(str(obs), str(pcopy))

    def test_parameterized(self):
        par = RBF(1, active_dims=[1]) + Linear(2, active_dims=[0,2]) + Bias(3) + White(3)
        par.gradient = 10
        par.randomize()
        pcopy = par.copy()
        self.assertIsInstance(pcopy.constraints, ParameterIndexOperations)
        self.assertIsInstance(pcopy.rbf.constraints, ParameterIndexOperationsView)
        self.assertIs(pcopy.constraints, pcopy.rbf.constraints._param_index_ops)
        self.assertIs(pcopy.constraints, pcopy.rbf.lengthscale.constraints._param_index_ops)
        self.assertIs(pcopy.constraints, pcopy.linear.constraints._param_index_ops)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        pcopy.gradient = 10 # gradient does not get copied anymore
        self.assertListEqual(par.gradient_full.tolist(), pcopy.gradient_full.tolist())
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assertIsNot(par.param_array, pcopy.param_array)
        self.assertIsNot(par.gradient_full, pcopy.gradient_full)
        with tempfile.TemporaryFile('w+b') as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        pcopy.gradient = 10
        np.testing.assert_allclose(par.linear.gradient_full, pcopy.linear.gradient_full)
        np.testing.assert_allclose(pcopy.linear.gradient_full, 10)
        self.assertSequenceEqual(str(par), str(pcopy))

    def test_model(self):
        par = toy_model()
        pcopy = par.copy()
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assertIsNot(par.param_array, pcopy.param_array)
        self.assertIsNot(par.gradient_full, pcopy.gradient_full)
        self.assertTrue(pcopy.checkgrad())
        self.assert_(np.any(pcopy.gradient!=0.0))
        with tempfile.TemporaryFile('w+b') as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assert_(pcopy.checkgrad())

    def test_modelrecreation(self):
        par = toy_model()
        pcopy = GPRegression(par.X.copy(), par.Y.copy(), kernel=par.kern.copy())
        np.testing.assert_allclose(par.param_array, pcopy.param_array)
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assertIsNot(par.param_array, pcopy.param_array)
        self.assertIsNot(par.gradient_full, pcopy.gradient_full)
        self.assertTrue(pcopy.checkgrad())
        self.assert_(np.any(pcopy.gradient!=0.0))
        np.testing.assert_allclose(pcopy.param_array, par.param_array, atol=1e-6)
        par.randomize()
        with tempfile.TemporaryFile('w+b') as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        np.testing.assert_allclose(par.param_array, pcopy.param_array)
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full, atol=1e-6)
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assert_(pcopy.checkgrad())

    def test_posterior(self):
        X = np.random.randn(3,5)
        Xv = np.random.rand(*X.shape)
        par = NormalPosterior(X,Xv)
        par.gradient = 10
        pcopy = par.copy()
        pcopy.gradient = 10
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        self.assertListEqual(par.gradient_full.tolist(), pcopy.gradient_full.tolist())
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assertIsNot(par.param_array, pcopy.param_array)
        self.assertIsNot(par.gradient_full, pcopy.gradient_full)
        with tempfile.TemporaryFile('w+b') as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        pcopy.gradient = 10
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        np.testing.assert_allclose(pcopy.mean.gradient_full, 10)
        self.assertSequenceEqual(str(par), str(pcopy))

    def test_model_concat(self):
        par = mrd_simulation(optimize=0, plot=0, plot_sim=0)
        par.randomize()
        pcopy = par.copy()
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        self.assertListEqual(par.gradient_full.tolist(), pcopy.gradient_full.tolist())
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assertIsNot(par.param_array, pcopy.param_array)
        self.assertIsNot(par.gradient_full, pcopy.gradient_full)
        self.assertTrue(par.checkgrad())
        self.assertTrue(pcopy.checkgrad())
        self.assert_(np.any(pcopy.gradient!=0.0))
        with tempfile.TemporaryFile('w+b') as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        self.assertListEqual(par.param_array.tolist(), pcopy.param_array.tolist())
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        self.assertSequenceEqual(str(par), str(pcopy))
        self.assert_(pcopy.checkgrad())

    def _callback(self, what, which):
        what.count += 1


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_parameter_index_operations']
    unittest.main()
