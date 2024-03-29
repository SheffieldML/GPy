"""
Created on 13 Mar 2014

@author: maxz
"""
# import cPickle as pickle
import pickle
import pytest
import numpy as np
import tempfile
from GPy.examples.dimensionality_reduction import mrd_simulation
from GPy.core.parameterization.variational import NormalPosterior
from GPy.models.gp_regression import GPRegression
import GPy


def toy_model():
    X = np.linspace(0, 1, 50)[:, None]
    Y = np.sin(X)
    m = GPRegression(X=X, Y=Y)
    return m


class ListDictTestCase:
    def assertListDictEquals(self, d1, d2, msg=None):
        # py3 fix
        # for k,v in d1.iteritems():
        for k, v in d1.items():
            self.assertListEqual(list(v), list(d2[k]), msg)

    def assertArrayListEquals(self, l1, l2):
        for a1, a2 in zip(l1, l2):
            np.testing.assert_array_equal(a1, a2)


class TestPickleSupport(ListDictTestCase):
    @pytest.mark.skip(reason="")  # why is this test skipped?
    def test_load_pickle(self):
        import os

        m = GPy.load(
            os.path.join(
                os.path.abspath(os.path.split(__file__)[0]), "pickle_test.pickle"
            )
        )
        assert m.checkgrad()
        assert m.log_likelihood(), -4.7351019830022087

    def test_model(self):
        par = toy_model()
        pcopy = par.copy()
        assert par.param_array.tolist() == pcopy.param_array.tolist()
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        assert str(par) == str(pcopy)
        assert np.all(par.param_array == pcopy.param_array)
        assert np.all(par.gradient_full == pcopy.gradient_full)
        assert pcopy.checkgrad()
        assert np.any(pcopy.gradient != 0.0)
        with tempfile.TemporaryFile("w+b") as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        assert par.param_array.tolist() == pcopy.param_array.tolist()
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        assert str(par) == str(pcopy)
        assert pcopy.checkgrad()

    def test_modelrecreation(self):
        par = toy_model()
        pcopy = GPRegression(par.X.copy(), par.Y.copy(), kernel=par.kern.copy())
        np.testing.assert_allclose(par.param_array, pcopy.param_array)
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        assert str(par) == str(pcopy)
        assert np.all(par.param_array == pcopy.param_array)
        assert np.all(par.gradient_full == pcopy.gradient_full)
        assert pcopy.checkgrad()
        assert np.any(pcopy.gradient != 0.0)
        np.testing.assert_allclose(pcopy.param_array, par.param_array, atol=1e-6)
        par.randomize()
        with tempfile.TemporaryFile("w+b") as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        np.testing.assert_allclose(par.param_array, pcopy.param_array)
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full, atol=1e-6)
        assert str(par) == str(pcopy)
        assert pcopy.checkgrad()

    def test_posterior(self):
        X = np.random.randn(3, 5)
        Xv = np.random.rand(*X.shape)
        par = NormalPosterior(X, Xv)
        par.gradient = 10
        pcopy = par.copy()
        pcopy.gradient = 10
        assert par.param_array.tolist() == pcopy.param_array.tolist()
        assert par.gradient_full.tolist() == pcopy.gradient_full.tolist()
        assert str(par) == str(pcopy)
        assert np.all(par.param_array == pcopy.param_array)
        assert np.all(par.gradient_full == pcopy.gradient_full)
        with tempfile.TemporaryFile("w+b") as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        assert par.param_array.tolist() == pcopy.param_array.tolist()
        pcopy.gradient = 10
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        np.testing.assert_allclose(pcopy.mean.gradient_full, 10)
        assert str(par) == str(pcopy)

    def test_model_concat(self):
        par = mrd_simulation(optimize=0, plot=0, plot_sim=0)
        par.randomize()
        pcopy = par.copy()
        assert par.param_array.tolist() == pcopy.param_array.tolist()
        assert par.gradient_full.tolist() == pcopy.gradient_full.tolist()
        assert str(par) == str(pcopy)
        assert np.all(par.param_array == pcopy.param_array)
        assert np.all(par.gradient_full == pcopy.gradient_full)
        assert par.checkgrad()
        assert pcopy.checkgrad()
        assert np.any(pcopy.gradient != 0.0)
        with tempfile.TemporaryFile("w+b") as f:
            par.pickle(f)
            f.seek(0)
            pcopy = pickle.load(f)
        assert par.param_array.tolist() == pcopy.param_array.tolist()
        np.testing.assert_allclose(par.gradient_full, pcopy.gradient_full)
        assert str(par) == str(pcopy)
        assert pcopy.checkgrad()

    def _callback(self, what, which):
        what.count += 1
