"""
Created on 4 Sep 2015

@author: maxz
"""

import numpy as np
import GPy
from GPy.core.parameterization.variational import NormalPosterior


class TestGP:
    def setup_method(self):
        np.random.seed(12345)
        self.N = 20
        self.N_new = 50
        self.D = 1
        self.X = np.random.uniform(-3.0, 3.0, (self.N, 1))
        self.Y = np.sin(self.X) + np.random.randn(self.N, self.D) * 0.05
        self.X_new = np.random.uniform(-3.0, 3.0, (self.N_new, 1))

    def test_setxy_bgplvm(self):
        self.setup_method()

        k = GPy.kern.RBF(1)
        m = GPy.models.BayesianGPLVM(self.Y, 1, kernel=k)
        mu, var = m.predict(m.X)
        X = m.X
        Xnew = NormalPosterior(m.X.mean[:10].copy(), m.X.variance[:10].copy())
        m.set_XY(Xnew, m.Y[:10].copy())
        assert m.checkgrad()

        assert m.num_data == m.X.shape[0]
        assert m.input_dim == m.X.shape[1]

        m.set_XY(X, self.Y)
        mu2, var2 = m.predict(m.X)
        np.testing.assert_allclose(mu, mu2)
        np.testing.assert_allclose(var, var2)

    def test_setxy_gplvm(self):
        self.setup_method()

        k = GPy.kern.RBF(1)
        m = GPy.models.GPLVM(self.Y, 1, kernel=k)
        mu, var = m.predict(m.X)
        X = m.X.copy()
        Xnew = X[:10].copy()
        m.set_XY(Xnew, m.Y[:10].copy())
        assert m.checkgrad()

        assert m.num_data == m.X.shape[0]
        assert m.input_dim == m.X.shape[1]

        m.set_XY(X, self.Y)
        mu2, var2 = m.predict(m.X)
        np.testing.assert_allclose(mu, mu2)
        np.testing.assert_allclose(var, var2)

    def test_setxy_gp(self):
        self.setup_method()

        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        mu, var = m.predict(m.X)
        X = m.X.copy()
        m.set_XY(m.X[:10], m.Y[:10])
        assert m.checkgrad()

        assert m.num_data == m.X.shape[0]
        assert m.input_dim == m.X.shape[1]

        m.set_XY(X, self.Y)
        mu2, var2 = m.predict(m.X)
        np.testing.assert_allclose(mu, mu2)
        np.testing.assert_allclose(var, var2)

    def test_mean_function(self):
        from GPy.core.parameterization.param import Param
        from GPy.core.mapping import Mapping

        self.setup_method()

        class Parabola(Mapping):
            def __init__(self, variance, degree=2, name="parabola"):
                super(Parabola, self).__init__(1, 1, name)
                self.variance = Param("variance", np.ones(degree + 1) * variance)
                self.degree = degree
                self.link_parameter(self.variance)

            def f(self, X):
                p = self.variance[0] * np.ones(X.shape)
                for i in range(1, self.degree + 1):
                    p += self.variance[i] * X ** (i)
                return p

            def gradients_X(self, dL_dF, X):
                grad = np.zeros(X.shape)
                for i in range(1, self.degree + 1):
                    grad += (i) * self.variance[i] * X ** (i - 1)
                return grad

            def update_gradients(self, dL_dF, X):
                for i in range(self.degree + 1):
                    self.variance.gradient[i] = (dL_dF * X ** (i)).sum()

        X = np.linspace(-2, 2, 100)[:, None]
        k = GPy.kern.RBF(1)
        k.randomize()
        p = Parabola(0.3)
        p.randomize()
        Y = (
            p.f(X)
            + np.random.multivariate_normal(np.zeros(X.shape[0]), k.K(X) + np.eye(X.shape[0]) * 1e-8)[:, None]
            + np.random.normal(0, 0.1, (X.shape[0], 1))
        )
        m = GPy.models.GPRegression(X, Y, mean_function=p)
        m.randomize()
        assert m.checkgrad()
        _ = m.predict(m.X)

    def test_param_array_reload_via_model_slice(self):
        """README load recipe: m[:] restores predictions; bare param_array[:] does not (#1217)."""
        self.setup_method()

        def lengthscale_scalar(model):
            return float(np.asarray(model.rbf.lengthscale).ravel()[0])

        m = GPy.models.GPRegression(self.X, self.Y, kernel=GPy.kern.RBF(1))
        m.optimize(messages=False, max_iters=50)
        params = m.param_array.copy()
        mu0, _ = m.predict(self.X)
        ls0 = lengthscale_scalar(m)

        # Official README path (initialize=False + model slice assignment)
        m_load = GPy.models.GPRegression(self.X.copy(), self.Y.copy(), kernel=GPy.kern.RBF(1), initialize=False)
        m_load.update_model(False)
        m_load.initialize_parameter()
        m_load[:] = params
        m_load.update_model(True)
        mu_load, _ = m_load.predict(self.X)
        np.testing.assert_allclose(mu_load, mu0)
        np.testing.assert_allclose(lengthscale_scalar(m_load), ls0)

        # Footgun: writing the shared buffer without notifying observers
        m_bad = GPy.models.GPRegression(self.X.copy(), self.Y.copy(), kernel=GPy.kern.RBF(1))
        m_bad.param_array[:] = params
        mu_bad, _ = m_bad.predict(self.X)
        assert abs(lengthscale_scalar(m_bad) - 1.0) < 1e-12
        assert np.max(np.abs(mu0 - mu_bad)) > 1e-6

        # Same buffer write is fine if observers are triggered explicitly
        m_trig = GPy.models.GPRegression(self.X.copy(), self.Y.copy(), kernel=GPy.kern.RBF(1))
        m_trig.param_array[:] = params
        m_trig._trigger_params_changed()
        mu_trig, _ = m_trig.predict(self.X)
        np.testing.assert_allclose(mu_trig, mu0)
