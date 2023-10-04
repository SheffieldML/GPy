"""
Created on 4 Sep 2015

@author: maxz
"""
import numpy as np
import GPy
from GPy.core.parameterization.variational import NormalPosterior


class TestGP:
    def setup(self):
        np.random.seed(12345)
        self.N = 20
        self.N_new = 50
        self.D = 1
        self.X = np.random.uniform(-3.0, 3.0, (self.N, 1))
        self.Y = np.sin(self.X) + np.random.randn(self.N, self.D) * 0.05
        self.X_new = np.random.uniform(-3.0, 3.0, (self.N_new, 1))

    def test_setxy_bgplvm(self):
        self.setup()

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
        self.setup()

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
        self.setup()

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

        self.setup()

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
                    self.variance.gradient[i] = (dL_dF * X ** (i)).sum(0)

        X = np.linspace(-2, 2, 100)[:, None]
        k = GPy.kern.RBF(1)
        k.randomize()
        p = Parabola(0.3)
        p.randomize()
        Y = (
            p.f(X)
            + np.random.multivariate_normal(
                np.zeros(X.shape[0]), k.K(X) + np.eye(X.shape[0]) * 1e-8
            )[:, None]
            + np.random.normal(0, 0.1, (X.shape[0], 1))
        )
        m = GPy.models.GPRegression(X, Y, mean_function=p)
        m.randomize()
        assert m.checkgrad()
        _ = m.predict(m.X)
