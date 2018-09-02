'''
Created on 14 Jul 2017, based on gp_tests

@author: javdrher
'''
import unittest
import numpy as np, GPy


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(12345)
        self.N = 20
        self.N_new = 50
        self.D = 1
        self.X = np.random.uniform(-3., 3., (self.N, 1))
        self.Y = np.sin(self.X) + np.random.randn(self.N, self.D) * 0.05
        self.X_new = np.random.uniform(-3., 3., (self.N_new, 1))

    def test_setxy_gp(self):
        k = GPy.kern.RBF(1) + GPy.kern.White(1)
        m = GPy.models.TPRegression(self.X, self.Y, kernel=k)
        mu, var = m.predict(m.X)
        X = m.X.copy()
        m.set_XY(m.X[:10], m.Y[:10])
        assert (m.checkgrad(tolerance=1e-2))
        m.set_XY(X, self.Y)
        mu2, var2 = m.predict(m.X)
        np.testing.assert_allclose(mu, mu2)
        np.testing.assert_allclose(var, var2)

    def test_mean_function(self):
        from GPy.core.parameterization.param import Param
        from GPy.core.mapping import Mapping

        class Parabola(Mapping):
            def __init__(self, variance, degree=2, name='parabola'):
                super(Parabola, self).__init__(1, 1, name)
                self.variance = Param('variance', np.ones(degree + 1) * variance)
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
        k = GPy.kern.RBF(1) + GPy.kern.White(1)
        k.randomize()
        p = Parabola(.3)
        p.randomize()
        Y = p.f(X) + np.random.multivariate_normal(np.zeros(X.shape[0]), k.K(X) + np.eye(X.shape[0]) * 1e-8)[:,
                     None] + np.random.normal(0, .1, (X.shape[0], 1))
        m = GPy.models.TPRegression(X, Y, kernel=k, mean_function=p)
        assert (m.checkgrad(tolerance=2e-1))
        _ = m.predict(m.X)

    def test_normalizer(self):
        k = GPy.kern.RBF(1) + GPy.kern.White(1)
        Y = self.Y
        mu, std = Y.mean(0), Y.std(0)
        m = GPy.models.TPRegression(self.X, Y, kernel=k, normalizer=True)
        m.optimize()
        assert (m.checkgrad())
        k = GPy.kern.RBF(1) + GPy.kern.White(1)
        m2 = GPy.models.TPRegression(self.X, (Y - mu) / std, kernel=k, normalizer=False)
        m2[:] = m[:]

        mu1, var1 = m.predict(m.X, full_cov=True)
        mu2, var2 = m2.predict(m2.X, full_cov=True)
        np.testing.assert_allclose(mu1, (mu2 * std) + mu)
        np.testing.assert_allclose(var1, var2 * std ** 2)

        mu1, var1 = m.predict(m.X, full_cov=False)
        mu2, var2 = m2.predict(m2.X, full_cov=False)

        np.testing.assert_allclose(mu1, (mu2 * std) + mu)
        np.testing.assert_allclose(var1, var2 * std ** 2)

        q50n = m.predict_quantiles(m.X, (50,))
        q50 = m2.predict_quantiles(m2.X, (50,))

        np.testing.assert_allclose(q50n[0], (q50[0] * std) + mu)

        # Test variance component:
        qs = np.array([2.5, 97.5])
        # The quantiles get computed before unormalization
        # And transformed using the mean transformation:
        c = np.random.choice(self.X.shape[0])
        q95 = m2.predict_quantiles(self.X[[c]], qs)
        mu, var = m2.predict(self.X[[c]])
        from scipy.stats import t
        np.testing.assert_allclose((mu + (t.ppf(qs / 100., m2.nu + m2.num_data) * np.sqrt(var))).flatten(),
                                   np.array(q95).flatten())

    def test_predict_equivalence(self):
        k = GPy.kern.RBF(1) + GPy.kern.White(1)
        m = GPy.models.TPRegression(self.X, self.Y, kernel=k)
        m.optimize()
        mu1, var1 = m.predict(m.X)
        mu2, var2 = m.predict_noiseless(m.X)
        mu3, var3 = m._raw_predict(m.X)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        np.testing.assert_allclose(mu1, mu3)
        np.testing.assert_allclose(var1, var3)

        m2 = GPy.models.TPRegression(self.X, self.Y, kernel=k, normalizer=True)
        m2.optimize()
        mu1, var1 = m2.predict(m.X)
        mu2, var2 = m2.predict_noiseless(m.X)
        mu3, var3 = m2._raw_predict(m.X)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)
        self.assertFalse(np.allclose(mu1, mu3))
        self.assertFalse(np.allclose(var1, var3))

    def test_gp_equivalence(self):
        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.optimize()
        mu1, var1 = m.predict(self.X)
        k1 = GPy.kern.RBF(1)
        k1[:] = k[:]
        k2 = GPy.kern.White(1, variance=m.likelihood.variance)
        m2 = GPy.models.TPRegression(self.X, self.Y, kernel=k1 + k2, deg_free=1e6)
        mu2, var2 = m2.predict(self.X)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)


if __name__ == "__main__":
    unittest.main()
