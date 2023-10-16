# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import pytest
import numpy as np
import GPy
from functools import reduce

try:
    import autograd
except ImportError:
    autograd = None


class TestMisc:
    def setup(self):
        self.N = 20
        self.N_new = 50
        self.D = 1
        self.X = np.random.uniform(-3.0, 3.0, (self.N, 1))
        self.Y = np.sin(self.X) + np.random.randn(self.N, self.D) * 0.05
        self.X_new = np.random.uniform(-3.0, 3.0, (self.N_new, 1))

    def test_setXY(self):
        self.setup()
        m = GPy.models.GPRegression(self.X, self.Y)
        m.set_XY(
            np.vstack([self.X, np.random.rand(1, self.X.shape[1])]),
            np.vstack([self.Y, np.random.rand(1, self.Y.shape[1])]),
        )
        m._trigger_params_changed()
        assert m.checkgrad()
        m.predict(m.X)

    def test_raw_predict_numerical_stability(self):
        """
        Test whether the predicted variance of normal GP goes negative under numerical unstable situation.
        Thanks simbartonels@github for reporting the bug and providing the following example.
        """
        self.setup()

        # set seed for reproducability
        np.random.seed(3)

        # Definition of the Branin test function
        def branin(X):
            y = (
                X[:, 1]
                - 5.1 / (4 * np.pi**2) * X[:, 0] ** 2
                + 5 * X[:, 0] / np.pi
                - 6
            ) ** 2
            y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(X[:, 0]) + 10
            return y

        # Training set defined as a 5*5 grid:
        xg1 = np.linspace(-5, 10, 5)
        xg2 = np.linspace(0, 15, 5)
        X = np.zeros((xg1.size * xg2.size, 2))
        for i, x1 in enumerate(xg1):
            for j, x2 in enumerate(xg2):
                X[i + xg1.size * j, :] = [x1, x2]
        Y = branin(X)[:, None]
        # Fit a GP
        # Create an exponentiated quadratic plus bias covariance function
        k = GPy.kern.RBF(input_dim=2, ARD=True)
        # Build a GP model
        m = GPy.models.GPRegression(X, Y, k)
        # fix the noise variance
        m.likelihood.variance.fix(1e-5)
        # Randomize the model and optimize
        m.randomize()
        m.optimize()
        # Compute the mean of model prediction on 1e5 Monte Carlo samples
        Xp = np.random.uniform(size=(int(1e5), 2))
        Xp[:, 0] = Xp[:, 0] * 15 - 5
        Xp[:, 1] = Xp[:, 1] * 15
        _, var = m.predict(Xp)
        assert np.all(var >= 0.0))

    def test_raw_predict(self):
        self.setup()
        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.randomize()
        m.likelihood.variance = 0.5
        Kinv = np.linalg.pinv(k.K(self.X) + np.eye(self.N) * m.likelihood.variance)
        K_hat = k.K(self.X_new) - k.K(self.X_new, self.X).dot(Kinv).dot(
            k.K(self.X, self.X_new)
        )
        mu_hat = k.K(self.X_new, self.X).dot(Kinv).dot(m.Y_normalized)

        mu, covar = m.predict_noiseless(self.X_new, full_cov=True)
        assert mu.shape == (self.N_new, self.D)
        assert covar.shape == (self.N_new, self.N_new)
        np.testing.assert_almost_equal(K_hat, covar)
        np.testing.assert_almost_equal(mu_hat, mu)

        mu, var = m.predict_noiseless(self.X_new)
        assert mu.shape == (self.N_new, self.D)
        assert var.shape == (self.N_new, 1)
        np.testing.assert_almost_equal(np.diag(K_hat)[:, None], var)
        np.testing.assert_almost_equal(mu_hat, mu)

    def test_normalizer(self):
        self.setup()
        k = GPy.kern.RBF(1)
        Y = self.Y
        mu, std = Y.mean(0), Y.std(0)
        m = GPy.models.GPRegression(self.X, Y, kernel=k, normalizer=True)
        m.optimize(messages=True)
        assert m.checkgrad()
        k = GPy.kern.RBF(1)
        m2 = GPy.models.GPRegression(self.X, (Y - mu) / std, kernel=k, normalizer=False)
        m2[:] = m[:]

        mu1, var1 = m.predict(m.X, full_cov=True)
        mu2, var2 = m2.predict(m2.X, full_cov=True)
        np.testing.assert_allclose(mu1, (mu2 * std) + mu)
        np.testing.assert_allclose(var1, var2 * std**2)

        mu1, var1 = m.predict(m.X, full_cov=False)
        mu2, var2 = m2.predict(m2.X, full_cov=False)

        np.testing.assert_allclose(mu1, (mu2 * std) + mu)
        np.testing.assert_allclose(var1, var2 * std**2)

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
        from scipy.stats import norm

        np.testing.assert_allclose(
            (mu + (norm.ppf(qs / 100.0) * np.sqrt(var))).flatten(),
            np.array(q95).flatten(),
        )

    def test_multioutput_regression_with_normalizer(self):
        """
        Test that normalizing works in multi-output case
        """
        self.setup()

        # Create test inputs
        X = self.X
        Y1 = np.sin(X) + np.random.randn(*X.shape) * 0.2
        Y2 = -np.sin(X) + np.random.randn(*X.shape) * 0.05
        Y = np.hstack((Y1, Y2))

        mu, std = Y.mean(0), Y.std(0)
        m = GPy.models.GPRegression(X, Y, normalizer=True)
        m.optimize(messages=True)
        assert m.checkgrad()
        k = GPy.kern.RBF(1)
        m2 = GPy.models.GPRegression(X, (Y - mu) / std, normalizer=False)
        m2[:] = m[:]

        mu1, var1 = m.predict(m.X, full_cov=True)
        mu2, var2 = m2.predict(m2.X, full_cov=True)
        np.testing.assert_allclose(mu1, (mu2 * std) + mu)
        np.testing.assert_allclose(var1, var2[:, :, None] * std[None, None, :] ** 2)

        mu1, var1 = m.predict(m.X, full_cov=False)
        mu2, var2 = m2.predict(m2.X, full_cov=False)

        np.testing.assert_allclose(mu1, (mu2 * std) + mu)
        np.testing.assert_allclose(var1, var2 * std[None, :] ** 2)

        q50n = m.predict_quantiles(m.X, (50,))
        q50 = m2.predict_quantiles(m2.X, (50,))

        np.testing.assert_allclose(q50n[0], (q50[0] * std) + mu)

        # Test variance component:
        qs = np.array([2.5, 97.5])
        # The quantiles get computed before unormalization
        # And transformed using the mean transformation:
        c = np.random.choice(X.shape[0])
        q95 = m2.predict_quantiles(X[[c]], qs)
        mu, var = m2.predict(X[[c]])
        from scipy.stats import norm

        np.testing.assert_allclose(
            (mu.T + (norm.ppf(qs / 100.0) * np.sqrt(var))).T.flatten(),
            np.array(q95).flatten(),
        )

    @pytest.mark.skipif(
        autograd is None, reason="autograd not available to check gradients"
    )
    def test_jacobian(self):
        import autograd.numpy as np, autograd as ag, GPy, matplotlib.pyplot as plt
        from GPy.models import GradientChecker, GPRegression

        def k(X, X2, alpha=1.0, lengthscale=None):
            if lengthscale is None:
                lengthscale = np.ones(X.shape[1])
            exp = 0.0
            for q in range(X.shape[1]):
                exp += ((X[:, [q]] - X2[:, [q]].T) / lengthscale[q]) ** 2
            # exp = np.sqrt(exp)
            return alpha * np.exp(-0.5 * exp)

        dk = ag.elementwise_grad(
            lambda x, x2: k(
                x, x2, alpha=ke.variance.values, lengthscale=ke.lengthscale.values
            )
        )
        dkdk = ag.elementwise_grad(dk, argnum=1)

        ke = GPy.kern.RBF(1, ARD=True)
        # ke.randomize()
        ke.variance = 0.2  # .randomize()
        ke.lengthscale[:] = 0.5
        ke.randomize()
        X = np.linspace(-1, 1, 1000)[:, None]
        X2 = np.array([[0.0]]).T
        np.testing.assert_allclose(ke.gradients_X([[1.0]], X, X), dk(X, X))
        np.testing.assert_allclose(ke.gradients_XX([[1.0]], X, X).sum(0), dkdk(X, X))
        np.testing.assert_allclose(ke.gradients_X([[1.0]], X, X2), dk(X, X2))
        np.testing.assert_allclose(ke.gradients_XX([[1.0]], X, X2).sum(0), dkdk(X, X2))

        m = GPRegression(self.X, self.Y)

        def f(x):
            m.X[:] = x
            return m.log_likelihood()

        def df(x):
            m.X[:] = x
            return m.kern.gradients_X(m.grad_dict["dL_dK"], X)

        def ddf(x):
            m.X[:] = x
            return m.kern.gradients_XX(m.grad_dict["dL_dK"], X).sum(0)

        gc = GradientChecker(f, df, self.X)
        gc2 = GradientChecker(df, ddf, self.X)
        assert gc.checkgrad()
        assert gc2.checkgrad()

    def test_predict_uncertain_inputs(self):
        """Projection of Gaussian through a linear function is still gaussian, and moments are analytical to compute, so we can check this case for predictions easily"""
        self.setup()

        X = np.linspace(-5, 5, 10)[:, None]
        Y = 2 * X + np.random.randn(*X.shape) * 1e-3
        m = GPy.models.BayesianGPLVM(
            Y, 1, X=X, kernel=GPy.kern.Linear(1), num_inducing=1
        )
        m.Gaussian_noise[:] = 1e-4
        m.X.mean[:] = X[:]
        m.X.variance[:] = 1e-5
        m.X.fix()
        m.optimize()
        X_pred_mu = np.random.randn(5, 1)
        X_pred_var = np.random.rand(5, 1) + 1e-5
        from GPy.core.parameterization.variational import NormalPosterior

        X_pred = NormalPosterior(X_pred_mu, X_pred_var)
        # mu = \int f(x)q(x|mu,S) dx = \int 2x.q(x|mu,S) dx = 2.mu
        # S = \int (f(x) - m)^2q(x|mu,S) dx = \int f(x)^2 q(x) dx - mu**2 = 4(mu^2 + S) - (2.mu)^2 = 4S
        Y_mu_true = 2 * X_pred_mu
        Y_var_true = 4 * X_pred_var
        Y_mu_pred, Y_var_pred = m.predict_noiseless(X_pred)
        np.testing.assert_allclose(Y_mu_true, Y_mu_pred, rtol=1e-3)
        np.testing.assert_allclose(Y_var_true, Y_var_pred, rtol=1e-3)

    def test_sparse_raw_predict(self):
        self.setup()

        k = GPy.kern.RBF(1)
        m = GPy.models.SparseGPRegression(self.X, self.Y, kernel=k)
        m.randomize()
        Z = m.Z[:]

        # Not easy to check if woodbury_inv is correct in itself as it requires a large derivation and expression
        Kinv = m.posterior.woodbury_inv
        K_hat = k.K(self.X_new) - k.K(self.X_new, Z).dot(Kinv).dot(k.K(Z, self.X_new))
        # K_hat = np.clip(K_hat, 1e-15, np.inf)

        mu, covar = m.predict_noiseless(self.X_new, full_cov=True)
        assert mu.shape == (self.N_new, self.D)
        assert covar.shape == (self.N_new, self.N_new)
        np.testing.assert_almost_equal(K_hat, covar)
        # np.testing.assert_almost_equal(mu_hat, mu)

        mu, var = m.predict_noiseless(self.X_new)
        assert mu.shape == (self.N_new, self.D)
        assert var.shape == (self.N_new, 1)
        np.testing.assert_almost_equal(np.diag(K_hat)[:, None], var)
        # np.testing.assert_almost_equal(mu_hat, mu)

    def test_likelihood_replicate(self):
        self.setup()

        m = GPy.models.GPRegression(self.X, self.Y)
        m2 = GPy.models.GPRegression(self.X, self.Y)
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[:] = m[""].values()
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[""] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[""] = m[""]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m.Gaussian_noise.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m[".*var"] = 2
        m2[".*var"] = m[".*var"]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

    def test_likelihood_set(self):
        self.setup()

        m = GPy.models.GPRegression(self.X, self.Y)
        m2 = GPy.models.GPRegression(self.X, self.Y)
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2.kern.lengthscale = m.kern.lengthscale
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2[".*lengthscale"] = m.kern.lengthscale
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2[".*lengthscale"] = m.kern[".*lengthscale"]
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2.kern.lengthscale = m.kern[".*lengthscale"]
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

    def test_missing_data(self):
        self.setup()

        Q = 4

        k = GPy.kern.Linear(Q, ARD=True) + GPy.kern.White(
            Q, np.exp(-2)
        )  # + kern.bias(Q)
        m = _create_missing_data_model(k, Q)
        assert m.checkgrad()
        mul, varl = m.predict(m.X)

        k = GPy.kern.RBF(Q, ARD=True) + GPy.kern.White(Q, np.exp(-2))  # + kern.bias(Q)
        m2 = _create_missing_data_model(k, Q)
        assert m.checkgrad()
        m2.kern.rbf.lengthscale[:] = 1e6

        m2.X[:] = m.X.param_array
        m2.likelihood[:] = m.likelihood[:]
        m2.kern.white[:] = m.kern.white[:]
        mu, var = m.predict(m.X)
        np.testing.assert_allclose(mul, mu)
        np.testing.assert_allclose(varl, var)

        q50 = m.predict_quantiles(m.X, (50,))
        np.testing.assert_allclose(mul, q50[0])

    def test_likelihood_replicate_kern(self):
        self.setup()

        m = GPy.models.GPRegression(self.X, self.Y)
        m2 = GPy.models.GPRegression(self.X, self.Y)
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[""] = m.kern[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[:] = m.kern[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[""] = m.kern[""]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[:] = m.kern[""].values()
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

    def test_big_model(self):
        self.setup()

        m = GPy.examples.dimensionality_reduction.mrd_simulation(
            optimize=0, plot=0, plot_sim=0
        )
        m.X.fix()
        print(m)
        m.unfix()
        m.checkgrad()
        print(m)
        m.fix()
        print(m)
        m.inducing_inputs.unfix()
        print(m)
        m.checkgrad()
        m.unfix()
        m.checkgrad()
        m.checkgrad()
        print(m)

    def test_mrd(self):
        from GPy.inference.latent_function_inference import InferenceMethodList, VarDTC
        from GPy.likelihoods import Gaussian

        self.setup()

        Y1 = np.random.normal(0, 1, (40, 13))
        Y2 = np.random.normal(0, 1, (40, 6))
        Y3 = np.random.normal(0, 1, (40, 8))
        Q = 5
        m = GPy.models.MRD(
            dict(data1=Y1, data2=Y2, data3=Y3),
            Q,
        )
        m.randomize()
        assert m.checkgrad()

        m = GPy.models.MRD(
            dict(data1=Y1, data2=Y2, data3=Y3),
            Q,
            initx="PCA_single",
            initz="random",
            kernel=[GPy.kern.RBF(Q, ARD=1) for _ in range(3)],
            inference_method=InferenceMethodList([VarDTC() for _ in range(3)]),
            likelihoods=[Gaussian(name="Gaussian_noise".format(i)) for i in range(3)],
        )
        m.randomize()
        assert m.checkgrad()

        m = GPy.models.MRD(
            dict(data1=Y1, data2=Y2, data3=Y3),
            Q,
            initx="random",
            initz="random",
            kernel=GPy.kern.RBF(Q, ARD=1),
        )
        m.randomize()
        assert m.checkgrad()

        m = GPy.models.MRD(
            dict(data1=Y1, data2=Y2, data3=Y3),
            Q,
            X=np.random.normal(0, 1, size=(40, Q)),
            X_variance=False,
            kernel=GPy.kern.RBF(Q, ARD=1),
            likelihoods=[Gaussian(name="Gaussian_noise".format(i)) for i in range(3)],
        )
        m.randomize()
        assert m.checkgrad()

    def test_model_set_params(self):
        self.setup()

        m = GPy.models.GPRegression(self.X, self.Y)
        lengthscale = np.random.uniform()
        m.kern.lengthscale = lengthscale
        np.testing.assert_equal(m.kern.lengthscale, lengthscale)
        m.kern.lengthscale *= 1
        m[".*var"] -= 0.1
        np.testing.assert_equal(m.kern.lengthscale, lengthscale)
        m.optimize()
        print(m)

    def test_model_updates(self):
        self.setup()

        Y1 = np.random.normal(0, 1, (40, 13))
        Y2 = np.random.normal(0, 1, (40, 6))
        m = GPy.models.MRD([Y1, Y2], 5)
        self.count = 0
        m.add_observer(self, self._count_updates, -2000)
        m.update_model(False)
        m[".*Gaussian"] = 0.001
        assert self.count == 0
        m[".*Gaussian"].constrain_bounded(0, 0.01)
        assert self.count == 0
        m.Z.fix()
        assert self.count == 0
        m.update_model(True)
        assert self.count == 1

    def _count_updates(self, me, which):
        self.count += 1

    def test_model_optimize(self):
        self.setup()

        X = np.random.uniform(-3.0, 3.0, (20, 1))
        Y = np.sin(X) + np.random.randn(20, 1) * 0.05
        m = GPy.models.GPRegression(X, Y)
        m.optimize()
        print(m)

    def test_input_warped_gp_identity(self):
        """
        A InputWarpedGP with the identity warping function should be
        equal to a standard GP.
        """
        self.setup()

        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.optimize()
        preds = m.predict(self.X)

        warp_k = GPy.kern.RBF(1)
        warp_f = GPy.util.input_warping_functions.IdentifyWarping()
        warp_m = GPy.models.InputWarpedGP(
            self.X, self.Y, kernel=warp_k, warping_function=warp_f
        )
        warp_m.optimize()
        warp_preds = warp_m.predict(self.X)

        np.testing.assert_almost_equal(preds, warp_preds, decimal=4)

    def test_kumar_warping_gradient(self):
        self.setup()

        n_X = 100
        np.random.seed(0)
        X = np.random.randn(n_X, 2)
        Y = np.sum(np.sin(X), 1).reshape(n_X, 1)

        k1 = GPy.kern.Linear(2)
        m1 = GPy.models.InputWarpedGP(X, Y, kernel=k1)
        m1.randomize()
        assert m1.checkgrad()

        k2 = GPy.kern.RBF(2)
        m2 = GPy.models.InputWarpedGP(X, Y, kernel=k2)
        m2.randomize()
        m2.checkgrad()
        assert m2.checkgrad()

        k3 = GPy.kern.Matern52(2)
        m3 = GPy.models.InputWarpedGP(X, Y, kernel=k3)
        m3.randomize()
        m3.checkgrad()
        assert m3.checkgrad()

    def test_kumar_warping_parameters(self):
        self.setup()

        np.random.seed(1)
        X = np.random.rand(5, 2)
        epsilon = 1e-6

        # testing warping indices
        warping_ind_1 = [0, 1, 2]
        warping_ind_2 = [-1, 1, 2]
        warping_ind_3 = [0, 1.5, 2]
        self.failUnlessRaises(
            ValueError, GPy.util.input_warping_functions.KumarWarping, X, warping_ind_1
        )
        self.failUnlessRaises(
            ValueError, GPy.util.input_warping_functions.KumarWarping, X, warping_ind_2
        )
        self.failUnlessRaises(
            ValueError, GPy.util.input_warping_functions.KumarWarping, X, warping_ind_3
        )

        # testing Xmin and Xmax
        Xmin_1, Xmax_1 = None, [1, 1]
        Xmin_2, Xmax_2 = [0, 0], None
        Xmin_3, Xmax_3 = [0, 0, 0], [1, 1]
        self.failUnlessRaises(
            ValueError,
            GPy.util.input_warping_functions.KumarWarping,
            X,
            [0, 1],
            epsilon,
            Xmin_1,
            Xmax_1,
        )
        self.failUnlessRaises(
            ValueError,
            GPy.util.input_warping_functions.KumarWarping,
            X,
            [0, 1],
            epsilon,
            Xmin_2,
            Xmax_2,
        )
        self.failUnlessRaises(
            ValueError,
            GPy.util.input_warping_functions.KumarWarping,
            X,
            [0, 1],
            epsilon,
            Xmin_3,
            Xmax_3,
        )

    def test_warped_gp_identity(self):
        """
        A WarpedGP with the identity warping function should be
        equal to a standard GP.
        """
        self.setup()

        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.optimize()
        preds = m.predict(self.X)

        warp_k = GPy.kern.RBF(1)
        warp_f = GPy.util.warping_functions.IdentityFunction(closed_inverse=False)
        warp_m = GPy.models.WarpedGP(
            self.X, self.Y, kernel=warp_k, warping_function=warp_f
        )
        warp_m.optimize()
        warp_preds = warp_m.predict(self.X)

        warp_k_exact = GPy.kern.RBF(1)
        warp_f_exact = GPy.util.warping_functions.IdentityFunction()
        warp_m_exact = GPy.models.WarpedGP(
            self.X, self.Y, kernel=warp_k_exact, warping_function=warp_f_exact
        )
        warp_m_exact.optimize()
        warp_preds_exact = warp_m_exact.predict(self.X)

        np.testing.assert_almost_equal(preds, warp_preds, decimal=4)
        np.testing.assert_almost_equal(preds, warp_preds_exact, decimal=4)

    def test_warped_gp_log(self):
        """
        A WarpedGP with the log warping function should be
        equal to a standard GP with log labels.
        Note that we predict the median here.
        """
        self.setup()

        k = GPy.kern.RBF(1)
        Y = np.abs(self.Y)
        logY = np.log(Y)
        m = GPy.models.GPRegression(self.X, logY, kernel=k)
        m.optimize()
        preds = m.predict(self.X)[0]

        warp_k = GPy.kern.RBF(1)
        warp_f = GPy.util.warping_functions.LogFunction(closed_inverse=False)
        warp_m = GPy.models.WarpedGP(self.X, Y, kernel=warp_k, warping_function=warp_f)
        warp_m.optimize()
        warp_preds = warp_m.predict(self.X, median=True)[0]

        warp_k_exact = GPy.kern.RBF(1)
        warp_f_exact = GPy.util.warping_functions.LogFunction()
        warp_m_exact = GPy.models.WarpedGP(
            self.X, Y, kernel=warp_k_exact, warping_function=warp_f_exact
        )
        warp_m_exact.optimize(messages=True)
        warp_preds_exact = warp_m_exact.predict(self.X, median=True)[0]

        np.testing.assert_almost_equal(np.exp(preds), warp_preds, decimal=4)
        np.testing.assert_almost_equal(np.exp(preds), warp_preds_exact, decimal=4)

    def test_warped_gp_cubic_sine(self):
        """
        A test replicating the cubic sine regression problem from
        Snelson's paper. This test doesn't have any assertions, it's
        just to ensure coverage of the tanh warping function code.
        """
        self.setup()
        max_iters = 100

        X = (2 * np.pi) * np.random.random(151) - np.pi
        Y = np.sin(X) + np.random.normal(0, 0.2, 151)
        Y = np.array([np.power(abs(y), float(1) / 3) * (1, -1)[y < 0] for y in Y])
        X = X[:, None]
        Y = Y[:, None]

        warp_m = GPy.models.WarpedGP(
            X, Y
        )  # , kernel=warp_k)#, warping_function=warp_f)
        warp_m[".*\.d"].constrain_fixed(1.0)
        warp_m.optimize_restarts(
            parallel=False, robust=False, num_restarts=5, max_iters=max_iters
        )
        warp_m.predict(X)
        warp_m.predict_quantiles(X)
        warp_m.log_predictive_density(X, Y)
        warp_m.predict_in_warped_space = False
        warp_m.plot()
        warp_m.predict_in_warped_space = True
        warp_m.plot()

    def test_offset_regression(self):
        # Tests GPy.models.GPOffsetRegression. Using two small time series
        # from a sine wave, we confirm the algorithm determines that the
        # likelihood is maximised when the offset hyperparameter is approximately
        # equal to the actual offset in X between the two time series.
        self.setup()

        offset = 3
        X1 = np.arange(0, 50, 5.0)[:, None]
        X2 = np.arange(0 + offset, 50 + offset, 5.0)[:, None]
        X = np.vstack([X1, X2])
        ind = np.vstack([np.zeros([10, 1]), np.ones([10, 1])])
        X = np.hstack([X, ind])
        Y = np.sin((X[0:10, 0]) / 30.0)[:, None]
        Y = np.vstack([Y, Y])

        m = GPy.models.GPOffsetRegression(X, Y)
        m.rbf.lengthscale = (
            5.0  # make it something other than one to check our gradients properly!
        )
        assert (
            m.checkgrad()
        ), "Gradients of offset parameters don't match numerical approximations."
        m.optimize()
        assert np.abs(m.offset[0] - offset) < 0.1, (
            "GPOffsetRegression model failing to estimate correct offset (value estimated = %0.2f instead of %0.2f)"
            % (m.offset[0], offset)
        )

    def test_logistic_basis_func_gradients(self):
        self.setup()

        X = np.random.uniform(-4, 4, (20, 5))
        points = np.random.uniform(X.min(0), X.max(0), X.shape[1])
        ks = []
        for i in range(points.shape[0]):
            if (i % 2 == 0) and (i % 3 != 0):
                self.assertRaises(
                    AssertionError,
                    GPy.kern.LogisticBasisFuncKernel,
                    1,
                    points,
                    ARD=i % 2 == 0,
                    ARD_slope=i % 3 == 0,
                    active_dims=[i],
                )
            else:
                ks.append(
                    GPy.kern.LogisticBasisFuncKernel(
                        1, points, ARD=i % 2 == 0, ARD_slope=i % 3 == 0, active_dims=[i]
                    )
                )
        k = GPy.kern.Add(ks)
        k.randomize()

        Y = np.random.normal(0, 1, (X.shape[0], 1))
        m = GPy.models.GPRegression(X, Y, kernel=k.copy())
        assert m.checkgrad()

    def test_posterior_inf_basis_funcs(self):
        self.setup()

        X = np.random.uniform(-4, 1, (50, 1))

        # Logistic:
        k = GPy.kern.LogisticBasisFuncKernel(1, [0, -2])

        true_w = [1, 2]
        true_slope = [5, -2]

        Y = 0
        for w, s, c in zip(true_w, true_slope, k.centers[0]):
            Y += w / (1 + np.exp(-s * (X - c)))
        Y += np.random.normal(0, 0.000001)

        m = GPy.models.GPRegression(X, Y, kernel=k.copy())
        # m.likelihood.fix(1e-6)
        m.optimize()

        wu, wv = m.kern.posterior_inf()
        # _sort = np.argsort(wu.flat)

        # from scipy.stats import norm
        # confidence_intervals = np.array(norm.interval(.95, loc=wu.flat[_sort], scale=np.sqrt(np.diag(wv))[_sort])).T
        # for i in range(wu.size):
        #    s,t = confidence_intervals[i]
        #    v = true_w[i]
        #    assert ((s<v)&(v<t)), "didnt find true w within the 95% confidence interval of the predicted values"

        np.testing.assert_allclose(np.sort(wu.flat), np.sort(true_w), rtol=1e-4)
        np.testing.assert_allclose(np.diag(wv), 0, atol=1e-4)
        np.testing.assert_allclose(
            np.sort(m.kern.slope.flat), np.sort(true_slope), rtol=1e-4
        )


class TestGradient:
    def setup(self):
        ######################################
        # # 1 dimensional example

        # sample inputs and outputs
        self.X1D = np.random.uniform(-3.0, 3.0, (20, 1))
        self.Y1D = np.sin(self.X1D) + np.random.randn(20, 1) * 0.05

        ######################################
        # # 2 dimensional example

        # sample inputs and outputs
        self.X2D = np.random.uniform(-3.0, 3.0, (40, 2))
        self.Y2D = (
            np.sin(self.X2D[:, 0:1]) * np.sin(self.X2D[:, 1:2])
            + np.random.randn(40, 1) * 0.05
        )

    def check_model(
        self, kern, model_type="GPRegression", dimension=1, uncertain_inputs=False
    ):
        # Get the correct gradients
        if dimension == 1:
            X = self.X1D
            Y = self.Y1D
        else:
            X = self.X2D
            Y = self.Y2D
        # Get model type (GPRegression, SparseGPRegression, etc)
        model_fit = getattr(GPy.models, model_type)

        # noise = GPy.kern.White(dimension)
        kern = kern  #  + noise
        if uncertain_inputs:
            m = model_fit(
                X, Y, kernel=kern, X_variance=np.random.rand(X.shape[0], X.shape[1])
            )
        else:
            m = model_fit(X, Y, kernel=kern)
        m.randomize()
        # contrain all parameters to be positive
        assert m.checkgrad()

    def test_GPRegression_rbf_1d(self):
        """Testing the GP regression with rbf kernel with white kernel on 1d data"""
        self.setup()
        rbf = GPy.kern.RBF(1)
        self.check_model(rbf, model_type="GPRegression", dimension=1)

    def test_GPRegression_rbf_2D(self):
        """Testing the GP regression with rbf kernel on 2d data"""
        self.setup()
        rbf = GPy.kern.RBF(2)
        self.check_model(rbf, model_type="GPRegression", dimension=2)

    def test_GPRegression_rbf_ARD_2D(self):
        """Testing the GP regression with rbf kernel on 2d data"""
        self.setup()
        k = GPy.kern.RBF(2, ARD=True)
        self.check_model(k, model_type="GPRegression", dimension=2)

    def test_GPRegression_mlp_1d(self):
        """Testing the GP regression with mlp kernel with white kernel on 1d data"""
        self.setup()
        mlp = GPy.kern.MLP(1)
        self.check_model(mlp, model_type="GPRegression", dimension=1)

    # TODO:
    # def test_GPRegression_poly_1d(self):
    #    ''' Testing the GP regression with polynomial kernel with white kernel on 1d data '''
    #    mlp = GPy.kern.Poly(1, degree=5)
    #    self.check_model(mlp, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern52_1D(self):
        """Testing the GP regression with matern52 kernel on 1d data"""
        self.setup()
        matern52 = GPy.kern.Matern52(1)
        self.check_model(matern52, model_type="GPRegression", dimension=1)

    def test_GPRegression_matern52_2D(self):
        """Testing the GP regression with matern52 kernel on 2d data"""
        self.setup()
        matern52 = GPy.kern.Matern52(2)
        self.check_model(matern52, model_type="GPRegression", dimension=2)

    def test_GPRegression_matern52_ARD_2D(self):
        """Testing the GP regression with matern52 kernel on 2d data"""
        self.setup()
        matern52 = GPy.kern.Matern52(2, ARD=True)
        self.check_model(matern52, model_type="GPRegression", dimension=2)

    def test_GPRegression_matern32_1D(self):
        """Testing the GP regression with matern32 kernel on 1d data"""
        self.setup()
        matern32 = GPy.kern.Matern32(1)
        self.check_model(matern32, model_type="GPRegression", dimension=1)

    def test_GPRegression_matern32_2D(self):
        """Testing the GP regression with matern32 kernel on 2d data"""
        self.setup()
        matern32 = GPy.kern.Matern32(2)
        self.check_model(matern32, model_type="GPRegression", dimension=2)

    def test_GPRegression_matern32_ARD_2D(self):
        """Testing the GP regression with matern32 kernel on 2d data"""
        self.setup()
        matern32 = GPy.kern.Matern32(2, ARD=True)
        self.check_model(matern32, model_type="GPRegression", dimension=2)

    def test_GPRegression_exponential_1D(self):
        """Testing the GP regression with exponential kernel on 1d data"""
        self.setup()
        exponential = GPy.kern.Exponential(1)
        self.check_model(exponential, model_type="GPRegression", dimension=1)

    def test_GPRegression_exponential_2D(self):
        """Testing the GP regression with exponential kernel on 2d data"""
        self.setup()
        exponential = GPy.kern.Exponential(2)
        self.check_model(exponential, model_type="GPRegression", dimension=2)

    def test_GPRegression_exponential_ARD_2D(self):
        """Testing the GP regression with exponential kernel on 2d data"""
        self.setup()
        exponential = GPy.kern.Exponential(2, ARD=True)
        self.check_model(exponential, model_type="GPRegression", dimension=2)

    def test_GPRegression_bias_kern_1D(self):
        """Testing the GP regression with bias kernel on 1d data"""
        self.setup()
        bias = GPy.kern.Bias(1)
        self.check_model(bias, model_type="GPRegression", dimension=1)

    def test_GPRegression_bias_kern_2D(self):
        """Testing the GP regression with bias kernel on 2d data"""
        self.setup()
        bias = GPy.kern.Bias(2)
        self.check_model(bias, model_type="GPRegression", dimension=2)

    def test_GPRegression_linear_kern_1D_ARD(self):
        """Testing the GP regression with linear kernel on 1d data"""
        self.setup()
        linear = GPy.kern.Linear(1, ARD=True)
        self.check_model(linear, model_type="GPRegression", dimension=1)

    def test_GPRegression_linear_kern_2D_ARD(self):
        """Testing the GP regression with linear kernel on 2d data"""
        self.setup()
        linear = GPy.kern.Linear(2, ARD=True)
        self.check_model(linear, model_type="GPRegression", dimension=2)

    def test_GPRegression_linear_kern_1D(self):
        """Testing the GP regression with linear kernel on 1d data"""
        self.setup()
        linear = GPy.kern.Linear(1)
        self.check_model(linear, model_type="GPRegression", dimension=1)

    def test_GPRegression_linear_kern_2D(self):
        """Testing the GP regression with linear kernel on 2d data"""
        self.setup()
        linear = GPy.kern.Linear(2)
        self.check_model(linear, model_type="GPRegression", dimension=2)

    def test_SparseGPRegression_rbf_white_kern_1d(self):
        """Testing the sparse GP regression with rbf kernel with white kernel on 1d data"""
        self.setup()
        rbf = GPy.kern.RBF(1)
        self.check_model(rbf, model_type="SparseGPRegression", dimension=1)

    def test_SparseGPRegression_rbf_white_kern_2D(self):
        """Testing the sparse GP regression with rbf kernel on 2d data"""
        self.setup()
        rbf = GPy.kern.RBF(2)
        self.check_model(rbf, model_type="SparseGPRegression", dimension=2)

    def test_SparseGPRegression_rbf_linear_white_kern_1D(self):
        """Testing the sparse GP regression with rbf kernel on 1d data"""
        self.setup()
        rbflin = GPy.kern.RBF(1) + GPy.kern.Linear(1) + GPy.kern.White(1, 1e-5)
        self.check_model(rbflin, model_type="SparseGPRegression", dimension=1)

    def test_SparseGPRegression_rbf_linear_white_kern_2D(self):
        """Testing the sparse GP regression with rbf kernel on 2d data"""
        self.setup()
        rbflin = GPy.kern.RBF(2) + GPy.kern.Linear(2)
        self.check_model(rbflin, model_type="SparseGPRegression", dimension=2)

    def test_SparseGPRegression_rbf_white_kern_2D_uncertain_inputs(self):
        """Testing the sparse GP regression with rbf, linear kernel on 2d data with uncertain inputs"""
        self.setup()
        rbflin = GPy.kern.RBF(2) + GPy.kern.White(2)
        self.check_model(
            rbflin, model_type="SparseGPRegression", dimension=2, uncertain_inputs=1
        )

    def test_SparseGPRegression_rbf_white_kern_1D_uncertain_inputs(self):
        """Testing the sparse GP regression with rbf, linear kernel on 1d data with uncertain inputs"""
        self.setup()
        rbflin = GPy.kern.RBF(1) + GPy.kern.White(1)
        self.check_model(
            rbflin, model_type="SparseGPRegression", dimension=1, uncertain_inputs=1
        )

    def test_TPRegression_matern52_1D(self):
        """Testing the TP regression with matern52 kernel on 1d data"""
        self.setup()
        matern52 = GPy.kern.Matern52(1) + GPy.kern.White(1)
        self.check_model(matern52, model_type="TPRegression", dimension=1)

    def test_TPRegression_rbf_2D(self):
        """Testing the TP regression with rbf kernel on 2d data"""
        self.setup()
        rbf = GPy.kern.RBF(2)
        self.check_model(rbf, model_type="TPRegression", dimension=2)

    def test_TPRegression_rbf_ARD_2D(self):
        """Testing the GP regression with rbf kernel on 2d data"""
        self.setup()
        k = GPy.kern.RBF(2, ARD=True)
        self.check_model(k, model_type="TPRegression", dimension=2)

    def test_TPRegression_matern52_2D(self):
        """Testing the TP regression with matern52 kernel on 2d data"""
        self.setup()
        matern52 = GPy.kern.Matern52(2)
        self.check_model(matern52, model_type="TPRegression", dimension=2)

    def test_TPRegression_matern52_ARD_2D(self):
        """Testing the TP regression with matern52 kernel on 2d data"""
        self.setup()
        matern52 = GPy.kern.Matern52(2, ARD=True)
        self.check_model(matern52, model_type="TPRegression", dimension=2)

    def test_TPRegression_matern32_1D(self):
        """Testing the TP regression with matern32 kernel on 1d data"""
        self.setup()
        matern32 = GPy.kern.Matern32(1)
        self.check_model(matern32, model_type="TPRegression", dimension=1)

    def test_TPRegression_matern32_2D(self):
        """Testing the TP regression with matern32 kernel on 2d data"""
        self.setup()
        matern32 = GPy.kern.Matern32(2)
        self.check_model(matern32, model_type="TPRegression", dimension=2)

    def test_TPRegression_matern32_ARD_2D(self):
        """Testing the TP regression with matern32 kernel on 2d data"""
        self.setup()
        matern32 = GPy.kern.Matern32(2, ARD=True)
        self.check_model(matern32, model_type="TPRegression", dimension=2)

    def test_GPLVM_rbf_bias_white_kern_2D(self):
        """Testing GPLVM with rbf + bias kernel"""
        self.setup()
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = (
            GPy.kern.RBF(input_dim, 0.5, 0.9 * np.ones((1,)))
            + GPy.kern.Bias(input_dim, 0.1)
            + GPy.kern.White(input_dim, 0.05)
            + GPy.kern.Matern32(input_dim)
            + GPy.kern.Matern52(input_dim)
        )
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, kernel=k)
        assert m.checkgrad()

    def test_SparseGPLVM_rbf_bias_white_kern_2D(self):
        """Testing GPLVM with rbf + bias kernel"""
        self.setup()
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = (
            GPy.kern.RBF(input_dim, 0.5, 0.9 * np.ones((1,)))
            + GPy.kern.Bias(input_dim, 0.1)
            + GPy.kern.White(input_dim, 0.05)
            + GPy.kern.Matern32(input_dim)
            + GPy.kern.Matern52(input_dim)
        )
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.SparseGPLVM(Y, input_dim, kernel=k)
        assert m.checkgrad()

    def test_BCGPLVM_rbf_bias_white_kern_2D(self):
        """Testing GPLVM with rbf + bias kernel"""
        self.setup()
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = (
            GPy.kern.RBF(input_dim, 0.5, 0.9 * np.ones((1,)))
            + GPy.kern.Bias(input_dim, 0.1)
            + GPy.kern.White(input_dim, 0.05)
        )
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.BCGPLVM(Y, input_dim, kernel=k)
        assert m.checkgrad()

    def test_GPLVM_rbf_linear_white_kern_2D(self):
        """Testing GPLVM with rbf + bias kernel"""
        self.setup()
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = (
            GPy.kern.Linear(input_dim)
            + GPy.kern.Bias(input_dim, 0.1)
            + GPy.kern.White(input_dim, 0.05)
        )
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, init="PCA", kernel=k)
        assert m.checkgrad()

    def test_GP_EP_probit(self):
        self.setup()
        N = 20
        Nhalf = int(N / 2)
        X = np.hstack([np.random.normal(5, 2, Nhalf), np.random.normal(10, 2, Nhalf)])[
            :, None
        ]
        Y = np.hstack([np.ones(Nhalf), np.zeros(Nhalf)])[:, None]
        kernel = GPy.kern.RBF(1)
        m = GPy.models.GPClassification(X, Y, kernel=kernel)
        assert m.checkgrad()

    def test_sparse_EP_DTC_probit(self):
        self.setup()
        N = 20
        Nhalf = int(N / 2)
        X = np.hstack([np.random.normal(5, 2, Nhalf), np.random.normal(10, 2, Nhalf)])[
            :, None
        ]
        Y = np.hstack([np.ones(Nhalf), np.zeros(Nhalf)])[:, None]
        Z = np.linspace(0, 15, 4)[:, None]
        kernel = GPy.kern.RBF(1)
        m = GPy.models.SparseGPClassification(X, Y, kernel=kernel, Z=Z)
        assert m.checkgrad()

    def test_sparse_EP_DTC_probit_uncertain_inputs(self):
        self.setup()
        N = 20
        Nhalf = int(N / 2)
        X = np.hstack([np.random.normal(5, 2, Nhalf), np.random.normal(10, 2, Nhalf)])[
            :, None
        ]
        Y = np.hstack([np.ones(Nhalf), np.zeros(Nhalf)])[:, None]
        Z = np.linspace(0, 15, 4)[:, None]
        X_var = np.random.uniform(0.1, 0.2, X.shape)
        kernel = GPy.kern.RBF(1)
        m = GPy.models.SparseGPClassificationUncertainInput(
            X, X_var, Y, kernel=kernel, Z=Z
        )
        assert m.checkgrad()

    def test_multioutput_regression_1D(self):
        self.setup()
        X1 = np.random.rand(50, 1) * 8
        X2 = np.random.rand(30, 1) * 5
        X = np.vstack((X1, X2))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
        Y2 = -np.sin(X2) + np.random.randn(*X2.shape) * 0.05
        Y = np.vstack((Y1, Y2))

        k1 = GPy.kern.RBF(1)
        m = GPy.models.GPCoregionalizedRegression(
            X_list=[X1, X2], Y_list=[Y1, Y2], kernel=k1
        )
        # import ipdb;ipdb.set_trace()
        # m.constrain_fixed('.*rbf_var', 1.)
        assert m.checkgrad()

    def test_simple_MultivariateGaussian_prior(self):
        self.setup()
        X = np.random.multivariate_normal(
            [1, 5], np.diag([0.5, 0.3]), (100, 1)
        ).reshape(100, 2)
        Y = X + np.random.randn(100, 2) * 0.05
        kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=1, ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(150, 5))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([20, 20]), np.diag([5, 5]))
        )
        m = GPy.models.GPRegression(X, Y, kernel=kernel)
        m.optimize()
        print(m.kern.variance)
        print(m.kern.lengthscale)

    def test_simple_MultivariateGaussian_prior_matrixmean(self):
        self.setup()
        X = np.random.multivariate_normal(
            [1, 5], np.diag([0.5, 0.3]), (100, 1)
        ).reshape(100, 2)
        Y = X + np.random.randn(100, 2) * 0.05
        kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=1, ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(150, 5))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([[20, 20]]), np.diag([5, 5]))
        )
        m = GPy.models.GPRegression(X, Y, kernel=kernel)
        m.optimize()
        print(m.kern.variance)
        print(m.kern.lengthscale)

    def test_multioutput_sparse_regression_1D(self):
        self.setup()
        X1 = np.random.rand(500, 1) * 8
        X2 = np.random.rand(300, 1) * 5
        X = np.vstack((X1, X2))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
        Y2 = -np.sin(X2) + np.random.randn(*X2.shape) * 0.05
        Y = np.vstack((Y1, Y2))

        k1 = GPy.kern.RBF(1)
        m = GPy.models.SparseGPCoregionalizedRegression(
            X_list=[X1, X2], Y_list=[Y1, Y2], kernel=k1
        )
        assert m.checkgrad()

    def test_gp_heteroscedastic_regression(self):
        self.setup()
        num_obs = 25
        X = np.random.randint(0, 140, num_obs)
        X = X[:, None]
        Y = 25.0 + np.sin(X / 20.0) * 2.0 + np.random.rand(num_obs)[:, None]
        kern = GPy.kern.Bias(1) + GPy.kern.RBF(1)
        m = GPy.models.GPHeteroscedasticRegression(X, Y, kern)
        assert m.checkgrad()

    def test_sparse_gp_heteroscedastic_regression(self):
        self.setup()
        num_obs = 25
        X = np.random.randint(0, 140, num_obs)
        X = X[:, None]
        Y = 25.0 + np.sin(X / 20.0) * 2.0 + np.random.rand(num_obs)[:, None]
        kern = GPy.kern.Bias(1) + GPy.kern.RBF(1)
        Y_metadata = {"output_index": np.arange(num_obs)[:, None]}
        noise_terms = np.unique(Y_metadata["output_index"].flatten())
        likelihoods_list = [
            GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" % j) for j in noise_terms
        ]
        likelihood = GPy.likelihoods.MixedNoise(likelihoods_list=likelihoods_list)
        m = GPy.core.SparseGP(
            X,
            Y,
            X[np.random.choice(num_obs, 10)],
            kern,
            likelihood,
            inference_method=GPy.inference.latent_function_inference.VarDTC(),
            Y_metadata=Y_metadata,
        )
        assert m.checkgrad()

    def test_gp_kronecker_gaussian(self):
        self.setup()
        np.random.seed(0)
        N1, N2 = 30, 20
        X1 = np.random.randn(N1, 1)
        X2 = np.random.randn(N2, 1)
        X1.sort(0)
        X2.sort(0)
        k1 = GPy.kern.RBF(1)  # + GPy.kern.White(1)
        k2 = GPy.kern.RBF(1)  # + GPy.kern.White(1)
        Y = np.random.randn(N1, N2)
        Y = Y - Y.mean(0)
        Y = Y / Y.std(0)
        m = GPy.models.GPKroneckerGaussianRegression(X1, X2, Y, k1, k2)

        # build the model the dumb way
        assert N1 * N2 < 1000, "too much data for standard GPs!"
        yy, xx = np.meshgrid(X2, X1)
        Xgrid = np.vstack((xx.flatten(order="F"), yy.flatten(order="F"))).T
        kg = GPy.kern.RBF(1, active_dims=[0]) * GPy.kern.RBF(1, active_dims=[1])
        mm = GPy.models.GPRegression(Xgrid, Y.reshape(-1, 1, order="F"), kernel=kg)

        m.randomize()
        mm[:] = m[:]
        assert np.allclose(m.log_likelihood(), mm.log_likelihood())
        assert np.allclose(m.gradient, mm.gradient)
        X1test = np.random.randn(100, 1)
        X2test = np.random.randn(100, 1)
        mean1, var1 = m.predict(X1test, X2test)
        yy, xx = np.meshgrid(X2test, X1test)
        Xgrid = np.vstack((xx.flatten(order="F"), yy.flatten(order="F"))).T
        mean2, var2 = mm.predict(Xgrid)
        assert np.allclose(mean1, mean2)
        assert np.allclose(var1, var2)

    def test_gp_VGPC(self):
        self.setup()
        np.random.seed(10)
        num_obs = 25
        X = np.random.randint(0, 140, num_obs)
        X = X[:, None]
        Y = 25.0 + np.sin(X / 20.0) * 2.0 + np.random.rand(num_obs)[:, None]
        kern = GPy.kern.Bias(1) + GPy.kern.RBF(1)
        lik = GPy.likelihoods.Gaussian()
        m = GPy.models.GPVariationalGaussianApproximation(
            X, Y, kernel=kern, likelihood=lik
        )
        m.randomize()
        assert m.checkgrad()

    def test_ssgplvm(self):
        from GPy import kern
        from GPy.models import SSGPLVM
        from GPy.examples.dimensionality_reduction import _simulate_matern

        self.setup()

        np.random.seed(10)
        D1, D2, D3, N, num_inducing, Q = 13, 5, 8, 45, 3, 9
        _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, False)
        Y = Ylist[0]
        k = kern.Linear(Q, ARD=True)  # + kern.white(Q, _np.exp(-2)) # + kern.bias(Q)
        # k = kern.RBF(Q, ARD=True, lengthscale=10.)
        m = SSGPLVM(
            Y, Q, init="rand", num_inducing=num_inducing, kernel=k, group_spike=True
        )
        m.randomize()
        assert m.checkgrad()

    def test_multiout_regression(self):
        np.random.seed(0)
        import GPy

        self.setup()

        N = 10
        N_train = 5
        D = 4
        noise_var = 0.3

        k = GPy.kern.RBF(1, lengthscale=0.1)
        x = np.random.rand(N, 1)
        cov = k.K(x)

        k_r = GPy.kern.RBF(2, lengthscale=0.4)
        x_r = np.random.rand(D, 2)
        cov_r = k_r.K(x_r)

        cov_all = np.kron(cov_r, cov)
        L = GPy.util.linalg.jitchol(cov_all)

        y_latent = L.dot(np.random.randn(N * D)).reshape(D, N).T

        x_test = x[N_train:]
        y_test = y_latent[N_train:]
        x = x[:N_train]
        y = y_latent[:N_train] + np.random.randn(N_train, D) * np.sqrt(noise_var)

        Mr = D
        Mc = x.shape[0]
        Qr = 5
        Qc = x.shape[1]

        m_mr = GPy.models.GPMultioutRegression(
            x,
            y,
            Xr_dim=Qr,
            kernel_row=GPy.kern.RBF(Qr, ARD=True),
            num_inducing=(Mc, Mr),
            init="GP",
        )
        m_mr.optimize_auto(max_iters=1)
        m_mr.randomize()
        assert m_mr.checkgrad()

        m_mr = GPy.models.GPMultioutRegression(
            x,
            y,
            Xr_dim=Qr,
            kernel_row=GPy.kern.RBF(Qr, ARD=True),
            num_inducing=(Mc, Mr),
            init="rand",
        )
        m_mr.optimize_auto(max_iters=1)
        m_mr.randomize()
        assert m_mr.checkgrad()

    def test_multiout_regression_md(self):
        import GPy

        np.random.seed(0)

        self.setup()

        N = 20
        N_train = 5
        D = 8
        noise_var = 0.3

        k = GPy.kern.RBF(1, lengthscale=0.1)
        x_raw = np.random.rand(N * D, 1)

        # dimension assignment
        D_list = []
        for i in range(2):
            while True:
                D_sub_list = []
                ratios = []
                r_p = 0.0
                for j in range(3):
                    ratios.append(np.random.rand() * (1 - r_p) + r_p)
                    D_sub_list.append(int((ratios[-1] - r_p) * 4 * N_train))
                    r_p = ratios[-1]
                D_sub_list.append(4 * N_train - np.sum(D_sub_list))
                if (np.array(D_sub_list) != 0).all():
                    D_list.extend([a + N - N_train for a in D_sub_list])
                    break

        cov = k.K(x_raw)

        k_r = GPy.kern.RBF(2, lengthscale=0.4)
        x_r = np.random.rand(D, 2)
        cov_r = k_r.K(x_r)

        cov_all = np.repeat(np.repeat(cov_r, D_list, axis=0), D_list, axis=1) * cov
        L = GPy.util.linalg.jitchol(cov_all)

        y_latent = L.dot(np.random.randn(N * D))

        x = np.zeros((D * N_train,))
        y = np.zeros((D * N_train,))
        x_test = np.zeros((D * (N - N_train),))
        y_test = np.zeros((D * (N - N_train),))
        indexD = np.zeros((D * N_train), dtype=np.int)
        indexD_test = np.zeros((D * (N - N_train)), dtype=np.int)

        offset_all = 0
        offset_train = 0
        offset_test = 0
        for i in range(D):
            D_test = N - N_train
            D_train = D_list[i] - N + N_train
            y[offset_train : offset_train + D_train] = y_latent[
                offset_all : offset_all + D_train
            ]
            x[offset_train : offset_train + D_train] = x_raw[
                offset_all : offset_all + D_train, 0
            ]
            y_test[offset_test : offset_test + D_test] = y_latent[
                offset_all + D_train : offset_all + D_train + D_test
            ]
            x_test[offset_test : offset_test + D_test] = x_raw[
                offset_all + D_train : offset_all + D_train + D_test, 0
            ]
            indexD[offset_train : offset_train + D_train] = i
            indexD_test[offset_test : offset_test + D_test] = i
            offset_train += D_train
            offset_test += D_test
            offset_all += D_train + D_test

        y_noisefree = y.copy()
        y += np.random.randn(*y.shape) * np.sqrt(noise_var)
        x_flat = x.flatten()[:, None]
        y_flat = y.flatten()[:, None]

        Mr, Mc, Qr, Qc = 4, 3, 2, 1

        m = GPy.models.GPMultioutRegressionMD(
            x_flat,
            y_flat,
            indexD,
            Xr_dim=Qr,
            kernel_row=GPy.kern.RBF(Qr, ARD=False),
            num_inducing=(Mc, Mr),
        )
        m.optimize_auto(max_iters=1)
        m.randomize()
        assert m.checkgrad()

        m = GPy.models.GPMultioutRegressionMD(
            x_flat,
            y_flat,
            indexD,
            Xr_dim=Qr,
            kernel_row=GPy.kern.RBF(Qr, ARD=False),
            num_inducing=(Mc, Mr),
            init="rand",
        )
        m.optimize_auto(max_iters=1)
        m.randomize()
        assert m.checkgrad()

    def test_posterior_covariance(self):
        self.setup()

        k = GPy.kern.Poly(2, order=1)
        X1 = np.array([[-2, 2], [-1, 1]])
        X2 = np.array([[2, 3], [-1, 3]])
        Y = np.array([[1], [2]])
        m = GPy.models.GPRegression(X1, Y, kernel=k)

        result = m._raw_posterior_covariance_between_points(X1, X2)
        expected = np.array([[0.4, 2.2], [1.0, 1.0]]) / 3.0

        assert np.allclose(result, expected)

    def test_posterior_covariance_missing_data(self):
        self.setup()

        Q = 4
        k = GPy.kern.Linear(Q, ARD=True)
        m = _create_missing_data_model(k, Q)

        with self.assertRaises(RuntimeError):
            m._raw_posterior_covariance_between_points(
                np.array([[1], [2]]), np.array([[3], [4]])
            )

    def test_multioutput_model_with_ep(self):
        self.setup()

        f = lambda x: np.sin(x) + 0.1 * (x - 2.0) ** 2 - 0.005 * x**3
        fd = lambda x: np.cos(x) + 0.2 * (x - 2.0) - 0.015 * x**2
        N = 10
        sigma = 0.05
        sigmader = 0.05
        x = np.array([np.linspace(1, 10, N)]).T
        y = f(x) + np.array(sigma * np.random.normal(0, 1, (N, 1)))

        M = 7
        xd = np.array([np.linspace(2, 8, M)]).T
        yd = 2 * (fd(xd) > 0) - 1

        # squared exponential kernel:
        se = GPy.kern.RBF(input_dim=1, lengthscale=1.5, variance=0.2)
        # We need to generate separate kernel for the derivative observations and give the created kernel as an input:
        se_der = GPy.kern.DiffKern(se, 0)

        # Then
        gauss = GPy.likelihoods.Gaussian(variance=sigma**2)
        probit = GPy.likelihoods.Binomial(
            gp_link=GPy.likelihoods.link_functions.ScaledProbit(nu=100)
        )

        # Then create the model, we give everything in lists
        m = GPy.models.MultioutputGP(
            X_list=[x, xd],
            Y_list=[y, yd],
            kernel_list=[se, se_der],
            likelihood_list=[gauss, probit],
            inference_method=GPy.inference.latent_function_inference.EP(
                ep_mode="nested"
            ),
        )

        assert m.checkgrad()

    def test_predictive_gradients_with_normalizer(self):
        """
        Check that model.predictive_gradients returns the gradients of
        model.predict when normalizer=True
        """
        self.setup()

        N, M, Q = 10, 15, 3
        X = np.random.rand(M, Q)
        Y = np.random.rand(M, 1)
        x = np.random.rand(N, Q)
        model = GPy.models.GPRegression(X=X, Y=Y, normalizer=True)
        from GPy.models import GradientChecker

        gm = GradientChecker(
            lambda x: model.predict(x)[0],
            lambda x: model.predictive_gradients(x)[0],
            x,
            "x",
        )
        gc = GradientChecker(
            lambda x: model.predict(x)[1],
            lambda x: model.predictive_gradients(x)[1],
            x,
            "x",
        )
        assert gm.checkgrad()
        assert gc.checkgrad()

    def test_posterior_covariance_between_points_with_normalizer(self):
        """
        Check that model.posterior_covariance_between_points returns
        the covariance from model.predict when normalizer=True
        """
        self.setup()

        np.random.seed(3)
        N, M, Q = 10, 15, 3
        X = np.random.rand(M, Q)
        Y = np.random.rand(M, 1)
        x = np.random.rand(2, Q)
        model = GPy.models.GPRegression(X=X, Y=Y, normalizer=True)

        c1 = model.posterior_covariance_between_points(x, x)
        c2 = model.predict(x, full_cov=True)[1]
        np.testing.assert_allclose(c1, c2)


class TestGradientMultioutputGPModel:
    def setup(self):
        # standard test function
        self.period = 3
        self.w = 2 * np.pi / self.period
        self.f = lambda x: np.sum(np.square(np.sin(self.w * x)), axis=1)
        self.df = lambda x: self.w * np.sin(2 * self.w * x)

        self.noise_std = 1e-2

        self.bounds = (-self.period, self.period)

        self.train_points = 5
        self.test_points = 25

    def approximate_predictive_gradients(self, model, x_test, D, step=1e-6):
        """
        Approximates gradients of predicted posterior means and variances.

        This function is used as the frameworks for GradientChecker and
        MultioutputGP do not easily combine when checking gradients of predicted
        partial derivative posteriors.
        """

        dmdx_aprx = np.zeros((x_test.shape[0] * (D + 1), D))
        dvdx_aprx = np.zeros((x_test.shape[0] * (D + 1), D))

        for d in range(D):
            x_over = x_test.copy()
            x_over[:, d] += step
            x_undr = x_test.copy()
            x_undr[:, d] -= step

            m_over, v_over = model.predict([x_over] * (D + 1))
            m_undr, v_undr = model.predict([x_undr] * (D + 1))

            dmdx_aprx[:, d, None] = (m_over - m_undr) / (2 * step)
            dvdx_aprx[:, d, None] = (v_over - v_undr) / (2 * step)

        return dmdx_aprx, dvdx_aprx

    def check_model(self, kern):
        """
        Checks predictions, hyperparameter gradients, and gradients of predicted
        posterior means and variances for MultioutputGP models that incorporate
        observed latent function gradient information.
        """

        D = kern.input_dim

        X_list = []
        Y_list = []
        for i in range(D + 1):
            # sample inputs for either latent function or partial derivatives
            X_i = np.random.uniform(*self.bounds, size=(self.train_points, D))
            # output of latent function or partial derivatives
            Y_i = (self.f(X_i) if (i == 0) else self.df(X_i)[:, i - 1])[:, None]
            # noisy observations
            Y_i += np.random.normal(scale=self.noise_std, size=Y_i.shape)

            X_list.append(X_i)
            Y_list.append(Y_i)

        # the kernel is accompanied with derivative kernels, one for each dimension
        kernel_list = [kern] + [GPy.kern.DiffKern(kern, d) for d in range(D)]

        # create model and check its hyperparameter gradient
        likelihood_list = [GPy.likelihoods.Gaussian(variance=self.noise_std**2)] * (
            D + 1
        )
        model = GPy.models.MultioutputGP(X_list, Y_list, kernel_list, likelihood_list)
        model.likelihood.constrain_fixed()
        assert model.checkgrad(step=1e-3)

        # optimize the model, and check its hyperparameter gradient again
        model.optimize()
        assert model.checkgrad(step=1e-3)

        # check predictions
        np.testing.assert_allclose(
            model.predict(X_list)[0], model.Y, atol=3 * self.noise_std
        )

        # test inputs for checking predictive gradients
        x_test = np.random.uniform(*self.bounds, size=(self.test_points, D))

        # predictive gradients
        dmdx, dvdx = model.predictive_gradients([x_test] * (D + 1))
        # approximated predictive gradients
        dmdx_aprx, dvdx_aprx = self.approximate_predictive_gradients(
            model, x_test, D, step=1e-3
        )
        # check predictive gradients
        np.testing.assert_allclose(dmdx, dmdx_aprx, atol=3 * self.noise_std)
        np.testing.assert_allclose(dvdx, dvdx_aprx, atol=3 * self.noise_std)

    def test_MultioutputGP_gradobs_RBF(self):
        """
        Testing gradient observing MultioutputGP model with an RBF kernel.
        """
        self.setup()
        for D in range(1, 4):
            kern = GPy.kern.RBF(input_dim=D)
            kern.randomize()
            self.check_model(kern)

    def test_MultioutputGP_gradobs_RBF_ARD(self):
        """
        Testing gradient observing MultioutputGP model with an RBF (ARD) kernel.
        """
        self.setup()
        for D in range(1, 4):
            kern = GPy.kern.RBF(input_dim=D, ARD=True)
            kern.randomize()
            self.check_model(kern)

    def test_MultioutputGP_gradobs_StdP(self):
        """
        Testing gradient observing MultioutputGP model with a StdP kernel.
        """
        self.setup()
        for D in range(1, 4):
            kern = GPy.kern.StdPeriodic(input_dim=D, period=self.period)
            kern.period.constrain_fixed()
            kern.randomize()
            self.check_model(kern)

    def test_MultioutputGP_gradobs_StdP_ARD(self):
        """
        Testing gradient observing MultioutputGP model with a StdP (ARD) kernel.
        """
        self.setup()
        for D in range(1, 4):
            kern = GPy.kern.StdPeriodic(
                input_dim=D, period=[self.period] * D, ARD1=True, ARD2=True
            )
            kern.period.constrain_fixed()
            kern.randomize()
            self.check_model(kern)

    def test_MultioutputGP_gradobs_prod_RBF(self):
        """
        Testing gradient observing MultioutputGP model with several RBF kernels.
        """
        self.setup()
        for D in range(2, 4):
            kerns = [GPy.kern.RBF(input_dim=1) for d in range(D)]
            kern = reduce(lambda k0, k1: k0 * k1, kerns)
            kern.randomize()
            self.check_model(kern)

    def test_MultioutputGP_gradobs_prod_StdP(self):
        """
        Testing gradient observing MultioutputGP model with several StdP kernels.
        """
        self.setup()
        for D in range(2, 4):
            kerns = [
                GPy.kern.StdPeriodic(input_dim=1, period=self.period) for d in range(D)
            ]
            kern = reduce(lambda k0, k1: k0 * k1, kerns)
            [k.period.constrain_fixed() for k in kern.parts]
            kern.randomize()
            self.check_model(kern)

    def test_MultioutputGP_gradobs_prod_mix(self):
        """
        Testing gradient observing MultioutputGP model with a mix of kernel types.
        """
        self.setup()
        for D in range(2, 4):
            kerns = []
            for d in range(D):
                if d % 2 == 0:
                    k = GPy.kern.RBF(input_dim=1)
                else:
                    k = GPy.kern.StdPeriodic(input_dim=1, period=self.period)
                    k.period.constrain_fixed()
                kerns.append(k)
            kern = reduce(lambda k0, k1: k0 * k1, kerns)
            kern.randomize()
            self.check_model(kern)


def _create_missing_data_model(kernel, Q):
    D1, D2, D3, N, num_inducing = 13, 5, 8, 400, 3
    _, _, Ylist = GPy.examples.dimensionality_reduction._simulate_matern(
        D1, D2, D3, N, num_inducing, False
    )
    Y = Ylist[0]

    inan = np.random.binomial(1, 0.9, size=Y.shape).astype(bool)  # 80% missing data
    Ymissing = Y.copy()
    Ymissing[inan] = np.nan

    m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
        Ymissing,
        Q,
        init="random",
        num_inducing=num_inducing,
        kernel=kernel,
        missing_data=True,
    )

    return m
