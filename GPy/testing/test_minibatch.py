"""
Created on 4 Sep 2015

@author: maxz
"""
import pytest
import numpy as np
import GPy

try:
    import climin
except ImportError:
    climin = None


class TestBGPLVM:
    def setup(self):
        np.random.seed(12345)
        X, W = np.random.normal(0, 1, (100, 6)), np.random.normal(0, 1, (6, 13))
        Y = X.dot(W) + np.random.normal(0, 0.1, (X.shape[0], W.shape[1]))
        self.inan = np.random.binomial(1, 0.1, Y.shape).astype(bool)
        self.X, self.W, self.Y = X, W, Y
        self.Q = 3
        self.m_full = GPy.models.BayesianGPLVM(Y, self.Q)

    def test_lik_comparisons_m1_s0(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, missing_data=True, stochastic=False
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_predict_missing_data(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            missing_data=True,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)

        self.assertRaises(NotImplementedError, m.predict, m.X, full_cov=True)

        mu1, var1 = m.predict(m.X, full_cov=False)
        mu2, var2 = self.m_full.predict(self.m_full.X, full_cov=False)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1, var2)

        mu1, var1 = m.predict(m.X.mean, full_cov=True)
        mu2, var2 = self.m_full.predict(self.m_full.X.mean, full_cov=True)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1[:, :, 0], var2)

        mu1, var1 = m.predict(m.X.mean, full_cov=False)
        mu2, var2 = self.m_full.predict(self.m_full.X.mean, full_cov=False)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(var1[:, [0]], var2)

    def test_lik_comparisons_m0_s0(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=self.m_full.X.variance.values,
            missing_data=False,
            stochastic=False,
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_lik_comparisons_m1_s1(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            missing_data=True,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_lik_comparisons_m0_s1(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            missing_data=False,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_gradients_missingdata(self):
        self.seutp()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            missing_data=True,
            stochastic=False,
            batchsize=self.Y.shape[1],
        )
        assert m.checkgrad()

    def test_gradients_missingdata_stochastics(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, missing_data=True, stochastic=True, batchsize=1
        )
        assert m.checkgrad()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, missing_data=True, stochastic=True, batchsize=4
        )
        assert m.checkgrad()

    def test_gradients_stochastics(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, missing_data=False, stochastic=True, batchsize=1
        )
        assert m.checkgrad()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, missing_data=False, stochastic=True, batchsize=4
        )
        assert m.checkgrad()

    def test_predict(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            missing_data=True,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()


class TestSparseGPMinibatch:
    def setup(self):
        np.random.seed(12345)
        X, W = np.random.normal(0, 1, (100, 6)), np.random.normal(0, 1, (6, 13))
        Y = X.dot(W) + np.random.normal(0, 0.1, (X.shape[0], W.shape[1]))
        self.inan = np.random.binomial(1, 0.1, Y.shape).astype(bool)
        self.X, self.W, self.Y = X, W, Y
        self.Q = 3
        self.m_full = GPy.models.SparseGPLVM(
            Y, self.Q, kernel=GPy.kern.RBF(self.Q, ARD=True)
        )

    def test_lik_comparisons_m1_s0(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, X_variance=False, missing_data=True, stochastic=False
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    @pytest.mark.skipif(climin is None, reason="climin not installed")
    def test_sparsegp_init(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        np.random.seed(1234)
        Z = self.X[np.random.choice(self.X.shape[0], replace=False, size=10)].copy()
        Q = Z.shape[1]
        m = GPy.models.sparse_gp_minibatch.SparseGPMiniBatch(
            self.X,
            self.Y,
            Z,
            GPy.kern.RBF(Q) + GPy.kern.Matern32(Q) + GPy.kern.Bias(Q),
            GPy.likelihoods.Gaussian(),
            missing_data=True,
            stochastic=False,
        )
        assert m.checkgrad()
        m.optimize("adadelta", max_iters=10)
        assert m.checkgrad()

        m = GPy.models.sparse_gp_minibatch.SparseGPMiniBatch(
            self.X,
            self.Y,
            Z,
            GPy.kern.RBF(Q) + GPy.kern.Matern32(Q) + GPy.kern.Bias(Q),
            GPy.likelihoods.Gaussian(),
            missing_data=True,
            stochastic=True,
        )
        assert m.checkgrad()
        m.optimize("rprop", max_iters=10)
        assert m.checkgrad()

        m = GPy.models.sparse_gp_minibatch.SparseGPMiniBatch(
            self.X,
            self.Y,
            Z,
            GPy.kern.RBF(Q) + GPy.kern.Matern32(Q) + GPy.kern.Bias(Q),
            GPy.likelihoods.Gaussian(),
            missing_data=False,
            stochastic=False,
        )
        assert m.checkgrad()
        m.optimize("rprop", max_iters=10)
        assert m.checkgrad()

        m = GPy.models.sparse_gp_minibatch.SparseGPMiniBatch(
            self.X,
            self.Y,
            Z,
            GPy.kern.RBF(Q) + GPy.kern.Matern32(Q) + GPy.kern.Bias(Q),
            GPy.likelihoods.Gaussian(),
            missing_data=False,
            stochastic=True,
        )
        assert m.checkgrad()
        m.optimize("adadelta", max_iters=10)
        assert m.checkgrad()

    def test_predict_missing_data(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=True,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)

        mu1, var1 = m.predict(m.X, full_cov=False)
        mu2, var2 = self.m_full.predict(self.m_full.X, full_cov=False)
        np.testing.assert_allclose(mu1, mu2)
        for i in range(var1.shape[1]):
            np.testing.assert_allclose(var1[:, [i]], var2)

        mu1, var1 = m.predict(m.X, full_cov=True)
        mu2, var2 = self.m_full.predict(self.m_full.X, full_cov=True)
        np.testing.assert_allclose(mu1, mu2)
        for i in range(var1.shape[2]):
            np.testing.assert_allclose(var1[:, :, i], var2)

    def test_lik_comparisons_m0_s0(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y, self.Q, X_variance=False, missing_data=False, stochastic=False
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_lik_comparisons_m1_s1(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=True,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_lik_comparisons_m0_s1(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=False,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()

    def test_gradients_missingdata(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=True,
            stochastic=False,
            batchsize=self.Y.shape[1],
        )
        assert m.checkgrad()

    def test_gradients_missingdata_stochastics(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=True,
            stochastic=True,
            batchsize=1,
        )
        assert m.checkgrad()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=True,
            stochastic=True,
            batchsize=4,
        )
        assert m.checkgrad()

    def test_gradients_stochastics(self):
        self.setup()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=False,
            stochastic=True,
            batchsize=1,
        )
        assert m.checkgrad()
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=False,
            stochastic=True,
            batchsize=4,
        )
        assert m.checkgrad()

    def test_predict(self):
        self.setup()
        # Test if the different implementations give the exact same likelihood as the full model.
        # All of the following settings should give the same likelihood and gradients as the full model:
        m = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
            self.Y,
            self.Q,
            X_variance=False,
            missing_data=True,
            stochastic=True,
            batchsize=self.Y.shape[1],
        )
        m[:] = self.m_full[:]
        np.testing.assert_almost_equal(
            m.log_likelihood(), self.m_full.log_likelihood(), 7
        )
        np.testing.assert_allclose(m.gradient, self.m_full.gradient)
        assert m.checkgrad()
