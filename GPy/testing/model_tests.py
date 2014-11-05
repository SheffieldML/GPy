# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import unittest
import numpy as np
import GPy

class MiscTests(unittest.TestCase):
    def setUp(self):
        self.N = 20
        self.N_new = 50
        self.D = 1
        self.X = np.random.uniform(-3., 3., (self.N, 1))
        self.Y = np.sin(self.X) + np.random.randn(self.N, self.D) * 0.05
        self.X_new = np.random.uniform(-3., 3., (self.N_new, 1))

    def test_raw_predict(self):
        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.randomize()
        m.likelihood.variance = .5
        Kinv = np.linalg.pinv(k.K(self.X) + np.eye(self.N) * m.likelihood.variance)
        K_hat = k.K(self.X_new) - k.K(self.X_new, self.X).dot(Kinv).dot(k.K(self.X, self.X_new))
        mu_hat = k.K(self.X_new, self.X).dot(Kinv).dot(m.Y_normalized)

        mu, covar = m._raw_predict(self.X_new, full_cov=True)
        self.assertEquals(mu.shape, (self.N_new, self.D))
        self.assertEquals(covar.shape, (self.N_new, self.N_new))
        np.testing.assert_almost_equal(K_hat, covar)
        np.testing.assert_almost_equal(mu_hat, mu)

        mu, var = m._raw_predict(self.X_new)
        self.assertEquals(mu.shape, (self.N_new, self.D))
        self.assertEquals(var.shape, (self.N_new, 1))
        np.testing.assert_almost_equal(np.diag(K_hat)[:, None], var)
        np.testing.assert_almost_equal(mu_hat, mu)

    def test_sparse_raw_predict(self):
        k = GPy.kern.RBF(1)
        m = GPy.models.SparseGPRegression(self.X, self.Y, kernel=k)
        m.randomize()
        Z = m.Z[:]
        X = self.X[:]

        # Not easy to check if woodbury_inv is correct in itself as it requires a large derivation and expression
        Kinv = m.posterior.woodbury_inv
        K_hat = k.K(self.X_new) - k.K(self.X_new, Z).dot(Kinv).dot(k.K(Z, self.X_new))

        mu, covar = m._raw_predict(self.X_new, full_cov=True)
        self.assertEquals(mu.shape, (self.N_new, self.D))
        self.assertEquals(covar.shape, (self.N_new, self.N_new))
        np.testing.assert_almost_equal(K_hat, covar)
        # np.testing.assert_almost_equal(mu_hat, mu)

        mu, var = m._raw_predict(self.X_new)
        self.assertEquals(mu.shape, (self.N_new, self.D))
        self.assertEquals(var.shape, (self.N_new, 1))
        np.testing.assert_almost_equal(np.diag(K_hat)[:, None], var)
        # np.testing.assert_almost_equal(mu_hat, mu)

    def test_likelihood_replicate(self):
        m = GPy.models.GPRegression(self.X, self.Y)
        m2 = GPy.models.GPRegression(self.X, self.Y)
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[:] = m[''].values()
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[''] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.randomize()
        m2[''] = m['']
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m.Gaussian_noise.randomize()
        m2[:] = m[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

        m['.*var'] = 2
        m2['.*var'] = m['.*var']
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())


    def test_likelihood_set(self):
        m = GPy.models.GPRegression(self.X, self.Y)
        m2 = GPy.models.GPRegression(self.X, self.Y)
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2.kern.lengthscale = m.kern.lengthscale
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2['.*lengthscale'] = m.kern.lengthscale
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2['.*lengthscale'] = m.kern['.*lengthscale']
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

        m.kern.lengthscale.randomize()
        m2.kern.lengthscale = m.kern['.*lengthscale']
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())

    def test_missing_data(self):
        from GPy import kern
        from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
        from GPy.examples.dimensionality_reduction import _simulate_matern

        D1, D2, D3, N, num_inducing, Q = 13, 5, 8, 400, 3, 4
        _, _, Ylist = _simulate_matern(D1, D2, D3, N, num_inducing, False)
        Y = Ylist[0]

        inan = np.random.binomial(1, .9, size=Y.shape).astype(bool) # 80% missing data
        Ymissing = Y.copy()
        Ymissing[inan] = np.nan

        k = kern.Linear(Q, ARD=True) + kern.White(Q, np.exp(-2)) # + kern.bias(Q)
        m = BayesianGPLVMMiniBatch(Ymissing, Q, init="random", num_inducing=num_inducing,
                          kernel=k, missing_data=True)
        assert(m.checkgrad())

        k = kern.RBF(Q, ARD=True) + kern.White(Q, np.exp(-2)) # + kern.bias(Q)
        m = BayesianGPLVMMiniBatch(Ymissing, Q, init="random", num_inducing=num_inducing,
                          kernel=k, missing_data=True)
        assert(m.checkgrad())

    def test_likelihood_replicate_kern(self):
        m = GPy.models.GPRegression(self.X, self.Y)
        m2 = GPy.models.GPRegression(self.X, self.Y)
        np.testing.assert_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[''] = m.kern[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[:] = m.kern[:]
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[''] = m.kern['']
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())
        m.kern.randomize()
        m2.kern[:] = m.kern[''].values()
        np.testing.assert_almost_equal(m.log_likelihood(), m2.log_likelihood())

    def test_big_model(self):
        m = GPy.examples.dimensionality_reduction.mrd_simulation(optimize=0, plot=0, plot_sim=0)
        m.X.fix()
        print m
        m.unfix()
        m.checkgrad()
        print m
        m.fix()
        print m
        m.inducing_inputs.unfix()
        print m
        m.checkgrad()
        m.unfix()
        m.checkgrad()
        m.checkgrad()
        print m

    def test_model_set_params(self):
        m = GPy.models.GPRegression(self.X, self.Y)
        lengthscale = np.random.uniform()
        m.kern.lengthscale = lengthscale
        np.testing.assert_equal(m.kern.lengthscale, lengthscale)
        m.kern.lengthscale *= 1
        m['.*var'] -= .1
        np.testing.assert_equal(m.kern.lengthscale, lengthscale)
        m.optimize()
        print m

    def test_model_optimize(self):
        X = np.random.uniform(-3., 3., (20, 1))
        Y = np.sin(X) + np.random.randn(20, 1) * 0.05
        m = GPy.models.GPRegression(X, Y)
        m.optimize()
        print m

class GradientTests(np.testing.TestCase):
    def setUp(self):
        ######################################
        # # 1 dimensional example

        # sample inputs and outputs
        self.X1D = np.random.uniform(-3., 3., (20, 1))
        self.Y1D = np.sin(self.X1D) + np.random.randn(20, 1) * 0.05

        ######################################
        # # 2 dimensional example

        # sample inputs and outputs
        self.X2D = np.random.uniform(-3., 3., (40, 2))
        self.Y2D = np.sin(self.X2D[:, 0:1]) * np.sin(self.X2D[:, 1:2]) + np.random.randn(40, 1) * 0.05

    def check_model(self, kern, model_type='GPRegression', dimension=1, uncertain_inputs=False):
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
            m = model_fit(X, Y, kernel=kern, X_variance=np.random.rand(X.shape[0], X.shape[1]))
        else:
            m = model_fit(X, Y, kernel=kern)
        m.randomize()
        # contrain all parameters to be positive
        self.assertTrue(m.checkgrad())

    def test_GPRegression_rbf_1d(self):
        ''' Testing the GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.RBF(1)
        self.check_model(rbf, model_type='GPRegression', dimension=1)

    def test_GPRegression_rbf_2D(self):
        ''' Testing the GP regression with rbf kernel on 2d data '''
        rbf = GPy.kern.RBF(2)
        self.check_model(rbf, model_type='GPRegression', dimension=2)

    def test_GPRegression_rbf_ARD_2D(self):
        ''' Testing the GP regression with rbf kernel on 2d data '''
        k = GPy.kern.RBF(2, ARD=True)
        self.check_model(k, model_type='GPRegression', dimension=2)

    def test_GPRegression_mlp_1d(self):
        ''' Testing the GP regression with mlp kernel with white kernel on 1d data '''
        mlp = GPy.kern.MLP(1)
        self.check_model(mlp, model_type='GPRegression', dimension=1)

    # TODO:
    # def test_GPRegression_poly_1d(self):
    #    ''' Testing the GP regression with polynomial kernel with white kernel on 1d data '''
    #    mlp = GPy.kern.Poly(1, degree=5)
    #    self.check_model(mlp, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern52_1D(self):
        ''' Testing the GP regression with matern52 kernel on 1d data '''
        matern52 = GPy.kern.Matern52(1)
        self.check_model(matern52, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern52_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2)
        self.check_model(matern52, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern52_ARD_2D(self):
        ''' Testing the GP regression with matern52 kernel on 2d data '''
        matern52 = GPy.kern.Matern52(2, ARD=True)
        self.check_model(matern52, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern32_1D(self):
        ''' Testing the GP regression with matern32 kernel on 1d data '''
        matern32 = GPy.kern.Matern32(1)
        self.check_model(matern32, model_type='GPRegression', dimension=1)

    def test_GPRegression_matern32_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2)
        self.check_model(matern32, model_type='GPRegression', dimension=2)

    def test_GPRegression_matern32_ARD_2D(self):
        ''' Testing the GP regression with matern32 kernel on 2d data '''
        matern32 = GPy.kern.Matern32(2, ARD=True)
        self.check_model(matern32, model_type='GPRegression', dimension=2)

    def test_GPRegression_exponential_1D(self):
        ''' Testing the GP regression with exponential kernel on 1d data '''
        exponential = GPy.kern.Exponential(1)
        self.check_model(exponential, model_type='GPRegression', dimension=1)

    def test_GPRegression_exponential_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.Exponential(2)
        self.check_model(exponential, model_type='GPRegression', dimension=2)

    def test_GPRegression_exponential_ARD_2D(self):
        ''' Testing the GP regression with exponential kernel on 2d data '''
        exponential = GPy.kern.Exponential(2, ARD=True)
        self.check_model(exponential, model_type='GPRegression', dimension=2)

    def test_GPRegression_bias_kern_1D(self):
        ''' Testing the GP regression with bias kernel on 1d data '''
        bias = GPy.kern.Bias(1)
        self.check_model(bias, model_type='GPRegression', dimension=1)

    def test_GPRegression_bias_kern_2D(self):
        ''' Testing the GP regression with bias kernel on 2d data '''
        bias = GPy.kern.Bias(2)
        self.check_model(bias, model_type='GPRegression', dimension=2)

    def test_GPRegression_linear_kern_1D_ARD(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.Linear(1, ARD=True)
        self.check_model(linear, model_type='GPRegression', dimension=1)

    def test_GPRegression_linear_kern_2D_ARD(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.Linear(2, ARD=True)
        self.check_model(linear, model_type='GPRegression', dimension=2)

    def test_GPRegression_linear_kern_1D(self):
        ''' Testing the GP regression with linear kernel on 1d data '''
        linear = GPy.kern.Linear(1)
        self.check_model(linear, model_type='GPRegression', dimension=1)

    def test_GPRegression_linear_kern_2D(self):
        ''' Testing the GP regression with linear kernel on 2d data '''
        linear = GPy.kern.Linear(2)
        self.check_model(linear, model_type='GPRegression', dimension=2)

    def test_SparseGPRegression_rbf_white_kern_1d(self):
        ''' Testing the sparse GP regression with rbf kernel with white kernel on 1d data '''
        rbf = GPy.kern.RBF(1)
        self.check_model(rbf, model_type='SparseGPRegression', dimension=1)

    def test_SparseGPRegression_rbf_white_kern_2D(self):
        ''' Testing the sparse GP regression with rbf kernel on 2d data '''
        rbf = GPy.kern.RBF(2)
        self.check_model(rbf, model_type='SparseGPRegression', dimension=2)

    def test_SparseGPRegression_rbf_linear_white_kern_1D(self):
        ''' Testing the sparse GP regression with rbf kernel on 2d data '''
        rbflin = GPy.kern.RBF(1) + GPy.kern.Linear(1)
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=1)

    def test_SparseGPRegression_rbf_linear_white_kern_2D(self):
        ''' Testing the sparse GP regression with rbf kernel on 2d data '''
        rbflin = GPy.kern.RBF(2) + GPy.kern.Linear(2)
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=2)

    # @unittest.expectedFailure
    def test_SparseGPRegression_rbf_linear_white_kern_2D_uncertain_inputs(self):
        ''' Testing the sparse GP regression with rbf, linear kernel on 2d data with uncertain inputs'''
        rbflin = GPy.kern.RBF(2) + GPy.kern.Linear(2)
        raise unittest.SkipTest("This is not implemented yet!")
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=2, uncertain_inputs=1)

    # @unittest.expectedFailure
    def test_SparseGPRegression_rbf_linear_white_kern_1D_uncertain_inputs(self):
        ''' Testing the sparse GP regression with rbf, linear kernel on 1d data with uncertain inputs'''
        rbflin = GPy.kern.RBF(1) + GPy.kern.Linear(1)
        raise unittest.SkipTest("This is not implemented yet!")
        self.check_model(rbflin, model_type='SparseGPRegression', dimension=1, uncertain_inputs=1)

    def test_GPLVM_rbf_bias_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias kernel """
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = GPy.kern.RBF(input_dim, 0.5, 0.9 * np.ones((1,))) + GPy.kern.Bias(input_dim, 0.1) + GPy.kern.White(input_dim, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, kernel=k)
        self.assertTrue(m.checkgrad())

    def test_GPLVM_rbf_linear_white_kern_2D(self):
        """ Testing GPLVM with rbf + bias kernel """
        N, input_dim, D = 50, 1, 2
        X = np.random.rand(N, input_dim)
        k = GPy.kern.Linear(input_dim) + GPy.kern.Bias(input_dim, 0.1) + GPy.kern.White(input_dim, 0.05)
        K = k.K(X)
        Y = np.random.multivariate_normal(np.zeros(N), K, input_dim).T
        m = GPy.models.GPLVM(Y, input_dim, init='PCA', kernel=k)
        self.assertTrue(m.checkgrad())

    def test_GP_EP_probit(self):
        N = 20
        X = np.hstack([np.random.normal(5, 2, N / 2), np.random.normal(10, 2, N / 2)])[:, None]
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        kernel = GPy.kern.RBF(1)
        m = GPy.models.GPClassification(X, Y, kernel=kernel)
        self.assertTrue(m.checkgrad())

    def test_sparse_EP_DTC_probit(self):
        N = 20
        X = np.hstack([np.random.normal(5, 2, N / 2), np.random.normal(10, 2, N / 2)])[:, None]
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        Z = np.linspace(0, 15, 4)[:, None]
        kernel = GPy.kern.RBF(1)
        m = GPy.models.SparseGPClassification(X, Y, kernel=kernel, Z=Z)
        # distribution = GPy.likelihoods.likelihood_functions.Bernoulli()
        # likelihood = GPy.likelihoods.EP(Y, distribution)
        # m = GPy.core.SparseGP(X, likelihood, kernel, Z)
        # m.ensure_default_constraints()
        self.assertTrue(m.checkgrad())

    @unittest.expectedFailure
    def test_generalized_FITC(self):
        N = 20
        X = np.hstack([np.random.rand(N / 2) + 1, np.random.rand(N / 2) - 1])[:, None]
        k = GPy.kern.RBF(1) + GPy.kern.White(1)
        Y = np.hstack([np.ones(N / 2), np.zeros(N / 2)])[:, None]
        m = GPy.models.FITCClassification(X, Y, kernel=k)
        m.update_likelihood_approximation()
        self.assertTrue(m.checkgrad())

    @unittest.expectedFailure
    def test_multioutput_regression_1D(self):
        X1 = np.random.rand(50, 1) * 8
        X2 = np.random.rand(30, 1) * 5
        X = np.vstack((X1, X2))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
        Y2 = -np.sin(X2) + np.random.randn(*X2.shape) * 0.05
        Y = np.vstack((Y1, Y2))

        k1 = GPy.kern.RBF(1)
        m = GPy.models.GPMultioutputRegression(X_list=[X1, X2], Y_list=[Y1, Y2], kernel_list=[k1])
        import ipdb;ipdb.set_trace()
        m.constrain_fixed('.*rbf_var', 1.)
        self.assertTrue(m.checkgrad())

    @unittest.expectedFailure
    def test_multioutput_sparse_regression_1D(self):
        X1 = np.random.rand(500, 1) * 8
        X2 = np.random.rand(300, 1) * 5
        X = np.vstack((X1, X2))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
        Y2 = -np.sin(X2) + np.random.randn(*X2.shape) * 0.05
        Y = np.vstack((Y1, Y2))

        k1 = GPy.kern.RBF(1)
        m = GPy.models.SparseGPMultioutputRegression(X_list=[X1, X2], Y_list=[Y1, Y2], kernel_list=[k1])
        m.constrain_fixed('.*rbf_var', 1.)
        self.assertTrue(m.checkgrad())

    def test_gp_heteroscedastic_regression(self):
        num_obs = 25
        X = np.random.randint(0, 140, num_obs)
        X = X[:, None]
        Y = 25. + np.sin(X / 20.) * 2. + np.random.rand(num_obs)[:, None]
        kern = GPy.kern.Bias(1) + GPy.kern.RBF(1)
        m = GPy.models.GPHeteroscedasticRegression(X, Y, kern)
        self.assertTrue(m.checkgrad())

    def test_sparse_gp_heteroscedastic_regression(self):
        num_obs = 25
        X = np.random.randint(0, 140, num_obs)
        X = X[:, None]
        Y = 25. + np.sin(X / 20.) * 2. + np.random.rand(num_obs)[:, None]
        kern = GPy.kern.Bias(1) + GPy.kern.RBF(1)
        Y_metadata = {'output_index':np.arange(num_obs)[:,None]}
        noise_terms = np.unique(Y_metadata['output_index'].flatten())
        likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j) for j in noise_terms]
        likelihood = GPy.likelihoods.MixedNoise(likelihoods_list=likelihoods_list)
        m = GPy.core.SparseGP(X, Y, X[np.random.choice(num_obs, 10)],
                              kern, likelihood,
                              GPy.inference.latent_function_inference.VarDTC(),
                              Y_metadata=Y_metadata)
        self.assertTrue(m.checkgrad())

    def test_gp_kronecker_gaussian(self):
        N1, N2 = 30, 20
        X1 = np.random.randn(N1, 1)
        X2 = np.random.randn(N2, 1)
        X1.sort(0); X2.sort(0)
        k1 = GPy.kern.RBF(1)  # + GPy.kern.White(1)
        k2 = GPy.kern.RBF(1)  # + GPy.kern.White(1)
        Y = np.random.randn(N1, N2)
        Y = Y - Y.mean(0)
        Y = Y / Y.std(0)
        m = GPy.models.GPKroneckerGaussianRegression(X1, X2, Y, k1, k2)

        # build the model the dumb way
        assert (N1 * N2 < 1000), "too much data for standard GPs!"
        yy, xx = np.meshgrid(X2, X1)
        Xgrid = np.vstack((xx.flatten(order='F'), yy.flatten(order='F'))).T
        kg = GPy.kern.RBF(1, active_dims=[0]) * GPy.kern.RBF(1, active_dims=[1])
        mm = GPy.models.GPRegression(Xgrid, Y.reshape(-1, 1, order='F'), kernel=kg)

        m.randomize()
        mm[:] = m[:]
        assert np.allclose(m.log_likelihood(), mm.log_likelihood())
        assert np.allclose(m.gradient, mm.gradient)
        X1test = np.random.randn(100, 1)
        X2test = np.random.randn(100, 1)
        mean1, var1 = m.predict(X1test, X2test)
        yy, xx = np.meshgrid(X2test, X1test)
        Xgrid = np.vstack((xx.flatten(order='F'), yy.flatten(order='F'))).T
        mean2, var2 = mm.predict(Xgrid)
        assert np.allclose(mean1, mean2)
        assert np.allclose(var1, var2)

    def test_gp_VGPC(self):
        num_obs = 25
        X = np.random.randint(0, 140, num_obs)
        X = X[:, None]
        Y = 25. + np.sin(X / 20.) * 2. + np.random.rand(num_obs)[:, None]
        kern = GPy.kern.Bias(1) + GPy.kern.RBF(1)
        m = GPy.models.GPVariationalGaussianApproximation(X, Y, kern)
        self.assertTrue(m.checkgrad())


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
