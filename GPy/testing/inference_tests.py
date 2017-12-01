# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The test cases for various inference algorithms
"""

import unittest
import numpy as np
import GPy
#np.seterr(invalid='raise')

class InferenceXTestCase(unittest.TestCase):

    def genData(self):
        np.random.seed(1111)
        Ylist = GPy.examples.dimensionality_reduction._simulate_matern(5, 1, 1, 10, 3, False)[0]
        return Ylist[0]

    def test_inferenceX_BGPLVM_Linear(self):
        Ys = self.genData()
        m = GPy.models.BayesianGPLVM(Ys,3,kernel=GPy.kern.Linear(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X.mean, mi.X.mean, decimal=2)
        np.testing.assert_array_almost_equal(m.X.variance, mi.X.variance, decimal=2)

    def test_inferenceX_BGPLVM_RBF(self):
        Ys = self.genData()
        m = GPy.models.BayesianGPLVM(Ys,3,kernel=GPy.kern.RBF(3,ARD=True))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X.mean, mi.X.mean, decimal=2)
        np.testing.assert_array_almost_equal(m.X.variance, mi.X.variance, decimal=2)

    def test_inferenceX_GPLVM_Linear(self):
        Ys = self.genData()
        m = GPy.models.GPLVM(Ys,3,kernel=GPy.kern.Linear(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X, mi.X, decimal=2)

    def test_inferenceX_GPLVM_RBF(self):
        Ys = self.genData()
        m = GPy.models.GPLVM(Ys,3,kernel=GPy.kern.RBF(3,ARD=True))
        m.optimize()
        x, mi = m.infer_newX(m.Y, optimize=True)
        np.testing.assert_array_almost_equal(m.X, mi.X, decimal=2)

class InferenceGPEP(unittest.TestCase):

    def genData(self):
        np.random.seed(1)
        k = GPy.kern.RBF(1, variance=7., lengthscale=0.2)
        X = np.random.rand(200,1)
        f = np.random.multivariate_normal(np.zeros(200), k.K(X) + 1e-5 * np.eye(X.shape[0]))
        lik = GPy.likelihoods.Bernoulli()
        p = lik.gp_link.transf(f) # squash the latent function
        Y = lik.samples(f).reshape(-1,1)
        return X, Y

    def genNoisyData(self):
        np.random.seed(1)
        X = np.random.rand(100,1)
        self.real_std = 0.1
        noise = np.random.randn(*X[:, 0].shape)*self.real_std
        Y = (np.sin(X[:, 0]*2*np.pi) + noise)[:, None]
        self.f = np.random.rand(X.shape[0],1)
        Y_extra_noisy = Y.copy()
        Y_extra_noisy[50] += 4.
        # Y_extra_noisy[80:83] -= 2.
        return X, Y, Y_extra_noisy

    def test_inference_EP(self):
        from paramz import ObsAr
        X, Y = self.genData()
        lik = GPy.likelihoods.Bernoulli()
        k = GPy.kern.RBF(1, variance=7., lengthscale=0.2)
        inf = GPy.inference.latent_function_inference.expectation_propagation.EP(max_iters=30, delta=0.5)
        self.model = GPy.core.GP(X=X,
                        Y=Y,
                        kernel=k,
                        inference_method=inf,
                        likelihood=lik)
        K = self.model.kern.K(X)
        mean_prior = np.zeros(K.shape[0])
        post_params, ga_approx, cav_params, log_Z_tilde = self.model.inference_method.expectation_propagation(mean_prior, K, ObsAr(Y), lik, None)

        mu_tilde = ga_approx.v / ga_approx.tau.astype(float)
        p, m, d = self.model.inference_method._inference(Y, mean_prior, K, ga_approx, cav_params, lik, Y_metadata=None,  Z_tilde=log_Z_tilde)
        p0, m0, d0 = super(GPy.inference.latent_function_inference.expectation_propagation.EP, inf).inference(k, X,lik ,mu_tilde[:,None], mean_function=None, variance=1./ga_approx.tau, K=K, Z_tilde=log_Z_tilde + np.sum(- 0.5*np.log(ga_approx.tau) + 0.5*(ga_approx.v*ga_approx.v*1./ga_approx.tau)))

        assert (np.sum(np.array([m - m0,
                    np.sum(d['dL_dK'] - d0['dL_dK']),
                    np.sum(d['dL_dthetaL'] - d0['dL_dthetaL']),
                    np.sum(d['dL_dm'] - d0['dL_dm']),
                    np.sum(p._woodbury_vector - p0._woodbury_vector),
                    np.sum(p.woodbury_inv - p0.woodbury_inv)])) < 1e6)

    # NOTE: adding a test like above for parameterized likelihood- the above test is
    # only for probit likelihood which does not have any tunable hyperparameter which is why
    # the term in dictionary of gradients: dL_dthetaL will always be zero. So here we repeat tests for
    # student-t likelihood and heterodescastic gaussian noise case. This test simply checks if the posterior
    # and gradients of log marginal are roughly the same for inference through EP and exact gaussian inference using
    # the gaussian approximation for the individual likelihood site terms. For probit likelihood, it is possible to
    # calculate moments analytically, but for other likelihoods, we will need to use numerical quadrature techniques,
    # and it is possible that any error might creep up because of quadrature implementation.
    def test_inference_EP_non_classification(self):
        from paramz import ObsAr
        X, Y, Y_extra_noisy = self.genNoisyData()
        deg_freedom = 5.
        init_noise_var = 0.08
        lik_studentT = GPy.likelihoods.StudentT(deg_free=deg_freedom, sigma2=init_noise_var)
        # like_gaussian_noise = GPy.likelihoods.MixedNoise()
        k = GPy.kern.RBF(1, variance=2., lengthscale=1.1)
        ep_inf_alt = GPy.inference.latent_function_inference.expectation_propagation.EP(max_iters=4, delta=0.5)
        # ep_inf_nested = GPy.inference.latent_function_inference.expectation_propagation.EP(ep_mode='nested', max_iters=100, delta=0.5)
        m = GPy.core.GP(X=X,Y=Y_extra_noisy,kernel=k,likelihood=lik_studentT,inference_method=ep_inf_alt)
        K = m.kern.K(X)
        mean_prior = np.zeros(K.shape[0])
        post_params, ga_approx, cav_params, log_Z_tilde = m.inference_method.expectation_propagation(mean_prior, K, ObsAr(Y_extra_noisy), lik_studentT, None)

        mu_tilde = ga_approx.v / ga_approx.tau.astype(float)
        p, m, d = m.inference_method._inference(Y_extra_noisy, mean_prior, K, ga_approx, cav_params, lik_studentT, Y_metadata=None,  Z_tilde=log_Z_tilde)
        p0, m0, d0 = super(GPy.inference.latent_function_inference.expectation_propagation.EP, ep_inf_alt).inference(k, X,lik_studentT ,mu_tilde[:,None], mean_function=None, variance=1./ga_approx.tau, K=K, Z_tilde=log_Z_tilde + np.sum(- 0.5*np.log(ga_approx.tau) + 0.5*(ga_approx.v*ga_approx.v*1./ga_approx.tau)))

        assert (np.sum(np.array([m - m0,
                    np.sum(d['dL_dK'] - d0['dL_dK']),
                    np.sum(d['dL_dthetaL'] - d0['dL_dthetaL']),
                    np.sum(d['dL_dm'] - d0['dL_dm']),
                    np.sum(p._woodbury_vector - p0._woodbury_vector),
                    np.sum(p.woodbury_inv - p0.woodbury_inv)])) < 1e6)

class VarDtcTest(unittest.TestCase):

    def test_var_dtc_inference_with_mean(self):
        """ Check dL_dm in var_dtc is calculated correctly"""
        np.random.seed(1)
        x = np.linspace(0.,2*np.pi,100)[:,None]
        y = -np.cos(x)+np.random.randn(*x.shape)*0.3+1
        m = GPy.models.SparseGPRegression(x,y, mean_function=GPy.mappings.Linear(input_dim=1, output_dim=1))
        self.assertTrue(m.checkgrad())


class HMCSamplerTest(unittest.TestCase):

    def test_sampling(self):
        np.random.seed(1)
        x = np.linspace(0.,2*np.pi,100)[:,None]
        y = -np.cos(x)+np.random.randn(*x.shape)*0.3+1

        m = GPy.models.GPRegression(x,y)
        m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
        m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
        m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))

        hmc = GPy.inference.mcmc.HMC(m,stepsize=1e-2)
        s = hmc.sample(num_samples=3)

class MCMCSamplerTest(unittest.TestCase):

    def test_sampling(self):
        np.random.seed(1)
        x = np.linspace(0.,2*np.pi,100)[:,None]
        y = -np.cos(x)+np.random.randn(*x.shape)*0.3+1

        m = GPy.models.GPRegression(x,y)
        m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
        m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
        m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))

        mcmc = GPy.inference.mcmc.Metropolis_Hastings(m)
        mcmc.sample(Ntotal=100, Nburn=10)

if __name__ == "__main__":
    unittest.main()
