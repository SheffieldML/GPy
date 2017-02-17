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
        mu, Sigma, mu_tilde, tau_tilde, log_Z_tilde = self.model.inference_method.expectation_propagation(K, ObsAr(Y), lik, None)

        v_tilde = mu_tilde * tau_tilde
        p, m, d = self.model.inference_method._inference(K, tau_tilde, v_tilde, lik, Y_metadata=None,  Z_tilde=log_Z_tilde.sum())
        p0, m0, d0 = super(GPy.inference.latent_function_inference.expectation_propagation.EP, inf).inference(k, X,lik ,mu_tilde[:,None], mean_function=None, variance=1./tau_tilde, K=K, Z_tilde=log_Z_tilde.sum() + np.sum(- 0.5*np.log(tau_tilde) + 0.5*(v_tilde*v_tilde*1./tau_tilde)))

        assert (np.sum(np.array([m - m0,
                    np.sum(d['dL_dK'] - d0['dL_dK']),
                    np.sum(d['dL_dthetaL'] - d0['dL_dthetaL']),
                    np.sum(d['dL_dm'] - d0['dL_dm']),
                    np.sum(p._woodbury_vector - p0._woodbury_vector),
                    np.sum(p.woodbury_inv - p0.woodbury_inv)])) < 1e6)


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
