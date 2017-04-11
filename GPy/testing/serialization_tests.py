'''
Created on 20 April 2017

@author: pgmoren
'''
import unittest, itertools
#import cPickle as pickle
import pickle
import numpy as np
import tempfile
import GPy
from nose import SkipTest
import numpy as np
fixed_seed = 11


class Test(unittest.TestCase):
    def test_serialize_deserialize_kernels(self):
        k1 = GPy.kern.RBF(2, variance=1.0, lengthscale=[1.0,1.0], ARD=True)
        k2 = GPy.kern.RatQuad(2, variance=2.0, lengthscale=1.0, power=2.0, active_dims = [0,1])
        k3 = GPy.kern.Bias(2, variance=2.0, active_dims = [1,0])
        k4 = GPy.kern.StdPeriodic(2, variance=2.0, lengthscale=1.0, period=1.0, active_dims = [1,1])
        k5 = GPy.kern.Linear(2, variances=[2.0, 1.0], ARD=True, active_dims = [1,1])
        k6 = GPy.kern.Exponential(2, variance=1., lengthscale=2)
        k7 = GPy.kern.Matern32(2, variance=1.0, lengthscale=[1.0,3.0], ARD=True, active_dims = [1,1])
        k8 = GPy.kern.Matern52(2, variance=2.0, lengthscale=[2.0,1.0], ARD=True, active_dims = [1,0])
        k9 = GPy.kern.ExpQuad(2, variance=3.0, lengthscale=[1.0,2.0], ARD=True, active_dims = [0,1])
        k10 = k1 + k1.copy() + k2 + k3 + k4 + k5 + k6
        k11 = k1 * k2 * k2.copy() * k3 * k4 * k5
        k12 = (k1 + k2) * (k3 + k4 + k5)
        k13 = ((k1 + k2) * k3) + k4 + k5 * k7
        k14 = ((k1 + k2) * k3) + k4 * k5 + k8
        k15 = ((k1 * k2) * k3) + k4 * k5 + k8 + k9

        k_list = [k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15]

        for kk in k_list:
            kk_dict = kk.to_dict()
            kk_r = GPy.kern.Kern.from_dict(kk_dict)
            assert type(kk) == type(kk_r)
            np.testing.assert_array_equal(kk[:], kk_r[:])
            np.testing.assert_array_equal(np.array(kk.active_dims), np.array(kk_r.active_dims))

    def test_serialize_deserialize_mappings(self):
        m1 = GPy.mappings.Identity(3,2)
        m2 = GPy.mappings.Constant(3,2,1)
        m2_r = GPy.core.mapping.Mapping.from_dict(m2.to_dict())
        np.testing.assert_array_equal(m2.C.values[:], m2_r.C.values[:])
        m3 = GPy.mappings.Linear(3,2)
        m3_r = GPy.core.mapping.Mapping.from_dict(m3.to_dict())
        assert np.all(m3.A == m3_r.A)

        m_list = [m1, m2, m3]
        for mm in m_list:
            mm_dict = mm.to_dict()
            mm_r = GPy.core.mapping.Mapping.from_dict(mm_dict)
            assert type(mm) == type(mm_r)
            assert type(mm.input_dim) == type(mm_r.input_dim)
            assert type(mm.output_dim) == type(mm_r.output_dim)

    def test_serialize_deserialize_likelihoods(self):
        l1 = GPy.likelihoods.Gaussian(GPy.likelihoods.link_functions.Identity(),variance=3.0)
        l1_r = GPy.likelihoods.likelihood.Likelihood.from_dict(l1.to_dict())
        l2 = GPy.likelihoods.Bernoulli(GPy.likelihoods.link_functions.Probit())
        l2_r = GPy.likelihoods.likelihood.Likelihood.from_dict(l2.to_dict())
        assert type(l1) == type(l1_r)
        assert np.all(l1.variance == l1_r.variance)
        assert type(l2) == type(l2_r)

    def test_serialize_deserialize_normalizers(self):
        n1 = GPy.util.normalizer.Standardize()
        n1.scale_by(np.random.rand(10))
        n1_r = GPy.util.normalizer._Norm.from_dict((n1.to_dict()))
        assert type(n1) == type(n1_r)
        assert np.all(n1.mean == n1_r.mean)
        assert np.all(n1.std == n1_r.std)

    def test_serialize_deserialize_link_functions(self):
        l1 = GPy.likelihoods.link_functions.Identity()
        l2 = GPy.likelihoods.link_functions.Probit()
        l_list = [l1, l2]
        for ll in l_list:
            ll_dict = ll.to_dict()
            ll_r = GPy.likelihoods.link_functions.GPTransformation.from_dict(ll_dict)
            assert type(ll) == type(ll_r)

    def test_serialize_deserialize_inference_methods(self):

        e1 = GPy.inference.latent_function_inference.expectation_propagation.EP(ep_mode="nested")
        e1.ga_approx_old = GPy.inference.latent_function_inference.expectation_propagation.gaussianApproximation(np.random.rand(10),np.random.rand(10))
        e1._ep_approximation = []
        e1._ep_approximation.append(GPy.inference.latent_function_inference.expectation_propagation.posteriorParams(np.random.rand(10),np.random.rand(100).reshape((10,10))))
        e1._ep_approximation.append(GPy.inference.latent_function_inference.expectation_propagation.gaussianApproximation(np.random.rand(10),np.random.rand(10)))
        e1._ep_approximation.append(GPy.inference.latent_function_inference.expectation_propagation.cavityParams(10))
        e1._ep_approximation[-1].v = np.random.rand(10)
        e1._ep_approximation[-1].tau = np.random.rand(10)
        e1._ep_approximation.append(np.random.rand(10))
        e1_r = GPy.inference.latent_function_inference.LatentFunctionInference.from_dict(e1.to_dict())

        assert type(e1) == type(e1_r)
        assert e1.epsilon==e1_r.epsilon
        assert e1.eta==e1_r.eta
        assert e1.delta==e1_r.delta
        assert e1.always_reset==e1_r.always_reset
        assert e1.max_iters==e1_r.max_iters
        assert e1.ep_mode==e1_r.ep_mode
        assert e1.parallel_updates==e1_r.parallel_updates

        np.testing.assert_array_equal(e1.ga_approx_old.tau[:], e1_r.ga_approx_old.tau[:])
        np.testing.assert_array_equal(e1.ga_approx_old.v[:], e1_r.ga_approx_old.v[:])
        np.testing.assert_array_equal(e1._ep_approximation[0].mu[:], e1_r._ep_approximation[0].mu[:])
        np.testing.assert_array_equal(e1._ep_approximation[0].Sigma[:], e1_r._ep_approximation[0].Sigma[:])
        np.testing.assert_array_equal(e1._ep_approximation[1].tau[:], e1_r._ep_approximation[1].tau[:])
        np.testing.assert_array_equal(e1._ep_approximation[1].v[:], e1_r._ep_approximation[1].v[:])
        np.testing.assert_array_equal(e1._ep_approximation[2].tau[:], e1_r._ep_approximation[2].tau[:])
        np.testing.assert_array_equal(e1._ep_approximation[2].v[:], e1_r._ep_approximation[2].v[:])
        np.testing.assert_array_equal(e1._ep_approximation[3][:], e1_r._ep_approximation[3][:])

        e2 = GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference()
        e2_r = GPy.inference.latent_function_inference.LatentFunctionInference.from_dict(e2.to_dict())

        assert type(e2) == type(e2_r)

    def test_serialize_deserialize_model(self):
        np.random.seed(fixed_seed)
        N = 20
        Nhalf = int(N/2)
        X = np.hstack([np.random.normal(5, 2, Nhalf), np.random.normal(10, 2, Nhalf)])[:, None]
        Y = np.hstack([np.ones(Nhalf), np.zeros(Nhalf)])[:, None]
        kernel = GPy.kern.RBF(1)
        likelihood = GPy.likelihoods.Bernoulli()
        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(ep_mode="nested")
        mean_function=None
        m = GPy.core.GP(X=X, Y=Y,  kernel=kernel, likelihood=likelihood, inference_method=inference_method, mean_function=mean_function, normalizer=True, name='gp_classification')
        m.optimize()
        m.save_model("temp_test_gp_with_data.json", compress=True, save_data=True)
        m.save_model("temp_test_gp_without_data.json", compress=True, save_data=False)
        m1_r = GPy.core.GP.load_model("temp_test_gp_with_data.json.zip")
        m2_r = GPy.core.GP.load_model("temp_test_gp_without_data.json.zip", (X,Y))
        import os
        os.remove("temp_test_gp_with_data.json.zip")
        os.remove("temp_test_gp_without_data.json.zip")
        var = m.predict(X)[0]
        var1_r = m1_r.predict(X)[0]
        var2_r = m2_r.predict(X)[0]
        np.testing.assert_array_equal(np.array(var).flatten(), np.array(var1_r).flatten())
        np.testing.assert_array_equal(np.array(var).flatten(), np.array(var2_r).flatten())

    def test_serialize_deserialize_inference_GPRegressor(self):
        np.random.seed(fixed_seed)
        N = 50
        N_new = 50
        D = 1
        X = np.random.uniform(-3., 3., (N, 1))
        Y = np.sin(X) + np.random.randn(N, D) * 0.05
        X_new = np.random.uniform(-3., 3., (N_new, 1))
        k = GPy.kern.RBF(input_dim=1, lengthscale=10)
        m = GPy.models.GPRegression(X,Y,k)
        m.optimize()
        m.save_model("temp_test_gp_regressor_with_data.json", compress=True, save_data=True)
        m.save_model("temp_test_gp_regressor_without_data.json", compress=True, save_data=False)
        m1_r = GPy.models.GPRegression.load_model("temp_test_gp_regressor_with_data.json.zip")
        m2_r = GPy.models.GPRegression.load_model("temp_test_gp_regressor_without_data.json.zip", (X,Y))
        import os
        os.remove("temp_test_gp_regressor_with_data.json.zip")
        os.remove("temp_test_gp_regressor_without_data.json.zip")

        Xp = np.random.uniform(size=(int(1e5),1))
        Xp[:,0] = Xp[:,0]*15-5

        _, var = m.predict(Xp)
        _, var1_r = m1_r.predict(Xp)
        _, var2_r = m2_r.predict(Xp)
        np.testing.assert_array_equal(var.flatten(), var1_r.flatten())
        np.testing.assert_array_equal(var.flatten(), var2_r.flatten())

    def test_serialize_deserialize_inference_GPClassifier(self):
        np.random.seed(fixed_seed)
        N = 50
        Nhalf = int(N/2)
        X = np.hstack([np.random.normal(5, 2, Nhalf), np.random.normal(10, 2, Nhalf)])[:, None]
        Y = np.hstack([np.ones(Nhalf), np.zeros(Nhalf)])[:, None]
        kernel = GPy.kern.RBF(1)
        m = GPy.models.GPClassification(X, Y, kernel=kernel)
        m.optimize()
        m.save_model("temp_test_gp_classifier_with_data.json", compress=True, save_data=True)
        m.save_model("temp_test_gp_classifier_without_data.json", compress=True, save_data=False)
        m1_r = GPy.models.GPClassification.load_model("temp_test_gp_classifier_with_data.json.zip")
        m2_r = GPy.models.GPClassification.load_model("temp_test_gp_classifier_without_data.json.zip", (X,Y))
        import os
        os.remove("temp_test_gp_classifier_with_data.json.zip")
        os.remove("temp_test_gp_classifier_without_data.json.zip")

        var = m.predict(X)[0]
        var1_r = m1_r.predict(X)[0]
        var2_r = m2_r.predict(X)[0]
        np.testing.assert_array_equal(np.array(var).flatten(), np.array(var1_r).flatten())
        np.testing.assert_array_equal(np.array(var).flatten(), np.array(var1_r).flatten())

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_parameter_index_operations']
    unittest.main()
