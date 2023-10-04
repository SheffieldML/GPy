import numpy as np
import unittest
import GPy
from GPy.models import GradientChecker

fixed_seed = 10
from nose.tools import with_setup, nottest


# this file will contain some high level tests, this is not unit testing, but will give us a higher level estimate
# if things are going well under the hood.
class TestObservationModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(fixed_seed)
        self.N = 100
        self.D = 2
        self.X = np.random.rand(self.N, self.D)

        self.real_noise_std = 0.05
        noise = np.random.randn(*self.X[:, 0].shape) * self.real_noise_std
        self.Y = (np.sin(self.X[:, 0] * 2 * np.pi) + noise)[:, None]
        self.num_points = self.X.shape[0]
        self.f = np.random.rand(self.N, 1)
        self.binary_Y = np.asarray(np.random.rand(self.N) > 0.5, dtype=np.int)[:, None]
        # self.binary_Y[self.binary_Y == 0.0] = -1.0
        self.positive_Y = np.exp(self.Y.copy())

        self.Y_noisy = self.Y.copy()
        self.Y_verynoisy = self.Y.copy()
        self.Y_noisy[75] += 1.3

        self.init_var = 0.15
        self.deg_free = 4.0
        censored = np.zeros_like(self.Y)
        random_inds = np.random.choice(self.N, int(self.N / 2), replace=True)
        censored[random_inds] = 1
        self.Y_metadata = dict()
        self.Y_metadata["censored"] = censored
        self.kernel1 = GPy.kern.RBF(self.X.shape[1]) + GPy.kern.White(self.X.shape[1])

    def tearDown(self):
        self.Y = None
        self.X = None
        self.binary_Y = None
        self.positive_Y = None
        self.kernel1 = None

    @with_setup(setUp, tearDown)
    def testEPClassification(self):
        bernoulli = GPy.likelihoods.Bernoulli()
        laplace_inf = GPy.inference.latent_function_inference.Laplace()

        ep_inf_alt = GPy.inference.latent_function_inference.EP(ep_mode="alternated")
        ep_inf_nested = GPy.inference.latent_function_inference.EP(ep_mode="nested")
        ep_inf_fractional = GPy.inference.latent_function_inference.EP(
            ep_mode="nested", eta=0.9
        )

        m1 = GPy.core.GP(
            self.X,
            self.binary_Y.copy(),
            kernel=self.kernel1.copy(),
            likelihood=bernoulli.copy(),
            inference_method=laplace_inf,
        )
        m1.randomize()

        m2 = GPy.core.GP(
            self.X,
            self.binary_Y.copy(),
            kernel=self.kernel1.copy(),
            likelihood=bernoulli.copy(),
            inference_method=ep_inf_alt,
        )
        m2.randomize()

        m3 = GPy.core.GP(
            self.X,
            self.binary_Y.copy(),
            kernel=self.kernel1.copy(),
            likelihood=bernoulli.copy(),
            inference_method=ep_inf_nested,
        )
        m3.randomize()
        #
        m4 = GPy.core.GP(
            self.X,
            self.binary_Y.copy(),
            kernel=self.kernel1.copy(),
            likelihood=bernoulli.copy(),
            inference_method=ep_inf_fractional,
        )
        m4.randomize()

        optimizer = "bfgs"

        # do gradcheck here ...
        # self.assertTrue(m1.checkgrad())
        # self.assertTrue(m2.checkgrad())
        # self.assertTrue(m3.checkgrad())
        # self.assertTrue(m4.checkgrad())

        m1.optimize(optimizer=optimizer, max_iters=300)
        m2.optimize(optimizer=optimizer, max_iters=300)
        m3.optimize(optimizer=optimizer, max_iters=300)
        m4.optimize(optimizer=optimizer, max_iters=300)

        # taking laplace predictions as the ground truth
        probs_mean_lap, probs_var_lap = m1.predict(self.X)
        probs_mean_ep_alt, probs_var_ep_alt = m2.predict(self.X)
        probs_mean_ep_nested, probs_var_ep_nested = m3.predict(self.X)

        # for simple single dimension data , marginal likelihood for laplace and EP approximations should not be so far apart.
        self.assertAlmostEqual(m1.log_likelihood(), m2.log_likelihood(), delta=1)
        self.assertAlmostEqual(m1.log_likelihood(), m3.log_likelihood(), delta=1)
        self.assertAlmostEqual(m1.log_likelihood(), m4.log_likelihood(), delta=5)

        GPy.util.classification.conf_matrix(probs_mean_lap, self.binary_Y)
        GPy.util.classification.conf_matrix(probs_mean_ep_alt, self.binary_Y)
        GPy.util.classification.conf_matrix(probs_mean_ep_nested, self.binary_Y)

    @nottest
    def rmse(self, Y, Ystar):
        return np.sqrt(np.mean((Y - Ystar) ** 2))

    @with_setup(setUp, tearDown)
    @unittest.skip(
        "Fails as a consequence of fixing the DSYR function. Needs to be reviewed!"
    )
    def test_EP_with_StudentT(self):
        studentT = GPy.likelihoods.StudentT(
            deg_free=self.deg_free, sigma2=self.init_var
        )
        laplace_inf = GPy.inference.latent_function_inference.Laplace()

        ep_inf_alt = GPy.inference.latent_function_inference.EP(ep_mode="alternated")
        ep_inf_nested = GPy.inference.latent_function_inference.EP(ep_mode="nested")
        ep_inf_frac = GPy.inference.latent_function_inference.EP(
            ep_mode="nested", eta=0.7
        )

        m1 = GPy.core.GP(
            self.X.copy(),
            self.Y_noisy.copy(),
            kernel=self.kernel1.copy(),
            likelihood=studentT.copy(),
            inference_method=laplace_inf,
        )
        # optimize
        m1[".*white"].constrain_fixed(1e-5)
        m1.randomize()

        m2 = GPy.core.GP(
            self.X.copy(),
            self.Y_noisy.copy(),
            kernel=self.kernel1.copy(),
            likelihood=studentT.copy(),
            inference_method=ep_inf_alt,
        )
        m2[".*white"].constrain_fixed(1e-5)
        # m2.constrain_bounded('.*t_scale2', 0.001, 10)
        m2.randomize()

        # m3 = GPy.core.GP(self.X, self.Y_noisy.copy(), kernel=self.kernel1, likelihood=studentT.copy(), inference_method=ep_inf_nested)
        # m3['.*white'].constrain_fixed(1e-5)
        # # m3.constrain_bounded('.*t_scale2', 0.001, 10)
        # m3.randomize()

        optimizer = "bfgs"
        m1.optimize(optimizer=optimizer, max_iters=400)
        m2.optimize(optimizer=optimizer, max_iters=400)
        # m3.optimize(optimizer=optimizer, max_iters=500)

        self.assertAlmostEqual(m1.log_likelihood(), m2.log_likelihood(), delta=200)

        # self.assertAlmostEqual(m1.log_likelihood(), m3.log_likelihood(), 3)

        preds_mean_lap, preds_var_lap = m1.predict(self.X)
        preds_mean_alt, preds_var_alt = m2.predict(self.X)
        # preds_mean_nested, preds_var_nested = m3.predict(self.X)
        rmse_lap = self.rmse(preds_mean_lap, self.Y)
        rmse_alt = self.rmse(preds_mean_alt, self.Y)
        # rmse_nested = self.rmse(preds_mean_nested, self.Y_noisy)

        if rmse_alt > rmse_lap:
            self.assertAlmostEqual(rmse_lap, rmse_alt, delta=1.5)
        # m3.optimize(optimizer=optimizer, max_iters=500)


if __name__ == "__main__":
    unittest.main()
