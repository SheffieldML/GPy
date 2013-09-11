import numpy as np
import unittest
import GPy
from GPy.models import GradientChecker
import functools

class LaplaceTests(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.D = 1
        self.X = np.linspace(0, 1, self.N)[:, None]

        self.real_std = 0.2
        noise = np.random.randn(*self.X.shape)*self.real_std
        self.Y = np.sin(self.X*2*np.pi) + noise

        self.f = np.random.rand(self.N, 1)

    def test_gaussian_dlik_df(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        link = functools.partial(gauss.link_function, self.Y)
        dlik_df = functools.partial(gauss.dlik_df, self.Y)
        grad = GradientChecker(link, dlik_df, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_gaussian_d2lik_d2f(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        dlik_df = functools.partial(gauss.dlik_df, self.Y)
        d2lik_d2f = functools.partial(gauss.d2lik_d2f, self.Y)
        grad = GradientChecker(dlik_df, d2lik_d2f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_gaussian_d3lik_d3f(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        d2lik_d2f = functools.partial(gauss.d2lik_d2f, self.Y)
        d3lik_d3f = functools.partial(gauss.d3lik_d3f, self.Y)
        grad = GradientChecker(d2lik_d2f, d3lik_d3f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_gaussian_dlik_dvar(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        #Since the function we are checking does not directly accept the variable we wish to tweak
        #We make function which makes the change (set params) then calls the function
        def p_link_var(var, likelihood, f, Y):
            likelihood._set_params(var)
            return likelihood.link_function(f, Y)

        def p_dlik_dvar(var, likelihood, f, Y):
            likelihood._set_params(var)
            return likelihood.dlik_dvar(f, Y)

        link = functools.partial(p_link_var, likelihood=gauss, f=self.f, Y=self.Y)
        dlik_dvar = functools.partial(p_dlik_dvar, likelihood=gauss, f=self.f, Y=self.Y)
        grad = GradientChecker(link, dlik_dvar, var, 'v')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_gaussian_dlik_df_dvar(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        def p_dlik_df(var, likelihood, f, Y):
            likelihood._set_params(var)
            return likelihood.dlik_df(f, Y)

        def p_dlik_df_dstd(var, likelihood, f, Y):
            likelihood._set_params(var)
            return likelihood.dlik_df_dvar(f, Y)

        dlik_df = functools.partial(p_dlik_df, likelihood=gauss, f=self.f, Y=self.Y)
        dlik_df_dstd = functools.partial(p_dlik_df_dstd, likelihood=gauss, f=self.f, Y=self.Y)
        grad = GradientChecker(dlik_df, dlik_df_dstd, var, 'v')
        grad.randomize()
        grad.checkgrad(verbose=1)

if __name__ == "__main__":
    print "Running unit tests"
    unittest.main()
