import numpy as np
import unittest
import GPy
from GPy.models import GradientChecker
import functools

def dparam_partial(inst_func, *args):
    """
    If we have a instance method that needs to be called but that doesn't
    take the parameter we wish to change to checkgrad, then this function
    will change the variable using set params.

    inst_func: should be a instance function of an object that we would like
                to change
    param: the param that will be given to set_params
    args: anything else that needs to be given to the function (for example
          the f or Y that are being used in the function whilst we tweak the
          param
    """
    def param_func(param, inst_func, args):
        inst_func.im_self._set_params(param)
        return inst_func(*args)
    return functools.partial(param_func, inst_func=inst_func, args=args)

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
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d2lik_d2f(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        dlik_df = functools.partial(gauss.dlik_df, self.Y)
        d2lik_d2f = functools.partial(gauss.d2lik_d2f, self.Y)
        grad = GradientChecker(dlik_df, d2lik_d2f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d3lik_d3f(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)
        d2lik_d2f = functools.partial(gauss.d2lik_d2f, self.Y)
        d3lik_d3f = functools.partial(gauss.d3lik_d3f, self.Y)
        grad = GradientChecker(d2lik_d2f, d3lik_d3f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_dlik_dvar(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)

        link = dparam_partial(gauss.link_function, self.Y, self.f)
        dlik_dvar = dparam_partial(gauss.dlik_dvar, self.Y, self.f)
        grad = GradientChecker(link, dlik_dvar, var, 'v')
        grad.constrain_positive('v')
        grad.randomize()
        grad.checkgrad(verbose=1)
        #self.assertTrue(grad.checkgrad())

    def test_gaussian_dlik_df_dvar(self):
        var = 0.1
        gauss = GPy.likelihoods.functions.Gaussian(var, self.D, self.N)

        dlik_df = dparam_partial(gauss.dlik_df, self.Y, self.f)
        dlik_df_dvar = dparam_partial(gauss.dlik_df_dvar, self.Y, self.f)
        grad = GradientChecker(dlik_df, dlik_df_dvar, var, 'v')
        grad.constrain_positive('v')
        grad.randomize()
        grad.checkgrad(verbose=1)
        #self.assertTrue(grad.checkgrad())

    def test_studentt_dlik_dvar(self):
        var = 0.1
        stu_t = GPy.likelihoods.functions.StudentT(deg_free=5, sigma2=var)

        link = dparam_partial(stu_t.link_function, self.Y, self.f)
        dlik_dvar = dparam_partial(stu_t.dlik_dvar, self.Y, self.f)
        grad = GradientChecker(link, dlik_dvar, var, 'v')
        grad.constrain_positive('v')
        grad.randomize()
        grad.checkgrad(verbose=1)
        #self.assertTrue(grad.checkgrad())

if __name__ == "__main__":
    print "Running unit tests"
    unittest.main()
