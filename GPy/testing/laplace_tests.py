import numpy as np
import unittest
import GPy
from GPy.models import GradientChecker
import functools
import inspect

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

def dparam_checkgrad(func, dfunc, params, args, constrain_positive=True, randomize=False, verbose=False):
    """
    checkgrad expects a f: R^N -> R^1 and df: R^N -> R^N
    However if we are holding other parameters fixed and moving something else
    We need to check the gradient of each of the fixed parameters
    (f and y for example) seperately.
    Whilst moving another parameter. otherwise f: gives back R^N and
    df: gives back R^NxM where M is
    The number of parameters and N is the number of data
    Need to take a slice out from f and a slice out of df
    """
    #print "\n{} likelihood: {} vs {}".format(func.im_self.__class__.__name__,
                                           #func.__name__, dfunc.__name__)
    partial_f = dparam_partial(func, *args)
    partial_df = dparam_partial(dfunc, *args)
    gradchecking = True
    for param in params:
        fnum = np.atleast_1d(partial_f(param)).shape[0]
        dfnum = np.atleast_1d(partial_df(param)).shape[0]
        for fixed_val in range(dfnum):
            #dlik and dlik_dvar gives back 1 value for each
            f_ind = min(fnum, fixed_val+1) - 1
            grad = GradientChecker(lambda x: np.atleast_1d(partial_f(x))[f_ind],
                                   lambda x : np.atleast_1d(partial_df(x))[fixed_val],
                                   param, 'p')
            if constrain_positive:
                grad.constrain_positive('p')
            if randomize:
                grad.randomize()
            print grad
            if verbose:
                grad.checkgrad(verbose=1)
            if not grad.checkgrad():
                gradchecking = False

    return gradchecking


class LaplaceTests(unittest.TestCase):
    def setUp(self):
        self.N = 1
        self.D = 5
        self.X = np.linspace(0, 1, self.N)[:, None]

        self.real_std = 0.2
        noise = np.random.randn(*self.X.shape)*self.real_std
        self.Y = np.sin(self.X*2*np.pi) + noise
        #self.Y = np.array([[1.0]])#np.sin(self.X*2*np.pi) + noise

        self.f = np.random.rand(self.N, 1)
        #self.f = np.array([[3.0]])#np.sin(self.X*2*np.pi) + noise

        self.var = np.random.rand(1)
        self.stu_t = GPy.likelihoods.functions.StudentT(deg_free=5, sigma2=self.var)
        self.gauss = GPy.likelihoods.functions.Gaussian(self.var, self.D, self.N)

    def tearDown(self):
        self.stu_t = None
        self.gauss = None
        self.Y = None
        self.f = None
        self.X = None

    def test_gaussian_dlik_df(self):
        print "\n{}".format(inspect.stack()[0][3])
        link = functools.partial(self.gauss.link_function, self.Y)
        dlik_df = functools.partial(self.gauss.dlik_df, self.Y)
        grad = GradientChecker(link, dlik_df, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d2lik_d2f(self):
        print "\n{}".format(inspect.stack()[0][3])
        dlik_df = functools.partial(self.gauss.dlik_df, self.Y)
        d2lik_d2f = functools.partial(self.gauss.d2lik_d2f, self.Y)
        grad = GradientChecker(dlik_df, d2lik_d2f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d3lik_d3f(self):
        print "\n{}".format(inspect.stack()[0][3])
        d2lik_d2f = functools.partial(self.gauss.d2lik_d2f, self.Y)
        d3lik_d3f = functools.partial(self.gauss.d3lik_d3f, self.Y)
        grad = GradientChecker(d2lik_d2f, d3lik_d3f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_dlik_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.link_function, self.gauss.dlik_dvar,
                    [self.var], args=(self.Y, self.f), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    def test_gaussian_dlik_df_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.dlik_df, self.gauss.dlik_df_dvar,
                    [self.var], args=(self.Y.copy(), self.f.copy()), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    def test_gaussian_d2lik_d2f_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.d2lik_d2f, self.gauss.d2lik_d2f_dvar,
                    [self.var], args=(self.Y, self.f), constrain_positive=True,
                    randomize=True, verbose=True)
                )

    def test_studentt_dlik_df(self):
        print "\n{}".format(inspect.stack()[0][3])
        link = functools.partial(self.stu_t.link_function, self.Y)
        dlik_df = functools.partial(self.stu_t.dlik_df, self.Y)
        grad = GradientChecker(link, dlik_df, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_studentt_d2lik_d2f(self):
        print "\n{}".format(inspect.stack()[0][3])
        dlik_df = functools.partial(self.stu_t.dlik_df, self.Y)
        d2lik_d2f = functools.partial(self.stu_t.d2lik_d2f, self.Y)
        grad = GradientChecker(dlik_df, d2lik_d2f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_studentt_d3lik_d3f(self):
        print "\n{}".format(inspect.stack()[0][3])
        d2lik_d2f = functools.partial(self.stu_t.d2lik_d2f, self.Y)
        d3lik_d3f = functools.partial(self.stu_t.d3lik_d3f, self.Y)
        grad = GradientChecker(d2lik_d2f, d3lik_d3f, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)

    def test_studentt_dlik_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.link_function, self.stu_t.dlik_dvar,
                    [self.var], args=(self.Y.copy(), self.f.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

    def test_studentt_dlik_df_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.dlik_df, self.stu_t.dlik_df_dvar,
                    [self.var], args=(self.Y.copy(), self.f.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

    def test_studentt_d2lik_d2f_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.d2lik_d2f, self.stu_t.d2lik_d2f_dvar,
                    [self.var], args=(self.Y.copy(), self.f.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

if __name__ == "__main__":
    print "Running unit tests"
    unittest.main()
