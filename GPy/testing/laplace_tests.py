import numpy as np
import unittest
import GPy
from GPy.models import GradientChecker
import functools
import inspect
from GPy.likelihoods.noise_models import gp_transformations

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
            print "fnum: {} dfnum: {} f_ind: {} fixed_val: {}".format(fnum, dfnum, f_ind, fixed_val)
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
        self.N = 5
        self.D = 3
        self.X = np.random.rand(self.N, self.D)*10

        self.real_std = 0.1
        noise = np.random.randn(*self.X[:, 0].shape)*self.real_std
        self.Y = (np.sin(self.X[:, 0]*2*np.pi) + noise)[:, None]
        self.f = np.random.rand(self.N, 1)

        self.var = 0.2

        self.var = np.random.rand(1)
        self.stu_t = GPy.likelihoods.student_t(deg_free=5, sigma2=self.var)
        self.gauss = GPy.likelihoods.gaussian(gp_transformations.Log(), variance=self.var, D=self.D, N=self.N)

        #Make a bigger step as lower bound can be quite curved
        self.step = 1e-3

    def tearDown(self):
        self.stu_t = None
        self.gauss = None
        self.Y = None
        self.f = None
        self.X = None

    def test_mass_logpdf(self):
        print "\n{}".format(inspect.stack()[0][3])
        np.testing.assert_almost_equal(
                               np.log(self.gauss.pdf(self.f.copy(), self.Y.copy())),
                               self.gauss.logpdf(self.f.copy(), self.Y.copy()))


    """ dGauss_df's """
    def test_gaussian_dlogpdf_df(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        logpdf = functools.partial(self.gauss.logpdf, y=self.Y)
        dlogpdf_df = functools.partial(self.gauss.dlogpdf_df, y=self.Y)
        grad = GradientChecker(logpdf, dlogpdf_df, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d2logpdf_df2(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        dlogpdf_df = functools.partial(self.gauss.dlogpdf_df, y=self.Y)
        d2logpdf_df2 = functools.partial(self.gauss.d2logpdf_df2, y=self.Y)
        grad = GradientChecker(dlogpdf_df, d2logpdf_df2, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d3logpdf_df3(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        d2logpdf_df2 = functools.partial(self.gauss.d2logpdf_df2, y=self.Y)
        d3logpdf_df3 = functools.partial(self.gauss.d3logpdf_df3, y=self.Y)
        grad = GradientChecker(d2logpdf_df2, d3logpdf_df3, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_dlogpdf_df_dvar(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.dlogpdf_df, self.gauss.dlogpdf_df_dtheta,
                    [self.var], args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    def test_gaussian_d2logpdf2_df2_dvar(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.d2logpdf_df2, self.gauss.d2logpdf_df2_dtheta,
                    [self.var], args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )


    """ dGauss_dlink's """
    def test_gaussian_dlogpdf_dlink(self):
        print "\n{}".format(inspect.stack()[0][3])
        logpdf = functools.partial(self.gauss.logpdf_link, y=self.Y)
        dlogpdf_dlink = functools.partial(self.gauss.dlogpdf_dlink, y=self.Y)
        grad = GradientChecker(logpdf, dlogpdf_dlink, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d2logpdf_dlink2(self):
        print "\n{}".format(inspect.stack()[0][3])
        dlogpdf_dlink = functools.partial(self.gauss.dlogpdf_dlink, y=self.Y)
        d2logpdf_dlink2 = functools.partial(self.gauss.d2logpdf_dlink2, y=self.Y)
        grad = GradientChecker(dlogpdf_dlink, d2logpdf_dlink2, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_d3logpdf_dlink3(self):
        print "\n{}".format(inspect.stack()[0][3])
        d2logpdf_dlink2 = functools.partial(self.gauss.d2logpdf_dlink2, y=self.Y)
        d3logpdf_dlink3 = functools.partial(self.gauss.d3logpdf_dlink3, y=self.Y)
        grad = GradientChecker(d2logpdf_dlink2, d3logpdf_dlink3, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_gaussian_dlogpdf_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.logpdf, self.gauss.dlogpdf_dtheta,
                    [self.var], args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    def test_gaussian_dlogpdf_dlink_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.dlogpdf_dlink, self.gauss.dlogpdf_dlink_dtheta,
                    [self.var], args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    def test_gaussian_d2logpdf2_dlink2_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.gauss.d2logpdf_dlink2, self.gauss.d2logpdf_dlink2_dtheta,
                    [self.var], args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )


    """ Gradchecker fault """
    @unittest.expectedFailure
    def test_gaussian_d2logpdf_df2_2(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.Y = None
        self.gauss = None

        self.N = 2
        self.D = 1
        self.X = np.linspace(0, self.D, self.N)[:, None]
        self.real_std = 0.2
        noise = np.random.randn(*self.X.shape)*self.real_std
        self.Y = np.sin(self.X*2*np.pi) + noise
        self.f = np.random.rand(self.N, 1)
        self.gauss = GPy.likelihoods.gaussian(variance=self.var, D=self.D, N=self.N)

        dlogpdf_df = functools.partial(self.gauss.dlogpdf_df, y=self.Y)
        d2logpdf_df2 = functools.partial(self.gauss.d2logpdf_df2, y=self.Y)
        grad = GradientChecker(dlogpdf_df, d2logpdf_df2, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    """ dStudentT_df's """
    def test_studentt_dlogpdf_df(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        link = functools.partial(self.stu_t.logpdf, y=self.Y)
        dlogpdf_df = functools.partial(self.stu_t.dlogpdf_df, y=self.Y)
        grad = GradientChecker(link, dlogpdf_df, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_studentt_d2logpdf_df2(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        dlogpdf_df = functools.partial(self.stu_t.dlogpdf_df, y=self.Y)
        d2logpdf_df2 = functools.partial(self.stu_t.d2logpdf_df2, y=self.Y)
        grad = GradientChecker(dlogpdf_df, d2logpdf_df2, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_studentt_d3lik_d3f(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        d2logpdf_df2 = functools.partial(self.stu_t.d2logpdf_df2, y=self.Y)
        d3logpdf_df3 = functools.partial(self.stu_t.d3logpdf_df3, y=self.Y)
        grad = GradientChecker(d2logpdf_df2, d3logpdf_df3, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_studentt_dlogpdf_df_dvar(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.dlogpdf_df, self.stu_t.dlogpdf_df_dtheta,
                    [self.var], args=(self.f.copy(), self.Y.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

    def test_studentt_d2logpdf_df2_dvar(self):
        #FIXME: Needs non-identity Link function
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.d2logpdf_df2, self.stu_t.d2logpdf_df2_dtheta,
                    [self.var], args=(self.f.copy(), self.Y.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

    """ dStudentT_dlink's """
    def test_studentt_dlogpdf_dlink(self):
        print "\n{}".format(inspect.stack()[0][3])
        logpdf = functools.partial(self.stu_t.logpdf, y=self.Y)
        dlogpdf_dlink = functools.partial(self.stu_t.dlogpdf_dlink, y=self.Y)
        grad = GradientChecker(logpdf, dlogpdf_dlink, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_studentt_d2logpdf_dlink2(self):
        print "\n{}".format(inspect.stack()[0][3])
        dlogpdf_dlink = functools.partial(self.stu_t.dlogpdf_dlink, y=self.Y)
        d2logpdf_dlink2 = functools.partial(self.stu_t.d2logpdf_dlink2, y=self.Y)
        grad = GradientChecker(dlogpdf_dlink, d2logpdf_dlink2, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_studentt_d3logpdf_dlink3(self):
        print "\n{}".format(inspect.stack()[0][3])
        d2logpdf_dlink2 = functools.partial(self.stu_t.d2logpdf_dlink2, y=self.Y)
        d3logpdf_dlink3 = functools.partial(self.stu_t.d3logpdf_dlink3, y=self.Y)
        grad = GradientChecker(d2logpdf_dlink2, d3logpdf_dlink3, self.f.copy(), 'f')
        grad.randomize()
        grad.checkgrad(verbose=1)
        self.assertTrue(grad.checkgrad())

    def test_studentt_dlogpdf_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.logpdf, self.stu_t.dlogpdf_dtheta,
                    [self.var], args=(self.f.copy(), self.Y.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

    def test_studentt_dlogpdf_dlink_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.dlogpdf_dlink, self.stu_t.dlogpdf_dlink_dtheta,
                    [self.var], args=(self.f.copy(), self.Y.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )

    def test_studentt_d2logpdf_dlink2_dvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.assertTrue(
                dparam_checkgrad(self.stu_t.d2logpdf_dlink2, self.stu_t.d2logpdf_dlink2_dtheta,
                    [self.var], args=(self.f.copy(), self.Y.copy()),
                    constrain_positive=True, randomize=True, verbose=True)
                )


    """ Grad check whole models (grad checking Laplace not just noise models """
    def test_gauss_rbf(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.Y = self.Y/self.Y.max()
        kernel = GPy.kern.rbf(self.X.shape[1]) + GPy.kern.white(self.X.shape[1])
        gauss_laplace = GPy.likelihoods.Laplace(self.Y.copy(), self.gauss)
        m = GPy.models.GPRegression(self.X, self.Y.copy(), kernel, likelihood=gauss_laplace)
        m.ensure_default_constraints()
        m.randomize()
        m.checkgrad(verbose=1, step=self.step)
        self.assertTrue(m.checkgrad(step=self.step))

    def test_studentt_approx_gauss_rbf(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.Y = self.Y/self.Y.max()
        self.stu_t = GPy.likelihoods.student_t(deg_free=1000, sigma2=self.var)
        kernel = GPy.kern.rbf(self.X.shape[1]) + GPy.kern.white(self.X.shape[1])
        stu_t_laplace = GPy.likelihoods.Laplace(self.Y.copy(), self.stu_t)
        m = GPy.models.GPRegression(self.X, self.Y.copy(), kernel, likelihood=stu_t_laplace)
        m.ensure_default_constraints()
        m.constrain_positive('t_noise')
        m.randomize()
        m.checkgrad(verbose=1, step=self.step)
        print m
        self.assertTrue(m.checkgrad(step=self.step))

    def test_studentt_rbf(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.Y = self.Y/self.Y.max()
        white_var = 0.001
        kernel = GPy.kern.rbf(self.X.shape[1]) + GPy.kern.white(self.X.shape[1])
        stu_t_laplace = GPy.likelihoods.Laplace(self.Y.copy(), self.stu_t)
        m = GPy.models.GPRegression(self.X, self.Y.copy(), kernel, likelihood=stu_t_laplace)
        m.ensure_default_constraints()
        m.constrain_positive('t_noise')
        m.constrain_fixed('white', white_var)
        m.randomize()
        m.checkgrad(verbose=1, step=self.step)
        print m
        self.assertTrue(m.checkgrad(step=self.step))

    """ With small variances its likely the implicit part isn't perfectly correct? """
    @unittest.expectedFailure
    def test_studentt_rbf_smallvar(self):
        print "\n{}".format(inspect.stack()[0][3])
        self.Y = self.Y/self.Y.max()
        white_var = 0.001
        kernel = GPy.kern.rbf(self.X.shape[1]) + GPy.kern.white(self.X.shape[1])
        stu_t_laplace = GPy.likelihoods.Laplace(self.Y.copy(), self.stu_t)
        m = GPy.models.GPRegression(self.X, self.Y.copy(), kernel, likelihood=stu_t_laplace)
        m.ensure_default_constraints()
        m.constrain_positive('t_noise')
        m.constrain_fixed('white', white_var)
        m['t_noise'] = 0.01
        m.randomize()
        m.checkgrad(verbose=1)
        print m
        self.assertTrue(m.checkgrad(step=self.step))

if __name__ == "__main__":
    print "Running unit tests"
    unittest.main()
