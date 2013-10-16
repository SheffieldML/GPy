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


from nose.tools import with_setup
class TestNoiseModels(object):
    """
    Generic model checker
    """
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

        #Make a bigger step as lower bound can be quite curved
        self.step = 1e-3

    def tearDown(self):
        self.Y = None
        self.f = None
        self.X = None

    def test_noise_models(self):
        self.setUp()
        """
        Dictionary where we nest models we would like to check
            Name: {
                "model": model_instance,
                "grad_params": {
                    "names": [names_of_params_we_want, to_grad_check],
                    "vals": [values_of_params, to_start_at],
                    "constrain_positive": [boolean_values, of_whether_to_constrain]
                    },
                "laplace": boolean_of_whether_model_should_work_for_laplace
                }
        """
        noise_models = {"Student_t_default": {
                            "model": GPy.likelihoods.student_t(deg_free=5, sigma2=self.var),
                            "grad_params": {
                                "names": ["t_noise"],
                                "vals": [self.var],
                                "constrain_positive": [True]
                                },
                            "laplace": True
                            },
                        "Student_t_small_var": {
                            "model": GPy.likelihoods.student_t(deg_free=5, sigma2=self.var),
                            "grad_params": {
                                "names": ["t_noise"],
                                "vals": [0.01],
                                "constrain_positive": [True]
                                },
                            "laplace": True
                            },
                        "Student_t_approx_gauss": {
                            "model": GPy.likelihoods.student_t(deg_free=1000, sigma2=self.var),
                            "grad_params": {
                                "names": ["t_noise"],
                                "vals": [self.var],
                                "constrain_positive": [True]
                                },
                            "laplace": True
                            },
                        "Student_t_log": {
                            "model": GPy.likelihoods.student_t(gp_link=gp_transformations.Log(), deg_free=5, sigma2=self.var),
                            "grad_params": {
                                "names": ["t_noise"],
                                "vals": [self.var],
                                "constrain_positive": [True]
                                },
                            "laplace": True
                            },
                        "Gaussian_default": {
                            "model": GPy.likelihoods.gaussian(variance=self.var, D=self.D, N=self.N),
                            "grad_params": {
                                "names": ["noise_model_variance"],
                                "vals": [self.var],
                                "constrain_positive": [True]
                                },
                            "laplace": True
                            },
                        "Gaussian_log": {
                            "model": GPy.likelihoods.gaussian(gp_link=gp_transformations.Log(), variance=self.var, D=self.D, N=self.N),
                            "grad_params": {
                                "names": ["noise_model_variance"],
                                "vals": [self.var],
                                "constrain_positive": [True]
                                },
                            "laplace": True
                            }
                        }

        for name, attributes in noise_models.iteritems():
            model = attributes["model"]
            params = attributes["grad_params"]
            param_vals = params["vals"]
            param_names= params["names"]
            constrain_positive = params["constrain_positive"]
            laplace = attributes["laplace"]

            if len(param_vals) > 1:
                raise NotImplementedError("Cannot support multiple params in likelihood yet!")

            #Required by all
            #Normal derivatives
            yield self.t_logpdf, model
            yield self.t_dlogpdf_df, model
            yield self.t_d2logpdf_df2, model
            #Link derivatives
            yield self.t_dlogpdf_dlink, model
            yield self.t_d2logpdf_dlink2, model
            yield self.t_d3logpdf_dlink3, model
            if laplace:
                #Laplace only derivatives
                yield self.t_d3logpdf_df3, model
                #Params
                yield self.t_dlogpdf_dparams, model, param_vals
                yield self.t_dlogpdf_df_dparams, model, param_vals
                yield self.t_d2logpdf2_df2_dparams, model, param_vals
                #Link params
                yield self.t_dlogpdf_link_dparams, model, param_vals
                yield self.t_dlogpdf_dlink_dparams, model, param_vals
                yield self.t_d2logpdf2_dlink2_dparams, model, param_vals

                #laplace likelihood gradcheck
                yield self.t_laplace_fit_rbf_white, model, param_vals, param_names, constrain_positive

        self.tearDown()

    #############
    # dpdf_df's #
    #############
    @with_setup(setUp, tearDown)
    def t_logpdf(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        np.testing.assert_almost_equal(
                               np.log(model.pdf(self.f.copy(), self.Y.copy())),
                               model.logpdf(self.f.copy(), self.Y.copy()))

    @with_setup(setUp, tearDown)
    def t_dlogpdf_df(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        self.description = "\n{}".format(inspect.stack()[0][3])
        logpdf = functools.partial(model.logpdf, y=self.Y)
        dlogpdf_df = functools.partial(model.dlogpdf_df, y=self.Y)
        grad = GradientChecker(logpdf, dlogpdf_df, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    @with_setup(setUp, tearDown)
    def t_d2logpdf_df2(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        dlogpdf_df = functools.partial(model.dlogpdf_df, y=self.Y)
        d2logpdf_df2 = functools.partial(model.d2logpdf_df2, y=self.Y)
        grad = GradientChecker(dlogpdf_df, d2logpdf_df2, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    @with_setup(setUp, tearDown)
    def t_d3logpdf_df3(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        d2logpdf_df2 = functools.partial(model.d2logpdf_df2, y=self.Y)
        d3logpdf_df3 = functools.partial(model.d3logpdf_df3, y=self.Y)
        grad = GradientChecker(d2logpdf_df2, d3logpdf_df3, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    ##############
    # df_dparams #
    ##############
    @with_setup(setUp, tearDown)
    def t_dlogpdf_dparams(self, model, params):
        print "\n{}".format(inspect.stack()[0][3])
        assert (
                dparam_checkgrad(model.logpdf, model.dlogpdf_dtheta,
                    params, args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    @with_setup(setUp, tearDown)
    def t_dlogpdf_df_dparams(self, model, params):
        print "\n{}".format(inspect.stack()[0][3])
        assert (
                dparam_checkgrad(model.dlogpdf_df, model.dlogpdf_df_dtheta,
                    params, args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    @with_setup(setUp, tearDown)
    def t_d2logpdf2_df2_dparams(self, model, params):
        print "\n{}".format(inspect.stack()[0][3])
        assert (
                dparam_checkgrad(model.d2logpdf_df2, model.d2logpdf_df2_dtheta,
                    params, args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    ################
    # dpdf_dlink's #
    ################
    @with_setup(setUp, tearDown)
    def t_dlogpdf_dlink(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        logpdf = functools.partial(model.logpdf_link, y=self.Y)
        dlogpdf_dlink = functools.partial(model.dlogpdf_dlink, y=self.Y)
        grad = GradientChecker(logpdf, dlogpdf_dlink, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    @with_setup(setUp, tearDown)
    def t_d2logpdf_dlink2(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        dlogpdf_dlink = functools.partial(model.dlogpdf_dlink, y=self.Y)
        d2logpdf_dlink2 = functools.partial(model.d2logpdf_dlink2, y=self.Y)
        grad = GradientChecker(dlogpdf_dlink, d2logpdf_dlink2, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    @with_setup(setUp, tearDown)
    def t_d3logpdf_dlink3(self, model):
        print "\n{}".format(inspect.stack()[0][3])
        d2logpdf_dlink2 = functools.partial(model.d2logpdf_dlink2, y=self.Y)
        d3logpdf_dlink3 = functools.partial(model.d3logpdf_dlink3, y=self.Y)
        grad = GradientChecker(d2logpdf_dlink2, d3logpdf_dlink3, self.f.copy(), 'g')
        grad.randomize()
        grad.checkgrad(verbose=1)
        assert grad.checkgrad()

    #################
    # dlink_dparams #
    #################
    @with_setup(setUp, tearDown)
    def t_dlogpdf_link_dparams(self, model, params):
        print "\n{}".format(inspect.stack()[0][3])
        assert (
                dparam_checkgrad(model.logpdf_link, model.dlogpdf_link_dtheta,
                    params, args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    @with_setup(setUp, tearDown)
    def t_dlogpdf_dlink_dparams(self, model, params):
        print "\n{}".format(inspect.stack()[0][3])
        assert (
                dparam_checkgrad(model.dlogpdf_dlink, model.dlogpdf_dlink_dtheta,
                    params, args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    @with_setup(setUp, tearDown)
    def t_d2logpdf2_dlink2_dparams(self, model, params):
        print "\n{}".format(inspect.stack()[0][3])
        assert (
                dparam_checkgrad(model.d2logpdf_dlink2, model.d2logpdf_dlink2_dtheta,
                    params, args=(self.f, self.Y), constrain_positive=True,
                    randomize=False, verbose=True)
                )

    ################
    # laplace test #
    ################
    @with_setup(setUp, tearDown)
    def t_laplace_fit_rbf_white(self, model, param_vals, param_names, constrain_positive):
        print "\n{}".format(inspect.stack()[0][3])
        self.Y = self.Y/self.Y.max()
        white_var = 0.001
        kernel = GPy.kern.rbf(self.X.shape[1]) + GPy.kern.white(self.X.shape[1])
        laplace_likelihood = GPy.likelihoods.Laplace(self.Y.copy(), model)
        m = GPy.models.GPRegression(self.X, self.Y.copy(), kernel, likelihood=laplace_likelihood)
        m.ensure_default_constraints()
        m.constrain_fixed('white', white_var)

        for param_num in range(len(param_names)):
            name = param_names[param_num]
            if constrain_positive[param_num]:
                m.constrain_positive(name)
            m[name] = param_vals[param_num]

        m.randomize()
        m.checkgrad(verbose=1, step=self.step)
        print m
        assert m.checkgrad(step=self.step)


class LaplaceTests(unittest.TestCase):
    """
    Specific likelihood tests, not general enough for the above tests
    """

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

if __name__ == "__main__":
    print "Running unit tests"
    unittest.main()
