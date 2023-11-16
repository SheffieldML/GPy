import numpy as np
import scipy
from scipy.special import cbrt
from GPy.models import GradientChecker
import random

_lim_val = np.finfo(np.float64).max
_lim_val_exp = np.log(_lim_val)
_lim_val_square = np.sqrt(_lim_val)
_lim_val_cube = cbrt(_lim_val)
from GPy.likelihoods.link_functions import (
    Identity,
    Probit,
    Cloglog,
    Log,
    Log_ex_1,
    Reciprocal,
    Heaviside,
    ScaledProbit,
)


class TestLinkFunction:
    def setup(self):
        self.small_f = np.array([[-1e-4]])
        self.zero_f = np.array([[1e-4]])
        self.mid_f = np.array([[5.0]])
        self.large_f = np.array([[1e4]])
        self.f_lower_lim = np.array(-np.inf)
        self.f_upper_lim = np.array(np.inf)

    def check_gradient(self, link_func, lim_of_inf, test_lim=False):
        grad = GradientChecker(link_func.transf, link_func.dtransf_df, x0=self.mid_f)
        assert grad.checkgrad(verbose=True)
        grad2 = GradientChecker(
            link_func.dtransf_df, link_func.d2transf_df2, x0=self.mid_f
        )
        assert grad2.checkgrad(verbose=True)
        grad3 = GradientChecker(
            link_func.d2transf_df2, link_func.d3transf_df3, x0=self.mid_f
        )
        assert grad3.checkgrad(verbose=True)

        grad = GradientChecker(link_func.transf, link_func.dtransf_df, x0=self.small_f)
        assert grad.checkgrad(verbose=True)
        grad2 = GradientChecker(
            link_func.dtransf_df, link_func.d2transf_df2, x0=self.small_f
        )
        assert grad2.checkgrad(verbose=True)
        grad3 = GradientChecker(
            link_func.d2transf_df2, link_func.d3transf_df3, x0=self.small_f
        )
        assert grad3.checkgrad(verbose=True)

        grad = GradientChecker(link_func.transf, link_func.dtransf_df, x0=self.zero_f)
        assert grad.checkgrad(verbose=True)
        grad2 = GradientChecker(
            link_func.dtransf_df, link_func.d2transf_df2, x0=self.zero_f
        )
        assert grad2.checkgrad(verbose=True)
        grad3 = GradientChecker(
            link_func.d2transf_df2, link_func.d3transf_df3, x0=self.zero_f
        )
        assert grad3.checkgrad(verbose=True)

        # Do a limit test if the large f value is too large
        large_f = np.clip(self.large_f, -np.inf, lim_of_inf - 1e-3)
        grad = GradientChecker(link_func.transf, link_func.dtransf_df, x0=large_f)
        assert grad.checkgrad(verbose=True)
        grad2 = GradientChecker(
            link_func.dtransf_df, link_func.d2transf_df2, x0=large_f
        )
        assert grad2.checkgrad(verbose=True)
        grad3 = GradientChecker(
            link_func.d2transf_df2, link_func.d3transf_df3, x0=large_f
        )
        assert grad3.checkgrad(verbose=True)

        if test_lim:
            print("Testing limits")
            # Remove some otherwise we are too close to the limit for gradcheck to work effectively
            lim_of_inf = lim_of_inf - 1e-4
            grad = GradientChecker(
                link_func.transf, link_func.dtransf_df, x0=lim_of_inf
            )
            assert grad.checkgrad(verbose=True)
            grad2 = GradientChecker(
                link_func.dtransf_df, link_func.d2transf_df2, x0=lim_of_inf
            )
            assert grad2.checkgrad(verbose=True)
            grad3 = GradientChecker(
                link_func.d2transf_df2, link_func.d3transf_df3, x0=lim_of_inf
            )
            assert grad3.checkgrad(verbose=True)

    def check_overflow(self, link_func, lim_of_inf):
        # Check that it does something sensible beyond this limit,
        # note this is not checking the value is correct, just that it isn't nan
        beyond_lim_of_inf = lim_of_inf + 100.0
        assert not np.isinf(link_func.transf(beyond_lim_of_inf))
        assert not np.isinf(link_func.dtransf_df(beyond_lim_of_inf))
        assert not np.isinf(link_func.d2transf_df2(beyond_lim_of_inf))

        assert not np.isnan(link_func.transf(beyond_lim_of_inf))
        assert not np.isnan(link_func.dtransf_df(beyond_lim_of_inf))
        assert not np.isnan(link_func.d2transf_df2(beyond_lim_of_inf))

    def test_log_overflow(self):
        self.setup()

        link = Log()
        lim_of_inf = _lim_val_exp

        np.testing.assert_almost_equal(np.exp(self.mid_f), link.transf(self.mid_f))
        assert np.isinf(np.exp(np.log(self.f_upper_lim)))
        # Check the clipping works
        np.testing.assert_almost_equal(link.transf(self.f_lower_lim), 0, decimal=5)
        assert np.isfinite(link.transf(self.f_upper_lim))
        self.check_overflow(link, lim_of_inf)

        # Check that it would otherwise fail
        beyond_lim_of_inf = lim_of_inf + 10.0
        old_err_state = np.seterr(over="ignore")
        assert np.isinf(np.exp(beyond_lim_of_inf))
        np.seterr(**old_err_state)

    def test_log_ex_1_overflow(self):
        self.setup()

        link = Log_ex_1()
        lim_of_inf = _lim_val_exp

        np.testing.assert_almost_equal(
            scipy.special.log1p(np.exp(self.mid_f)), link.transf(self.mid_f)
        )
        assert np.isinf(scipy.special.log1p(np.exp(np.log(self.f_upper_lim))))
        # Check the clipping works
        np.testing.assert_almost_equal(link.transf(self.f_lower_lim), 0, decimal=5)
        # Need to look at most significant figures here rather than the decimals
        np.testing.assert_approx_equal(
            link.transf(self.f_upper_lim), scipy.special.log1p(_lim_val), significant=5
        )
        self.check_overflow(link, lim_of_inf)

        # Check that it would otherwise fail
        beyond_lim_of_inf = lim_of_inf + 10.0
        old_err_state = np.seterr(over="ignore")
        assert np.isinf(scipy.special.log1p(np.exp(beyond_lim_of_inf)))
        np.seterr(**old_err_state)

    def test_log_gradients(self):
        # transf dtransf_df d2transf_df2 d3transf_df3
        self.setup()

        link = Log()
        lim_of_inf = _lim_val_exp
        self.check_gradient(link, lim_of_inf, test_lim=True)

    def test_identity_gradients(self):
        self.setup()
        link = Identity()
        lim_of_inf = _lim_val
        # FIXME: Should be able to think of a way to test the limits of this
        self.check_gradient(link, lim_of_inf, test_lim=False)

    def test_probit_gradients(self):
        self.setup()
        link = Probit()
        lim_of_inf = _lim_val
        self.check_gradient(link, lim_of_inf, test_lim=True)

    def test_scaledprobit_gradients(self):
        self.setup()
        link = ScaledProbit(nu=random.random())
        lim_of_inf = _lim_val
        self.check_gradient(link, lim_of_inf, test_lim=True)

    def test_Cloglog_gradients(self):
        self.setup()
        link = Cloglog()
        lim_of_inf = _lim_val_exp
        self.check_gradient(link, lim_of_inf, test_lim=True)

    def test_Log_ex_1_gradients(self):
        self.setup()
        link = Log_ex_1()
        lim_of_inf = _lim_val_exp
        self.check_gradient(link, lim_of_inf, test_lim=True)
        self.check_overflow(link, lim_of_inf)

    def test_reciprocal_gradients(self):
        self.setup()
        link = Reciprocal()
        lim_of_inf = _lim_val
        # Does not work with much smaller values, and values closer to zero than 1e-5
        self.check_gradient(link, lim_of_inf, test_lim=True)
