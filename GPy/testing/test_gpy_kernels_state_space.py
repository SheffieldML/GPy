# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Testing state space related functions.
"""
import unittest
import numpy as np
import GPy
import GPy.models.state_space_model as SS_model
from .state_space_main_tests import (
    generate_x_points,
    generate_sine_data,
    generate_linear_data,
    generate_brownian_data,
    generate_linear_plus_sin,
)

# from state_space_main_tests import generate_x_points, generate_sine_data, \
#    generate_linear_data, generate_brownian_data, generate_linear_plus_sin


class TestStateSpaceKernels:
    def run_for_model(
        self,
        X,
        Y,
        ss_kernel,
        kalman_filter_type="regular",
        use_cython=False,
        check_gradients=True,
        optimize=True,
        optimize_max_iters=250,
        predict_X=None,
        compare_with_GP=True,
        gp_kernel=None,
        mean_compare_decimal=10,
        var_compare_decimal=7,
    ):
        m1 = SS_model.StateSpace(
            X,
            Y,
            ss_kernel,
            kalman_filter_type=kalman_filter_type,
            use_cython=use_cython,
        )

        m1.likelihood[:] = Y.var() / 100.0

        if check_gradients:
            assert m1.checkgrad()

        if 1:  # optimize:
            m1.optimize(optimizer="lbfgsb", max_iters=1)

        if compare_with_GP and (predict_X is None):
            predict_X = X

        assert compare_with_GP
        if compare_with_GP:
            m2 = GPy.models.GPRegression(X, Y, gp_kernel)

            m2[:] = m1[:]

            if predict_X is not None:
                x_pred_reg_1 = m1.predict(predict_X)
                x_quant_reg_1 = m1.predict_quantiles(predict_X)

            x_pred_reg_2 = m2.predict(predict_X)
            x_quant_reg_2 = m2.predict_quantiles(predict_X)

            np.testing.assert_array_almost_equal(
                x_pred_reg_1[0], x_pred_reg_2[0], mean_compare_decimal
            )
            np.testing.assert_array_almost_equal(
                x_pred_reg_1[1], x_pred_reg_2[1], var_compare_decimal
            )
            np.testing.assert_array_almost_equal(
                x_quant_reg_1[0], x_quant_reg_2[0], mean_compare_decimal
            )
            np.testing.assert_array_almost_equal(
                x_quant_reg_1[1], x_quant_reg_2[1], mean_compare_decimal
            )
            np.testing.assert_array_almost_equal(
                m1.gradient, m2.gradient, var_compare_decimal
            )
            np.testing.assert_almost_equal(
                m1.log_likelihood(), m2.log_likelihood(), var_compare_decimal
            )

    def test_matern32_kernel(
        self,
    ):
        np.random.seed(234)  # seed the random number generator
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=10.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )
        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_Matern32(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel = GPy.kern.Matern32(
            1,
            active_dims=[
                0,
            ],
        )

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            predict_X=X,
            compare_with_GP=True,
            gp_kernel=gp_kernel,
            mean_compare_decimal=5,
            var_compare_decimal=5,
        )

    def test_matern52_kernel(
        self,
    ):
        np.random.seed(234)  # seed the random number generator
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=10.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )
        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_Matern52(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel = GPy.kern.Matern52(
            1,
            active_dims=[
                0,
            ],
        )

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            optimize=True,
            predict_X=X,
            compare_with_GP=True,
            gp_kernel=gp_kernel,
            mean_compare_decimal=5,
            var_compare_decimal=5,
        )

    def test_rbf_kernel(
        self,
    ):
        # import pdb;pdb.set_trace()

        np.random.seed(234)  # seed the random number generator
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=10.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )
        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_RBF(
            1,
            110.0,
            1.5,
            active_dims=[
                0,
            ],
            balance=True,
            approx_order=10,
        )
        gp_kernel = GPy.kern.RBF(
            1,
            110.0,
            1.5,
            active_dims=[
                0,
            ],
        )

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            optimize_max_iters=1000,
            mean_compare_decimal=2,
            var_compare_decimal=1,
        )

    def test_periodic_kernel(
        self,
    ):
        np.random.seed(322)  # seed the random number generator
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=10.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )
        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        ss_kernel.lengthscale.constrain_bounded(0.27, 1000)
        ss_kernel.period.constrain_bounded(0.17, 100)

        gp_kernel = GPy.kern.StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel.lengthscale.constrain_bounded(0.27, 1000)
        gp_kernel.period.constrain_bounded(0.17, 100)

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=3,
            var_compare_decimal=3,
        )

    def test_quasi_periodic_kernel(
        self,
    ):
        np.random.seed(329)  # seed the random number generator
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=10.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )
        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_Matern32(1) * GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        ss_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        ss_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        gp_kernel = GPy.kern.Matern32(1) * GPy.kern.StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        gp_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=1,
            var_compare_decimal=2,
        )

    def test_linear_kernel(
        self,
    ):
        np.random.seed(234)  # seed the random number generator
        (X, Y) = generate_linear_data(
            x_points=None,
            tangent=2.0,
            add_term=20.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_Linear(
            1,
            X,
            active_dims=[
                0,
            ],
        ) + GPy.kern.sde_Bias(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel = GPy.kern.Linear(
            1,
            active_dims=[
                0,
            ],
        ) + GPy.kern.Bias(
            1,
            active_dims=[
                0,
            ],
        )

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=False,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=5,
            var_compare_decimal=5,
        )

    def test_brownian_kernel(
        self,
    ):
        np.random.seed(234)  # seed the random number generator
        (X, Y) = generate_brownian_data(
            x_points=None,
            kernel_var=2.0,
            noise_var=0.1,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_Brownian()
        gp_kernel = GPy.kern.Brownian()

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=4,
            var_compare_decimal=4,
        )

    def test_exponential_kernel(
        self,
    ):
        np.random.seed(12345)  # seed the random number generator
        (X, Y) = generate_linear_data(
            x_points=None,
            tangent=1.0,
            add_term=20.0,
            noise_var=2.0,
            plot=False,
            points_num=10,
            x_interval=(0, 20),
            random=True,
        )

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        ss_kernel = GPy.kern.sde_Exponential(
            1,
            Y.var(),
            X.ptp() / 2.0,
            active_dims=[
                0,
            ],
        )
        gp_kernel = GPy.kern.Exponential(
            1,
            Y.var(),
            X.ptp() / 2.0,
            active_dims=[
                0,
            ],
        )

        Y -= Y.mean()

        self.run_for_model(
            X,
            Y,
            ss_kernel,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            optimize_max_iters=1000,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )

    def test_kernel_addition_svd(
        self,
    ):
        # np.random.seed(329) # seed the random number generator
        np.random.seed(42)
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=5.0,
            noise_var=2.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        (X1, Y1) = generate_linear_data(
            x_points=X,
            tangent=1.0,
            add_term=20.0,
            noise_var=0.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        # Sine data <-
        Y = Y + Y1
        Y -= Y.mean()

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        def get_new_kernels():
            ss_kernel = GPy.kern.sde_Linear(
                1, X, variances=1
            ) + GPy.kern.sde_StdPeriodic(
                1,
                period=5.0,
                variance=300,
                lengthscale=3,
                active_dims=[
                    0,
                ],
            )
            # ss_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
            # ss_kernel.std_periodic.period.constrain_bounded(3, 8)

            gp_kernel = GPy.kern.Linear(1, variances=1) + GPy.kern.StdPeriodic(
                1,
                period=5.0,
                variance=300,
                lengthscale=3,
                active_dims=[
                    0,
                ],
            )
            # gp_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
            # gp_kernel.std_periodic.period.constrain_bounded(3, 8)

            return ss_kernel, gp_kernel

        # Cython is available only with svd.
        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(
            X,
            Y,
            ss_kernel,
            kalman_filter_type="svd",
            use_cython=True,
            optimize_max_iters=10,
            check_gradients=False,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=3,
            var_compare_decimal=3,
        )

        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(
            X,
            Y,
            ss_kernel,
            kalman_filter_type="svd",
            use_cython=False,
            optimize_max_iters=10,
            check_gradients=False,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=3,
            var_compare_decimal=3,
        )

    def test_kernel_addition_regular(
        self,
    ):
        # np.random.seed(329) # seed the random number generator
        np.random.seed(42)
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=5.0,
            noise_var=2.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        (X1, Y1) = generate_linear_data(
            x_points=X,
            tangent=1.0,
            add_term=20.0,
            noise_var=0.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        # Sine data <-
        Y = Y + Y1
        Y -= Y.mean()

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        def get_new_kernels():
            ss_kernel = GPy.kern.sde_Linear(
                1, X, variances=1
            ) + GPy.kern.sde_StdPeriodic(
                1,
                period=5.0,
                variance=300,
                lengthscale=3,
                active_dims=[
                    0,
                ],
            )
            # ss_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
            # ss_kernel.std_periodic.period.constrain_bounded(3, 8)

            gp_kernel = GPy.kern.Linear(1, variances=1) + GPy.kern.StdPeriodic(
                1,
                period=5.0,
                variance=300,
                lengthscale=3,
                active_dims=[
                    0,
                ],
            )
            # gp_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
            # gp_kernel.std_periodic.period.constrain_bounded(3, 8)

            return ss_kernel, gp_kernel

        ss_kernel, gp_kernel = get_new_kernels()
        try:
            self.run_for_model(
                X,
                Y,
                ss_kernel,
                kalman_filter_type="regular",
                use_cython=False,
                optimize_max_iters=10,
                check_gradients=True,
                predict_X=X,
                gp_kernel=gp_kernel,
                mean_compare_decimal=2,
                var_compare_decimal=2,
            )
        except AssertionError:
            raise SkipTest(
                "Skipping Regular kalman filter for kernel addition, because it is not stable (normal situation) for this data."
            )

    def test_kernel_multiplication(
        self,
    ):
        np.random.seed(329)  # seed the random number generator
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=10.0,
            noise_var=2.0,
            plot=False,
            points_num=50,
            x_interval=(0, 20),
            random=True,
        )

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)

        def get_new_kernels():
            ss_kernel = GPy.kern.sde_Matern32(1) * GPy.kern.sde_Matern52(1)
            gp_kernel = GPy.kern.Matern32(1) * GPy.kern.sde_Matern52(1)

            return ss_kernel, gp_kernel

        ss_kernel, gp_kernel = get_new_kernels()

        # import ipdb;ipdb.set_trace()
        self.run_for_model(
            X,
            Y,
            ss_kernel,
            kalman_filter_type="svd",
            use_cython=True,
            optimize_max_iters=10,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )

        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(
            X,
            Y,
            ss_kernel,
            kalman_filter_type="regular",
            use_cython=False,
            optimize_max_iters=10,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )

        ss_kernel, gp_kernel = get_new_kernels()
        self.run_for_model(
            X,
            Y,
            ss_kernel,
            kalman_filter_type="svd",
            use_cython=False,
            optimize_max_iters=10,
            check_gradients=True,
            predict_X=X,
            gp_kernel=gp_kernel,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )

    def test_forecast_regular(
        self,
    ):
        # Generate data ->
        np.random.seed(339)  # seed the random number generator
        # import pdb; pdb.set_trace()
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=5.0,
            noise_var=2.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        (X1, Y1) = generate_linear_data(
            x_points=X,
            tangent=1.0,
            add_term=20.0,
            noise_var=0.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        Y = Y + Y1

        X_train = X[X <= 20]
        Y_train = Y[X <= 20]
        X_test = X[X > 20]
        Y_test = Y[X > 20]

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)
        X_train.shape = (X_train.shape[0], 1)
        Y_train.shape = (Y_train.shape[0], 1)
        X_test.shape = (X_test.shape[0], 1)
        Y_test.shape = (Y_test.shape[0], 1)
        # Generate data <-

        # import pdb; pdb.set_trace()

        periodic_kernel = GPy.kern.StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel = (
            GPy.kern.Linear(
                1,
                active_dims=[
                    0,
                ],
            )
            + GPy.kern.Bias(
                1,
                active_dims=[
                    0,
                ],
            )
            + periodic_kernel
        )
        gp_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        gp_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        periodic_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        ss_kernel = (
            GPy.kern.sde_Linear(
                1,
                X,
                active_dims=[
                    0,
                ],
            )
            + GPy.kern.sde_Bias(
                1,
                active_dims=[
                    0,
                ],
            )
            + periodic_kernel
        )

        ss_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        ss_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        self.run_for_model(
            X_train,
            Y_train,
            ss_kernel,
            kalman_filter_type="regular",
            use_cython=False,
            optimize_max_iters=30,
            check_gradients=True,
            predict_X=X_test,
            gp_kernel=gp_kernel,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )

    def test_forecast_svd(
        self,
    ):
        # Generate data ->
        np.random.seed(339)  # seed the random number generator
        # import pdb; pdb.set_trace()
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=5.0,
            noise_var=2.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        (X1, Y1) = generate_linear_data(
            x_points=X,
            tangent=1.0,
            add_term=20.0,
            noise_var=0.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        Y = Y + Y1

        X_train = X[X <= 20]
        Y_train = Y[X <= 20]
        X_test = X[X > 20]
        Y_test = Y[X > 20]

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)
        X_train.shape = (X_train.shape[0], 1)
        Y_train.shape = (Y_train.shape[0], 1)
        X_test.shape = (X_test.shape[0], 1)
        Y_test.shape = (Y_test.shape[0], 1)
        # Generate data <-

        # import pdb; pdb.set_trace()

        periodic_kernel = GPy.kern.StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel = (
            GPy.kern.Linear(
                1,
                active_dims=[
                    0,
                ],
            )
            + GPy.kern.Bias(
                1,
                active_dims=[
                    0,
                ],
            )
            + periodic_kernel
        )
        gp_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        gp_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        periodic_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        ss_kernel = (
            GPy.kern.sde_Linear(
                1,
                X,
                active_dims=[
                    0,
                ],
            )
            + GPy.kern.sde_Bias(
                1,
                active_dims=[
                    0,
                ],
            )
            + periodic_kernel
        )

        ss_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        ss_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        self.run_for_model(
            X_train,
            Y_train,
            ss_kernel,
            kalman_filter_type="svd",
            use_cython=False,
            optimize_max_iters=30,
            check_gradients=False,
            predict_X=X_test,
            gp_kernel=gp_kernel,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )

    def test_forecast_svd_cython(
        self,
    ):
        # Generate data ->
        np.random.seed(339)  # seed the random number generator
        # import pdb; pdb.set_trace()
        (X, Y) = generate_sine_data(
            x_points=None,
            sin_period=5.0,
            sin_ampl=5.0,
            noise_var=2.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        (X1, Y1) = generate_linear_data(
            x_points=X,
            tangent=1.0,
            add_term=20.0,
            noise_var=0.0,
            plot=False,
            points_num=100,
            x_interval=(0, 40),
            random=True,
        )

        Y = Y + Y1

        X_train = X[X <= 20]
        Y_train = Y[X <= 20]
        X_test = X[X > 20]
        Y_test = Y[X > 20]

        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)
        X_train.shape = (X_train.shape[0], 1)
        Y_train.shape = (Y_train.shape[0], 1)
        X_test.shape = (X_test.shape[0], 1)
        Y_test.shape = (Y_test.shape[0], 1)
        # Generate data <-

        # import pdb; pdb.set_trace()

        periodic_kernel = GPy.kern.StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        gp_kernel = (
            GPy.kern.Linear(
                1,
                active_dims=[
                    0,
                ],
            )
            + GPy.kern.Bias(
                1,
                active_dims=[
                    0,
                ],
            )
            + periodic_kernel
        )
        gp_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        gp_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        periodic_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        ss_kernel = (
            GPy.kern.sde_Linear(
                1,
                X,
                active_dims=[
                    0,
                ],
            )
            + GPy.kern.sde_Bias(
                1,
                active_dims=[
                    0,
                ],
            )
            + periodic_kernel
        )

        ss_kernel.std_periodic.lengthscale.constrain_bounded(0.25, 1000)
        ss_kernel.std_periodic.period.constrain_bounded(0.15, 100)

        self.run_for_model(
            X_train,
            Y_train,
            ss_kernel,
            kalman_filter_type="svd",
            use_cython=True,
            optimize_max_iters=30,
            check_gradients=False,
            predict_X=X_test,
            gp_kernel=gp_kernel,
            mean_compare_decimal=2,
            var_compare_decimal=2,
        )
