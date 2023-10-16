# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Test module for state_space_main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import GPy.models.state_space_setup as ss_setup
import GPy.models.state_space_main as ssm


def generate_x_points(points_num=100, x_interval=(0, 20), random=True):
    """
    Function generates (sorted) points on the x axis.

    Input:
    ---------------------------
        points_num: int
            How many points to generate
        x_interval: tuple (a,b)
            On which interval to generate points
        random: bool
            Regular points or random

    Output:
    ---------------------------
        x_points: np.array
            Generated points
    """

    x_interval = np.asarray(x_interval)

    if random:
        x_points = (
            np.random.rand(points_num) * (x_interval[1] - x_interval[0]) + x_interval[0]
        )
        x_points = np.sort(x_points)
    else:
        x_points = np.linspace(x_interval[0], x_interval[1], num=points_num)

    return x_points


def generate_sine_data(
    x_points=None,
    sin_period=2.0,
    sin_ampl=10.0,
    noise_var=2.0,
    plot=False,
    points_num=100,
    x_interval=(0, 20),
    random=True,
):
    """
    Function generates sinusoidal data.

    Input:
    --------------------------------

    x_points: np.array
        Previously generated X points
    sin_period: float
        Sine period
    sin_ampl: float
        Sine amplitude
    noise_var: float
        Gaussian noise variance added to the sine function
    plot: bool
        Whether to plot generated data

    (if x_points is None, the the following parameters are used to generate
    those. They are the same as in 'generate_x_points' function)

    points_num: int

    x_interval: tuple (a,b)

    random: bool
    """

    sin_function = lambda xx: sin_ampl * np.sin(2 * np.pi / sin_period * xx)

    if x_points is None:
        x_points = generate_x_points(points_num, x_interval, random)

    y_points = sin_function(x_points) + np.random.randn(len(x_points)) * np.sqrt(
        noise_var
    )

    if plot:
        pass

    return x_points, y_points


def generate_linear_data(
    x_points=None,
    tangent=2.0,
    add_term=1.0,
    noise_var=2.0,
    plot=False,
    points_num=100,
    x_interval=(0, 20),
    random=True,
):
    """
    Function generates linear data.

    Input:
    --------------------------------

    x_points: np.array
        Previously generated X points
    tangent: float
        Factor with which independent variable is multiplied in linear equation.
    add_term: float
        Additive term in linear equation.
    noise_var: float
        Gaussian noise variance added to the sine function
    plot: bool
        Whether to plot generated data

    (if x_points is None, the the following parameters are used to generate
    those. They are the same as in 'generate_x_points' function)

    points_num: int

    x_interval: tuple (a,b)

    random: bool
    """

    linear_function = lambda xx: tangent * xx + add_term

    if x_points is None:
        x_points = generate_x_points(points_num, x_interval, random)

    y_points = linear_function(x_points) + np.random.randn(len(x_points)) * np.sqrt(
        noise_var
    )

    if plot:
        pass

    return x_points, y_points


def generate_brownian_data(
    x_points=None,
    kernel_var=2.0,
    noise_var=2.0,
    plot=False,
    points_num=100,
    x_interval=(0, 20),
    random=True,
):
    """
    Generate brownian data - data from Brownian motion.
    First point is always 0, and \Beta(0) = 0  - standard conditions for Brownian motion.

    Input:
    --------------------------------

    x_points: np.array
        Previously generated X points
    variance: float
        Gaussian noise variance added to the sine function
    plot: bool
        Whether to plot generated data

    (if x_points is None, the the following parameters are used to generate
    those. They are the same as in 'generate_x_points' function)

    points_num: int

    x_interval: tuple (a,b)

    random: bool

    """
    if x_points is None:
        x_points = generate_x_points(points_num, x_interval, random)
        if x_points[0] != 0:
            x_points[0] = 0

    y_points = np.zeros((points_num,))
    for i in range(1, points_num):
        noise = np.random.randn() * np.sqrt(
            kernel_var * (x_points[i] - x_points[i - 1])
        )
        y_points[i] = y_points[i - 1] + noise

    y_points += np.random.randn(len(x_points)) * np.sqrt(noise_var)

    return x_points, y_points


def generate_linear_plus_sin(
    x_points=None,
    tangent=2.0,
    add_term=1.0,
    noise_var=2.0,
    sin_period=2.0,
    sin_ampl=10.0,
    plot=False,
    points_num=100,
    x_interval=(0, 20),
    random=True,
):
    """
    Generate the sum of linear trend and the sine function.

    For parameters see the 'generate_linear' and 'generate_sine'.

    Comment: Gaussian noise variance is added only once (for linear function).
    """

    x_points, y_linear_points = generate_linear_data(
        x_points, tangent, add_term, noise_var, False, points_num, x_interval, random
    )

    x_points, y_sine_points = generate_sine_data(
        x_points, sin_period, sin_ampl, 0.0, False, points_num, x_interval, random
    )

    y_points = y_linear_points + y_sine_points

    if plot:
        pass

    return x_points, y_points


def generate_random_y_data(samples, dim, ts_no):
    """
    Generate data:

    Input:
    ------------------

    samples - how many samples
    dim - dimensionality of the data
    ts_no - number of time series

    Output:
    --------------------------
        Y: np.array((samples, dim, ts_no))
    """

    Y = np.empty((samples, dim, ts_no))

    for i in range(0, samples):
        for j in range(0, ts_no):
            sample = np.random.randn(dim)
            Y[i, :, j] = sample

    if Y.shape[2] == 1:  # ts_no = 1
        Y.shape = (Y.shape[0], Y.shape[1])
    return Y


class TestStateSpaceKernels:
    def run_descr_model(
        self,
        measurements,
        A,
        Q,
        H,
        R,
        true_states=None,
        mean_compare_decimal=8,
        m_init=None,
        P_init=None,
        dA=None,
        dQ=None,
        dH=None,
        dR=None,
        use_cython=False,
        kalman_filter_type="regular",
        calc_log_likelihood=True,
        calc_grad_log_likelihood=True,
    ):
        # import pdb; pdb.set_trace()

        state_dim = 1 if not isinstance(A, np.ndarray) else A.shape[0]
        ts_no = 1 if (len(measurements.shape) < 3) else measurements.shape[2]
        import importlib

        grad_params_no = None if dA is None else dA.shape[2]

        ss_setup.use_cython = use_cython
        global ssm
        if (ssm.cython_code_available) and (ssm.use_cython != use_cython):
            importlib.reload(ssm.DescreteStateSpace)

        grad_calc_params = None
        if calc_grad_log_likelihood:
            grad_calc_params = {}
            grad_calc_params["dA"] = dA
            grad_calc_params["dQ"] = dQ
            grad_calc_params["dH"] = dH
            grad_calc_params["dR"] = dR

        (
            f_mean,
            f_var,
            loglikelhood,
            g_loglikelhood,
            dynamic_callables_smoother,
        ) = ssm.DescreteStateSpace.kalman_filter(
            A,
            Q,
            H,
            R,
            measurements,
            index=None,
            m_init=m_init,
            P_init=P_init,
            p_kalman_filter_type=kalman_filter_type,
            calc_log_likelihood=calc_log_likelihood,
            calc_grad_log_likelihood=calc_grad_log_likelihood,
            grad_params_no=grad_params_no,
            grad_calc_params=grad_calc_params,
        )

        f_mean_squeezed = np.squeeze(f_mean[1:, :])  # exclude initial value
        _f_var_squeezed = np.squeeze(f_var[1:, :])  # exclude initial value

        if true_states is not None:
            # print np.max(np.abs(f_mean_squeezed-true_states))
            np.testing.assert_almost_equal(
                np.max(np.abs(f_mean_squeezed - true_states)),
                0,
                decimal=mean_compare_decimal,
            )

        np.testing.assert_equal(
            f_mean.shape, (measurements.shape[0] + 1, state_dim, ts_no)
        )
        np.testing.assert_equal(
            f_var.shape, (measurements.shape[0] + 1, state_dim, state_dim)
        )

        (_M_smooth, _P_smooth) = ssm.DescreteStateSpace.rts_smoother(
            state_dim, dynamic_callables_smoother, f_mean, f_var
        )

        return f_mean, f_var

    def run_continuous_model(
        self,
        F,
        L,
        Qc,
        p_H,
        p_R,
        P_inf,
        X_data,
        Y_data,
        index=None,
        m_init=None,
        P_init=None,
        use_cython=False,
        kalman_filter_type="regular",
        calc_log_likelihood=True,
        calc_grad_log_likelihood=True,
        grad_params_no=0,
        grad_calc_params=None,
    ):
        # import pdb; pdb.set_trace()

        state_dim = 1 if not isinstance(F, np.ndarray) else F.shape[0]
        ts_no = 1 if (len(Y_data.shape) < 3) else Y_data.shape[2]

        import importlib

        ss_setup.use_cython = use_cython
        global ssm
        if (ssm.cython_code_available) and (ssm.use_cython != use_cython):
            importlib.reload(ssm)

        (
            f_mean,
            f_var,
            loglikelhood,
            g_loglikelhood,
            dynamic_callables_smoother,
        ) = ssm.ContDescrStateSpace.cont_discr_kalman_filter(
            F,
            L,
            Qc,
            p_H,
            p_R,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=None,
            p_kalman_filter_type="regular",
            calc_log_likelihood=False,
            calc_grad_log_likelihood=False,
            grad_params_no=0,
            grad_calc_params=grad_calc_params,
        )

        _f_mean_squeezed = np.squeeze(f_mean[1:, :])  # exclude initial value
        _f_var_squeezed = np.squeeze(f_var[1:, :])  # exclude initial value

        np.testing.assert_equal(f_mean.shape, (Y_data.shape[0] + 1, state_dim, ts_no))
        np.testing.assert_equal(
            f_var.shape, (Y_data.shape[0] + 1, state_dim, state_dim)
        )

        (_M_smooth, _P_smooth) = ssm.ContDescrStateSpace.cont_discr_rts_smoother(
            state_dim, f_mean, f_var, dynamic_callables_smoother
        )

        return f_mean, f_var

    def test_discrete_ss_first(self, plot=False):
        """
        Tests discrete State-Space model - first test.
        """
        np.random.seed(235)  # seed the random number generator

        A = 1.0  # For cython code to run properly need float input
        H = 1.0
        Q = 1.0
        R = 1.0

        steps_num = 100

        # generate data ->
        true_states = np.zeros((steps_num,))
        init_state = 0
        measurements = np.zeros((steps_num,))

        for s in range(0, steps_num):
            if s == 0:
                true_states[0] = init_state + np.sqrt(Q) * np.random.randn()
            else:
                true_states[s] = true_states[s - 1] + np.sqrt(R) * np.random.randn()
            measurements[s] = true_states[s] + np.sqrt(R) * np.random.randn()
        # generate data <-

        # descrete kalman filter ->
        m_init = 0
        P_init = 1
        d_num = 1000
        state_discr = np.linspace(-10, 10, d_num)

        state_trans_matrix = np.empty((d_num, d_num))
        for i in range(d_num):
            state_trans_matrix[:, i] = norm.pdf(
                state_discr, loc=A * state_discr[i], scale=np.sqrt(Q)
            )

        m_prev = norm.pdf(state_discr, loc=m_init, scale=np.sqrt(P_init))
        # m_prev / np.sum(m_prev)
        m = np.zeros((d_num, steps_num))
        i_mean = np.zeros((steps_num,))

        for s in range(0, steps_num):
            # Prediction step:
            if s == 0:
                m[:, s] = np.dot(state_trans_matrix, m_prev)
            else:
                m[:, s] = np.dot(state_trans_matrix, m[:, s - 1])
            # Update step:
            # meas_ind = np.argmin(np.abs(state_discr - measurements[s])
            y_vec = np.zeros((d_num,))
            for i in range(d_num):
                y_vec[i] = norm.pdf(
                    measurements[s], loc=H * state_discr[i], scale=np.sqrt(R)
                )
            norm_const = np.dot(y_vec, m[:, s])
            m[:, s] = y_vec * m[:, s] / norm_const

            i_mean[s] = state_discr[np.argmax(m[:, s])]
        # descrete kalman filter <-

        (f_mean, f_var) = self.run_descr_model(
            measurements,
            A,
            Q,
            H,
            R,
            true_states=i_mean,
            mean_compare_decimal=1,
            m_init=m_init,
            P_init=P_init,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=False,
        )

        (f_mean, f_var) = self.run_descr_model(
            measurements,
            A,
            Q,
            H,
            R,
            true_states=i_mean,
            mean_compare_decimal=1,
            m_init=m_init,
            P_init=P_init,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=False,
        )

        (f_mean, f_var) = self.run_descr_model(
            measurements,
            A,
            Q,
            H,
            R,
            true_states=i_mean,
            mean_compare_decimal=1,
            m_init=m_init,
            P_init=P_init,
            use_cython=True,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=False,
        )

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(true_states, "g.-", label="true states")
            # plt.plot( measurements, 'b.-', label='measurements')
            plt.plot(f_mean, "r.-", label="Kalman filter estimates")
            plt.plot(i_mean, "k.-", label="Discretization")

            plt.plot(f_mean + 2 * np.sqrt(f_var), "r.--")
            plt.plot(f_mean - 2 * np.sqrt(f_var), "r.--")
            plt.legend()
            plt.show()
            # plotting <-
        return None

    def test_discrete_ss_1D(self, plot=False):
        """
        This function tests Kalman filter and smoothing when the state
        dimensionality is one dimensional.
        """

        np.random.seed(234)  # seed the random number generator

        # 1D ss model
        state_dim = 1
        param_num = 2  # sigma_Q, sigma_R - parameters
        measurement_dim = 1  # dimensionality od measurement

        A = 1.0
        Q = 2.0
        dA = np.zeros((state_dim, state_dim, param_num))
        dQ = np.zeros((state_dim, state_dim, param_num))
        dQ[0, 0, 0] = 1.0

        # measurement related parameters (subject to change) ->
        H = np.ones((measurement_dim, state_dim))
        R = 0.5 * np.eye(measurement_dim)
        dH = np.zeros((measurement_dim, state_dim, param_num))
        dR = np.zeros((measurement_dim, measurement_dim, param_num))
        dR[:, :, 1] = np.eye(measurement_dim)
        # measurement related parameters (subject to change) <-

        # 1D measurement, 1 ts_no ->
        data = generate_random_y_data(10, 1, 1)  # np.array((samples, dim, ts_no))

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=True,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(np.squeeze(data), "g.-", label="measurements")
            plt.plot(np.squeeze(f_mean[1:]), "b.-", label="Kalman filter estimates")
            plt.plot(np.squeeze(f_mean[1:] + H * f_var[1:] * H), "b--")
            plt.plot(np.squeeze(f_mean[1:] - H * f_var[1:] * H), "b--")
            #            plt.plot( np.squeeze(M_sm[1:]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:]+H*P_sm[1:]*H), 'r--')
            #            plt.plot( np.squeeze(M_sm[1:]-H*P_sm[1:]*H), 'r--')
            plt.legend()
            plt.title("1D state-space, 1D measurements, 1 ts_no")
            plt.show()
            # plotting <-
        # 1D measurement, 1 ts_no <-

        # 1D measurement, 3 ts_no ->
        data = generate_random_y_data(10, 1, 3)  # np.array((samples, dim, ts_no))

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=True,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        # import pdb; pdb.set_trace()
        if plot:
            # plotting ->
            plt.figure()
            plt.plot(np.squeeze(data[:, :, 1]), "g.-", label="measurements")
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1]), "b.-", label="Kalman filter estimates"
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1]) + np.squeeze(H * f_var[1:] * H), "b--"
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1]) - np.squeeze(H * f_var[1:] * H), "b--"
            )
            #            plt.plot( np.squeeze(M_sm[1:,0,1]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])+H*np.squeeze(P_sm[1:])*H, 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])-H*np.squeeze(P_sm[1:])*H, 'r--')
            plt.legend()
            plt.title("1D state-space, 1D measurements, 3 ts_no. 2-nd ts ploted")
            plt.show()
            # plotting <-
        # 1D measurement, 3 ts_no <-
        measurement_dim = 2  # dimensionality of measurement

        H = np.ones((measurement_dim, state_dim))
        R = 0.5 * np.eye(measurement_dim)
        dH = np.zeros((measurement_dim, state_dim, param_num))
        dR = np.zeros((measurement_dim, measurement_dim, param_num))
        dR[:, :, 1] = np.eye(measurement_dim)
        # measurement related parameters (subject to change) <

        data = generate_random_y_data(10, 2, 3)  # np.array((samples, dim, ts_no))

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        #        (f_mean, f_var) = self.run_descr_model(data, A,Q,H,R, true_states=None,
        #                          mean_compare_decimal=16,
        #                          m_init=None, P_init=None, dA=dA,dQ=dQ,
        #                          dH=dH,dR=dR, use_cython=True,
        #                          kalman_filter_type='svd',
        #                          calc_log_likelihood=True,
        #                          calc_grad_log_likelihood=True)

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(np.squeeze(data[:, 0, 1]), "g.-", label="measurements")
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1]), "b.-", label="Kalman filter estimates"
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #            plt.plot( np.squeeze(M_sm[1:,0,1]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title(
                "1D state-space, 2D measurements, 3 ts_no. 1-st measurement, 2-nd ts ploted"
            )
            plt.show()
            # plotting <-
        # 2D measurement, 3 ts_no <-

    def test_discrete_ss_2D(self, plot=False):
        """
        This function tests Kalman filter and smoothing when the state
        dimensionality is two dimensional.
        """

        np.random.seed(234)  # seed the random number generator

        # 1D ss model
        state_dim = 2
        param_num = 3  # sigma_Q, sigma_R, one parameters in A - parameters
        measurement_dim = 1  # dimensionality od measurement

        A = np.eye(state_dim)
        A[0, 0] = 0.5
        Q = np.ones((state_dim, state_dim))
        dA = np.zeros((state_dim, state_dim, param_num))
        dA[1, 1, 2] = 1
        dQ = np.zeros((state_dim, state_dim, param_num))
        dQ[:, :, 1] = np.eye(measurement_dim)

        # measurement related parameters (subject to change) ->
        H = np.ones((measurement_dim, state_dim))
        R = 0.5 * np.eye(measurement_dim)
        dH = np.zeros((measurement_dim, state_dim, param_num))
        dR = np.zeros((measurement_dim, measurement_dim, param_num))
        dR[:, :, 1] = np.eye(measurement_dim)
        # measurement related parameters (subject to change) <-

        # 1D measurement, 1 ts_no ->
        data = generate_random_y_data(10, 1, 1)  # np.array((samples, dim, ts_no))

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=True,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )
        if plot:
            # plotting ->
            plt.figure()
            plt.plot(np.squeeze(data), "g.-", label="measurements")
            plt.plot(np.squeeze(f_mean[1:, 0]), "b.-", label="Kalman filter estimates")
            plt.plot(
                np.squeeze(f_mean[1:, 0])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #            plt.plot( np.squeeze(M_sm[1:,0]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,0])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,0])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title("2D state-space, 1D measurements, 1 ts_no")
            plt.show()
            # plotting <-
        # 1D measurement, 1 ts_no <-

        # 1D measurement, 3 ts_no ->
        data = generate_random_y_data(10, 1, 3)  # np.array((samples, dim, ts_no))
        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=True,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )
        if plot:
            # plotting ->
            plt.figure()
            plt.plot(np.squeeze(data[:, :, 1]), "g.-", label="measurements")
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1]), "b.-", label="Kalman filter estimates"
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #            plt.plot( np.squeeze(M_sm[1:,0,1]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title("2D state-space, 1D measurements, 3 ts_no. 2-nd ts ploted")
            plt.show()
            # plotting <-
        # 1D measurement, 3 ts_no <-

        # 2D measurement, 3 ts_no ->
        # measurement related parameters (subject to change) ->
        measurement_dim = 2  # dimensionality od measurement

        H = np.ones((measurement_dim, state_dim))
        R = 0.5 * np.eye(measurement_dim)
        dH = np.zeros((measurement_dim, state_dim, param_num))
        dR = np.zeros((measurement_dim, measurement_dim, param_num))
        dR[:, :, 1] = np.eye(measurement_dim)
        # measurement related parameters (subject to change) <

        data = generate_random_y_data(10, 2, 3)  # np.array((samples, dim, ts_no))
        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        (f_mean, f_var) = self.run_descr_model(
            data,
            A,
            Q,
            H,
            R,
            true_states=None,
            mean_compare_decimal=16,
            m_init=None,
            P_init=None,
            dA=dA,
            dQ=dQ,
            dH=dH,
            dR=dR,
            use_cython=False,
            kalman_filter_type="svd",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
        )

        #        (f_mean, f_var) = self.run_descr_model(data, A,Q,H,R, true_states=None,
        #                          mean_compare_decimal=16,
        #                          m_init=None, P_init=None, dA=dA,dQ=dQ,
        #                          dH=dH,dR=dR, use_cython=True,
        #                          kalman_filter_type='svd',
        #                          calc_log_likelihood=True,
        #                          calc_grad_log_likelihood=True)

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(np.squeeze(data[:, 0, 1]), "g.-", label="measurements")
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1]), "b.-", label="Kalman filter estimates"
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                np.squeeze(f_mean[1:, 0, 1])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #            plt.plot( np.squeeze(M_sm[1:,0,1]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,0,1])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title(
                "2D state-space, 2D measurements, 3 ts_no. 1-st measurement, 2-nd ts ploted"
            )
            plt.show()
            # plotting <-
        # 2D measurement, 3 ts_no <-

    def test_continuous_ss(self, plot=False):
        """
        This function tests the continuous state-space model.
        """

        # 1D measurements, 1 ts_no ->
        measurement_dim = 1  # dimensionality of measurement

        X_data = generate_x_points(points_num=10, x_interval=(0, 20), random=True)
        Y_data = generate_random_y_data(10, 1, 1)  # np.array((samples, dim, ts_no))

        try:
            import GPy
        except ImportError as e:
            return None

        periodic_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        (F, L, Qc, H, P_inf, P0, dFt, dQct, dP_inft, dP0) = periodic_kernel.sde()

        state_dim = dFt.shape[0]
        param_num = dFt.shape[2]

        grad_calc_params = {}
        grad_calc_params["dP_inf"] = dP_inft
        grad_calc_params["dF"] = dFt
        grad_calc_params["dQc"] = dQct
        grad_calc_params["dR"] = np.zeros((measurement_dim, measurement_dim, param_num))
        grad_calc_params["dP_init"] = dP0
        # dH matrix is None

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            1.5,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            1.5,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=False,
            kalman_filter_type="rbc",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            1.5,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=True,
            kalman_filter_type="rbc",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(X_data, np.squeeze(Y_data[:, 0]), "g.-", label="measurements")
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15]),
                "b.-",
                label="Kalman filter estimates",
            )
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #        plt.plot( np.squeeze(M_sm[1:,15]), 'r.-',label='Smoother Estimates')
            #        plt.plot( np.squeeze(M_sm[1:,15])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #        plt.plot( np.squeeze(M_sm[1:,15])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title("1D measurements, 1 ts_no")
            plt.show()
            # plotting <-
        # 1D measurements, 1 ts_no <-

        # 1D measurements, 3 ts_no ->
        measurement_dim = 1  # dimensionality od measurement

        X_data = generate_x_points(points_num=10, x_interval=(0, 20), random=True)
        Y_data = generate_random_y_data(10, 1, 3)  # np.array((samples, dim, ts_no))

        periodic_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        (F, L, Qc, H, P_inf, P0, dFt, dQct, dP_inft, dP0) = periodic_kernel.sde()

        state_dim = dFt.shape[0]
        param_num = dFt.shape[2]

        grad_calc_params = {}
        grad_calc_params["dP_inf"] = dP_inft
        grad_calc_params["dF"] = dFt
        grad_calc_params["dQc"] = dQct
        grad_calc_params["dR"] = np.zeros((measurement_dim, measurement_dim, param_num))
        grad_calc_params["dP_init"] = dP0
        # dH matrix is None

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            1.5,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            1.5,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=False,
            kalman_filter_type="rbc",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            1.5,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=True,
            kalman_filter_type="rbc",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(X_data, np.squeeze(Y_data[:, 0, 1]), "g.-", label="measurements")
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15, 1]),
                "b.-",
                label="Kalman filter estimates",
            )
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15, 1])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15, 1])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #            plt.plot( np.squeeze(M_sm[1:,15,1]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,15,1])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,15,1])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title("1D measurements, 3 ts_no. 2-nd ts ploted")
            plt.show()
            # plotting <-
        # 1D measurements, 3 ts_no <-

        # 2D measurements, 3 ts_no ->
        measurement_dim = 2  # dimensionality od measurement

        X_data = generate_x_points(points_num=10, x_interval=(0, 20), random=True)
        Y_data = generate_random_y_data(10, 2, 3)  # np.array((samples, dim, ts_no))

        periodic_kernel = GPy.kern.sde_StdPeriodic(
            1,
            active_dims=[
                0,
            ],
        )
        (F, L, Qc, H, P_inf, P0, dFt, dQct, dP_inft, dP0) = periodic_kernel.sde()
        H = np.vstack((H, H))  # make 2D measurements
        R = 1.5 * np.eye(measurement_dim)

        state_dim = dFt.shape[0]
        param_num = dFt.shape[2]

        grad_calc_params = {}
        grad_calc_params["dP_inf"] = dP_inft
        grad_calc_params["dF"] = dFt
        grad_calc_params["dQc"] = dQct
        grad_calc_params["dR"] = np.zeros((measurement_dim, measurement_dim, param_num))
        grad_calc_params["dP_init"] = dP0
        # dH matrix is None

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            R,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=False,
            kalman_filter_type="regular",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        (f_mean, f_var) = self.run_continuous_model(
            F,
            L,
            Qc,
            H,
            R,
            P_inf,
            X_data,
            Y_data,
            index=None,
            m_init=None,
            P_init=P0,
            use_cython=False,
            kalman_filter_type="rbc",
            calc_log_likelihood=True,
            calc_grad_log_likelihood=True,
            grad_params_no=param_num,
            grad_calc_params=grad_calc_params,
        )

        #        (f_mean, f_var) = self.run_continuous_model(F, L, Qc, H, R, P_inf, X_data, Y_data, index = None,
        #                          m_init=None, P_init=P0, use_cython=True,
        #                          kalman_filter_type='rbc',
        #                          calc_log_likelihood=True,
        #                          calc_grad_log_likelihood=True,
        #                          grad_params_no=param_num, grad_calc_params=grad_calc_params)

        if plot:
            # plotting ->
            plt.figure()
            plt.plot(X_data, np.squeeze(Y_data[:, 0, 1]), "g.-", label="measurements")
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15, 1]),
                "b.-",
                label="Kalman filter estimates",
            )
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15, 1])
                + np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            plt.plot(
                X_data,
                np.squeeze(f_mean[1:, 15, 1])
                - np.einsum("ij,ajk,kl", H, f_var[1:], H.T)[:, 0, 0],
                "b--",
            )
            #            plt.plot( np.squeeze(M_sm[1:,15,1]), 'r.-',label='Smoother Estimates')
            #            plt.plot( np.squeeze(M_sm[1:,15,1])+np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            #            plt.plot( np.squeeze(M_sm[1:,15,1])-np.einsum('ij,ajk,kl', H, P_sm[1:], H.T)[:,0,0], 'r--')
            plt.legend()
            plt.title("1D measurements, 3 ts_no. 2-nd ts ploted")
            plt.show()
            # plotting <-
        # 2D measurements, 3 ts_no <-


# def test_EM_gradient(plot=False):
#    """
#    Test EM gradient calculation. This method works (the formulas are such)
#    that it works only for time invariant matrices A, Q, H, R. For the continuous
#    model it means that time intervals are the same.
#    """
#
#    np.random.seed(234) # seed the random number generator
#
#    # 1D measurements, 1 ts_no ->
#    measurement_dim = 1 # dimensionality of measurement
#
#    x_data = generate_x_points(points_num=10, x_interval = (0, 20), random=False)
#    data = generate_random_y_data(10, 1, 1) # np.array((samples, dim, ts_no))
#
#    import GPy
#    #periodic_kernel = GPy.kern.sde_Matern32(1,active_dims=[0,])
#    periodic_kernel = GPy.kern.sde_StdPeriodic(1,active_dims=[0,])
#    (F,L,Qc,H,P_inf,P0, dFt,dQct,dP_inft,dP0t) = periodic_kernel.sde()
#
#    state_dim = dFt.shape[0];
#    param_num = dFt.shape[2]
#
#    grad_calc_params = {}
#    grad_calc_params['dP_inf'] = dP_inft
#    grad_calc_params['dF'] = dFt
#    grad_calc_params['dQc'] = dQct
#    grad_calc_params['dR'] = np.zeros((measurement_dim,measurement_dim,param_num))
#    grad_calc_params['dP_init'] = dP0t
#    # dH matrix is None
#
#
#    #(F,L,Qc,H,P_inf,dF,dQc,dP_inf) = ssm.balance_ss_model(F,L,Qc,H,P_inf,dF,dQc,dP_inf)
#    # Use the Kalman filter to evaluate the likelihood
#
#    #import pdb; pdb.set_trace()
#    (M_kf, P_kf, log_likelihood,
#     grad_log_likelihood,SmootherMatrObject) = ss.ContDescrStateSpace.cont_discr_kalman_filter(F,
#                                  L, Qc, H, 1.5, P_inf, x_data, data, m_init=None,
#                                  P_init=P0, calc_log_likelihood=True,
#                                  calc_grad_log_likelihood=True,
#                                  grad_params_no=param_num,
#                                  grad_calc_params=grad_calc_params)
#
#    if plot:
#        # plotting ->
#        plt.figure()
#        plt.plot( np.squeeze(data[:,0]), 'g.-', label='measurements')
#        plt.plot( np.squeeze(M_kf[1:,15]), 'b.-',label='Kalman filter estimates')
#        plt.plot( np.squeeze(M_kf[1:,15])+np.einsum('ij,ajk,kl', H, P_kf[1:], H.T)[:,0,0], 'b--')
#        plt.plot( np.squeeze(M_kf[1:,15])-np.einsum('ij,ajk,kl', H, P_kf[1:], H.T)[:,0,0], 'b--')
#        plt.title("1D measurements, 1 ts_no")
#        plt.show()
#        # plotting <-
#    # 1D measurements, 1 ts_no <-
