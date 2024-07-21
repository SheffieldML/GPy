# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Main functionality for state-space inference.
"""

import types  # for cheking whether a variable is a function

import numpy as np
import scipy as sp
import scipy.linalg as linalg
from typing import Iterable

try:
    from . import state_space_setup

    setup_available = True
except ImportError as e:
    setup_available = False


print_verbose = False

try:
    import state_space_cython

    cython_code_available = True
    if print_verbose:
        print("state_space: cython is available")
except ImportError as e:
    cython_code_available = False

# cython_code_available = False
# Use cython by default
use_cython = False
if setup_available:
    use_cython = state_space_setup.use_cython

if print_verbose:
    if use_cython:
        print("state_space: cython is used")
    else:
        print("state_space: cython is NOT used")

# When debugging external module can set some value to this variable (e.g.)
# 'model' and in this module this variable can be seen.s
tmp_buffer = None


class Dynamic_Callables_Python(object):
    def f_a(self, k, m, A):
        """
        p_a: function (k, x_{k-1}, A_{k}). Dynamic function.
            k (iteration number), starts at 0
            x_{k-1} State from the previous step
            A_{k} Jacobian matrices of f_a. In the linear case it is exactly
            A_{k}.
        """

        raise NotImplemented("f_a is not implemented!")

    def Ak(self, k, m, P):  # returns state iteration matrix
        """
        function (k, m, P) return Jacobian of dynamic function, it is passed
        into p_a.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        """

        raise NotImplemented("Ak is not implemented!")

    def Qk(self, k):
        """
        function (k). Returns noise matrix of dynamic model on iteration k.
                k (iteration number). starts at 0
        """
        raise NotImplemented("Qk is not implemented!")

    def Q_srk(self, k):
        """
        function (k). Returns the square root of noise matrix of dynamic model
        on iteration k.

        k (iteration number). starts at 0

        This function is implemented to use SVD prediction step.
        """

        raise NotImplemented("Q_srk is not implemented!")

    def dAk(self, k):
        """
        function (k). Returns the derivative of A on iteration k.
                k (iteration number). starts at 0
        """
        raise NotImplemented("dAk is not implemented!")

    def dQk(self, k):
        """
        function (k). Returns the derivative of Q on iteration k.
                k (iteration number). starts at 0
        """
        raise NotImplemented("dQk is not implemented!")

    def reset(self, compute_derivatives=False):
        """
        Return the state of this object to the beginning of iteration
        (to k eq. 0).
        """

        raise NotImplemented("reset is not implemented!")


if use_cython:
    Dynamic_Callables_Class = state_space_cython.Dynamic_Callables_Cython
else:
    Dynamic_Callables_Class = Dynamic_Callables_Python


class Measurement_Callables_Python(object):
    def f_h(self, k, m_pred, Hk):
        """
        function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number), starts at 0
            x_{k} state
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly
            H_{k}.
        """

        raise NotImplemented("f_a is not implemented!")

    def Hk(self, k, m_pred, P_pred):  # returns state iteration matrix
        """
        function (k, m, P) return Jacobian of measurement function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        """

        raise NotImplemented("Hk is not implemented!")

    def Rk(self, k):
        """
        function (k). Returns noise matrix of measurement equation
            on iteration k.
            k (iteration number). starts at 0
        """
        raise NotImplemented("Rk is not implemented!")

    def R_isrk(self, k):
        """
        function (k). Returns the square root of the noise matrix of
            measurement equation on iteration k.
            k (iteration number). starts at 0

        This function is implemented to use SVD prediction step.
        """

        raise NotImplemented("Q_srk is not implemented!")

    def dHk(self, k):
        """
        function (k). Returns the derivative of H on iteration k.
                k (iteration number). starts at 0
        """
        raise NotImplemented("dAk is not implemented!")

    def dRk(self, k):
        """
        function (k). Returns the derivative of R on iteration k.
                k (iteration number). starts at 0
        """
        raise NotImplemented("dQk is not implemented!")

    def reset(self, compute_derivatives=False):
        """
        Return the state of this object to the beginning of iteration
        (to k eq. 0)
        """

        raise NotImplemented("reset is not implemented!")


if use_cython:
    Measurement_Callables_Class = state_space_cython.Measurement_Callables_Cython
else:
    Measurement_Callables_Class = Measurement_Callables_Python


class R_handling_Python(Measurement_Callables_Class):
    """
    The calss handles noise matrix R.
    """

    def __init__(self, R, index, R_time_var_index, unique_R_number, dR=None):
        """
        Input:
        ---------------
        R - array with noise on various steps. The result of preprocessing
            the noise input.

        index - for each step of Kalman filter contains the corresponding index
                in the array.

        R_time_var_index - another index in the array R. Computed earlier and
                            is passed here.

        unique_R_number - number of unique noise matrices below which square
                roots are cached and above which they are computed each time.

        dR: 3D array[:, :, param_num]
            derivative of R. Derivative is supported only when R do not change
            over time

        Output:
        --------------
        Object which has two necessary functions:
            f_R(k)
            inv_R_square_root(k)
        """
        self.R = R
        self.index = np.asarray(index, np.int_)
        self.R_time_var_index = int(R_time_var_index)
        self.dR = dR

        if len(np.unique(index)) > unique_R_number:
            self.svd_each_time = True
        else:
            self.svd_each_time = False

        self.R_square_root = {}

    def Rk(self, k):
        return self.R[:, :, int(self.index[self.R_time_var_index, k])]

    def dRk(self, k):
        if self.dR is None:
            raise ValueError("dR derivative is None")

        return self.dR  # the same dirivative on each iteration

    def R_isrk(self, k):
        """
        Function returns the inverse square root of R matrix on step k.
        """
        ind = int(self.index[self.R_time_var_index, k])
        R = self.R[:, :, ind]

        if R.shape[0] == 1:  # 1-D case handle simplier. No storage
            # of the result, just compute it each time.
            inv_square_root = np.sqrt(1.0 / R)
        else:
            if self.svd_each_time:
                (U, S, Vh) = sp.linalg.svd(
                    R,
                    full_matrices=False,
                    compute_uv=True,
                    overwrite_a=False,
                    check_finite=True,
                )

                inv_square_root = U * 1.0 / np.sqrt(S)
            else:
                if ind in self.R_square_root:
                    inv_square_root = self.R_square_root[ind]
                else:
                    (U, S, Vh) = sp.linalg.svd(
                        R,
                        full_matrices=False,
                        compute_uv=True,
                        overwrite_a=False,
                        check_finite=True,
                    )

                    inv_square_root = U * 1.0 / np.sqrt(S)

                    self.R_square_root[ind] = inv_square_root

        return inv_square_root


if use_cython:
    R_handling_Class = state_space_cython.R_handling_Cython
else:
    R_handling_Class = R_handling_Python


class Std_Measurement_Callables_Python(R_handling_Class):
    def __init__(
        self,
        H,
        H_time_var_index,
        R,
        index,
        R_time_var_index,
        unique_R_number,
        dH=None,
        dR=None,
    ):
        super(Std_Measurement_Callables_Python, self).__init__(
            R, index, R_time_var_index, unique_R_number, dR
        )

        self.H = H
        self.H_time_var_index = int(H_time_var_index)
        self.dH = dH

    def f_h(self, k, m, H):
        """
        function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number), starts at 0
            x_{k} state
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly
            H_{k}.
        """

        return np.dot(H, m)

    def Hk(self, k, m_pred, P_pred):  # returns state iteration matrix
        """
        function (k, m, P) return Jacobian of measurement function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        """

        return self.H[:, :, int(self.index[self.H_time_var_index, k])]

    def dHk(self, k):
        if self.dH is None:
            raise ValueError("dH derivative is None")

        return self.dH  # the same dirivative on each iteration


if use_cython:
    Std_Measurement_Callables_Class = (
        state_space_cython.Std_Measurement_Callables_Cython
    )
else:
    Std_Measurement_Callables_Class = Std_Measurement_Callables_Python


class Q_handling_Python(Dynamic_Callables_Class):
    def __init__(self, Q, index, Q_time_var_index, unique_Q_number, dQ=None):
        """
        Input:
        ---------------
        R - array with noise on various steps. The result of preprocessing
            the noise input.

        index - for each step of Kalman filter contains the corresponding index
                in the array.

        R_time_var_index - another index in the array R. Computed earlier and
                            passed here.

        unique_R_number - number of unique noise matrices below which square
            roots are cached and above which they are computed each time.

        dQ: 3D array[:, :, param_num]
            derivative of Q. Derivative is supported only when Q do not
            change over time

        Output:
        --------------
        Object which has two necessary functions:
            f_R(k)
            inv_R_square_root(k)
        """

        self.Q = Q
        self.index = np.asarray(index, np.int_)
        self.Q_time_var_index = Q_time_var_index
        self.dQ = dQ

        if len(np.unique(index)) > unique_Q_number:
            self.svd_each_time = True
        else:
            self.svd_each_time = False

        self.Q_square_root = {}

    def Qk(self, k):
        """
        function (k). Returns noise matrix of dynamic model on iteration k.
                k (iteration number). starts at 0
        """
        return self.Q[:, :, self.index[self.Q_time_var_index, k]]

    def dQk(self, k):
        if self.dQ is None:
            raise ValueError("dQ derivative is None")

        return self.dQ  # the same dirivative on each iteration

    def Q_srk(self, k):
        """
        function (k). Returns the square root of noise matrix of dynamic model
                      on iteration k.
            k (iteration number). starts at 0

        This function is implemented to use SVD prediction step.
        """
        ind = self.index[self.Q_time_var_index, k]
        Q = self.Q[:, :, ind]

        if Q.shape[0] == 1:  # 1-D case handle simplier. No storage
            # of the result, just compute it each time.
            square_root = np.sqrt(Q)
        else:
            if self.svd_each_time:
                (U, S, Vh) = sp.linalg.svd(
                    Q,
                    full_matrices=False,
                    compute_uv=True,
                    overwrite_a=False,
                    check_finite=True,
                )

                square_root = U * np.sqrt(S)
            else:
                if ind in self.Q_square_root:
                    square_root = self.Q_square_root[ind]
                else:
                    (U, S, Vh) = sp.linalg.svd(
                        Q,
                        full_matrices=False,
                        compute_uv=True,
                        overwrite_a=False,
                        check_finite=True,
                    )

                    square_root = U * np.sqrt(S)

                    self.Q_square_root[ind] = square_root

        return square_root


if use_cython:
    Q_handling_Class = state_space_cython.Q_handling_Cython
else:
    Q_handling_Class = Q_handling_Python


class Std_Dynamic_Callables_Python(Q_handling_Class):
    def __init__(
        self,
        A,
        A_time_var_index,
        Q,
        index,
        Q_time_var_index,
        unique_Q_number,
        dA=None,
        dQ=None,
    ):
        super(Std_Dynamic_Callables_Python, self).__init__(
            Q, index, Q_time_var_index, unique_Q_number, dQ
        )

        self.A = A
        self.A_time_var_index = np.asarray(A_time_var_index, np.int_)
        self.dA = dA

    def f_a(self, k, m, A):
        """
        f_a: function (k, x_{k-1}, A_{k}). Dynamic function.
        k (iteration number), starts at 0
        x_{k-1} State from the previous step
        A_{k} Jacobian matrices of f_a. In the linear case it is exactly
        A_{k}.
        """
        return np.dot(A, m)

    def Ak(self, k, m_pred, P_pred):  # returns state iteration matrix
        """
        function (k, m, P) return Jacobian of measurement function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        """

        return self.A[:, :, self.index[self.A_time_var_index, k]]

    def dAk(self, k):
        if self.dA is None:
            raise ValueError("dA derivative is None")

        return self.dA  # the same dirivative on each iteration

    def reset(self, compute_derivatives=False):
        """
        Return the state of this object to the beginning of iteration
        (to k eq. 0)
        """

        return self


if use_cython:
    Std_Dynamic_Callables_Class = state_space_cython.Std_Dynamic_Callables_Cython
else:
    Std_Dynamic_Callables_Class = Std_Dynamic_Callables_Python


class AddMethodToClass(object):
    def __init__(self, func=None, tp="staticmethod"):
        """
        Input:
        --------------
        func: function to add
        tp: string
        Type of the method: normal, staticmethod, classmethod
        """
        if func is None:
            raise ValueError("Function can not be None")

        self.func = func
        self.tp = tp

    def __get__(self, obj, klass=None, *args, **kwargs):
        if self.tp == "staticmethod":
            return self.func
        elif self.tp == "normal":

            def newfunc(obj, *args, **kwargs):
                return self.func

        elif self.tp == "classmethod":

            def newfunc(klass, *args, **kwargs):
                return self.func

        return newfunc


class DescreteStateSpaceMeta(type):
    """
    Substitute necessary methods from cython.
    """

    def __new__(typeclass, name, bases, attributes):
        """
        After thos method the class object is created
        """

        if use_cython:
            if "_kalman_prediction_step_SVD" in attributes:
                attributes["_kalman_prediction_step_SVD"] = AddMethodToClass(
                    state_space_cython._kalman_prediction_step_SVD_Cython
                )

            if "_kalman_update_step_SVD" in attributes:
                attributes["_kalman_update_step_SVD"] = AddMethodToClass(
                    state_space_cython._kalman_update_step_SVD_Cython
                )

            if "_cont_discr_kalman_filter_raw" in attributes:
                attributes["_cont_discr_kalman_filter_raw"] = AddMethodToClass(
                    state_space_cython._cont_discr_kalman_filter_raw_Cython
                )

        return super(DescreteStateSpaceMeta, typeclass).__new__(
            typeclass, name, bases, attributes
        )


class DescreteStateSpace(object):
    """
    This class implents state-space inference for linear and non-linear
    state-space models.
    Linear models are:
    x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
    y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})

    Nonlinear:
    x_{k} =  f_a(k, x_{k-1}, A_{k}) + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
    y_{k} =  f_h(k, x_{k}, H_{k}) + r_{k};           r_{k-1} ~ N(0, R_{k})
    Here f_a and f_h are some functions of k (iteration number), x_{k-1} or
    x_{k} (state value on certain iteration), A_{k} and  H_{k} - Jacobian
    matrices of f_a and f_h respectively. In the linear case they are exactly
    A_{k} and  H_{k}.


    Currently two nonlinear Gaussian filter algorithms are implemented:
    Extended Kalman Filter (EKF), Statistically linearized Filter (SLF), which
    implementations are very similar.

    """

    __metaclass__ = DescreteStateSpaceMeta

    @staticmethod
    def _reshape_input_data(shape, desired_dim=3):
        """
        Static function returns the column-wise shape for for an input shape.

        Input:
        --------------
            shape: tuple
                Shape of an input array, so that it is always a column.

            desired_dim: int
                desired shape of output. For Y data it should be 3
                (sample_no, dimension, ts_no). For X data - 2 (sample_no, 1)
        Output:
        --------------
            new_shape: tuple
                New shape of the measurements array. Idea is that samples are
                along dimension 0, sample dimension - dimension 1, different
                time series - dimension 2.
            old_shape: tuple or None
                If the shape has been modified, return old shape, otherwise
                None.
        """

        if len(shape) > 3:
            raise ValueError(
                """Input array is not supposed to be more
                                than 3 dimensional."""
            )

        if len(shape) > desired_dim:
            raise ValueError("Input array shape is more than desired shape.")
        elif len(shape) == 1:
            if desired_dim == 3:
                return ((shape[0], 1, 1), shape)  # last dimension is the
                # time serime_series_no
            elif desired_dim == 2:
                return ((shape[0], 1), shape)

        elif len(shape) == 2:
            if desired_dim == 3:
                return (
                    ((shape[1], 1, 1), shape)
                    if (shape[0] == 1)
                    else ((shape[0], shape[1], 1), shape)
                )  # convert to column
                # vector
            elif desired_dim == 2:
                return (
                    ((shape[1], 1), shape)
                    if (shape[0] == 1)
                    else ((shape[0], shape[1]), None)
                )  # convert to column vector

        else:  # len(shape) == 3
            return (shape, None)  # do nothing

    @classmethod
    def kalman_filter(
        cls,
        p_A,
        p_Q,
        p_H,
        p_R,
        Y,
        index=None,
        m_init=None,
        P_init=None,
        p_kalman_filter_type="regular",
        calc_log_likelihood=False,
        calc_grad_log_likelihood=False,
        grad_params_no=None,
        grad_calc_params=None,
    ):
        """
        This function implements the basic Kalman Filter algorithm
        These notations for the State-Space model are assumed:
            x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})

        Returns estimated filter distributions x_{k} ~ N(m_{k}, P(k))

        Current Features:
        ----------------------------------------
        1) The function generaly do not modify the passed parameters. If
        it happens then it is an error. There are several exeprions: scalars
        can be modified into a matrix, in some rare cases shapes of
        the derivatives matrices may be changed, it is ignored for now.

        2) Copies of p_A, p_Q, index are created in memory to be used later
        in smoother. References to copies are kept in "matrs_for_smoother"
        return parameter.

        3) Function support "multiple time series mode" which means that exactly
        the same State-Space model is used to filter several sets of measurements.
        In this case third dimension of Y should include these state-space measurements
        Log_likelihood and Grad_log_likelihood have the corresponding dimensions then.

        4) Calculation of Grad_log_likelihood is not supported if matrices A,Q,
        H, or R changes over time. (later may be changed)

        5) Measurement may include missing values. In this case update step is
        not done for this measurement. (later may be changed)

        Input:
        -----------------

        p_A: scalar, square matrix, 3D array
            A_{k} in the model. If matrix then A_{k} = A - constant.
            If it is 3D array then A_{k} = p_A[:,:, index[0,k]]

        p_Q: scalar, square symmetric matrix, 3D array
            Q_{k-1} in the model. If matrix then Q_{k-1} = Q - constant.
            If it is 3D array then Q_{k-1} = p_Q[:,:, index[1,k]]

        p_H: scalar, matrix (measurement_dim, state_dim) , 3D array
            H_{k} in the model. If matrix then H_{k} = H - constant.
            If it is 3D array then H_{k} = p_Q[:,:, index[2,k]]

        p_R: scalar, square symmetric matrix, 3D array
            R_{k} in the model. If matrix then R_{k} = R - constant.
            If it is 3D array then R_{k} = p_R[:,:, index[3,k]]

        Y: matrix or vector or 3D array
            Data. If Y is matrix then samples are along 0-th dimension and
            features along the 1-st. If 3D array then third dimension
            correspond to "multiple time series mode".

        index: vector
            Which indices (on 3-rd dimension) from arrays p_A, p_Q,p_H, p_R to use
            on every time step. If this parameter is None then it is assumed
            that p_A, p_Q, p_H, p_R do not change over time and indices are not needed.
            index[0,:] - correspond to A, index[1,:] - correspond to Q
            index[2,:] - correspond to H, index[3,:] - correspond to R.
            If index.shape[0] == 1, it is assumed that indides for all matrices
            are the same.

        m_init: vector or matrix
            Initial distribution mean. If None it is assumed to be zero.
            For "multiple time series mode" it is matrix, second dimension of
            which correspond to different time series. In regular case ("one
            time series mode") it is a vector.

        P_init: square symmetric matrix or scalar
            Initial covariance of the states. If the parameter is scalar
            then it is assumed that initial covariance matrix is unit matrix
            multiplied by this scalar. If None the unit matrix is used instead.
            "multiple time series mode" does not affect it, since it does not
            affect anything related to state variaces.

        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.

        calc_grad_log_likelihood: boolean
            Whether to calculate gradient of the marginal likelihood
            of the state-space model. If true then "grad_calc_params" parameter must
            provide the extra parameters for gradient calculation.

        grad_params_no: int
            If previous parameter is true, then this parameters gives the
            total number of parameters in the gradient.

        grad_calc_params: dictionary
            Dictionary with derivatives of model matrices with respect
            to parameters "dA", "dQ", "dH", "dR", "dm_init", "dP_init".
            They can be None, in this case zero matrices (no dependence on parameters)
            is assumed. If there is only one parameter then third dimension is
            automatically added.

        Output:
        --------------

        M: (no_steps+1,state_dim) matrix or (no_steps+1,state_dim, time_series_no) 3D array
            Filter estimates of the state means. In the extra step the initial
            value is included. In the "multiple time series mode" third dimension
            correspond to different timeseries.

        P: (no_steps+1, state_dim, state_dim) 3D array
            Filter estimates of the state covariances. In the extra step the initial
            value is included.

        log_likelihood: double or (1, time_series_no) 3D array.
            If the parameter calc_log_likelihood was set to true, return
            logarithm of marginal likelihood of the state-space model. If
            the parameter was false, return None. In the "multiple time series mode" it is a vector
            providing log_likelihood for each time series.

        grad_log_likelihood: column vector or (grad_params_no, time_series_no) matrix
            If calc_grad_log_likelihood is true, return gradient of log likelihood
            with respect to parameters. It returns it column wise, so in
            "multiple time series mode" gradients for each time series is in the
            corresponding column.

        matrs_for_smoother: dict
            Dictionary with model functions for smoother. The intrinsic model
            functions are computed in this functions and they are returned to
            use in smoother for convenience. They are: 'p_a', 'p_f_A', 'p_f_Q'
            The dictionary contains the same fields.
        """

        # import pdb; pdb.set_trace()

        # Parameters checking ->
        # index
        p_A = np.atleast_1d(p_A)
        p_Q = np.atleast_1d(p_Q)
        p_H = np.atleast_1d(p_H)
        p_R = np.atleast_1d(p_R)

        # Reshape and check measurements:
        Y.shape, old_Y_shape = cls._reshape_input_data(Y.shape)
        measurement_dim = Y.shape[1]
        time_series_no = Y.shape[2]  # multiple time series mode

        if (
            ((len(p_A.shape) == 3) and (len(p_A.shape[2]) != 1))
            or ((len(p_Q.shape) == 3) and (len(p_Q.shape[2]) != 1))
            or ((len(p_H.shape) == 3) and (len(p_H.shape[2]) != 1))
            or ((len(p_R.shape) == 3) and (len(p_R.shape[2]) != 1))
        ):
            model_matrices_chage_with_time = True
        else:
            model_matrices_chage_with_time = False

        # Check index
        old_index_shape = None
        if index is None:
            if (
                (len(p_A.shape) == 3)
                or (len(p_Q.shape) == 3)
                or (len(p_H.shape) == 3)
                or (len(p_R.shape) == 3)
            ):
                raise ValueError(
                    "Parameter index can not be None for time varying matrices (third dimension is present)"
                )
            else:  # matrices do not change in time, so form dummy zero indices.
                index = np.zeros((1, Y.shape[0]))
        else:
            if len(index.shape) == 1:
                index.shape = (1, index.shape[0])
                old_index_shape = (index.shape[0],)

            if index.shape[1] != Y.shape[0]:
                raise ValueError(
                    "Number of measurements must be equal the number of A_{k}, Q_{k}, H_{k}, R_{k}"
                )

        if index.shape[0] == 1:
            A_time_var_index = 0
            Q_time_var_index = 0
            H_time_var_index = 0
            R_time_var_index = 0
        elif index.shape[0] == 4:
            A_time_var_index = 0
            Q_time_var_index = 1
            H_time_var_index = 2
            R_time_var_index = 3
        else:
            raise ValueError("First Dimension of index must be either 1 or 4.")

        state_dim = p_A.shape[0]
        # Check and make right shape for model matrices. On exit they all are 3 dimensional. Last dimension
        # correspond to change in time.
        (p_A, old_A_shape) = cls._check_SS_matrix(
            p_A, state_dim, measurement_dim, which="A"
        )
        (p_Q, old_Q_shape) = cls._check_SS_matrix(
            p_Q, state_dim, measurement_dim, which="Q"
        )
        (p_H, old_H_shape) = cls._check_SS_matrix(
            p_H, state_dim, measurement_dim, which="H"
        )
        (p_R, old_R_shape) = cls._check_SS_matrix(
            p_R, state_dim, measurement_dim, which="R"
        )

        # m_init
        if m_init is None:
            m_init = np.zeros((state_dim, time_series_no))
        else:
            m_init = np.atleast_2d(m_init).T

        # P_init
        if P_init is None:
            P_init = np.eye(state_dim)
        elif not isinstance(P_init, Iterable):  # scalar
            P_init = P_init * np.eye(state_dim)

        if p_kalman_filter_type not in ("regular", "svd"):
            raise ValueError("Kalman filer type neither 'regular nor 'svd'.")

        # Functions to pass to the kalman_filter algorithm:
        # Parameters:
        # k - number of Kalman filter iteration
        # m - vector for calculating matrices. Required for EKF. Not used here.

        c_p_A = (
            p_A.copy()
        )  # create a copy because this object is passed to the smoother
        c_p_Q = (
            p_Q.copy()
        )  # create a copy because this object is passed to the smoother
        c_index = (
            index.copy()
        )  # create a copy because this object is passed to the smoother

        if calc_grad_log_likelihood:
            if model_matrices_chage_with_time:
                raise ValueError(
                    "When computing likelihood gradient A and Q can not change over time."
                )

            dA = cls._check_grad_state_matrices(
                grad_calc_params.get("dA"), state_dim, grad_params_no, which="dA"
            )
            dQ = cls._check_grad_state_matrices(
                grad_calc_params.get("dQ"), state_dim, grad_params_no, which="dQ"
            )
            dH = cls._check_grad_measurement_matrices(
                grad_calc_params.get("dH"),
                state_dim,
                grad_params_no,
                measurement_dim,
                which="dH",
            )
            dR = cls._check_grad_measurement_matrices(
                grad_calc_params.get("dR"),
                state_dim,
                grad_params_no,
                measurement_dim,
                which="dR",
            )

            dm_init = grad_calc_params.get("dm_init")
            if dm_init is None:
                # multiple time series mode. Keep grad_params always as a last dimension
                dm_init = np.zeros((state_dim, time_series_no, grad_params_no))

            dP_init = grad_calc_params.get("dP_init")
            if dP_init is None:
                dP_init = np.zeros((state_dim, state_dim, grad_params_no))
        else:
            dA = None
            dQ = None
            dH = None
            dR = None
            dm_init = None
            dP_init = None

        dynamic_callables = Std_Dynamic_Callables_Class(
            c_p_A, A_time_var_index, c_p_Q, c_index, Q_time_var_index, 20, dA, dQ
        )
        measurement_callables = Std_Measurement_Callables_Class(
            p_H, H_time_var_index, p_R, index, R_time_var_index, 20, dH, dR
        )

        (
            M,
            P,
            log_likelihood,
            grad_log_likelihood,
            dynamic_callables,
        ) = cls._kalman_algorithm_raw(
            state_dim,
            dynamic_callables,
            measurement_callables,
            Y,
            m_init,
            P_init,
            p_kalman_filter_type=p_kalman_filter_type,
            calc_log_likelihood=calc_log_likelihood,
            calc_grad_log_likelihood=calc_grad_log_likelihood,
            grad_params_no=grad_params_no,
            dm_init=dm_init,
            dP_init=dP_init,
        )

        # restore shapes so that input parameters are unchenged
        if old_index_shape is not None:
            index.shape = old_index_shape

        if old_Y_shape is not None:
            Y.shape = old_Y_shape

        if old_A_shape is not None:
            p_A.shape = old_A_shape

        if old_Q_shape is not None:
            p_Q.shape = old_Q_shape

        if old_H_shape is not None:
            p_H.shape = old_H_shape

        if old_R_shape is not None:
            p_R.shape = old_R_shape
        # Return values

        return (M, P, log_likelihood, grad_log_likelihood, dynamic_callables)

    @classmethod
    def extended_kalman_filter(
        cls,
        p_state_dim,
        p_a,
        p_f_A,
        p_f_Q,
        p_h,
        p_f_H,
        p_f_R,
        Y,
        m_init=None,
        P_init=None,
        calc_log_likelihood=False,
    ):
        """
        Extended Kalman Filter

        Input:
        -----------------

        p_state_dim: integer

        p_a: if None - the function from the linear model is assumed. No non-
             linearity in the dynamic is assumed.

             function (k, x_{k-1}, A_{k}). Dynamic function.
             k: (iteration number),
             x_{k-1}: (previous state)
             x_{k}: Jacobian matrices of f_a. In the linear case it is exactly A_{k}.

        p_f_A: matrix - in this case function which returns this matrix is assumed.
               Look at this parameter description in kalman_filter function.

               function (k, m, P) return Jacobian of dynamic function, it is
               passed into p_a.

               k: (iteration number),
               m: point where Jacobian is evaluated
               P: parameter for Jacobian, usually covariance matrix.

        p_f_Q: matrix. In this case function which returns this matrix is asumed.
               Look at this parameter description in kalman_filter function.

               function (k). Returns noise matrix of dynamic model on iteration k.
               k: (iteration number).

        p_h: if None - the function from the linear measurement model is assumed.
             No nonlinearity in the measurement is assumed.

             function (k, x_{k}, H_{k}). Measurement function.
             k: (iteration number),
             x_{k}: (current state)
             H_{k}: Jacobian matrices of f_h. In the linear case it is exactly H_{k}.

        p_f_H: matrix - in this case function which returns this matrix is assumed.
               function (k, m, P) return Jacobian of dynamic function, it is
               passed into p_h.
               k: (iteration number),
               m: point where Jacobian is evaluated
               P: parameter for Jacobian, usually covariance matrix.

        p_f_R: matrix. In this case function which returns this matrix is asumed.
               function (k). Returns noise matrix of measurement equation
               on iteration k.
               k: (iteration number).

        Y: matrix or vector
            Data. If Y is matrix then samples are along 0-th dimension and
            features along the 1-st. May have missing values.

        p_mean: vector
            Initial distribution mean. If None it is assumed to be zero

        P_init: square symmetric matrix or scalar
            Initial covariance of the states. If the parameter is scalar
            then it is assumed that initial covariance matrix is unit matrix
            multiplied by this scalar. If None the unit matrix is used instead.

        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.
        """

        # Y
        Y.shape, old_Y_shape = cls._reshape_input_data(Y.shape)

        # m_init
        if m_init is None:
            m_init = np.zeros((p_state_dim, 1))
        else:
            m_init = np.atleast_2d(m_init).T

        # P_init
        if P_init is None:
            P_init = np.eye(p_state_dim)
        elif not isinstance(P_init, Iterable):  # scalar
            P_init = P_init * np.eye(p_state_dim)

        if p_a is None:
            p_a = lambda k, m, A: np.dot(A, m)

        old_A_shape = None
        if not isinstance(p_f_A, types.FunctionType):  # not a function but array
            p_f_A = np.atleast_1d(p_f_A)
            (p_A, old_A_shape) = cls._check_A_matrix(p_f_A)

            p_f_A = lambda k, m, P: p_A[:, :, 0]  # make function
        else:
            if p_f_A(1, m_init, P_init).shape[0] != m_init.shape[0]:
                raise ValueError("p_f_A function returns matrix of wrong size")

        old_Q_shape = None
        if not isinstance(p_f_Q, types.FunctionType):  # not a function but array
            p_f_Q = np.atleast_1d(p_f_Q)
            (p_Q, old_Q_shape) = cls._check_Q_matrix(p_f_Q)

            p_f_Q = lambda k: p_Q[:, :, 0]  # make function
        else:
            if p_f_Q(1).shape[0] != m_init.shape[0]:
                raise ValueError("p_f_Q function returns matrix of wrong size")

        if p_h is None:
            lambda k, m, H: np.dot(H, m)

        old_H_shape = None
        if not isinstance(p_f_H, types.FunctionType):  # not a function but array
            p_f_H = np.atleast_1d(p_f_H)
            (p_H, old_H_shape) = cls._check_H_matrix(p_f_H)

            p_f_H = lambda k, m, P: p_H  # make function
        else:
            if p_f_H(1, m_init, P_init).shape[0] != Y.shape[1]:
                raise ValueError("p_f_H function returns matrix of wrong size")

        old_R_shape = None
        if not isinstance(p_f_R, types.FunctionType):  # not a function but array
            p_f_R = np.atleast_1d(p_f_R)
            (p_R, old_R_shape) = cls._check_H_matrix(p_f_R)

            p_f_R = lambda k: p_R  # make function
        else:
            if p_f_R(1).shape[0] != m_init.shape[0]:
                raise ValueError("p_f_R function returns matrix of wrong size")

        #        class dynamic_callables_class(Dynamic_Model_Callables):
        #
        #            Ak =
        #            Qk =

        class measurement_callables_class(R_handling_Class):
            def __init__(self, R, index, R_time_var_index, unique_R_number):
                super(measurement_callables_class, self).__init__(
                    R, index, R_time_var_index, unique_R_number
                )

            Hk = AddMethodToClass(f_H)
            f_h = AddMethodToClass(f_hl)

        (M, P, log_likelihood, grad_log_likelihood) = cls._kalman_algorithm_raw(
            p_state_dim,
            p_a,
            p_f_A,
            p_f_Q,
            p_h,
            p_f_H,
            p_f_R,
            Y,
            m_init,
            P_init,
            calc_log_likelihood,
            calc_grad_log_likelihood=False,
            grad_calc_params=None,
        )

        if old_Y_shape is not None:
            Y.shape = old_Y_shape

        if old_A_shape is not None:
            p_A.shape = old_A_shape

        if old_Q_shape is not None:
            p_Q.shape = old_Q_shape

        if old_H_shape is not None:
            p_H.shape = old_H_shape

        if old_R_shape is not None:
            p_R.shape = old_R_shape

        return (M, P)

    @classmethod
    def _kalman_algorithm_raw(
        cls,
        state_dim,
        p_dynamic_callables,
        p_measurement_callables,
        Y,
        m_init,
        P_init,
        p_kalman_filter_type="regular",
        calc_log_likelihood=False,
        calc_grad_log_likelihood=False,
        grad_params_no=None,
        dm_init=None,
        dP_init=None,
    ):
        """
        General nonlinear filtering algorithm for inference in the state-space
        model:

        x_{k} =  f_a(k, x_{k-1}, A_{k}) + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
        y_{k} =  f_h(k, x_{k}, H_{k}) + r_{k};           r_{k-1} ~ N(0, R_{k})

        Returns estimated filter distributions x_{k} ~ N(m_{k}, P(k))

        Current Features:
        ----------------------------------------

        1) Function support "multiple time series mode" which means that exactly
        the same State-Space model is used to filter several sets of measurements.
        In this case third dimension of Y should include these state-space measurements
        Log_likelihood and Grad_log_likelihood have the corresponding dimensions then.

        2) Measurement may include missing values. In this case update step is
        not done for this measurement. (later may be changed)

        Input:
        -----------------
        state_dim: int
            Demensionality of the states

        p_a: function (k, x_{k-1}, A_{k}). Dynamic function.
            k (iteration number),
            x_{k-1}
            A_{k} Jacobian matrices of f_a. In the linear case it is exactly A_{k}.

        p_f_A: function (k, m, P) return Jacobian of dynamic function, it is
            passed into p_a.
            k (iteration number),
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.

        p_f_Q: function (k). Returns noise matrix of dynamic model on iteration k.
            k (iteration number).

        p_h: function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number),
            x_{k}
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly H_{k}.

        p_f_H: function (k, m, P) return Jacobian of dynamic function, it is
            passed into p_h.
            k (iteration number),
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.

        p_f_R: function (k). Returns noise matrix of measurement equation
            on iteration k.
            k (iteration number).

        Y: matrix or vector or 3D array
            Data. If Y is matrix then samples are along 0-th dimension and
            features along the 1-st. If 3D array then third dimension
            correspond to "multiple time series mode".

        m_init: vector or matrix
            Initial distribution mean. For "multiple time series mode"
            it is matrix, second dimension of which correspond to different
            time series. In regular case ("one time series mode") it is a
            vector.

        P_init: matrix or scalar
            Initial covariance of the states. Must be not None
            "multiple time series mode" does not affect it, since it does not
            affect anything related to state variaces.

        p_kalman_filter_type: string


        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.

        calc_grad_log_likelihood: boolean
            Whether to calculate gradient of the marginal likelihood
            of the state-space model. If true then the next parameter must
            provide the extra parameters for gradient calculation.

        grad_calc_params: dictionary
            Dictionary with derivatives of model matrices with respect
            to parameters "dA", "dQ", "dH", "dR", "dm_init", "dP_init".

        Output:
        --------------

        M: (no_steps+1,state_dim) matrix or (no_steps+1,state_dim, time_series_no) 3D array
            Filter estimates of the state means. In the extra step the initial
            value is included. In the "multiple time series mode" third dimension
            correspond to different timeseries.

        P: (no_steps+1, state_dim, state_dim) 3D array
            Filter estimates of the state covariances. In the extra step the initial
            value is included.

        log_likelihood: double or (1, time_series_no) 3D array.
            If the parameter calc_log_likelihood was set to true, return
            logarithm of marginal likelihood of the state-space model. If
            the parameter was false, return None. In the "multiple time series mode" it is a vector
            providing log_likelihood for each time series.

        grad_log_likelihood: column vector or (grad_params_no, time_series_no) matrix
            If calc_grad_log_likelihood is true, return gradient of log likelihood
            with respect to parameters. It returns it column wise, so in
            "multiple time series mode" gradients for each time series is in the
            corresponding column.

        """

        steps_no = Y.shape[0]  # number of steps in the Kalman Filter
        time_series_no = Y.shape[2]  # multiple time series mode

        # Allocate space for results
        # Mean estimations. Initial values will be included
        M = np.empty(((steps_no + 1), state_dim, time_series_no))
        M[0, :, :] = m_init  # Initialize mean values
        # Variance estimations. Initial values will be included
        P = np.empty(((steps_no + 1), state_dim, state_dim))
        P_init = 0.5 * (
            P_init + P_init.T
        )  # symmetrize initial covariance. In some ustable cases this is uiseful
        P[0, :, :] = P_init  # Initialize initial covariance matrix

        if p_kalman_filter_type == "svd":
            (U, S, Vh) = sp.linalg.svd(
                P_init,
                full_matrices=False,
                compute_uv=True,
                overwrite_a=False,
                check_finite=True,
            )
            S[(S == 0)] = 1e-17  # allows to run algorithm for singular initial variance
            P_upd = (P_init, S, U)

        log_likelihood = 0 if calc_log_likelihood else None
        grad_log_likelihood = 0 if calc_grad_log_likelihood else None

        # setting initial values for derivatives update
        dm_upd = dm_init
        dP_upd = dP_init
        # Main loop of the Kalman filter
        for k in range(0, steps_no):
            # In this loop index for new estimations is (k+1), old - (k)
            # This happened because initial values are stored at 0-th index.

            prev_mean = M[k, :, :]  # mean from the previous step

            if p_kalman_filter_type == "svd":
                m_pred, P_pred, dm_pred, dP_pred = cls._kalman_prediction_step_SVD(
                    k,
                    prev_mean,
                    P_upd,
                    p_dynamic_callables,
                    calc_grad_log_likelihood=calc_grad_log_likelihood,
                    p_dm=dm_upd,
                    p_dP=dP_upd,
                )
            else:
                m_pred, P_pred, dm_pred, dP_pred = cls._kalman_prediction_step(
                    k,
                    prev_mean,
                    P[k, :, :],
                    p_dynamic_callables,
                    calc_grad_log_likelihood=calc_grad_log_likelihood,
                    p_dm=dm_upd,
                    p_dP=dP_upd,
                )

            k_measurment = Y[k, :, :]

            if np.any(np.isnan(k_measurment)) == False:
                if p_kalman_filter_type == "svd":
                    (
                        m_upd,
                        P_upd,
                        log_likelihood_update,
                        dm_upd,
                        dP_upd,
                        d_log_likelihood_update,
                    ) = cls._kalman_update_step_SVD(
                        k,
                        m_pred,
                        P_pred,
                        p_measurement_callables,
                        k_measurment,
                        calc_log_likelihood=calc_log_likelihood,
                        calc_grad_log_likelihood=calc_grad_log_likelihood,
                        p_dm=dm_pred,
                        p_dP=dP_pred,
                    )

                #                m_upd, P_upd, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update = \
                #                cls._kalman_update_step(k,  m_pred , P_pred[0], f_h, f_H, p_R.f_R, k_measurment,
                #                        calc_log_likelihood=calc_log_likelihood,
                #                        calc_grad_log_likelihood=calc_grad_log_likelihood,
                #                        p_dm = dm_pred, p_dP = dP_pred, grad_calc_params_2 = (dH, dR))
                #
                #                (U,S,Vh) = sp.linalg.svd( P_upd,full_matrices=False, compute_uv=True,
                #                      overwrite_a=False,check_finite=True)
                #                P_upd = (P_upd, S,U)
                else:
                    (
                        m_upd,
                        P_upd,
                        log_likelihood_update,
                        dm_upd,
                        dP_upd,
                        d_log_likelihood_update,
                    ) = cls._kalman_update_step(
                        k,
                        m_pred,
                        P_pred,
                        p_measurement_callables,
                        k_measurment,
                        calc_log_likelihood=calc_log_likelihood,
                        calc_grad_log_likelihood=calc_grad_log_likelihood,
                        p_dm=dm_pred,
                        p_dP=dP_pred,
                    )

            else:
                #                if k_measurment.shape != (1,1):
                #                    raise ValueError("Nan measurements are currently not supported for \
                #                                     multidimensional output and multiple time series.")
                #                else:
                #                    m_upd = m_pred; P_upd = P_pred; dm_upd = dm_pred; dP_upd = dP_pred
                #                    log_likelihood_update = 0.0;
                #                    d_log_likelihood_update = 0.0;

                if not np.all(np.isnan(k_measurment)):
                    raise ValueError(
                        """Nan measurements are currently not supported if
                                     they are intermixed with not NaN measurements"""
                    )
                else:
                    m_upd = m_pred
                    P_upd = P_pred
                    dm_upd = dm_pred
                    dP_upd = dP_pred
                    if calc_log_likelihood:
                        log_likelihood_update = np.zeros((time_series_no,))
                    if calc_grad_log_likelihood:
                        d_log_likelihood_update = np.zeros(
                            (grad_params_no, time_series_no)
                        )

            if calc_log_likelihood:
                log_likelihood += log_likelihood_update

            if calc_grad_log_likelihood:
                grad_log_likelihood += d_log_likelihood_update

            M[k + 1, :, :] = m_upd  # separate mean value for each time series

            if p_kalman_filter_type == "svd":
                P[k + 1, :, :] = P_upd[0]
            else:
                P[k + 1, :, :] = P_upd

        # !!!Print statistics! Print sizes of matrices
        # !!!Print statistics! Print iteration time base on another boolean variable
        return (
            M,
            P,
            log_likelihood,
            grad_log_likelihood,
            p_dynamic_callables.reset(False),
        )

    @staticmethod
    def _kalman_prediction_step(
        k,
        p_m,
        p_P,
        p_dyn_model_callable,
        calc_grad_log_likelihood=False,
        p_dm=None,
        p_dP=None,
    ):
        """
        Desctrete prediction function

        Input:
            k:int
                Iteration No. Starts at 0. Total number of iterations equal to the
                number of measurements.

            p_m: matrix of size (state_dim, time_series_no)
                Mean value from the previous step. For "multiple time series mode"
                it is matrix, second dimension of which correspond to different
                time series.

            p_P:
                Covariance matrix from the previous step.

            p_dyn_model_callable: class


            calc_grad_log_likelihood: boolean
                Whether to calculate gradient of the marginal likelihood
                of the state-space model. If true then the next parameter must
                provide the extra parameters for gradient calculation.

            p_dm: 3D array (state_dim, time_series_no, parameters_no)
                Mean derivatives from the previous step. For "multiple time series mode"
                it is 3D array, second dimension of which correspond to different
                time series.

            p_dP: 3D array (state_dim, state_dim, parameters_no)
                Mean derivatives from the previous step

        Output:
        ----------------------------
        m_pred, P_pred, dm_pred, dP_pred: metrices, 3D objects
            Results of the prediction steps.

        """

        # index correspond to values from previous iteration.
        A = p_dyn_model_callable.Ak(
            k, p_m, p_P
        )  # state transition matrix (or Jacobian)
        Q = p_dyn_model_callable.Qk(k)  # state noise matrix

        # Prediction step ->
        m_pred = p_dyn_model_callable.f_a(k, p_m, A)  # predicted mean
        P_pred = A.dot(p_P).dot(A.T) + Q  # predicted variance
        # Prediction step <-

        if calc_grad_log_likelihood:
            dA_all_params = p_dyn_model_callable.dAk(
                k
            )  # derivatives of A wrt parameters
            dQ_all_params = p_dyn_model_callable.dQk(
                k
            )  # derivatives of Q wrt parameters

            param_number = p_dP.shape[2]

            # p_dm, p_dP - derivatives form the previoius step
            dm_pred = np.empty(p_dm.shape)
            dP_pred = np.empty(p_dP.shape)

            for j in range(param_number):
                dA = dA_all_params[:, :, j]
                dQ = dQ_all_params[:, :, j]

                dP = p_dP[:, :, j]
                dm = p_dm[:, :, j]
                dm_pred[:, :, j] = np.dot(dA, p_m) + np.dot(A, dm)
                # prediction step derivatives for current parameter:

                dP_pred[:, :, j] = np.dot(dA, np.dot(p_P, A.T))
                dP_pred[:, :, j] += dP_pred[:, :, j].T
                dP_pred[:, :, j] += np.dot(A, np.dot(dP, A.T)) + dQ

                dP_pred[:, :, j] = 0.5 * (
                    dP_pred[:, :, j] + dP_pred[:, :, j].T
                )  # symmetrize
        else:
            dm_pred = None
            dP_pred = None

        return m_pred, P_pred, dm_pred, dP_pred

    @staticmethod
    def _kalman_prediction_step_SVD(
        k,
        p_m,
        p_P,
        p_dyn_model_callable,
        calc_grad_log_likelihood=False,
        p_dm=None,
        p_dP=None,
    ):
        """
        Desctrete prediction function

        Input:
            k:int
                Iteration No. Starts at 0. Total number of iterations equal to the
                number of measurements.

            p_m: matrix of size (state_dim, time_series_no)
                Mean value from the previous step. For "multiple time series mode"
                it is matrix, second dimension of which correspond to different
                time series.

            p_P: tuple (Prev_cov, S, V)
                Covariance matrix from the previous step and its SVD decomposition.
                Prev_cov = V * S * V.T The tuple is (Prev_cov, S, V)

            p_dyn_model_callable: object

            calc_grad_log_likelihood: boolean
                Whether to calculate gradient of the marginal likelihood
                of the state-space model. If true then the next parameter must
                provide the extra parameters for gradient calculation.

            p_dm: 3D array (state_dim, time_series_no, parameters_no)
                Mean derivatives from the previous step. For "multiple time series mode"
                it is 3D array, second dimension of which correspond to different
                time series.

            p_dP: 3D array (state_dim, state_dim, parameters_no)
                Mean derivatives from the previous step

        Output:
        ----------------------------
        m_pred, P_pred, dm_pred, dP_pred: metrices, 3D objects
            Results of the prediction steps.

        """

        # covariance from the previous step and its SVD decomposition
        # p_prev_cov = v * S * V.T
        Prev_cov, S_old, V_old = p_P
        # p_prev_cov_tst = np.dot(p_V, (p_S * p_V).T) # reconstructed covariance from the previous step

        # index correspond to values from previous iteration.
        A = p_dyn_model_callable.Ak(
            k, p_m, Prev_cov
        )  # state transition matrix (or Jacobian)
        Q = p_dyn_model_callable.Qk(
            k
        )  # state noise matrx. This is necessary for the square root calculation (next step)
        Q_sr = p_dyn_model_callable.Q_srk(k)
        # Prediction step ->
        m_pred = p_dyn_model_callable.f_a(k, p_m, A)  # predicted mean

        # coavariance prediction have changed:
        svd_1_matr = np.vstack(((np.sqrt(S_old) * np.dot(A, V_old)).T, Q_sr.T))
        (U, S, Vh) = sp.linalg.svd(
            svd_1_matr,
            full_matrices=False,
            compute_uv=True,
            overwrite_a=False,
            check_finite=True,
        )

        # predicted variance computed by the regular method. For testing
        # P_pred_tst = A.dot(Prev_cov).dot(A.T) + Q
        V_new = Vh.T
        S_new = S**2

        P_pred = np.dot(V_new * S_new, V_new.T)  # prediction covariance
        P_pred = (P_pred, S_new, Vh.T)
        # Prediction step <-

        # derivatives
        if calc_grad_log_likelihood:
            dA_all_params = p_dyn_model_callable.dAk(
                k
            )  # derivatives of A wrt parameters
            dQ_all_params = p_dyn_model_callable.dQk(
                k
            )  # derivatives of Q wrt parameters

            param_number = p_dP.shape[2]

            # p_dm, p_dP - derivatives form the previoius step
            dm_pred = np.empty(p_dm.shape)
            dP_pred = np.empty(p_dP.shape)

            for j in range(param_number):
                dA = dA_all_params[:, :, j]
                dQ = dQ_all_params[:, :, j]

                # dP = p_dP[:,:,j]
                # dm = p_dm[:,:,j]
                dm_pred[:, :, j] = np.dot(dA, p_m) + np.dot(A, p_dm[:, :, j])
                # prediction step derivatives for current parameter:

                dP_pred[:, :, j] = np.dot(dA, np.dot(Prev_cov, A.T))
                dP_pred[:, :, j] += dP_pred[:, :, j].T
                dP_pred[:, :, j] += np.dot(A, np.dot(p_dP[:, :, j], A.T)) + dQ

                dP_pred[:, :, j] = 0.5 * (
                    dP_pred[:, :, j] + dP_pred[:, :, j].T
                )  # symmetrize
        else:
            dm_pred = None
            dP_pred = None

        return m_pred, P_pred, dm_pred, dP_pred

    @staticmethod
    def _kalman_update_step(
        k,
        p_m,
        p_P,
        p_meas_model_callable,
        measurement,
        calc_log_likelihood=False,
        calc_grad_log_likelihood=False,
        p_dm=None,
        p_dP=None,
    ):
        """
        Input:

        k: int
              Iteration No. Starts at 0. Total number of iterations equal to the
              number of measurements.

        m_P: matrix of size (state_dim, time_series_no)
             Mean value from the previous step. For "multiple time series mode"
                it is matrix, second dimension of which correspond to different
                time series.

        p_P:
             Covariance matrix from the prediction step.

        p_meas_model_callable: object

        measurement: (measurement_dim, time_series_no) matrix
            One measurement used on the current update step. For
            "multiple time series mode" it is matrix, second dimension of
            which correspond to different time series.

        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.

        calc_grad_log_likelihood: boolean
                Whether to calculate gradient of the marginal likelihood
                of the state-space model. If true then the next parameter must
                provide the extra parameters for gradient calculation.

        p_dm: 3D array (state_dim, time_series_no, parameters_no)
                Mean derivatives from the prediction step. For "multiple time series mode"
                it is 3D array, second dimension of which correspond to different
                time series.

        p_dP: array
            Covariance derivatives from the prediction step.

        Output:
        ----------------------------
        m_upd, P_upd, dm_upd, dP_upd: metrices, 3D objects
            Results of the prediction steps.

        log_likelihood_update: double or 1D array
            Update to the log_likelihood from this step

        d_log_likelihood_update: (grad_params_no, time_series_no) matrix
            Update to the gradient of log_likelihood, "multiple time series mode"
            adds extra columns to the gradient.

        """
        # import pdb; pdb.set_trace()

        m_pred = p_m  # from prediction step
        P_pred = p_P  # from prediction step

        H = p_meas_model_callable.Hk(k, m_pred, P_pred)
        R = p_meas_model_callable.Rk(k)

        time_series_no = p_m.shape[1]  # number of time serieses

        log_likelihood_update = None
        dm_upd = None
        dP_upd = None
        d_log_likelihood_update = None
        # Update step (only if there is data)
        # if not np.any(np.isnan(measurement)): # TODO: if some dimensions are missing, do properly computations for other.
        v = measurement - p_meas_model_callable.f_h(k, m_pred, H)
        S = H.dot(P_pred).dot(H.T) + R
        if measurement.shape[0] == 1:  # measurements are one dimensional
            if S < 0:
                raise ValueError("Kalman Filter Update: S is negative step %i" % k)
                # import pdb; pdb.set_trace()

            K = P_pred.dot(H.T) / S
            if calc_log_likelihood:
                log_likelihood_update = -0.5 * (
                    np.log(2 * np.pi) + np.log(S) + v * v / S
                )
                # log_likelihood_update = log_likelihood_update[0,0] # to make int
                if np.any(
                    np.isnan(log_likelihood_update)
                ):  # some member in P_pred is None.
                    raise ValueError("Nan values in likelihood update!")
            LL = None
            islower = None
        else:
            LL, islower = linalg.cho_factor(S)
            K = linalg.cho_solve((LL, islower), H.dot(P_pred.T)).T

            if calc_log_likelihood:
                log_likelihood_update = -0.5 * (
                    v.shape[0] * np.log(2 * np.pi)
                    + 2 * np.sum(np.log(np.diag(LL)))
                    + np.sum((linalg.cho_solve((LL, islower), v)) * v, axis=0)
                )  # diagonal of v.T*S^{-1}*v

        if calc_grad_log_likelihood:
            dm_pred_all_params = p_dm  # derivativas of the prediction phase
            dP_pred_all_params = p_dP

            param_number = p_dP.shape[2]

            dH_all_params = p_meas_model_callable.dHk(k)
            dR_all_params = p_meas_model_callable.dRk(k)

            dm_upd = np.empty(dm_pred_all_params.shape)
            dP_upd = np.empty(dP_pred_all_params.shape)

            # firts dimension parameter_no, second - time series number
            d_log_likelihood_update = np.empty((param_number, time_series_no))
            for param in range(param_number):
                dH = dH_all_params[:, :, param]
                dR = dR_all_params[:, :, param]

                dm_pred = dm_pred_all_params[:, :, param]
                dP_pred = dP_pred_all_params[:, :, param]

                # Terms in the likelihood derivatives
                dv = -np.dot(dH, m_pred) - np.dot(H, dm_pred)
                dS = np.dot(dH, np.dot(P_pred, H.T))
                dS += dS.T
                dS += np.dot(H, np.dot(dP_pred, H.T)) + dR

                # TODO: maybe symmetrize dS

                # dm and dP for the next stem
                if LL is not None:  # the state vector is not a scalar
                    tmp1 = linalg.cho_solve((LL, islower), H).T
                    tmp2 = linalg.cho_solve((LL, islower), dH).T
                    tmp3 = linalg.cho_solve((LL, islower), dS).T
                else:  # the state vector is a scalar
                    tmp1 = H.T / S
                    tmp2 = dH.T / S
                    tmp3 = dS.T / S

                dK = (
                    np.dot(dP_pred, tmp1)
                    + np.dot(P_pred, tmp2)
                    - np.dot(P_pred, np.dot(tmp1, tmp3))
                )

                # terms required for the next step, save this for each parameter
                dm_upd[:, :, param] = dm_pred + np.dot(dK, v) + np.dot(K, dv)

                dP_upd[:, :, param] = -np.dot(dK, np.dot(S, K.T))
                dP_upd[:, :, param] += dP_upd[:, :, param].T
                dP_upd[:, :, param] += dP_pred - np.dot(K, np.dot(dS, K.T))

                dP_upd[:, :, param] = 0.5 * (
                    dP_upd[:, :, param] + dP_upd[:, :, param].T
                )  # symmetrize
                # computing the likelihood change for each parameter:
                if LL is not None:  # the state vector is not 1D
                    # tmp4 = linalg.cho_solve((LL,islower), dv)
                    tmp5 = linalg.cho_solve((LL, islower), v)
                else:  # the state vector is a scalar
                    # tmp4 = dv / S
                    tmp5 = v / S

                d_log_likelihood_update[param, :] = -(
                    0.5 * np.sum(np.diag(tmp3))
                    + np.sum(tmp5 * dv, axis=0)
                    - 0.5 * np.sum(tmp5 * np.dot(dS, tmp5), axis=0)
                )
                # Before
                # d_log_likelihood_update[param,0] = -(0.5*np.sum(np.diag(tmp3)) + \
                # np.dot(tmp5.T, dv) - 0.5 * np.dot(tmp5.T ,np.dot(dS, tmp5)) )

        # Compute the actual updates for mean and variance of the states.
        m_upd = m_pred + K.dot(v)

        # Covariance update and ensure it is symmetric
        P_upd = K.dot(S).dot(K.T)
        P_upd = 0.5 * (P_upd + P_upd.T)
        P_upd = P_pred - P_upd  # this update matrix is symmetric

        return (
            m_upd,
            P_upd,
            log_likelihood_update,
            dm_upd,
            dP_upd,
            d_log_likelihood_update,
        )

    @staticmethod
    def _kalman_update_step_SVD(
        k,
        p_m,
        p_P,
        p_meas_model_callable,
        measurement,
        calc_log_likelihood=False,
        calc_grad_log_likelihood=False,
        p_dm=None,
        p_dP=None,
    ):
        """
        Input:

        k: int
              Iteration No. Starts at 0. Total number of iterations equal to the
              number of measurements.

        m_P: matrix of size (state_dim, time_series_no)
             Mean value from the previous step. For "multiple time series mode"
                it is matrix, second dimension of which correspond to different
                time series.

        p_P: tuple (P_pred, S, V)
             Covariance matrix from the prediction step and its SVD decomposition.
             P_pred = V * S * V.T The tuple is (P_pred, S, V)

        p_h: function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number), starts at 0
            x_{k} state
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly H_{k}.

        p_f_H: function (k, m, P) return Jacobian of measurement function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.

        p_f_R: function (k). Returns noise matrix of measurement equation
            on iteration k.
            k (iteration number). starts at 0

        p_f_iRsr: function (k). Returns the square root of the noise matrix of
            measurement equation on iteration k.
            k (iteration number). starts at 0

        measurement: (measurement_dim, time_series_no) matrix
            One measurement used on the current update step. For
            "multiple time series mode" it is matrix, second dimension of
            which correspond to different time series.

        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.

        calc_grad_log_likelihood: boolean
                Whether to calculate gradient of the marginal likelihood
                of the state-space model. If true then the next parameter must
                provide the extra parameters for gradient calculation.

        p_dm: 3D array (state_dim, time_series_no, parameters_no)
                Mean derivatives from the prediction step. For "multiple time series mode"
                it is 3D array, second dimension of which correspond to different
                time series.

        p_dP: array
            Covariance derivatives from the prediction step.

        grad_calc_params_2: List or None
            List with derivatives. The first component is 'f_dH' - function(k)
            which returns the derivative of H. The second element is 'f_dR'
             - function(k). Function which returns the derivative of R.

        Output:
        ----------------------------
        m_upd, P_upd, dm_upd, dP_upd: metrices, 3D objects
            Results of the prediction steps.

        log_likelihood_update: double or 1D array
            Update to the log_likelihood from this step

        d_log_likelihood_update: (grad_params_no, time_series_no) matrix
            Update to the gradient of log_likelihood, "multiple time series mode"
            adds extra columns to the gradient.

        """

        # import pdb; pdb.set_trace()

        m_pred = p_m  # from prediction step
        P_pred, S_pred, V_pred = p_P  # from prediction step

        H = p_meas_model_callable.Hk(k, m_pred, P_pred)
        R = p_meas_model_callable.Rk(k)
        R_isr = p_meas_model_callable.R_isrk(
            k
        )  # square root of the inverse of R matrix

        time_series_no = p_m.shape[1]  # number of time serieses

        log_likelihood_update = None
        dm_upd = None
        dP_upd = None
        d_log_likelihood_update = None
        # Update step (only if there is data)
        # if not np.any(np.isnan(measurement)): # TODO: if some dimensions are missing, do properly computations for other.
        v = measurement - p_meas_model_callable.f_h(k, m_pred, H)

        svd_2_matr = np.vstack(
            (np.dot(R_isr.T, np.dot(H, V_pred)), np.diag(1.0 / np.sqrt(S_pred)))
        )

        (U, S, Vh) = sp.linalg.svd(
            svd_2_matr,
            full_matrices=False,
            compute_uv=True,
            overwrite_a=False,
            check_finite=True,
        )

        # P_upd = U_upd S_upd**2 U_upd.T
        U_upd = np.dot(V_pred, Vh.T)
        S_upd = (1.0 / S) ** 2

        P_upd = np.dot(U_upd * S_upd, U_upd.T)  # update covariance
        P_upd = (P_upd, S_upd, U_upd)  # tuple to pass to the next step

        # stil need to compute S and K for derivative computation
        S = H.dot(P_pred).dot(H.T) + R
        if measurement.shape[0] == 1:  # measurements are one dimensional
            if S < 0:
                raise ValueError("Kalman Filter Update: S is negative step %i" % k)
                # import pdb; pdb.set_trace()

            K = P_pred.dot(H.T) / S
            if calc_log_likelihood:
                log_likelihood_update = -0.5 * (
                    np.log(2 * np.pi) + np.log(S) + v * v / S
                )
                # log_likelihood_update = log_likelihood_update[0,0] # to make int
                if np.any(
                    np.isnan(log_likelihood_update)
                ):  # some member in P_pred is None.
                    raise ValueError("Nan values in likelihood update!")
            LL = None
            islower = None
        else:
            LL, islower = linalg.cho_factor(S)
            K = linalg.cho_solve((LL, islower), H.dot(P_pred.T)).T

            if calc_log_likelihood:
                log_likelihood_update = -0.5 * (
                    v.shape[0] * np.log(2 * np.pi)
                    + 2 * np.sum(np.log(np.diag(LL)))
                    + np.sum((linalg.cho_solve((LL, islower), v)) * v, axis=0)
                )  # diagonal of v.T*S^{-1}*v

        # Old  method of computing updated covariance (for testing) ->
        # P_upd_tst = K.dot(S).dot(K.T)
        # P_upd_tst = 0.5*(P_upd_tst + P_upd_tst.T)
        # P_upd_tst =  P_pred - P_upd_tst# this update matrix is symmetric
        # Old  method of computing updated covariance (for testing) <-

        if calc_grad_log_likelihood:
            dm_pred_all_params = p_dm  # derivativas of the prediction phase
            dP_pred_all_params = p_dP

            param_number = p_dP.shape[2]

            dH_all_params = p_meas_model_callable.dHk(k)
            dR_all_params = p_meas_model_callable.dRk(k)

            dm_upd = np.empty(dm_pred_all_params.shape)
            dP_upd = np.empty(dP_pred_all_params.shape)

            # firts dimension parameter_no, second - time series number
            d_log_likelihood_update = np.empty((param_number, time_series_no))
            for param in range(param_number):
                dH = dH_all_params[:, :, param]
                dR = dR_all_params[:, :, param]

                dm_pred = dm_pred_all_params[:, :, param]
                dP_pred = dP_pred_all_params[:, :, param]

                # Terms in the likelihood derivatives
                dv = -np.dot(dH, m_pred) - np.dot(H, dm_pred)
                dS = np.dot(dH, np.dot(P_pred, H.T))
                dS += dS.T
                dS += np.dot(H, np.dot(dP_pred, H.T)) + dR

                # TODO: maybe symmetrize dS

                # dm and dP for the next stem
                if LL is not None:  # the state vector is not a scalar
                    tmp1 = linalg.cho_solve((LL, islower), H).T
                    tmp2 = linalg.cho_solve((LL, islower), dH).T
                    tmp3 = linalg.cho_solve((LL, islower), dS).T
                else:  # the state vector is a scalar
                    tmp1 = H.T / S
                    tmp2 = dH.T / S
                    tmp3 = dS.T / S

                dK = (
                    np.dot(dP_pred, tmp1)
                    + np.dot(P_pred, tmp2)
                    - np.dot(P_pred, np.dot(tmp1, tmp3))
                )

                # terms required for the next step, save this for each parameter
                dm_upd[:, :, param] = dm_pred + np.dot(dK, v) + np.dot(K, dv)

                dP_upd[:, :, param] = -np.dot(dK, np.dot(S, K.T))
                dP_upd[:, :, param] += dP_upd[:, :, param].T
                dP_upd[:, :, param] += dP_pred - np.dot(K, np.dot(dS, K.T))

                dP_upd[:, :, param] = 0.5 * (
                    dP_upd[:, :, param] + dP_upd[:, :, param].T
                )  # symmetrize
                # computing the likelihood change for each parameter:
                if LL is not None:  # the state vector is not 1D
                    tmp5 = linalg.cho_solve((LL, islower), v)
                else:  # the state vector is a scalar
                    tmp5 = v / S

                d_log_likelihood_update[param, :] = -(
                    0.5 * np.sum(np.diag(tmp3))
                    + np.sum(tmp5 * dv, axis=0)
                    - 0.5 * np.sum(tmp5 * np.dot(dS, tmp5), axis=0)
                )
                # Before
                # d_log_likelihood_update[param,0] = -(0.5*np.sum(np.diag(tmp3)) + \
                # np.dot(tmp5.T, dv) - 0.5 * np.dot(tmp5.T ,np.dot(dS, tmp5)) )

        # Compute the actual updates for mean of the states. Variance update
        # is computed earlier.
        m_upd = m_pred + K.dot(v)

        return (
            m_upd,
            P_upd,
            log_likelihood_update,
            dm_upd,
            dP_upd,
            d_log_likelihood_update,
        )

    @staticmethod
    def _rts_smoother_update_step(
        k,
        p_m,
        p_P,
        p_m_pred,
        p_P_pred,
        p_m_prev_step,
        p_P_prev_step,
        p_dynamic_callables,
    ):
        """
        Rauch–Tung–Striebel(RTS) update step

        Input:
        -----------------------------
        k: int
              Iteration No. Starts at 0. Total number of iterations equal to the
              number of measurements.

        p_m: matrix of size (state_dim, time_series_no)
             Filter mean on step k

        p_P:  matrix of size (state_dim,state_dim)
             Filter Covariance on step k

        p_m_pred: matrix of size (state_dim, time_series_no)
             Means from the smoother prediction step.

        p_P_pred:
             Covariance from the smoother prediction step.

        p_m_prev_step
            Smoother mean from the previous step.

        p_P_prev_step:
            Smoother covariance from the previous step.

        p_f_A: function (k, m, P) return Jacobian of dynamic function, it is
            passed into p_a.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.

        """

        A = p_dynamic_callables.Ak(k, p_m, p_P)  # state transition matrix (or Jacobian)

        tmp = np.dot(A, p_P.T)
        if A.shape[0] == 1:  # 1D states
            G = tmp.T / p_P_pred  # P[:,:,k] is symmetric
        else:
            try:
                LL, islower = linalg.cho_factor(p_P_pred)
                G = linalg.cho_solve((LL, islower), tmp).T
            except:
                # It happende that p_P_pred has several near zero eigenvalues
                # hence the Cholesky method does not work.
                res = sp.linalg.lstsq(p_P_pred, tmp)
                G = res[0].T

        m_upd = p_m + G.dot(p_m_prev_step - p_m_pred)
        P_upd = p_P + G.dot(p_P_prev_step - p_P_pred).dot(G.T)

        P_upd = 0.5 * (P_upd + P_upd.T)

        return m_upd, P_upd, G

    @classmethod
    def rts_smoother(cls, state_dim, p_dynamic_callables, filter_means, filter_covars):
        """
        This function implements Rauch–Tung–Striebel(RTS) smoother algorithm
        based on the results of kalman_filter_raw.
        These notations are the same:
            x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})

        Returns estimated smoother distributions x_{k} ~ N(m_{k}, P(k))

        Input:
        --------------

        p_a: function (k, x_{k-1}, A_{k}). Dynamic function.
                k (iteration number), starts at 0
                x_{k-1} State from the previous step
                A_{k} Jacobian matrices of f_a. In the linear case it is exactly A_{k}.

        p_f_A: function (k, m, P) return Jacobian of dynamic function, it is
            passed into p_a.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.

        p_f_Q: function (k). Returns noise matrix of dynamic model on iteration k.
            k (iteration number). starts at 0

        filter_means: (no_steps+1,state_dim) matrix or (no_steps+1,state_dim, time_series_no) 3D array
            Results of the Kalman Filter means estimation.

        filter_covars: (no_steps+1, state_dim, state_dim) 3D array
            Results of the Kalman Filter covariance estimation.

        Output:
        -------------

        M: (no_steps+1, state_dim) matrix
            Smoothed estimates of the state means

        P: (no_steps+1, state_dim, state_dim) 3D array
            Smoothed estimates of the state covariances
        """

        no_steps = (
            filter_covars.shape[0] - 1
        )  # number of steps (minus initial covariance)

        M = np.empty(filter_means.shape)  # smoothed means
        P = np.empty(filter_covars.shape)  # smoothed covars
        # G = np.empty( (no_steps,state_dim,state_dim)  ) # G from the update step of the smoother

        M[-1, :] = filter_means[-1, :]
        P[-1, :, :] = filter_covars[-1, :, :]
        for k in range(no_steps - 1, -1, -1):
            m_pred, P_pred, tmp1, tmp2 = cls._kalman_prediction_step(
                k,
                filter_means[k, :],
                filter_covars[k, :, :],
                p_dynamic_callables,
                calc_grad_log_likelihood=False,
            )
            p_m = filter_means[k, :]
            if len(p_m.shape) < 2:
                p_m.shape = (p_m.shape[0], 1)

            p_m_prev_step = M[k + 1, :]
            if len(p_m_prev_step.shape) < 2:
                p_m_prev_step.shape = (p_m_prev_step.shape[0], 1)

            m_upd, P_upd, G_tmp = cls._rts_smoother_update_step(
                k,
                p_m,
                filter_covars[k, :, :],
                m_pred,
                P_pred,
                p_m_prev_step,
                P[k + 1, :, :],
                p_dynamic_callables,
            )

            M[k, :] = m_upd  # np.squeeze(m_upd)
            P[k, :, :] = P_upd
            # G[k,:,:] = G_upd.T # store transposed G.
        # Return values

        return (M, P)  # , G)

    @staticmethod
    def _EM_gradient(
        A,
        Q,
        H,
        R,
        m_init,
        P_init,
        measurements,
        M,
        P,
        G,
        dA,
        dQ,
        dH,
        dR,
        dm_init,
        dP_init,
    ):
        """
        Gradient computation with the EM algorithm.

        Input:
        -----------------

        M: Means from the smoother
        P: Variances from the smoother
        G: Gains? from the smoother
        """
        import pdb

        pdb.set_trace()

        param_number = dA.shape[-1]
        d_log_likelihood_update = np.empty((param_number, 1))

        sample_no = measurements.shape[0]
        P_1 = P[1:, :, :]  # remove 0-th step
        P_2 = P[0:-1, :, :]  # remove 0-th step

        M_1 = M[1:, :]  # remove 0-th step
        M_2 = M[0:-1, :]  # remove the last step

        Sigma = np.mean(P_1, axis=0) + np.dot(M_1.T, M_1) / sample_no  #
        Phi = np.mean(P_2, axis=0) + np.dot(M_2.T, M_2) / sample_no  #

        B = np.dot(measurements.T, M_1) / sample_no
        C = (sp.einsum("ijk,ikl", P_1, G) + np.dot(M_1.T, M_2)) / sample_no  #

        #        C1 = np.zeros( (P_1.shape[1],P_1.shape[1]) )
        #        for k in range(P_1.shape[0]):
        #            C1 += np.dot(P_1[k,:,:],G[k,:,:]) + sp.outer( M_1[k,:], M_2[k,:] )
        #        C1 = C1 / sample_no

        D = np.dot(measurements.T, measurements) / sample_no

        try:
            P_init_inv = sp.linalg.inv(P_init)

            if np.max(np.abs(P_init_inv)) > 10e13:
                compute_P_init_terms = False
            else:
                compute_P_init_terms = True
        except np.linalg.LinAlgError:
            compute_P_init_terms = False

        try:
            Q_inv = sp.linalg.inv(Q)

            if np.max(np.abs(Q_inv)) > 10e13:
                compute_Q_terms = False
            else:
                compute_Q_terms = True
        except np.linalg.LinAlgError:
            compute_Q_terms = False

        try:
            R_inv = sp.linalg.inv(R)

            if np.max(np.abs(R_inv)) > 10e13:
                compute_R_terms = False
            else:
                compute_R_terms = True
        except np.linalg.LinAlgError:
            compute_R_terms = False

        d_log_likelihood_update = np.zeros((param_number, 1))
        for j in range(param_number):
            if compute_P_init_terms:
                d_log_likelihood_update[j, :] -= 0.5 * np.sum(
                    P_init_inv * dP_init[:, :, j].T
                )  # p #m

                M0_smoothed = M[0]
                M0_smoothed.shape = (M0_smoothed.shape[0], 1)
                tmp1 = np.dot(
                    dP_init[:, :, j],
                    np.dot(
                        P_init_inv,
                        (
                            P[0, :, :]
                            + sp.outer((M0_smoothed - m_init), (M0_smoothed - m_init))
                        ),
                    ),
                )  # p #m
                d_log_likelihood_update[j, :] += 0.5 * np.sum(P_init_inv * tmp1.T)

                tmp2 = sp.outer(dm_init[:, j], M0_smoothed)
                tmp2 += tmp2.T
                d_log_likelihood_update[j, :] += 0.5 * np.sum(P_init_inv * tmp2.T)

            if compute_Q_terms:
                d_log_likelihood_update[j, :] -= (
                    sample_no / 2.0 * np.sum(Q_inv * dQ[:, :, j].T)
                )  # m

                tmp1 = np.dot(C, A.T)
                tmp1 += tmp1.T
                tmp1 = Sigma - tmp1 + np.dot(A, np.dot(Phi, A.T))  # m
                tmp1 = np.dot(dQ[:, :, j], np.dot(Q_inv, tmp1))
                d_log_likelihood_update[j, :] += (
                    sample_no / 2.0 * np.sum(Q_inv * tmp1.T)
                )

                tmp2 = np.dot(dA[:, :, j], C.T)
                tmp2 += tmp2.T
                tmp3 = np.dot(dA[:, :, j], np.dot(Phi, A.T))
                tmp3 += tmp3.T
                d_log_likelihood_update[j, :] -= (
                    sample_no / 2.0 * np.sum(Q_inv.T * (tmp3 - tmp2))
                )

            if compute_R_terms:
                d_log_likelihood_update[j, :] -= (
                    sample_no / 2.0 * np.sum(R_inv * dR[:, :, j].T)
                )

                tmp1 = np.dot(B, H.T)
                tmp1 += tmp1.T
                tmp1 = D - tmp1 + np.dot(H, np.dot(Sigma, H.T))
                tmp1 = np.dot(dR[:, :, j], np.dot(R_inv, tmp1))
                d_log_likelihood_update[j, :] += (
                    sample_no / 2.0 * np.sum(R_inv * tmp1.T)
                )

                tmp2 = np.dot(dH[:, :, j], B.T)
                tmp2 += tmp2.T
                tmp3 = np.dot(dH[:, :, j], np.dot(Sigma, H.T))
                tmp3 += tmp3.T
                d_log_likelihood_update[j, :] -= (
                    sample_no / 2.0 * np.sum(R_inv.T * (tmp3 - tmp2))
                )

        return d_log_likelihood_update

    @staticmethod
    def _check_SS_matrix(p_M, state_dim, measurement_dim, which="A"):
        """
        Veryfy that on exit the matrix has appropriate shape for the KF algorithm.

        Input:
            p_M: matrix
                As it is given for the user
            state_dim: int
                State dimensioanlity
            measurement_dim: int
                Measurement dimensionality
            which: string
                One of: 'A', 'Q', 'H', 'R'
        Output:
        ---------------
            p_M: matrix of the right shape

            old_M_shape: tuple
                Old Shape
        """

        old_M_shape = None
        if len(p_M.shape) < 3:  # new shape is 3 dimensional
            old_M_shape = p_M.shape  # save shape to restore it on exit
            if len(p_M.shape) == 2:  # matrix
                p_M.shape = (p_M.shape[0], p_M.shape[1], 1)
            elif len(p_M.shape) == 1:  # scalar but in array already
                if p_M.shape[0] != 1:
                    raise ValueError(
                        "Matrix %s is an 1D array, while it must be a matrix or scalar",
                        which,
                    )
                else:
                    p_M.shape = (1, 1, 1)

        if (which == "A") or (which == "Q"):
            if (p_M.shape[0] != state_dim) or (p_M.shape[1] != state_dim):
                raise ValueError(
                    "%s must be a square matrix of size (%i,%i)"
                    % (which, state_dim, state_dim)
                )
        if which == "H":
            if (p_M.shape[0] != measurement_dim) or (p_M.shape[1] != state_dim):
                raise ValueError(
                    "H must be of shape (measurement_dim, state_dim) (%i,%i)"
                    % (measurement_dim, state_dim)
                )
        if which == "R":
            if (p_M.shape[0] != measurement_dim) or (p_M.shape[1] != measurement_dim):
                raise ValueError(
                    "R must be of shape (measurement_dim, measurement_dim) (%i,%i)"
                    % (measurement_dim, measurement_dim)
                )

        return (p_M, old_M_shape)

    @staticmethod
    def _check_grad_state_matrices(dM, state_dim, grad_params_no, which="dA"):
        """
        Function checks (mostly check dimensions) matrices for marginal likelihood
        gradient parameters calculation. It check dA, dQ matrices.

        Input:
        -------------
            dM: None, scaler or 3D matrix
                It is supposed to be (state_dim,state_dim,grad_params_no) matrix.
                If None then zero matrix is assumed. If scalar then the function
                checks consistency with "state_dim" and "grad_params_no".

            state_dim: int
                State dimensionality

            grad_params_no: int
                How many parrameters of likelihood gradient in total.

            which: string
                'dA' or 'dQ'


        Output:
        --------------
            function of (k) which returns the parameters matrix.

        """

        if dM is None:
            dM = np.zeros((state_dim, state_dim, grad_params_no))
        elif isinstance(dM, np.ndarray):
            if state_dim == 1:
                if len(dM.shape) < 3:
                    dM.shape = (1, 1, 1)
            else:
                if len(dM.shape) < 3:
                    dM.shape = (state_dim, state_dim, 1)
        elif isinstance(dM, int):
            if state_dim > 1:
                raise ValueError(
                    "When computing likelihood gradient wrong %s dimension." % which
                )
            else:
                dM = np.ones((1, 1, 1)) * dM

        #        if not isinstance(dM, types.FunctionType):
        #            f_dM = lambda k: dM
        #        else:
        #            f_dM = dM

        return dM

    @staticmethod
    def _check_grad_measurement_matrices(
        dM, state_dim, grad_params_no, measurement_dim, which="dH"
    ):
        """
        Function checks (mostly check dimensions) matrices for marginal likelihood
        gradient parameters calculation. It check dH, dR matrices.

        Input:
        -------------
            dM: None, scaler or 3D matrix
                It is supposed to be
                (measurement_dim ,state_dim,grad_params_no) for "dH" matrix.
                (measurement_dim,measurement_dim,grad_params_no) for "dR"

                If None then zero matrix is assumed. If scalar then the function
                checks consistency with "state_dim" and "grad_params_no".

            state_dim: int
                State dimensionality

            grad_params_no: int
                How many parrameters of likelihood gradient in total.

            measurement_dim: int
                Dimensionality of measurements.

            which: string
                'dH' or 'dR'


        Output:
        --------------
            function of (k) which returns the parameters matrix.
        """

        if dM is None:
            if which == "dH":
                dM = np.zeros((measurement_dim, state_dim, grad_params_no))
            elif which == "dR":
                dM = np.zeros((measurement_dim, measurement_dim, grad_params_no))
        elif isinstance(dM, np.ndarray):
            if state_dim == 1:
                if len(dM.shape) < 3:
                    dM.shape = (1, 1, 1)
            else:
                if len(dM.shape) < 3:
                    if which == "dH":
                        dM.shape = (measurement_dim, state_dim, 1)
                    elif which == "dR":
                        dM.shape = (measurement_dim, measurement_dim, 1)
        elif isinstance(dM, int):
            if state_dim > 1:
                raise ValueError(
                    "When computing likelihood gradient wrong dH dimension."
                )
            else:
                dM = np.ones((1, 1, 1)) * dM

        #        if not isinstance(dM, types.FunctionType):
        #            f_dM = lambda k: dM
        #        else:
        #            f_dM = dM

        return dM


class Struct(object):
    pass


class ContDescrStateSpace(DescreteStateSpace):
    """
    Class for continuous-discrete Kalman filter. State equation is
    continuous while measurement equation is discrete.

        d x(t)/ dt = F x(t) + L q;        where q~ N(0, Qc)
        y_{t_k} = H_{k} x_{t_k} + r_{k};        r_{k-1} ~ N(0, R_{k})

    """

    class AQcompute_once(Q_handling_Class):
        """
        Class for calculating matrices A, Q, dA, dQ of the discrete Kalman Filter
        from the matrices F, L, Qc, P_ing, dF, dQc, dP_inf of the continuos state
        equation. dt - time steps.

        It has the same interface as AQcompute_batch.

        It computes matrices for only one time step. This object is used when
        there are many different time steps and storing matrices for each of them
        would take too much memory.
        """

        def __init__(
            self,
            F,
            L,
            Qc,
            dt,
            compute_derivatives=False,
            grad_params_no=None,
            P_inf=None,
            dP_inf=None,
            dF=None,
            dQc=None,
        ):
            """
            Constructor. All necessary parameters are passed here and stored
            in the opject.

            Input:
            -------------------
                F, L, Qc, P_inf : matrices
                    Parameters of corresponding continuous state model
                dt: array
                    All time steps
                compute_derivatives: bool
                    Whether to calculate derivatives

                dP_inf, dF, dQc: 3D array
                    Derivatives if they are required

            Output:
            -------------------
            Nothing
            """
            # Copies are done because this object is used later in smoother
            # and these parameters must not change.
            self.F = F.copy()
            self.L = L.copy()
            self.Qc = Qc.copy()

            self.dt = dt  # copy is not taken because dt is internal parameter

            # Parameters are used to calculate derivatives but derivatives
            # are not used in the smoother. Therefore copies are not taken.
            self.P_inf = P_inf
            self.dP_inf = dP_inf
            self.dF = dF
            self.dQc = dQc

            self.compute_derivatives = compute_derivatives
            self.grad_params_no = grad_params_no

            self.last_k = 0
            self.last_k_computed = False
            self.v_Ak = None
            self.v_Qk = None
            self.v_dAk = None
            self.v_dQk = None

            self.square_root_computed = False
            self.Q_inverse_computed = False
            self.Q_svd_computed = False
            # !!!Print statistics! Which object is created

        def f_a(self, k, m, A):
            """
            Dynamic model
            """

            return np.dot(A, m)  # default dynamic model

        def _recompute_for_new_k(self, k):
            """
            Computes the necessary matrices for an index k and store the results.

            Input:
            ----------------------
            k: int
                Index in the time differences array dt where to compute matrices

            Output:
            ----------------------
                Ak,Qk, dAk, dQk: matrices and/or 3D arrays
                    A, Q, dA dQ on step k
            """
            if (self.last_k != k) or (self.last_k_computed == False):
                v_Ak, v_Qk, tmp, v_dAk, v_dQk = ContDescrStateSpace.lti_sde_to_descrete(
                    self.F,
                    self.L,
                    self.Qc,
                    self.dt[k],
                    self.compute_derivatives,
                    grad_params_no=self.grad_params_no,
                    P_inf=self.P_inf,
                    dP_inf=self.dP_inf,
                    dF=self.dF,
                    dQc=self.dQc,
                )

                self.last_k = k
                self.last_k_computed = True
                self.v_Ak = v_Ak
                self.v_Qk = v_Qk
                self.v_dAk = v_dAk
                self.v_dQk = v_dQk

                self.Q_square_root_computed = False
                self.Q_inverse_computed = False
                self.Q_svd_computed = False
            else:
                v_Ak = self.v_Ak
                v_Qk = self.v_Qk
                v_dAk = self.v_dAk
                v_dQk = self.v_dQk

            # !!!Print statistics! Print sizes of matrices

            return v_Ak, v_Qk, v_dAk, v_dQk

        def reset(self, compute_derivatives):
            """
            For reusing this object e.g. in smoother computation. Actually,
            this object can not be reused because it computes the matrices on
            every iteration. But this method is written for keeping the same
            interface with the class AQcompute_batch.
            """

            self.last_k = 0
            self.last_k_computed = False
            self.compute_derivatives = compute_derivatives

            self.Q_square_root_computed = False
            self.Q_inverse_computed = False
            self.Q_svd_computed = False
            self.Q_eigen_computed = False
            return self

        def Ak(self, k, m, P):
            v_Ak, v_Qk, v_dAk, v_dQk = self._recompute_for_new_k(k)
            return v_Ak

        def Qk(self, k):
            v_Ak, v_Qk, v_dAk, v_dQk = self._recompute_for_new_k(k)
            return v_Qk

        def dAk(self, k):
            v_Ak, v_Qk, v_dAk, v_dQk = self._recompute_for_new_k(k)
            return v_dAk

        def dQk(self, k):
            v_Ak, v_Qk, v_dAk, v_dQk = self._recompute_for_new_k(k)
            return v_dQk

        def Q_srk(self, k):
            """
            Check square root, maybe rewriting for Spectral decomposition is needed.
            Square root of the noise matrix Q
            """

            if (self.last_k == k) and (self.last_k_computed == True):
                if not self.Q_square_root_computed:
                    if not self.Q_svd_computed:
                        (U, S, Vh) = sp.linalg.svd(
                            self.v_Qk,
                            full_matrices=False,
                            compute_uv=True,
                            overwrite_a=False,
                            check_finite=False,
                        )
                        self.Q_svd = (U, S, Vh)
                        self.Q_svd_computed = True
                    else:
                        (U, S, Vh) = self.Q_svd

                    square_root = U * np.sqrt(S)
                    self.square_root_computed = True
                    self.Q_square_root = square_root
                else:
                    square_root = self.Q_square_root
            else:
                raise ValueError("Square root of Q can not be computed")

            return square_root

        def Q_inverse(self, k, p_largest_cond_num, p_regularization_type):
            """
            Function inverts Q matrix and regularizes the inverse.
            Regularization is useful when original matrix is badly conditioned.
            Function is currently used only in SparseGP code.

            Inputs:
            ------------------------------
            k: int
            Iteration number.

            p_largest_cond_num: float
            Largest condition value for the inverted matrix. If cond. number is smaller than that
            no regularization happen.

            regularization_type: 1 or 2
            Regularization type.

            regularization_type: int (1 or 2)

                type 1: 1/(S[k] + regularizer) regularizer is computed
                type 2: S[k]/(S^2[k] + regularizer) regularizer is computed
            """

            # import pdb; pdb.set_trace()

            if (self.last_k == k) and (self.last_k_computed == True):
                if not self.Q_inverse_computed:
                    if not self.Q_svd_computed:
                        (U, S, Vh) = sp.linalg.svd(
                            self.v_Qk,
                            full_matrices=False,
                            compute_uv=True,
                            overwrite_a=False,
                            check_finite=False,
                        )
                        self.Q_svd = (U, S, Vh)
                        self.Q_svd_computed = True
                    else:
                        (U, S, Vh) = self.Q_svd

                    Q_inverse_r = psd_matrix_inverse(
                        k,
                        0.5 * (self.v_Qk + self.v_Qk.T),
                        U,
                        S,
                        p_largest_cond_num,
                        p_regularization_type,
                    )

                    self.Q_inverse_computed = True
                    self.Q_inverse_r = Q_inverse_r

                else:
                    Q_inverse_r = self.Q_inverse_r
            else:
                raise ValueError(
                    """Inverse of Q can not be computed, because Q has not been computed.
                                     This requires some programming"""
                )

            return Q_inverse_r

        def return_last(self):
            """
            Function returns last computed matrices.
            """
            if not self.last_k_computed:
                raise ValueError("Matrices are not computed.")
            else:
                k = self.last_k
                A = self.v_Ak
                Q = self.v_Qk
                dA = self.v_dAk
                dQ = self.v_dQk

            return k, A, Q, dA, dQ

    class AQcompute_batch_Python(Q_handling_Class):
        """
        Class for calculating matrices A, Q, dA, dQ of the discrete Kalman Filter
        from the matrices F, L, Qc, P_ing, dF, dQc, dP_inf of the continuos state
        equation. dt - time steps.

        It has the same interface as AQcompute_once.

        It computes matrices for all time steps. This object is used when
        there are not so many (controlled by internal variable)
        different time steps and storing all the matrices do not take too much memory.

        Since all the matrices are computed all together, this object can be used
        in smoother without repeating the computations.
        """

        def __init__(
            self,
            F,
            L,
            Qc,
            dt,
            compute_derivatives=False,
            grad_params_no=None,
            P_inf=None,
            dP_inf=None,
            dF=None,
            dQc=None,
        ):
            """
            Constructor. All necessary parameters are passed here and stored
            in the opject.

            Input:
            -------------------
                F, L, Qc, P_inf : matrices
                    Parameters of corresponding continuous state model
                dt: array
                    All time steps
                compute_derivatives: bool
                    Whether to calculate derivatives

                dP_inf, dF, dQc: 3D array
                    Derivatives if they are required

            Output:
            -------------------
            Nothing
            """
            (
                As,
                Qs,
                reconstruct_indices,
                dAs,
                dQs,
            ) = ContDescrStateSpace.lti_sde_to_descrete(
                F,
                L,
                Qc,
                dt,
                compute_derivatives,
                grad_params_no=grad_params_no,
                P_inf=P_inf,
                dP_inf=dP_inf,
                dF=dF,
                dQc=dQc,
            )

            self.As = As
            self.Qs = Qs
            self.dAs = dAs
            self.dQs = dQs
            self.reconstruct_indices = reconstruct_indices
            self.total_size_of_data = (
                self.As.nbytes
                + self.Qs.nbytes
                + (self.dAs.nbytes if (self.dAs is not None) else 0)
                + (self.dQs.nbytes if (self.dQs is not None) else 0)
                + (
                    self.reconstruct_indices.nbytes
                    if (self.reconstruct_indices is not None)
                    else 0
                )
            )

            self.Q_svd_dict = {}
            self.Q_square_root_dict = {}
            self.Q_inverse_dict = {}

            self.last_k = None
            # !!!Print statistics! Which object is created
            # !!!Print statistics! Print sizes of matrices

        def f_a(self, k, m, A):
            """
            Dynamic model
            """
            return np.dot(A, m)  # default dynamic model

        def reset(self, compute_derivatives=False):
            """
            For reusing this object e.g. in smoother computation. It makes sence
            because necessary matrices have been already computed for all
            time steps.
            """
            return self

        def Ak(self, k, m, P):
            self.last_k = k
            return self.As[:, :, self.reconstruct_indices[k]]

        def Qk(self, k):
            self.last_k = k
            return self.Qs[:, :, self.reconstruct_indices[k]]

        def dAk(self, k):
            self.last_k = k
            return self.dAs[:, :, :, self.reconstruct_indices[k]]

        def dQk(self, k):
            self.last_k = k
            return self.dQs[:, :, :, self.reconstruct_indices[k]]

        def Q_srk(self, k):
            """
            Square root of the noise matrix Q
            """
            matrix_index = self.reconstruct_indices[k]
            if matrix_index in self.Q_square_root_dict:
                square_root = self.Q_square_root_dict[matrix_index]
            else:
                if matrix_index in self.Q_svd_dict:
                    (U, S, Vh) = self.Q_svd_dict[matrix_index]
                else:
                    (U, S, Vh) = sp.linalg.svd(
                        self.Qs[:, :, matrix_index],
                        full_matrices=False,
                        compute_uv=True,
                        overwrite_a=False,
                        check_finite=False,
                    )
                    self.Q_svd_dict[matrix_index] = (U, S, Vh)

                square_root = U * np.sqrt(S)
                self.Q_square_root_dict[matrix_index] = square_root

            return square_root

        def Q_inverse(self, k, p_largest_cond_num, p_regularization_type):
            """
            Function inverts Q matrix and regularizes the inverse.
            Regularization is useful when original matrix is badly conditioned.
            Function is currently used only in SparseGP code.

            Inputs:
            ------------------------------
            k: int
            Iteration number.

            p_largest_cond_num: float
            Largest condition value for the inverted matrix. If cond. number is smaller than that
            no regularization happen.

            regularization_type: 1 or 2
            Regularization type.

            regularization_type: int (1 or 2)

                type 1: 1/(S[k] + regularizer) regularizer is computed
                type 2: S[k]/(S^2[k] + regularizer) regularizer is computed
            """
            # import pdb; pdb.set_trace()

            matrix_index = self.reconstruct_indices[k]
            if matrix_index in self.Q_inverse_dict:
                Q_inverse_r = self.Q_inverse_dict[matrix_index]
            else:
                if matrix_index in self.Q_svd_dict:
                    (U, S, Vh) = self.Q_svd_dict[matrix_index]
                else:
                    (U, S, Vh) = sp.linalg.svd(
                        self.Qs[:, :, matrix_index],
                        full_matrices=False,
                        compute_uv=True,
                        overwrite_a=False,
                        check_finite=False,
                    )
                    self.Q_svd_dict[matrix_index] = (U, S, Vh)

                Q_inverse_r = psd_matrix_inverse(
                    k,
                    0.5 * (self.Qs[:, :, matrix_index] + self.Qs[:, :, matrix_index].T),
                    U,
                    S,
                    p_largest_cond_num,
                    p_regularization_type,
                )
                self.Q_inverse_dict[matrix_index] = Q_inverse_r

            return Q_inverse_r

        def return_last(self):
            """
            Function returns last available matrices.
            """

            if self.last_k is None:
                raise ValueError("Matrices are not computed.")
            else:
                ind = self.reconstruct_indices[self.last_k]
                A = self.As[:, :, ind]
                Q = self.Qs[:, :, ind]
                dA = self.dAs[:, :, :, ind]
                dQ = self.dQs[:, :, :, ind]

            return self.last_k, A, Q, dA, dQ

    @classmethod
    def cont_discr_kalman_filter(
        cls,
        F,
        L,
        Qc,
        p_H,
        p_R,
        P_inf,
        X,
        Y,
        index=None,
        m_init=None,
        P_init=None,
        p_kalman_filter_type="regular",
        calc_log_likelihood=False,
        calc_grad_log_likelihood=False,
        grad_params_no=0,
        grad_calc_params=None,
    ):
        """
        This function implements the continuous-discrete Kalman Filter algorithm
        These notations for the State-Space model are assumed:
            d/dt x(t) =  F * x(t) + L * w(t);         w(t) ~ N(0, Qc)
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})

        Returns estimated filter distributions x_{k} ~ N(m_{k}, P(k))

        Current Features:
        ----------------------------------------
        1) The function generaly do not modify the passed parameters. If
        it happens then it is an error. There are several exeprions: scalars
        can be modified into a matrix, in some rare cases shapes of
        the derivatives matrices may be changed, it is ignored for now.

        2) Copies of F,L,Qc are created in memory because they may be used later
        in smoother. References to copies are kept in "AQcomp" object
        return parameter.

        3) Function support "multiple time series mode" which means that exactly
        the same State-Space model is used to filter several sets of measurements.
        In this case third dimension of Y should include these state-space measurements
        Log_likelihood and Grad_log_likelihood have the corresponding dimensions then.

        4) Calculation of Grad_log_likelihood is not supported if matrices
        H, or R changes overf time (with index k). (later may be changed)

        5) Measurement may include missing values. In this case update step is
        not done for this measurement. (later may be changed)

        Input:
        -----------------

        F: (state_dim, state_dim) matrix
            F in the model.

        L: (state_dim, noise_dim) matrix
            L in the model.

        Qc: (noise_dim, noise_dim) matrix
            Q_c in the model.

        p_H: scalar, matrix (measurement_dim, state_dim) , 3D array
            H_{k} in the model. If matrix then H_{k} = H - constant.
            If it is 3D array then H_{k} = p_Q[:,:, index[2,k]]

        p_R: scalar, square symmetric matrix, 3D array
            R_{k} in the model. If matrix then R_{k} = R - constant.
            If it is 3D array then R_{k} = p_R[:,:, index[3,k]]

        P_inf: (state_dim, state_dim) matrix
            State varince matrix on infinity.

        X: 1D array
            Time points of measurements. Needed for converting continuos
            problem to the discrete one.

        Y: matrix or vector or 3D array
            Data. If Y is matrix then samples are along 0-th dimension and
            features along the 1-st. If 3D array then third dimension
            correspond to "multiple time series mode".

        index: vector
            Which indices (on 3-rd dimension) from arrays p_H, p_R to use
            on every time step. If this parameter is None then it is assumed
            that p_H, p_R do not change over time and indices are not needed.
            index[0,:] - correspond to H, index[1,:] - correspond to R
            If index.shape[0] == 1, it is assumed that indides for all matrices
            are the same.

        m_init: vector or matrix
            Initial distribution mean. If None it is assumed to be zero.
            For "multiple time series mode" it is matrix, second dimension of
            which correspond to different time series. In regular case ("one
            time series mode") it is a vector.

        P_init: square symmetric matrix or scalar
            Initial covariance of the states. If the parameter is scalar
            then it is assumed that initial covariance matrix is unit matrix
            multiplied by this scalar. If None the unit matrix is used instead.
            "multiple time series mode" does not affect it, since it does not
            affect anything related to state variaces.

        p_kalman_filter_type: string, one of ('regular', 'svd')
            Which Kalman Filter is used. Regular or SVD. SVD is more numerically
            stable, in particular, Covariace matrices are guarantied to be
            positive semi-definite. However, 'svd' works slower, especially for
            small data due to SVD call overhead.

        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.

        calc_grad_log_likelihood: boolean
            Whether to calculate gradient of the marginal likelihood
            of the state-space model. If true then "grad_calc_params" parameter must
            provide the extra parameters for gradient calculation.

        grad_params_no: int
            If previous parameter is true, then this parameters gives the
            total number of parameters in the gradient.

        grad_calc_params: dictionary
            Dictionary with derivatives of model matrices with respect
            to parameters "dF", "dL", "dQc", "dH", "dR", "dm_init", "dP_init".
            They can be None, in this case zero matrices (no dependence on parameters)
            is assumed. If there is only one parameter then third dimension is
            automatically added.

        Output:
        --------------

        M: (no_steps+1,state_dim) matrix or (no_steps+1,state_dim, time_series_no) 3D array
            Filter estimates of the state means. In the extra step the initial
            value is included. In the "multiple time series mode" third dimension
            correspond to different timeseries.

        P: (no_steps+1, state_dim, state_dim) 3D array
            Filter estimates of the state covariances. In the extra step the initial
            value is included.

        log_likelihood: double or (1, time_series_no) 3D array.

            If the parameter calc_log_likelihood was set to true, return
            logarithm of marginal likelihood of the state-space model. If
            the parameter was false, return None. In the "multiple time series mode" it is a vector
            providing log_likelihood for each time series.

        grad_log_likelihood: column vector or (grad_params_no, time_series_no) matrix
            If calc_grad_log_likelihood is true, return gradient of log likelihood
            with respect to parameters. It returns it column wise, so in
            "multiple time series mode" gradients for each time series is in the
            corresponding column.

        AQcomp: object
            Contains some pre-computed values for converting continuos model into
            discrete one. It can be used later in the smoothing pahse.
        """

        p_H = np.atleast_1d(p_H)
        p_R = np.atleast_1d(p_R)

        X.shape, old_X_shape = cls._reshape_input_data(
            X.shape, 2
        )  # represent as column
        if X.shape[1] != 1:
            raise ValueError("Only one dimensional X data is supported.")

        Y.shape, old_Y_shape = cls._reshape_input_data(Y.shape)  # represent as column

        state_dim = F.shape[0]
        measurement_dim = Y.shape[1]
        time_series_no = Y.shape[2]  # multiple time series mode

        if ((len(p_H.shape) == 3) and (len(p_H.shape[2]) != 1)) or (
            (len(p_R.shape) == 3) and (len(p_R.shape[2]) != 1)
        ):
            model_matrices_chage_with_time = True
        else:
            model_matrices_chage_with_time = False

        # Check index
        old_index_shape = None
        if index is None:
            if (len(p_H.shape) == 3) or (len(p_R.shape) == 3):
                raise ValueError(
                    "Parameter index can not be None for time varying matrices (third dimension is present)"
                )
            else:  # matrices do not change in time, so form dummy zero indices.
                index = np.zeros((1, Y.shape[0]))
        else:
            if len(index.shape) == 1:
                index.shape = (1, index.shape[0])
                old_index_shape = (index.shape[0],)

            if index.shape[1] != Y.shape[0]:
                raise ValueError(
                    "Number of measurements must be equal the number of H_{k}, R_{k}"
                )

        if index.shape[0] == 1:
            H_time_var_index = 0
            R_time_var_index = 0
        elif index.shape[0] == 4:
            H_time_var_index = 0
            R_time_var_index = 1
        else:
            raise ValueError("First Dimension of index must be either 1 or 2.")

        (p_H, old_H_shape) = cls._check_SS_matrix(
            p_H, state_dim, measurement_dim, which="H"
        )
        (p_R, old_R_shape) = cls._check_SS_matrix(
            p_R, state_dim, measurement_dim, which="R"
        )

        if m_init is None:
            m_init = np.zeros((state_dim, time_series_no))
        else:
            m_init = np.atleast_2d(m_init).T

        if P_init is None:
            P_init = P_inf.copy()

        if p_kalman_filter_type not in ("regular", "svd"):
            raise ValueError("Kalman filer type neither 'regular nor 'svd'.")

        # Functions to pass to the kalman_filter algorithm:
        # Parameters:
        # k - number of Kalman filter iteration
        # m - vector for calculating matrices. Required for EKF. Not used here.
        # f_hl = lambda k,m,H: np.dot(H, m)
        # f_H = lambda k,m,P: p_H[:,:, index[H_time_var_index, k]]
        # f_R = lambda k: p_R[:,:, index[R_time_var_index, k]]
        # o_R = R_handling( p_R, index, R_time_var_index, 20)

        if calc_grad_log_likelihood:
            dF = cls._check_grad_state_matrices(
                grad_calc_params.get("dF"), state_dim, grad_params_no, which="dA"
            )
            dQc = cls._check_grad_state_matrices(
                grad_calc_params.get("dQc"), state_dim, grad_params_no, which="dQ"
            )
            dP_inf = cls._check_grad_state_matrices(
                grad_calc_params.get("dP_inf"), state_dim, grad_params_no, which="dA"
            )

            dH = cls._check_grad_measurement_matrices(
                grad_calc_params.get("dH"),
                state_dim,
                grad_params_no,
                measurement_dim,
                which="dH",
            )
            dR = cls._check_grad_measurement_matrices(
                grad_calc_params.get("dR"),
                state_dim,
                grad_params_no,
                measurement_dim,
                which="dR",
            )

            dm_init = grad_calc_params.get(
                "dm_init"
            )  # Initial values for the Kalman Filter
            if dm_init is None:
                # multiple time series mode. Keep grad_params always as a last dimension
                dm_init = np.zeros((state_dim, time_series_no, grad_params_no))

            dP_init = grad_calc_params.get(
                "dP_init"
            )  # Initial values for the Kalman Filter
            if dP_init is None:
                dP_init = dP_inf(
                    0
                ).copy()  # get the dP_init matrix, because now it is a function

        else:
            dP_inf = None
            dF = None
            dQc = None
            dH = None
            dR = None
            dm_init = None
            dP_init = None

        measurement_callables = Std_Measurement_Callables_Class(
            p_H, H_time_var_index, p_R, index, R_time_var_index, 20, dH, dR
        )
        # import pdb; pdb.set_trace()

        dynamic_callables = cls._cont_to_discrete_object(
            X,
            F,
            L,
            Qc,
            compute_derivatives=calc_grad_log_likelihood,
            grad_params_no=grad_params_no,
            P_inf=P_inf,
            dP_inf=dP_inf,
            dF=dF,
            dQc=dQc,
        )

        if print_verbose:
            print("General: run Continuos-Discrete Kalman Filter")
        # Also for dH, dR and probably for all derivatives
        (
            M,
            P,
            log_likelihood,
            grad_log_likelihood,
            AQcomp,
        ) = cls._cont_discr_kalman_filter_raw(
            state_dim,
            dynamic_callables,
            measurement_callables,
            X,
            Y,
            m_init=m_init,
            P_init=P_init,
            p_kalman_filter_type=p_kalman_filter_type,
            calc_log_likelihood=calc_log_likelihood,
            calc_grad_log_likelihood=calc_grad_log_likelihood,
            grad_params_no=grad_params_no,
            dm_init=dm_init,
            dP_init=dP_init,
        )

        if old_index_shape is not None:
            index.shape = old_index_shape

        if old_X_shape is not None:
            X.shape = old_X_shape

        if old_Y_shape is not None:
            Y.shape = old_Y_shape

        if old_H_shape is not None:
            p_H.shape = old_H_shape

        if old_R_shape is not None:
            p_R.shape = old_R_shape

        return (M, P, log_likelihood, grad_log_likelihood, AQcomp)

    @classmethod
    def _cont_discr_kalman_filter_raw(
        cls,
        state_dim,
        p_dynamic_callables,
        p_measurement_callables,
        X,
        Y,
        m_init,
        P_init,
        p_kalman_filter_type="regular",
        calc_log_likelihood=False,
        calc_grad_log_likelihood=False,
        grad_params_no=None,
        dm_init=None,
        dP_init=None,
    ):
        """
        General filtering algorithm for inference in the continuos-discrete
        state-space model:

            d/dt x(t) =  F * x(t) + L * w(t);         w(t) ~ N(0, Qc)
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})

        Returns estimated filter distributions x_{k} ~ N(m_{k}, P(k))

        Current Features:
        ----------------------------------------

        1) Function support "multiple time series mode" which means that exactly
        the same State-Space model is used to filter several sets of measurements.
        In this case third dimension of Y should include these state-space measurements
        Log_likelihood and Grad_log_likelihood have the corresponding dimensions then.

        2) Measurement may include missing values. In this case update step is
        not done for this measurement. (later may be changed)

        Input:
        -----------------
        state_dim: int
            Demensionality of the states

        F: (state_dim, state_dim) matrix
            F in the model.

        L: (state_dim, noise_dim) matrix
            L in the model.

        Qc: (noise_dim, noise_dim) matrix
            Q_c in the model.

        P_inf: (state_dim, state_dim) matrix
            State varince matrix on infinity.

        p_h: function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number),
            x_{k}
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly H_{k}.

        f_H: function (k, m, P) return Jacobian of dynamic function, it is
            passed into p_h.
            k (iteration number),
            m: point where Jacobian is evaluated,
            P: parameter for Jacobian, usually covariance matrix.

        p_f_R: function (k). Returns noise matrix of measurement equation
            on iteration k.
            k (iteration number).

        m_init: vector or matrix
            Initial distribution mean. For "multiple time series mode"
            it is matrix, second dimension of which correspond to different
            time series. In regular case ("one time series mode") it is a
            vector.

        P_init: matrix or scalar
            Initial covariance of the states. Must be not None
            "multiple time series mode" does not affect it, since it does not
            affect anything related to state variaces.

        p_kalman_filter_type: string, one of ('regular', 'svd')
            Which Kalman Filter is used. Regular or SVD. SVD is more numerically
            stable, in particular, Covariace matrices are guarantied to be
            positive semi-definite. However, 'svd' works slower, especially for
            small data due to SVD call overhead.

        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.

        calc_grad_log_likelihood: boolean
            Whether to calculate gradient of the marginal likelihood
            of the state-space model. If true then the next parameter must
            provide the extra parameters for gradient calculation.

        grad_params_no: int
            Number of gradient parameters

        dP_inf, dF, dQc, dH, dR, dm_init, dP_init: matrices or 3D arrays.
            Necessary parameters for derivatives calculation.

        """

        # import pdb; pdb.set_trace()
        steps_no = Y.shape[0]  # number of steps in the Kalman Filter
        time_series_no = Y.shape[2]  # multiple time series mode

        # Allocate space for results
        # Mean estimations. Initial values will be included
        M = np.empty(((steps_no + 1), state_dim, time_series_no))
        M[0, :, :] = m_init  # Initialize mean values
        # Variance estimations. Initial values will be included
        P = np.empty(((steps_no + 1), state_dim, state_dim))
        P_init = 0.5 * (
            P_init + P_init.T
        )  # symmetrize initial covariance. In some ustable cases this is uiseful
        P[0, :, :] = P_init  # Initialize initial covariance matrix

        # import pdb;pdb.set_trace()
        if p_kalman_filter_type == "svd":
            (U, S, Vh) = sp.linalg.svd(
                P_init,
                full_matrices=False,
                compute_uv=True,
                overwrite_a=False,
                check_finite=True,
            )
            S[(S == 0)] = 1e-17  # allows to run algorithm for singular initial variance
            P_upd = (P_init, S, U)
        # log_likelihood = 0
        # grad_log_likelihood = np.zeros((grad_params_no,1))
        log_likelihood = 0 if calc_log_likelihood else None
        grad_log_likelihood = 0 if calc_grad_log_likelihood else None

        # setting initial values for derivatives update
        dm_upd = dm_init
        dP_upd = dP_init
        # Main loop of the Kalman filter
        for k in range(0, steps_no):
            # In this loop index for new estimations is (k+1), old - (k)
            # This happened because initial values are stored at 0-th index.
            # import pdb; pdb.set_trace()

            prev_mean = M[k, :, :]  # mean from the previous step

            if p_kalman_filter_type == "svd":
                m_pred, P_pred, dm_pred, dP_pred = cls._kalman_prediction_step_SVD(
                    k,
                    prev_mean,
                    P_upd,
                    p_dynamic_callables,
                    calc_grad_log_likelihood=calc_grad_log_likelihood,
                    p_dm=dm_upd,
                    p_dP=dP_upd,
                )
            else:
                m_pred, P_pred, dm_pred, dP_pred = cls._kalman_prediction_step(
                    k,
                    prev_mean,
                    P[k, :, :],
                    p_dynamic_callables,
                    calc_grad_log_likelihood=calc_grad_log_likelihood,
                    p_dm=dm_upd,
                    p_dP=dP_upd,
                )

            # import pdb; pdb.set_trace()
            k_measurment = Y[k, :, :]

            if np.any(np.isnan(k_measurment)) == False:
                if p_kalman_filter_type == "svd":
                    (
                        m_upd,
                        P_upd,
                        log_likelihood_update,
                        dm_upd,
                        dP_upd,
                        d_log_likelihood_update,
                    ) = cls._kalman_update_step_SVD(
                        k,
                        m_pred,
                        P_pred,
                        p_measurement_callables,
                        k_measurment,
                        calc_log_likelihood=calc_log_likelihood,
                        calc_grad_log_likelihood=calc_grad_log_likelihood,
                        p_dm=dm_pred,
                        p_dP=dP_pred,
                    )

                #                m_upd, P_upd, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update = \
                #                cls._kalman_update_step(k,  m_pred , P_pred[0], f_h, f_H, p_R.f_R, k_measurment,
                #                        calc_log_likelihood=calc_log_likelihood,
                #                        calc_grad_log_likelihood=calc_grad_log_likelihood,
                #                        p_dm = dm_pred, p_dP = dP_pred, grad_calc_params_2 = (dH, dR))
                #
                #                (U,S,Vh) = sp.linalg.svd( P_upd,full_matrices=False, compute_uv=True,
                #                      overwrite_a=False,check_finite=True)
                #                P_upd = (P_upd, S,U)
                else:
                    (
                        m_upd,
                        P_upd,
                        log_likelihood_update,
                        dm_upd,
                        dP_upd,
                        d_log_likelihood_update,
                    ) = cls._kalman_update_step(
                        k,
                        m_pred,
                        P_pred,
                        p_measurement_callables,
                        k_measurment,
                        calc_log_likelihood=calc_log_likelihood,
                        calc_grad_log_likelihood=calc_grad_log_likelihood,
                        p_dm=dm_pred,
                        p_dP=dP_pred,
                    )
            else:
                if k_measurment.shape != (1, 1):
                    raise ValueError(
                        "Nan measurements are currently not supported for \
                                     multidimensional output and multiple tiem series."
                    )
                else:
                    m_upd = m_pred
                    P_upd = P_pred
                    dm_upd = dm_pred
                    dP_upd = dP_pred
                    log_likelihood_update = 0.0
                    d_log_likelihood_update = 0.0

            if calc_log_likelihood:
                log_likelihood += log_likelihood_update

            if calc_grad_log_likelihood:
                grad_log_likelihood += d_log_likelihood_update

            M[k + 1, :, :] = m_upd  # separate mean value for each time series

            if p_kalman_filter_type == "svd":
                P[k + 1, :, :] = P_upd[0]
            else:
                P[k + 1, :, :] = P_upd
            # print("kf it: %i" % k)
            # !!!Print statistics! Print sizes of matrices
            # !!!Print statistics! Print iteration time base on another boolean variable
        return (
            M,
            P,
            log_likelihood,
            grad_log_likelihood,
            p_dynamic_callables.reset(False),
        )

    @classmethod
    def cont_discr_rts_smoother(
        cls,
        state_dim,
        filter_means,
        filter_covars,
        p_dynamic_callables=None,
        X=None,
        F=None,
        L=None,
        Qc=None,
    ):
        """

        Continuos-discrete Rauch–Tung–Striebel(RTS) smoother.

        This function implements Rauch–Tung–Striebel(RTS) smoother algorithm
        based on the results of _cont_discr_kalman_filter_raw.

        Model:
            d/dt x(t) =  F * x(t) + L * w(t);         w(t) ~ N(0, Qc)
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})

        Returns estimated smoother distributions x_{k} ~ N(m_{k}, P(k))

        Input:
        --------------

        filter_means: (no_steps+1,state_dim) matrix or (no_steps+1,state_dim, time_series_no) 3D array
            Results of the Kalman Filter means estimation.

        filter_covars: (no_steps+1, state_dim, state_dim) 3D array
            Results of the Kalman Filter covariance estimation.

        Dynamic_callables: object or None
            Object form the filter phase which provides functions for computing
            A, Q, dA, dQ fro  discrete model from the continuos model.

         X, F, L, Qc: matrices
             If AQcomp is None, these matrices are used to create this object from scratch.

        Output:
        -------------

        M: (no_steps+1,state_dim) matrix
            Smoothed estimates of the state means

        P: (no_steps+1,state_dim, state_dim) 3D array
            Smoothed estimates of the state covariances
        """

        f_a = lambda k, m, A: np.dot(A, m)  # state dynamic model
        if p_dynamic_callables is None:  # make this object from scratch
            p_dynamic_callables = cls._cont_to_discrete_object(
                cls,
                X,
                F,
                L,
                Qc,
                f_a,
                compute_derivatives=False,
                grad_params_no=None,
                P_inf=None,
                dP_inf=None,
                dF=None,
                dQc=None,
            )

        no_steps = (
            filter_covars.shape[0] - 1
        )  # number of steps (minus initial covariance)

        M = np.empty(filter_means.shape)  # smoothed means
        P = np.empty(filter_covars.shape)  # smoothed covars

        if print_verbose:
            print("General: run Continuos-Discrete Kalman Smoother")

        M[-1, :, :] = filter_means[-1, :, :]
        P[-1, :, :] = filter_covars[-1, :, :]
        for k in range(no_steps - 1, -1, -1):
            prev_mean = filter_means[k, :]  # mean from the previous step
            m_pred, P_pred, tmp1, tmp2 = cls._kalman_prediction_step(
                k,
                prev_mean,
                filter_covars[k, :, :],
                p_dynamic_callables,
                calc_grad_log_likelihood=False,
            )
            p_m = filter_means[k, :]
            p_m_prev_step = M[(k + 1), :]

            m_upd, P_upd, tmp_G = cls._rts_smoother_update_step(
                k,
                p_m,
                filter_covars[k, :, :],
                m_pred,
                P_pred,
                p_m_prev_step,
                P[(k + 1), :, :],
                p_dynamic_callables,
            )

            M[k, :, :] = m_upd
            P[k, :, :] = P_upd
        # Return values
        return (M, P)

    @classmethod
    def _cont_to_discrete_object(
        cls,
        X,
        F,
        L,
        Qc,
        compute_derivatives=False,
        grad_params_no=None,
        P_inf=None,
        dP_inf=None,
        dF=None,
        dQc=None,
        dt0=None,
    ):
        """
        Function return the object which is used in Kalman filter and/or
        smoother to obtain matrices A, Q and their derivatives for discrete model
        from the continuous model.

        There are 2 objects AQcompute_once and AQcompute_batch and the function
        returs the appropriate one based on the number of different time steps.

        Input:
        ----------------------
        X, F, L, Qc: matrices
            Continuous model matrices

        f_a: function
            Dynamic Function is attached to the Dynamic_Model_Callables class
        compute_derivatives: boolean
            Whether to compute derivatives

        grad_params_no: int
            Number of parameters in the gradient

        P_inf, dP_inf, dF, dQ: matrices and 3D objects
            Data necessary to compute derivatives.

        Output:
        --------------------------
        AQcomp: object
            Its methods return matrices (and optionally derivatives) for the
            discrete state-space model.

        """

        unique_round_decimals = 10
        threshold_number_of_unique_time_steps = (
            20  # above which matrices are separately each time
        )
        dt = np.empty((X.shape[0],))
        dt[1:] = np.diff(X[:, 0], axis=0)
        if dt0 is None:
            dt[0] = 0  # dt[1]
        else:
            if isinstance(dt0, str):
                dt = dt[1:]
            else:
                dt[0] = dt0

        unique_indices = np.unique(np.round(dt, decimals=unique_round_decimals))
        number_unique_indices = len(unique_indices)

        # import pdb; pdb.set_trace()
        if use_cython:

            class AQcompute_batch(state_space_cython.AQcompute_batch_Cython):
                def __init__(
                    self,
                    F,
                    L,
                    Qc,
                    dt,
                    compute_derivatives=False,
                    grad_params_no=None,
                    P_inf=None,
                    dP_inf=None,
                    dF=None,
                    dQc=None,
                ):
                    (
                        As,
                        Qs,
                        reconstruct_indices,
                        dAs,
                        dQs,
                    ) = ContDescrStateSpace.lti_sde_to_descrete(
                        F,
                        L,
                        Qc,
                        dt,
                        compute_derivatives,
                        grad_params_no=grad_params_no,
                        P_inf=P_inf,
                        dP_inf=dP_inf,
                        dF=dF,
                        dQc=dQc,
                    )

                    super(AQcompute_batch, self).__init__(
                        As, Qs, reconstruct_indices, dAs, dQs
                    )

        else:
            AQcompute_batch = cls.AQcompute_batch_Python

        if number_unique_indices > threshold_number_of_unique_time_steps:
            AQcomp = cls.AQcompute_once(
                F,
                L,
                Qc,
                dt,
                compute_derivatives=compute_derivatives,
                grad_params_no=grad_params_no,
                P_inf=P_inf,
                dP_inf=dP_inf,
                dF=dF,
                dQc=dQc,
            )
            if print_verbose:
                print("CDO:  Continue-to-discrete INSTANTANEOUS object is created.")
                print(
                    "CDO:  Number of different time steps: %i"
                    % (number_unique_indices,)
                )

        else:
            AQcomp = AQcompute_batch(
                F,
                L,
                Qc,
                dt,
                compute_derivatives=compute_derivatives,
                grad_params_no=grad_params_no,
                P_inf=P_inf,
                dP_inf=dP_inf,
                dF=dF,
                dQc=dQc,
            )
            if print_verbose:
                print("CDO:  Continue-to-discrete BATCH object is created.")
                print(
                    "CDO:  Number of different time steps: %i"
                    % (number_unique_indices,)
                )
                print("CDO:  Total size if its data: %i" % (AQcomp.total_size_of_data,))

        return AQcomp

    @staticmethod
    def lti_sde_to_descrete(
        F,
        L,
        Qc,
        dt,
        compute_derivatives=False,
        grad_params_no=None,
        P_inf=None,
        dP_inf=None,
        dF=None,
        dQc=None,
    ):
        """
        Linear Time-Invariant Stochastic Differential Equation (LTI SDE):

            dx(t) = F x(t) dt + L d \beta  ,where

                x(t): (vector) stochastic process
                \beta: (vector) Brownian motion process
                F, L: (time invariant) matrices of corresponding dimensions
                Qc: covariance of noise.

        This function rewrites it into the corresponding state-space form:

            x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})

        TODO: this function can be redone to "preprocess dataset", when
        close time points are handeled properly (with rounding parameter) and
        values are averaged accordingly.

        Input:
        --------------
        F,L: LTI SDE matrices of corresponding dimensions

        Qc: matrix (n,n)
            Covarince between different dimensions of noise \beta.
            n is the dimensionality of the noise.

        dt: double or iterable
            Time difference used on this iteration.
            If dt is iterable, then A and Q_noise are computed for every
            unique dt

        compute_derivatives: boolean
            Whether derivatives of A and Q are required.

        grad_params_no: int
            Number of gradient parameters

        P_inf: (state_dim. state_dim) matrix

        dP_inf

        dF: 3D array
            Derivatives of F

        dQc: 3D array
            Derivatives of Qc

        dR: 3D array
            Derivatives of R

        Output:
        --------------

        A: matrix
            A_{k}. Because we have LTI SDE only dt can affect on matrix
            difference for different k.

        Q_noise: matrix
            Covariance matrix of (vector) q_{k-1}. Only dt can affect the
            matrix difference for different k.

        reconstruct_index: array
            If dt was iterable return three dimensinal arrays A and Q_noise.
            Third dimension of these arrays correspond to unique dt's.
            This reconstruct_index contain indices of the original dt's
            in the uninue dt sequence. A[:,:, reconstruct_index[5]]
            is matrix A of 6-th(indices start from zero) dt in the original
            sequence.
        dA: 3D array
            Derivatives of A

        dQ: 3D array
            Derivatives of Q
        """
        # Dimensionality
        n = F.shape[0]

        if not isinstance(dt, Iterable):  # not iterable, scalar
            # import pdb; pdb.set_trace()
            # The dynamical model
            A = matrix_exponent(F * dt)

            # The covariance matrix Q by matrix fraction decomposition ->
            Phi = np.zeros((2 * n, 2 * n))
            Phi[:n, :n] = F
            Phi[:n, n:] = L.dot(Qc).dot(L.T)
            Phi[n:, n:] = -F.T
            AB = matrix_exponent(Phi * dt)
            AB = np.dot(AB, np.vstack((np.zeros((n, n)), np.eye(n))))

            Q_noise_1 = linalg.solve(AB[n:, :].T, AB[:n, :].T)
            Q_noise_2 = P_inf - A.dot(P_inf).dot(A.T)
            # The covariance matrix Q by matrix fraction decomposition <-

            if compute_derivatives:
                dA = np.zeros([n, n, grad_params_no])
                dQ = np.zeros([n, n, grad_params_no])

                # AA  = np.zeros([2*n, 2*n, nparam])
                FF = np.zeros([2 * n, 2 * n])
                AA = np.zeros([2 * n, 2 * n, grad_params_no])

                for p in range(0, grad_params_no):
                    FF[:n, :n] = F
                    FF[n:, :n] = dF[:, :, p]
                    FF[n:, n:] = F

                    # Solve the matrix exponential
                    AA[:, :, p] = matrix_exponent(FF * dt)

                    # Solve the differential equation
                    # foo         = AA[:,:,p].dot(np.vstack([m, dm[:,p]]))
                    # mm          = foo[:n,:]
                    # dm[:,p] = foo[n:,:]

                    # The discrete-time dynamical model*
                    if p == 0:
                        A = AA[:n, :n, p]
                        Q_noise_3 = P_inf - A.dot(P_inf).dot(A.T)
                        Q_noise = Q_noise_3
                        # PP = A.dot(P).dot(A.T) + Q_noise_2

                    # The derivatives of A and Q
                    dA[:, :, p] = AA[n:, :n, p]
                    tmp = dA[:, :, p].dot(P_inf).dot(A.T)
                    dQ[:, :, p] = (
                        dP_inf[:, :, p] - tmp - A.dot(dP_inf[:, :, p]).dot(A.T) - tmp.T
                    )

                    dQ[:, :, p] = 0.5 * (dQ[:, :, p] + dQ[:, :, p].T)  # Symmetrize
            else:
                dA = None
                dQ = None
                Q_noise = Q_noise_2
            # Innacuracies have been observed when Q_noise_1 was used.

            # Q_noise = Q_noise_1

            Q_noise = 0.5 * (Q_noise + Q_noise.T)  # Symmetrize
            return A, Q_noise, None, dA, dQ

        else:  # iterable, array
            # Time discretizations (round to 14 decimals to avoid problems)
            dt_unique, tmp, reconstruct_index = np.unique(
                np.round(dt, 8), return_index=True, return_inverse=True
            )
            del tmp
            # Allocate space for A and Q
            A = np.empty((n, n, dt_unique.shape[0]))
            Q_noise = np.empty((n, n, dt_unique.shape[0]))

            if compute_derivatives:
                dA = np.empty((n, n, grad_params_no, dt_unique.shape[0]))
                dQ = np.empty((n, n, grad_params_no, dt_unique.shape[0]))
            else:
                dA = None
                dQ = None
            # Call this function for each unique dt
            for j in range(0, dt_unique.shape[0]):
                (
                    A[:, :, j],
                    Q_noise[:, :, j],
                    tmp1,
                    dA_t,
                    dQ_t,
                ) = ContDescrStateSpace.lti_sde_to_descrete(
                    F,
                    L,
                    Qc,
                    dt_unique[j],
                    compute_derivatives=compute_derivatives,
                    grad_params_no=grad_params_no,
                    P_inf=P_inf,
                    dP_inf=dP_inf,
                    dF=dF,
                    dQc=dQc,
                )
                if compute_derivatives:
                    dA[:, :, :, j] = dA_t
                    dQ[:, :, :, j] = dQ_t

            # Return
            return A, Q_noise, reconstruct_index, dA, dQ


def matrix_exponent(M):
    """
    The function computes matrix exponent and handles some special cases
    """

    if M.shape[0] == 1:  # 1*1 matrix
        Mexp = np.array(((np.exp(M[0, 0]),),))

    else:  # matrix is larger
        method = None
        try:
            Mexp = linalg.expm(M)
            method = 1
        except (Exception,) as e:
            Mexp = linalg.expm3(M)
            method = 2
        finally:
            if np.any(np.isnan(Mexp)):
                if method == 2:
                    raise ValueError("Matrix Exponent is not computed 1")
                else:
                    Mexp = linalg.expm3(M)
                    method = 2
                    if np.any(np.isnan(Mexp)):
                        raise ValueError("Matrix Exponent is not computed 2")

    return Mexp


def balance_matrix(A):
    """
    Balance matrix, i.e. finds such similarity transformation of the original
    matrix A:  A = T * bA * T^{-1}, where norms of columns of bA and of rows of bA
    are as close as possible. It is usually used as a preprocessing step in
    eigenvalue calculation routine. It is useful also for State-Space models.

    See also:
        [1] Beresford N. Parlett and Christian Reinsch (1969). Balancing
            a matrix for calculation of eigenvalues and eigenvectors.
            Numerische Mathematik, 13(4): 293-304.

    Input:
    ----------------------
    A: square matrix
        Matrix to be balanced

    Output:
    ----------------
        bA: matrix
            Balanced matrix

        T: matrix
            Left part of the similarity transformation

        T_inv: matrix
            Right part of the similarity transformation.
    """

    if len(A.shape) != 2 or (A.shape[0] != A.shape[1]):
        raise ValueError("balance_matrix: Expecting square matrix")

    N = A.shape[0]  # matrix size

    gebal = sp.linalg.lapack.get_lapack_funcs("gebal", (A,))
    bA, lo, hi, pivscale, info = gebal(A, permute=True, scale=True, overwrite_a=False)
    if info < 0:
        raise ValueError(
            "balance_matrix: Illegal value in %d-th argument of internal gebal " % -info
        )

    # calculating the similarity transforamtion:
    def perm_matr(D, c1, c2):
        """
        Function creates the permutation matrix which swaps columns c1 and c2.

        Input:
        --------------
        D: int
            Size of the permutation matrix
        c1: int
            Column 1. Numeration starts from 1...D
        c2: int
            Column 2. Numeration starts from 1...D
        """
        i1 = c1 - 1
        i2 = c2 - 1  # indices
        P = np.eye(D)
        P[i1, i1] = 0.0
        P[i2, i2] = 0.0
        # nullify diagonal elements
        P[i1, i2] = 1.0
        P[i2, i1] = 1.0

        return P

    P = np.eye(N)  # permutation matrix
    if hi != N - 1:  # there are row permutations
        for k in range(N - 1, hi, -1):
            new_perm = perm_matr(N, k + 1, pivscale[k])
            P = np.dot(P, new_perm)
    if lo != 0:
        for k in range(0, lo, 1):
            new_perm = perm_matr(N, k + 1, pivscale[k])
            P = np.dot(P, new_perm)
    D = pivscale.copy()
    D[0:lo] = 1.0
    D[hi + 1 : N] = 1.0  # thesee scaling factors must be set to one.
    # D = np.diag(D) # make a diagonal matrix

    T = np.dot(P, np.diag(D))  # similarity transformation in question
    T_inv = np.dot(np.diag(D ** (-1)), P.T)

    # print( np.max(A - np.dot(T, np.dot(bA, T_inv) )) )
    return bA.copy(), T, T_inv


def balance_ss_model(F, L, Qc, H, Pinf, P0, dF=None, dQc=None, dPinf=None, dP0=None):
    """
    Balances State-Space model for more numerical stability

    This is based on the following:

     dx/dt = F x + L w
         y = H x

     Let T z = x, which gives

     dz/dt = inv(T) F T z + inv(T) L w
         y = H T z
    """

    bF, T, T_inv = balance_matrix(F)

    bL = np.dot(T_inv, L)
    bQc = Qc  # not affected

    bH = np.dot(H, T)

    bPinf = np.dot(T_inv, np.dot(Pinf, T_inv.T))

    # import pdb; pdb.set_trace()
    #    LL,islower = linalg.cho_factor(Pinf)
    #    inds = np.triu_indices(Pinf.shape[0],k=1)
    #    LL[inds] = 0.0
    #    bLL = np.dot(T_inv, LL)
    #    bPinf = np.dot( bLL, bLL.T)

    bP0 = np.dot(T_inv, np.dot(P0, T_inv.T))

    if dF is not None:
        bdF = np.zeros(dF.shape)
        for i in range(dF.shape[2]):
            bdF[:, :, i] = np.dot(T_inv, np.dot(dF[:, :, i], T))

    else:
        bdF = None

    if dPinf is not None:
        bdPinf = np.zeros(dPinf.shape)
        for i in range(dPinf.shape[2]):
            bdPinf[:, :, i] = np.dot(T_inv, np.dot(dPinf[:, :, i], T_inv.T))

    #            LL,islower = linalg.cho_factor(dPinf[:,:,i])
    #            inds = np.triu_indices(dPinf[:,:,i].shape[0],k=1)
    #            LL[inds] = 0.0
    #            bLL = np.dot(T_inv, LL)
    #            bdPinf[:,:,i] = np.dot( bLL, bLL.T)

    else:
        bdPinf = None

    if dP0 is not None:
        bdP0 = np.zeros(dP0.shape)
        for i in range(dP0.shape[2]):
            bdP0[:, :, i] = np.dot(T_inv, np.dot(dP0[:, :, i], T_inv.T))
    else:
        bdP0 = None

    bdQc = dQc  # not affected

    # (F,L,Qc,H,Pinf,P0,dF,dQc,dPinf,dP0)

    return bF, bL, bQc, bH, bPinf, bP0, bdF, bdQc, bdPinf, bdP0
