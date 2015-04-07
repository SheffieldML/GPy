# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:32:32 2015


"""

import collections # for cheking whether a variable is iterable
import types # for cheking whether a variable is a function 

import numpy as np
import scipy as sp
import scipy.linalg as linalg

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
    
    @staticmethod
    def reshape_input_data(shape):
        """
        Function returns the column-wise shape for for an input shape.
        
        Inputs:
        --------------
            shape: tuple
                Shape of an input array, so that it is always a column.
            old_shape: tuple or None
                If the shape has been modified, return old shape, otherwise None
        """
    
    
        if (len(shape) > 2):
            raise ValueError("Input array is not supposed to be 3 or more dimensional")            
        elif len(shape) == 1:
            return ((shape[0],1), shape)
        else: # len(shape) == 2
            return ((shape[1],1), shape) if (shape[0] == 1) else (shape,None)
            
    
#    def __init__(self, X, Y, kernel):
#        """
#        Inputs:
#        --------------
#        
#            X: double array (N_samples,1)
#                Time points of the observations
#            Y: double array (N_samples,N_dim)
#                Observations
#        """        
#        
#        # If X is 1D array make it column (x.shape[0],1)
#        X.shape, old_Y_shape  =  DescreteStateSpace.reshape_input_data(X.shape)         
#        Y.shape, old_X_shape  =  DescreteStateSpace.reshape_input_data(Y.shape)  
#        
#        self.num_data, input_dim = X.shape
#        assert input_dim==1, "State space methods for time only"
#        num_data_Y, self.output_dim = Y.shape
#        assert num_data_Y == self.num_data, "Number of samples in X and Y don't match"
#        assert self.output_dim == 1, "State space methods for single outputs only"
#
#        # Make sure the observations are ordered in time
#        sort_index = np.argsort(X[:,0])
#        self.X = X[sort_index]
#        self.Y = Y[sort_index]
#
#        # Noise variance ??? - TODO: figure out
#        self.sigma2 = 1 
#        
#        self.kern = kernel
#
#        # Make sure all parameters are positive
#        #self.ensure_default_constraints()
#
#        # Assert that the kernel is supported
#        #assert self.kern.sde() not False, "This kernel is not supported for state space estimation"

    @classmethod
    def kalman_filter(cls,p_A, p_Q, p_H, p_R, Y, index = None, m_init=None,
                      P_init=None, calc_log_likelihood=False, 
                      calc_grad_log_likelihood=False, grad_params_no=None,
                      grad_calc_params=None):
        """
        This function implements the basic (raw) Kalman Filter algorithm
        These notations are assumed:
            x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})
            
        Returns estimated filter distributions x_{k} ~ N(m_{k}, P(k))            
            
        Currently, H_{k} and R_{k} are not supposed to change over time.
        I. e. They are assumed to be constant. Later they may be done time
        dependent by analogy with A_{k} and Q_{k-1}.


        Input:            
        -----------------
        
        p_A: scalar, square matrix, 3D array
            A_{k} in the model. If matrix then A_{k} = A - constant.
            If it is 3D array then A_{k} = p_A[:,:, index[k]]            
            
        p_Q: scalar, square symmetric matrix, 3D array
            Q_{k-1} in the model. If matrix then Q_{k-1} = Q - constant.
            If it is 3D array then Q_{k-1} = p_Q[:,:, index[k]]
        p_H: scalar, matrix
            H_{k} = p_H. Assumed to be constant.
        
        p_R: scalar, square symmetric matrix
            H_{k} = p_H. Assumed to be constant.
            
        Y: matrix or vector
            Data. If Y is matrix then samples are along 0-th dimension and
            features along the 1-st. May have missing values.
            
        index: vector
            Which indices (on 3-rd dimension) from arrays p_A and p_Q to use
            on every time step. If this parameter is None then it is assumed 
            that p_A and p_Q are matrices and indices are not needed.
            
        p_mean: vector
            Initial distribution mean. If None it is assumed to be zero
            
        P_init: square symmetric matrix or scalar
            Initial covariance of the states. If the parameter is scalar
            then it is assumed that initial covariance matrix is unit matrix 
            multiplied by this scalar. If None the unit matrix is used instead.
            
        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.
            
        calc_grad_log_likelihood: boolean
            Whether to calculate gradient of the marginal likelihood 
            of the state-space model. If true then the next parameter must 
            provide the extra parameters for gradient calculation.
            
        grad_params_no: int
            Number of parameters 
        
        Output:
        --------------
        
        M: (state_dim, no_steps) matrix 
            Filter estimates of the state means
            
        P: (state_dim, state_dim, no_steps) 3D array
            Filter estimates of the state covariances
        
        log_likelihood: double
            If the parameter calc_log_likelihood was set to true, return
            logarithm of marginal likelihood of the state-space model. If
            the parameter was false, return None
            
        """
    
        # Parameters checking ->
        # index        
        p_A = np.atleast_1d(p_A)
        p_Q = np.atleast_1d(p_Q)
        if ((len(p_A.shape) >= 3) and (p_A.shape[2] != 1)) or \
            ((len(p_Q.shape) >= 3) and (p_Q.shape[2] != 1)):
            if index is None:
                raise ValueError("A and Q can not change over time if indices are not given")
            else:
                A_or_Q_change_over_time = True
        else:
            index = np.zeros((Y.shape[0],) )
            A_or_Q_change_over_time = False
            
        # p_A
        (p_A, old_A_shape) = cls._check_A_matrix(p_A)

        # p_Q
        (p_Q,old_Q_shape) = cls._check_Q_matrix(p_Q)
                    
        # p_H
        p_H = np.atleast_1d(p_H)
        (p_H,old_H_shape) = cls._check_H_matrix(p_H)
        
        # p_R
        p_R = np.atleast_1d(p_R)
        (p_R,old_R_shape) = cls._check_R_matrix(p_R)
        
        state_dim = p_A.shape[0]
        # m_init
        if m_init is None:
            m_init = np.zeros((state_dim,1))
        else:
            m_init = np.atleast_2d(m_init).T
        
        # P_init
        if P_init is None:
            P_init = np.eye(state_dim)    
        elif not isinstance(P_init, collections.Iterable): #scalar
            P_init = P_init*np.eye(state_dim)
            
        # Y
        Y.shape, old_Y_shape  = cls.reshape_input_data(Y.shape)
        if (Y.shape[0] != len(index)):
            raise ValueError("Number of measurements must be equal the number of A_{k} and Q_{k}")
                        
        measurement_dim = Y.shape[1]
        # Functions to pass to the kalman_filter algorithm:
        # PArameters:
        # k - number of Kalman filter iteration
        # m - vector for calculating matrices. Required for EKF. Not used here.
        fa = lambda k,m,A: np.dot(A, m)
        f_A = lambda k,m,P: p_A[:,:, index[k]]
        f_Q = lambda k: p_Q[:,:, index[k]]
        fh = lambda k,m,H: np.dot(H, m)
        f_H = lambda k,m,P: p_H
        f_R = lambda k: p_R
        
        
        grad_calc_params_pass_further = None
        if calc_grad_log_likelihood:
            if A_or_Q_change_over_time:
                raise ValueError("When computing likelihood gradient A and Q can not change over time.")    
            
            f_dA = cls._check_grad_state_matrices(grad_calc_params.get('dA'), state_dim, grad_params_no, which = 'dA')
            
            f_dQ = cls._check_grad_state_matrices(grad_calc_params.get('dQ'), state_dim, grad_params_no, which = 'dQ')
            
            f_dH = cls._check_grad_measurement_matrices(grad_calc_params.get('dH'), state_dim, grad_params_no, measurement_dim, which = 'dH')
             
            f_dR = cls._check_grad_measurement_matrices(grad_calc_params.get('dH'), state_dim, grad_params_no, measurement_dim, which = 'dR')
            
            dm_init = grad_calc_params.get('dm_init')
            if dm_init is None:
                dm_init = np.zeros( (state_dim,grad_params_no) )
            
            dP_init = grad_calc_params.get('dP_init')
            if dP_init is None:
                dP_init = np.zeros( (state_dim,state_dim,grad_params_no) )
            
            grad_calc_params_pass_further = {}
            grad_calc_params_pass_further['f_dA'] = f_dA
            grad_calc_params_pass_further['f_dQ'] = f_dQ
            grad_calc_params_pass_further['f_dH'] = f_dH
            grad_calc_params_pass_further['f_dR'] = f_dR
            grad_calc_params_pass_further['dm_init'] = dm_init
            grad_calc_params_pass_further['dP_init'] = dP_init
            
        (M, P,log_likelihood, grad_log_likelihood) = cls._kalman_algorithm_raw(state_dim, fa, f_A, f_Q, fh, f_H, f_R, Y, m_init,
                          P_init, calc_log_likelihood=calc_log_likelihood, 
                          calc_grad_log_likelihood=calc_grad_log_likelihood, 
                          grad_calc_params=grad_calc_params_pass_further)
        # restore shapes so that input parameters are unchenged
        if old_A_shape is not None:
            p_A.shape = old_A_shape
            
        if old_Q_shape is not None:
            p_Q.shape = old_Q_shape
            
        if old_H_shape is not None:
            p_H.shape = old_H_shape
            
        if old_R_shape is not None:
            p_R.shape = old_R_shape
            
        if old_Y_shape is not None:
            Y.shape = old_Y_shape
        # Return values
        return (M, P,log_likelihood, grad_log_likelihood)
        
    @classmethod        
    def extended_kalman_filter(cls,p_state_dim, p_a, p_f_A, p_f_Q, p_h, p_f_H, p_f_R, Y, m_init=None,
                          P_init=None,calc_log_likelihood=False):
                              
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
        Y.shape, old_Y_shape  =  cls.reshape_input_data(Y.shape)
        
         # m_init
        if m_init is None:
            m_init = np.zeros((p_state_dim,1))
        else:
            m_init = np.atleast_2d(m_init).T
            
        # P_init
        if P_init is None:
            P_init = np.eye(p_state_dim)    
        elif not isinstance(P_init, collections.Iterable): #scalar
            P_init = P_init*np.eye(p_state_dim)
        
        if p_a is None:
            p_a = lambda k,m,A: np.dot(A, m)
        
        old_A_shape = None
        if not isinstance(p_f_A, types.FunctionType): # not a function but array
            p_f_A = np.atleast_1d(p_f_A)
            (p_A, old_A_shape) = cls._check_A_matrix(p_f_A)
            
            p_f_A = lambda k, m, P: p_A[:,:, 0] # make function            
        else:
            if p_f_A(1, m_init, P_init).shape[0] != m_init.shape[0]:
                raise ValueError("p_f_A function returns matrix of wrong size")
        
        old_Q_shape = None                        
        if not isinstance(p_f_Q, types.FunctionType): # not a function but array
            p_f_Q = np.atleast_1d(p_f_Q)
            (p_Q, old_Q_shape) = cls._check_Q_matrix(p_f_Q)
            
            p_f_Q = lambda k: p_Q[:,:, 0] # make function            
        else:
            if p_f_Q(1).shape[0] != m_init.shape[0]:
                raise ValueError("p_f_Q function returns matrix of wrong size")   
        
        if p_h is None:
            lambda k,m,H: np.dot(H, m)
            
        old_H_shape = None
        if not isinstance(p_f_H, types.FunctionType): # not a function but array
            p_f_H = np.atleast_1d(p_f_H)
            (p_H, old_H_shape) = cls._check_H_matrix(p_f_H)
            
            p_f_H = lambda k, m, P: p_H # make function            
        else:
            if p_f_H(1, m_init, P_init).shape[0] != Y.shape[1]:
                raise ValueError("p_f_H function returns matrix of wrong size") 

        old_R_shape = None
        if not isinstance(p_f_R, types.FunctionType): # not a function but array
            p_f_R = np.atleast_1d(p_f_R)
            (p_R, old_R_shape) = cls._check_H_matrix(p_f_R)
            
            p_f_R = lambda k: p_R # make function            
        else:
            if p_f_R(1).shape[0] != m_init.shape[0]:
                raise ValueError("p_f_R function returns matrix of wrong size")

        (M, P,log_likelihood, grad_log_likelihood)  = cls._kalman_algorithm_raw(p_state_dim, p_a, p_f_A, p_f_Q, p_h, p_f_H, p_f_R, Y, m_init,
                          P_init, calc_log_likelihood, 
                          calc_grad_log_likelihood=False, grad_calc_params=None)
        
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
    def _kalman_algorithm_raw(cls,state_dim, p_a, p_f_A, p_f_Q, p_h, p_f_H, p_f_R, Y, m_init,
                          P_init, calc_log_likelihood=False, 
                          calc_grad_log_likelihood=False, grad_calc_params=None):
        """
        General nonlinear filtering algorithm for inference in the state-space 
        model:

        x_{k} =  f_a(k, x_{k-1}, A_{k}) + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
        y_{k} =  f_h(k, x_{k}, H_{k}) + r_{k};           r_{k-1} ~ N(0, R_{k})
        
        Returns estimated filter distributions x_{k} ~ N(m_{k}, P(k))            
        
        Input:            
        -----------------
        p_a: function (k, x_{k-1}, A_{k}). Dynamic function.        
            k (iteration number), 
            x_{k-1} 
            x_{k} Jacobian matrices of f_a. In the linear case it is exactly A_{k}.
            
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
            
        Y: matrix or vector
            Data. If Y is matrix then samples are along 0-th dimension and
            features along the 1-st. May have missing values.
            
        m_init: vector
            Initial distribution mean. Must be not None
            
        P_init: matrix or scalar
            Initial covariance of the states. Must be not None
            
        calc_log_likelihood: boolean
            Whether to calculate marginal likelihood of the state-space model.
            
        calc_grad_log_likelihood: boolean
            Whether to calculate gradient of the marginal likelihood 
            of the state-space model. If true then the next parameter must 
            provide the extra parameters for gradient calculation.
        
        
        Output:
        --------------
        
        M: (state_dim, no_steps) matrix 
            Filter estimates of the state means
            
        P: (state_dim, state_dim, no_steps) 3D array
            Filter estimates of the state covariances
        
        log_likelihood: double
            If the parameter calc_log_likelihood was set to true, return
            logarithm of marginal likelihood of the state-space model. If
            the parameter was false, return None
            
        """
            
        steps_no = Y.shape[0] # number of steps in the Kalman Filter
        
        if calc_grad_log_likelihood:
            grad_calc_params_1 = (grad_calc_params['f_dA'], grad_calc_params['f_dQ'])
            grad_calc_params_2 = (grad_calc_params['f_dH'],grad_calc_params['f_dR'])
            
            dm_init = grad_calc_params['dm_init']; dP_init = grad_calc_params['dP_init']
        else:
            grad_calc_params_1 = None
            grad_calc_params_2 = None
        
            dm_init = None; dP_init = None
                    
        # Allocate space for results
        # Mean estimations. Initial values will be included
        M = np.empty((state_dim,(steps_no+1)))
        # Variance estimations. Initial values will be included
        P = np.empty((state_dim,state_dim,(steps_no+1)))

        # Initialize
        M[:,0] = m_init
        P[:,:,0] = P_init
        
        if calc_log_likelihood:
            log_likelihood = 0
        else:
            log_likelihood = None
            
        if calc_grad_log_likelihood:
            grad_log_likelihood = 0
        else:
            grad_log_likelihood = None
        # Main loop of the Kalman filter
        for k in range(0,steps_no):
            # In this loop index for new estimations is (k+1), old - (k)
            # This happened because initial values are stored at 0-th index.                 
            
            if k == 0: #settinginitial values
                dm_upd = dm_init
                dP_upd = dP_init
                
            m_pred, P_pred, dm_pred, dP_pred = \
            cls._kalman_prediction_step(k, M[:,k] ,P[:,:,k], p_a, p_f_A, p_f_Q, 
                calc_grad_log_likelihood=calc_grad_log_likelihood, 
                p_dm = dm_upd, p_dP = dP_upd, grad_calc_params_1 = grad_calc_params_1)
            
            m_upd, P_upd, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update = \
            cls._kalman_update_step(k,  m_pred , P_pred, p_h, p_f_H, p_f_R, Y, 
                        calc_log_likelihood=calc_log_likelihood, 
                        calc_grad_log_likelihood=calc_grad_log_likelihood, 
                        p_dm = dm_pred, p_dP = dP_pred, grad_calc_params_2 = grad_calc_params_2)                
            
            if calc_log_likelihood:
                log_likelihood += log_likelihood_update
            
            if calc_grad_log_likelihood:
                grad_log_likelihood += d_log_likelihood_update
            
            M[:,k+1] = m_upd
            P[:,:,k+1] = P_upd
        # Return values, get rid of initial values
        #return (M[:,1:], P[:,:,1:], log_likelihood, grad_log_likelihood)
        return (M, P, log_likelihood, grad_log_likelihood)
        
    @staticmethod    
    def _kalman_prediction_step(k, p_m , p_P, p_a, p_f_A, p_f_Q, calc_grad_log_likelihood=False, 
                                p_dm = None, p_dP = None, grad_calc_params_1 = None):
        """
        Desctrete prediction function        
        
        Input:
            k:int
                Iteration No. Starts at 0. Total number of iterations equal to the 
                number of measurements.
              
            p_m: matrix of size (state_dim, time_series_no)
                Mean value from the previous step                
            p_P:
                Covariance matrix from the previous step.
                
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
                
            calc_grad_log_likelihood: boolean
                Whether to calculate gradient of the marginal likelihood 
                of the state-space model. If true then the next parameter must 
                provide the extra parameters for gradient calculation.
            p_dm: 3D array (state_dim, time_series_no, parameters_no)
                Mean derivatives from the previous step
                
            p_dP: 3D array (state_dim, state_dim, parameters_no)
                Mean derivatives from the previous step
            
        """
        
        # index correspond to values from previous iteration.
        A = p_f_A(k,p_m,p_P) # state transition matrix (or Jacobian)
        Q = p_f_Q(k) # state noise matrix 
                  
        
        # Prediction step ->
        m_pred = p_a(k, p_m, A) # predicted mean
        P_pred = A.dot(p_P).dot(A.T) + Q # predicted variance
        # Prediction step <-
        
        
        if calc_grad_log_likelihood:
            p_f_dA = grad_calc_params_1[0]; dA_all_params = p_f_dA(k) # derivatives of A wrt parameters 
            p_f_dQ = grad_calc_params_1[1]; dQ_all_params = p_f_dQ(k) # derivatives of Q wrt parameters
            dm_all_params = p_dm # derivarites from the previous step 
            dP_all_params = p_dP 
            
            param_number = p_dP.shape[2]
            
            dm_pred = np.empty(dm_all_params.shape)
            dP_pred = np.empty(dP_all_params.shape)
            
            for j in range(param_number):
                dA = dA_all_params[:,:,j]
                dQ = dQ_all_params[:,:,j]
                
                
                dm = dm_all_params[:,j]                 
                dP = dP_all_params[:,:,j]
                
                # prediction step derivatives for current parameter:            
                dm_pred[:,j] =  np.dot(dA, p_m) + np.dot(A, dm)
                dP_pred[:,:,j] = np.dot( dA ,np.dot(p_P, A.T))
                dP_pred[:,:,j] += dP_pred[:,:,j].T            
                dP_pred[:,:,j] += np.dot( A ,np.dot(dP, A.T)) + dQ
                
        else:
            dm_pred = None
            dP_pred = None
            
        return m_pred, P_pred, dm_pred, dP_pred
    
    
    @staticmethod
    def _kalman_update_step(k,   p_m , p_P, p_h, p_f_H, p_f_R, Y, calc_log_likelihood= False, 
                            calc_grad_log_likelihood=False, p_dm = None, p_dP = None, grad_calc_params_2 = None):
        """
        Input:
        
        k: int
              Iteration No. Starts at 0. Total number of iterations equal to the 
              number of measurements.
          
        m_P: matrix of size (state_dim, time_series_no)
             Mean value from the previous step.
                
        p_P:
             Covariance matrix from the previous step.
             
        p_h: function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number), starts at 0
            x_{k} state 
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly H_{k}.
        
        p_f_H: function (k, m, P) return Jacobian of dynamic function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        
        p_f_R: function (k). Returns noise matrix of measurement equation 
            on iteration k.
            k (iteration number). starts at 0
        calc_grad_log_likelihood: int
        
        p_dm: array
            Mean derivatives from the prediction step
        
        p_dP: array
            Covariance derivatives from the prediction step
        """        
        
        m_pred = p_m
        P_pred = p_P        
        
        H = p_f_H(k, m_pred, P_pred)
        R = p_f_R(k)
        measurement = Y[k,:].T # measurement as column
        
        log_likelihood_update=None; dm_upd=None; dP_upd=None; d_log_likelihood_update=None
        # Update step (only if there is data)
        if not np.any(np.isnan(measurement)): # TODO: if some dimensions are missing, do properly computations for other.
             v = measurement-p_h(k, m_pred, H)
             S = H.dot(P_pred).dot(H.T) + R
             if Y.shape[1]==1: # measurements are one dimensional
                 K = P_pred.dot(H.T) / S
                 if calc_log_likelihood:
                     log_likelihood_update = -0.5 * ( np.log(2*np.pi) + np.log(S) +
                                         v*v / S)
                     log_likelihood_update = log_likelihood_update[0,0] # to make int
                     if np.isnan(log_likelihood_update):
                         pass
                 LL = None; islower = None
             else:
                 LL,islower = linalg.cho_factor(S)
                 K = linalg.cho_solve((LL,islower), H.dot(P_pred.T)).T
                 
                 if calc_log_likelihood:
                     log_likelihood_update = -0.5 * ( v.shape[0]*np.log(2*np.pi) + 
                         2*np.sum( np.log(np.diag(LL)) ) + 
                         np.dot(v.T, linalg.cho_solve((LL,islower),v)) )
                     log_likelihood_update = log_likelihood_update[0,0] # to make int  
                 
            
             if calc_grad_log_likelihood:
                 dm_pred_all_params = p_dm # derivativas of the prediction phase 
                 dP_pred_all_params = p_dP
                 
                 param_number = p_dP.shape[2]
                 
                 p_f_dH = grad_calc_params_2[0]; dH_all_params = p_f_dH(k)
                 p_f_dR = grad_calc_params_2[1]; dR_all_params = p_f_dR(k)
                 
                 dm_upd = np.empty(dm_pred_all_params.shape)
                 dP_upd = np.empty(dP_pred_all_params.shape)
                 
                 d_log_likelihood_update = np.empty((param_number,1))
                 for param in range(param_number):
                
                    dH = dH_all_params[:,:,param]
                    dR = dR_all_params[:,:,param]
                    
                    dm_pred = dm_pred_all_params[:,param]
                    dP_pred = dP_pred_all_params[:,:,param]
                
                    # Terms in the likelihood derivatives
                    dv = - np.dot( dH, m_pred) -  np.dot( H, dm_pred)           
                    dS = np.dot(dH, np.dot( P_pred, H.T))
                    dS += dS.T
                    dS += np.dot(H, np.dot( dP_pred, H.T)) + dR
                
                    # TODO: maybe symmetrize dS
                
                    #dm and dP for the next stem
                    if LL is not None: # the state vector is not a scalar
                        tmp1 = linalg.cho_solve((LL,islower), H).T
                        tmp2 = linalg.cho_solve((LL,islower), dH).T
                        tmp3 = linalg.cho_solve((LL,islower), dS).T
                    else: # the state vector is a scalar
                        tmp1 = H.T / S
                        tmp2 = dH.T / S
                        tmp3 = dS.T / S
                        
                    dK = np.dot( dP_pred, tmp1) + np.dot( P_pred, tmp2) - \
                         np.dot( P_pred, np.dot( tmp1, tmp3 ) )
                    
                    # terms required for the next step, save this for each parameter
                    dm_upd[:,param] = dm_pred + np.dot(dK, v) + np.dot(K, dv)
                    dP_upd[:,:,param] = -np.dot(dK, np.dot(S, K.T))      
                    dP_upd[:,:,param] += dP_upd[:,:,param].T
                    dP_upd[:,:,param] += dP_pred - np.dot(K , np.dot( dS, K.T))
                    
                    dP_upd[:,:,param] = 0.5*(dP_upd[:,:,param] + dP_upd[:,:,param].T) #symmetrize
                    # computing the likelihood change for each parameter:
                    if LL is not None: # the state vector is not 1D
                        #tmp4 = linalg.cho_solve((LL,islower), dv)
                        tmp5 = linalg.cho_solve((LL,islower), v)
                    else: # the state vector is a scalar
                        #tmp4 = dv / S
                        tmp5 = v / S
                        
                        
                    d_log_likelihood_update[param,0] = -(0.5*np.sum(np.diag(tmp3)) + \
                    np.dot(tmp5.T, dv) - 0.5 * np.dot(tmp5.T ,np.dot(dS, tmp5)) ) 
                 
            
            
            # Compute the actual updates for mean and variance of the states.
             m_upd = m_pred + K.dot( v )
             
             # Covariance update and ensure it is symmetric
             P_upd = K.dot(S).dot(K.T)
             P_upd = 0.5*(P_upd + P_upd.T)
             P_upd =  P_pred - P_upd# this update matrix is symmetric
        
             return m_upd, P_upd, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update
    
    @staticmethod
    def _rts_smoother_update_step(k, p_m , p_P, p_m_pred, p_P_pred, p_m_prev_step, 
                                  p_P_prev_step, p_f_A):
        """
        Input:
        
        k: int
              Iteration No. Starts at 0. Total number of iterations equal to the 
              number of measurements.
          
        p_m: matrix of size (state_dim, time_series_no)
             Filter mean on step k
                
        p_P:
             Filter Covariance on step k
        
        p_m_pred: matrix of size (state_dim, time_series_no)
             Prediction mean
                
        p_P_pred:
             Prediction covariance:
             
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
        
        A = p_f_A(k,p_m,p_P) # state transition matrix (or Jacobian)
        
        tmp = np.dot( A, p_P.T)
        if A.shape[0] == 1: # 1D states
            G = tmp.T / p_P_pred # P[:,:,k] is symmetric
        else:
            LL,islower = linalg.cho_factor(p_P_pred)
            G = linalg.cho_solve((LL,islower),tmp).T
        
        m_upd = p_m + G.dot( p_m_prev_step-p_m_pred )
        P_upd = p_P + G.dot( p_P_prev_step-p_P_pred).dot(G.T)
         
        P_upd = 0.5*(P_upd + P_upd.T)
        
        return m_upd, P_upd
             
    @classmethod  
    def _rts_smoother_raw(cls,state_dim, p_a, p_f_A, p_f_Q, filter_means, 
                          filter_covars):
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
            
        filter_means:
            Include initial values
            
        filter_covars
            Include initial values
            
        Output:
        -------------
        
        M: (state_dim, no_steps) matrix 
            Smoothed estimates of the state means
            
        P: (state_dim, state_dim, no_steps) 3D array
            Smoothed estimates of the state covariances
        """

        no_steps = filter_covars.shape[-1] # number 
      
        M = np.empty(filter_means.shape) # smoothed means
        P = np.empty(filter_covars.shape) # smoothed covars
        
        M[:,-1] = filter_means[:,-1]
        P[:,:,-1] = filter_covars[:,:,-1]
        for k in range(no_steps-2,-1,-1): 
            
            m_pred, P_pred, tmp1, tmp2 = \
                    cls._kalman_prediction_step(k, filter_means[:,k], 
                                                filter_covars[:,:,k], p_a, p_f_A, p_f_Q, 
                                                calc_grad_log_likelihood=False) 
            
            m_upd, P_upd = cls._rts_smoother_update_step(k, 
                            filter_means[:,k] ,filter_covars[:,:,k], 
                            m_pred, P_pred, M[:,k+1] ,P[:,:,k+1], p_f_A)
                      
            M[:,k] = m_upd
            P[:,:,k] = P_upd
        # Return values
            
        return (M, P)
    
    @staticmethod
    def _check_A_matrix(p_A):
        """
        Check A matrix so that it has right shape
        """
        
        old_A_shape = None
        if len(p_A.shape) < 3:         
            old_A_shape = p_A.shape # save shape to restore it on exit
            if len(p_A.shape) == 2: # matrix
                p_A.shape = (p_A.shape[0],p_A.shape[1],1)
                if p_A.shape[0] != p_A.shape[1]:
                    raise ValueError("p_A must be a square matrix")
                
            elif len(p_A.shape) == 1: # scalar but in array already 
                if (p_A.shape[0] != 1):
                    raise ValueError("Parameter p_A is an 1D array, while it must be matrix or scalar")
                else:
                    p_A.shape = (1,1,1)
                    
        return (p_A,old_A_shape)
        
    @staticmethod
    def _check_Q_matrix(p_Q):
        """
        Check Q matrix
        """
        
        old_Q_shape = None
        if len(p_Q.shape) < 3:         
            old_Q_shape = p_Q.shape # save shape to restore it on exit
            if len(p_Q.shape) == 2: # matrix
                p_Q.shape = (p_Q.shape[0],p_Q.shape[1],1)
                if p_Q.shape[0] != p_Q.shape[1]:
                    raise ValueError("p_Q must be a square matrix")
                    
            elif len(p_Q.shape) == 1: # scalar but in array already 
                if (p_Q.shape[0] != 1):
                    raise ValueError("Parameter p_Q is an 1D array, while it must be matrix or scalar")
                else:
                    p_Q.shape = (1,1,1)
                    
        return (p_Q,old_Q_shape)
        
    @staticmethod 
    def _check_H_matrix(p_H):
        """
        Check H matrix
        """

        old_H_shape = None
        if (len(p_H.shape) == 1):
            if p_H.shape[0] != 1:
                raise ValueError("Ambiguity in the shape of p_H")
            else:
                old_H_shape = p_H.shape
                p_H.shape = (1,1)
                
        return (p_H, old_H_shape)
        
    @staticmethod        
    def _check_R_matrix(p_R):
        """
        Check R matrix
        """        
        
        old_R_shape = None
        if (len(p_R.shape) == 1):
            if p_R.shape[0] != 1:
                raise ValueError("Ambiguity in the shape of p_H")
            else:
                old_R_shape = p_R.shape
                p_R.shape = (1,1)
                
        return (p_R, old_R_shape)
        
    @staticmethod        
    def _check_grad_state_matrices(dM, state_dim, grad_params_no, which = 'dA'):
        """
        Check dA, dQ matrices
        
        Input:
            
            which: string 'dA' or 'dQ'
        """        
        
        
        if dM is None:
            dM=np.zeros((state_dim,state_dim,grad_params_no))                                
        elif isinstance(dM, np.ndarray):
            if state_dim == 1:
                if len(dM.shape) < 3:
                    dM.shape = (1,1,1)
            else:
                if len(dM.shape) < 3:
                    dM.shape = (state_dim,state_dim,1)
        elif isinstance(dM, np.int):
            if state_dim > 1:
                raise ValueError("When computing likelihood gradient wrong %s dimension." % which)
            else:
                dM = np.ones((1,1,1)) * dM
        if not isinstance(dM, types.FunctionType):
            f_dM = lambda k: dM
        else:
            f_dM = dM
                
        return f_dM
        
        
    @staticmethod        
    def _check_grad_measurement_matrices(dM, state_dim, grad_params_no, measurement_dim, which = 'dH'):
        """
        Check dA, dQ matrices
        
        Input:
            
            which: string 'dH' or 'dR'
        """  
        
        if dM is None:
            if which == 'dH':
                dM=np.zeros((measurement_dim ,state_dim,grad_params_no))
            elif  which == 'dR':
                dM=np.zeros((measurement_dim,measurement_dim,grad_params_no))               
        elif isinstance(dM, np.ndarray):
            if state_dim == 1:
                if len(dM.shape) < 3:
                    dM.shape = (1,1,1)
            else:
                if len(dM.shape) < 3:
                     if which == 'dH':
                        dM.shape = (measurement_dim,state_dim,1)
                     elif  which == 'dR':
                        dM.shape = (measurement_dim,measurement_dim,1)   
        elif isinstance(dM, np.int):
            if state_dim > 1:
                raise ValueError("When computing likelihood gradient wrong dH dimension.")
            else:
                dM = np.ones((1,1,1)) * dM
        if not isinstance(dM, types.FunctionType):
            f_dM = lambda k: dM
        else:
            f_dM = dM
            
        return f_dM
        
        
        
class Struct(object):
    pass

class ContDescrStateSpace(DescreteStateSpace):
    """
    Class for continuous-discrete Kalman filter. State equation is
    continuous while measurement equation is discrete.
    
        d x(t)/ dt = F x(t) + L q;        where q~ N(0, Qc)
        y_{t_k} = H_{k} x_{t_k} + r_{k};        r_{k-1} ~ N(0, R_{k})
    
    """
    
    class AQcompute_once(object):
        """
        Class for providing the functions for A and Q dA and dQ calculation
        and for caching the result of the call
        """
        
        def __init__(self, F,L,Qc,dt,compute_derivatives=False, grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None):
            """
            Input:
            
                dt: array
                    Array of dt steps
            """
            self.dt = dt                    
            self.F = F
            self.L = L
            self.Qc = Qc
            
            self.P_inf = P_inf            
            self.dP_inf = dP_inf
            self.dF = dF
            self.dQc = dQc            
            
            self.compute_derivatives = compute_derivatives             
            self.grad_params_no =  grad_params_no           
                      
            
            self.last_k = 0
            self.last_k_computed = False
            self.Ak = None
            self.Qk = None
        
        def recompute_for_new_k(self,k):
            """
            """
            
            if self.last_k == k:
                if (self.last_k_computed == False):        
                    Ak,Qk, tmp, dAk, dQk = ContDescrStateSpace.lti_sde_to_descrete(self.F,
                        self.L,self.Qc,self.dt[k],self.compute_derivatives, 
                        grad_params_no=self.grad_params_no, P_inf=self.P_inf, dP_inf=self.dP_inf, dF=self.dF, dQc=self.dQc)
                    self.Ak = Ak
                    self.Qk = Qk
                    self.dAk = dAk
                    self.dQk = dQk
                    self.last_k_computed = True
                else:
                    Ak = self.Ak
                    Qk = self.Qk
                    dAk = self.dAk
                    dQk = self.dQk
            else:
                Ak,Qk, tmp, dAk, dQk = ContDescrStateSpace.lti_sde_to_descrete(self.F,
                        self.L,self.Qc,self.dt[k],self.compute_derivatives, 
                        grad_params_no=self.grad_params_no, P_inf=self.P_inf, dP_inf=self.dP_inf, dF=self.dF, dQc=self.dQc)
                
                self.last_k = k
                self.last_k_computed = True
                self.Ak = Ak
                self.Qk = Qk
                self.dAk = dAk
                self.dQk = dQk
            return Ak,Qk, dAk, dQk 
        
        def reset(self, compute_derivatives):
            """
            For reusing this object e.g. in smoother computation
            """
            
            self.last_k = 0
            self.last_k_computed = False
            self.compute_derivatives = compute_derivatives
            
            return self
            
        def f_A(self,k,m,P):
            
            Ak,Qk, dAk, dQk = self.recompute_for_new_k(k)
            return Ak
            
        def f_Q(self,k):
            
            Ak,Qk, dAk, dQk = self.recompute_for_new_k(k)
            
            return Qk
            
        def f_dA(self, k):
            
            Ak,Qk, dAk, dQk = self.recompute_for_new_k(k)
            
            return dAk
        
        def f_dQ(self, k):
            
            Ak,Qk, dAk, dQk = self.recompute_for_new_k(k)
            
            return dQk
            
            
    class AQcompute_batch(object):
        """
        """
        def __init__(self, F,L,Qc,dt,compute_derivatives=False, grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None):
            """
            Input:
            
                dt: array
                    Array of dt steps
            """
            As, Qs, reconstruct_indices, dAs, dQs = ContDescrStateSpace.lti_sde_to_descrete(F,
                        L,Qc,dt,compute_derivatives, 
                        grad_params_no=grad_params_no, P_inf=P_inf, dP_inf=dP_inf, dF=dF, dQc=dQc)
                        
            self.As = As
            self.Qs = Qs
            self.dAs = dAs
            self.dQs = dQs
            self.reconstruct_indices = reconstruct_indices
        
        def reset(self, compute_derivatives):
            """
            For reusing this object e.g. in smoother computation
            """
            
            return self
            
        def f_A(self,k,m,P): 
            """
            """
            return self.As[:,:, self.reconstruct_indices[k]]
            
        def f_Q(self,k):
            """
            """
            return self.Qs[:,:, self.reconstruct_indices[k]]
        
        def f_dA(self,k): 
            """
            """
            return self.dAs[:,:, :, self.reconstruct_indices[k]]
        
        def f_dQ(self,k): 
            """
            """
            return self.dQs[:,:, :, self.reconstruct_indices[k]]
    
    @classmethod
    def cont_discr_kalman_filter(cls,F,L,Qc,H,R,P_inf,X,Y,m_init=None,
                      P_init=None, calc_log_likelihood=False, 
                      calc_grad_log_likelihood=False, grad_params_no=None, grad_calc_params=None):
        """
        
        """
        
        # Time step lengths
        state_dim = F.shape[0]
        measurement_dim = Y.shape[1]
        
        if m_init is None:
            m_init = np.zeros((state_dim,1))
        
        if P_init is None:
            P_init = P_inf.copy()
        
        if calc_grad_log_likelihood:
            
            dF = cls._check_grad_state_matrices( grad_calc_params.get('dF'), state_dim, grad_params_no, which = 'dA') 
            dQc = cls._check_grad_state_matrices( grad_calc_params.get('dQc'), state_dim, grad_params_no, which = 'dQ')
            dP_inf = cls._check_grad_state_matrices(grad_calc_params.get('dP_inf'), state_dim, grad_params_no, which = 'dQ')
            
            dH = cls._check_grad_measurement_matrices(grad_calc_params.get('dH'), state_dim, grad_params_no, measurement_dim, which = 'dH')
            dR = cls._check_grad_measurement_matrices(grad_calc_params.get('dR'), state_dim, grad_params_no, measurement_dim, which = 'dR')
            
            dm_init = grad_calc_params.get('dm_init') # Initial values for the Kalman Filter
            if dm_init is None:
                dm_init = np.zeros( (state_dim,grad_params_no) )
            
            dP_init = grad_calc_params.get('dP_init') # Initial values for the Kalman Filter
            if dP_init is None:
                dP_init = dP_inf(0).copy() # get the dP_init matrix, because now it is a function
                
        else:
            dP_inf = None
            dF = None
            dQc = None
            dH = None
            dR = None
        
            
        # TODO: Defaults for m_init, P_init, dm_init, dP_init. !!!
        # Also for dH, dR and probably for all derivatives
        (M, P, log_likelihood, grad_log_likelihood, AQcomp) = cls._cont_discr_kalman_filter_raw(state_dim,F,L,Qc,H,R,P_inf,X,Y,m_init=m_init,
                      P_init=P_init, calc_log_likelihood=calc_log_likelihood, 
                      calc_grad_log_likelihood=calc_grad_log_likelihood, grad_params_no=grad_params_no, dP_inf=dP_inf, 
                      dF = dF, dQc=dQc, dH=dH, dR=dR, dm_init=dm_init, dP_init=dP_init)
                      
        return (M, P, log_likelihood, grad_log_likelihood,AQcomp)
        
    @classmethod
    def _cont_discr_kalman_filter_raw(cls,state_dim,F,L,Qc,H,R,P_inf,X,Y,m_init=None,
                      P_init=None, calc_log_likelihood=False, 
                      calc_grad_log_likelihood=False, grad_params_no=None, dP_inf=None, 
                      dF = None, dQc=None, dH=None, dR=None, dm_init=None, dP_init=None):
        """
        Kalman filter for continuos dynamic model.
        
        Input:
            F:
                Dynamic model matrix
            
            X:
                Time steps
        """

        steps_no = X.shape[0] # number of steps in the Kalman Filter
        
        # Return object which computes A, Q and possibly derivatives on the way, pass the derivative matrices not functions
        AQcomp = cls._cont_to_discrete_object(X, F,L,Qc,compute_derivatives=calc_grad_log_likelihood, 
                                              grad_params_no=grad_params_no, 
                                              P_inf=P_inf, dP_inf=dP_inf(0), 
                                              dF = dF(0), dQc=dQc(0))
        
        # Functions required for Kalman filter
        f_a = lambda k,m,A: np.dot(A, m) # state dynamic model
        f_h = lambda k,m,H: np.dot(H, m) # measurement model
        f_H = lambda k,m,P: H
        f_R = lambda k: R
        
        # Allocate space for results
        # Mean estimations. Initial values will be included
        M = np.empty((state_dim,(steps_no+1)))
        # Variance estimations. Initial values will be included
        P = np.empty((state_dim,state_dim,(steps_no+1)))

        # Initialize
        M[:,0] = m_init[:,0]
        P[:,:,0] = P_init
        
        log_likelihood = 0
        grad_log_likelihood = np.zeros((grad_params_no,1))
        # Main loop of the Kalman filter
        for k in range(0,steps_no):
            # In this loop index for new estimations is (k+1), old - (k)
            # This happened because initial values are stored at 0-th index.                 
            
            if k == 0: #setting initial values
                dm_upd = dm_init
                dP_upd = dP_init
                
            m_pred, P_pred, dm_pred, dP_pred = \
            cls._kalman_prediction_step(k, M[:,k] ,P[:,:,k], f_a, AQcomp.f_A, AQcomp.f_Q, 
                calc_grad_log_likelihood=calc_grad_log_likelihood, 
                p_dm = dm_upd, p_dP = dP_upd, grad_calc_params_1 = (AQcomp.f_dA, AQcomp.f_dQ) )
            
            m_upd, P_upd, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update = \
            cls._kalman_update_step(k,  m_pred , P_pred, f_h, f_H, f_R, Y, 
                        calc_log_likelihood=calc_log_likelihood, 
                        calc_grad_log_likelihood=calc_grad_log_likelihood, 
                        p_dm = dm_pred, p_dP = dP_pred, grad_calc_params_2 = (dH, dR))                
            
            if calc_log_likelihood:
                log_likelihood += log_likelihood_update
            
            if calc_grad_log_likelihood:
                grad_log_likelihood += d_log_likelihood_update
            
            M[:,k+1] = m_upd
            P[:,:,k+1] = P_upd
        # Return values, get rid of initial values
        #return (M[:,1:], P[:,:,1:], log_likelihood, grad_log_likelihood)
        # Return values, get rid of initial values
        #return (M[:,1:], P[:,:,1:], log_likelihood, grad_log_likelihood)
        return (M, P, log_likelihood, grad_log_likelihood, AQcomp.reset(False))
    
    @classmethod  
    def cont_discr_rts_smoother(cls,state_dim, filter_means, filter_covars, 
                                AQcomp=None, X=None, F=None,L=None,Qc=None):
        """
        
        UPDATE Description!
        
        This function implements Rauch–Tung–Striebel(RTS) smoother algorithm
        based on the results of kalman_filter_raw.
        These notations are the same:
            x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
            y_{k} = H_{k} * x_{k} + r_{k};            r_{k-1} ~ N(0, R_{k})
            
        Returns estimated smoother distributions x_{k} ~ N(m_{k}, P(k))            
            
        Input:
        --------------
        
        filter_means:
            Include initial values
            
        filter_covars
            Include initial values
            
        Output:
        -------------
        
        M: (state_dim, no_steps) matrix 
            Smoothed estimates of the state means
            
        P: (state_dim, state_dim, no_steps) 3D array
            Smoothed estimates of the state covariances
        """
        
        if AQcomp is None: # make this object from scratch
            AQcomp = cls._cont_to_discrete_object(cls, X, F,L,Qc,compute_derivatives=False, 
                                                  grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None)
        
        f_a = lambda k,m,A: np.dot(A, m) # state dynamic model
        
        no_steps = filter_covars.shape[-1] # number 
      
        M = np.empty(filter_means.shape) # smoothed means
        P = np.empty(filter_covars.shape) # smoothed covars
        
        M[:,-1] = filter_means[:,-1]
        P[:,:,-1] = filter_covars[:,:,-1]
        for k in range(no_steps-2,-1,-1): 
            
            m_pred, P_pred, tmp1, tmp2 = \
                    cls._kalman_prediction_step(k, filter_means[:,k], 
                                                filter_covars[:,:,k], f_a, AQcomp.f_A, AQcomp.f_Q, 
                                                calc_grad_log_likelihood=False) 
            
            m_upd, P_upd = cls._rts_smoother_update_step(k, 
                            filter_means[:,k] ,filter_covars[:,:,k], 
                            m_pred, P_pred, M[:,k+1] ,P[:,:,k+1], AQcomp.f_A)
                      
            M[:,k] = m_upd
            P[:,:,k] = P_upd
        # Return values
            
        return (M, P)
    
    @classmethod
    def _cont_to_discrete_object(cls, X, F,L,Qc,compute_derivatives=False,grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None):
        """
        Function return the object which can be used in Kalman filter and/or
        smoother to obtain matrices A and Q as well as necessary derivatives(clarify).
        
        """        
        
        unique_round_decimals = 8
        dt = np.empty((X.shape[0],))
        dt[1:] = np.diff(X[:,0],axis=0)
        dt[0]  = dt[1]
        unique_indices = np.unique(np.round(dt, decimals=unique_round_decimals))
    
        if len(unique_indices) > 20:        
            AQcomp = cls.AQcompute_once(F,L,Qc, dt,compute_derivatives=compute_derivatives,
                                    grad_params_no=grad_params_no, P_inf=P_inf, dP_inf=dP_inf, dF=dF, dQc=dQc)
        else:
            AQcomp = cls.AQcompute_batch(F,L,Qc,dt,compute_derivatives=compute_derivatives,
                                    grad_params_no=grad_params_no, P_inf=P_inf, dP_inf=dP_inf, dF=dF, dQc=dQc) 
    
        return AQcomp
        
    @staticmethod
    def lti_sde_to_descrete(F,L,Qc,dt,compute_derivatives=False, 
                            grad_params_no=None, P_inf=None, 
                            dP_inf=None, dF = None, dQc=None):
        """
        Linear Time-Invariant Stochastic Differential Equation (LTI SDE):
        
            dx(t) = F x(t) dt + L d \beta  ,where
            
                x(t): (vector) stochastic process
                \beta: (vector) Brownian motion process
                F, L: (time invariant) matrices of corresponding dimensions 
                Qc: covariance of noise.
    
        This function rewrites it into the corresponding state-space form:
        
            x_{k} =  A_{k} * x_{k-1} + q_{k-1};       q_{k-1} ~ N(0, Q_{k-1})
            
        
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
        
        compute_derivatives
        
        param_num: int
            Number of parameters
        
        P_inf
        
        dP_inf
        
        dF
        
        dQc
        
        dR
        
        dm_prev: 
            Mean derivatives from the previos step
        dP_prev:
            Variance derivatives from the previous step
        
        
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
        """         
         
        # Dimensionality
        n = F.shape[0]
        
        if not isinstance(dt, collections.Iterable): # not iterable, scalar
 
            # The dynamical model
            A  = linalg.expm(F*dt) 
 
            # The covariance matrix Q by matrix fraction decomposition ->
            Phi = np.zeros((2*n,2*n))
            Phi[:n,:n] = F
            Phi[:n,n:] = L.dot(Qc).dot(L.T)
            Phi[n:,n:] = -F.T
            AB = linalg.expm(Phi*dt).dot(np.vstack((np.zeros((n,n)),np.eye(n))))
            Q_noise_1 = linalg.solve(AB[n:,:].T,AB[:n,:].T)
            # The covariance matrix Q by matrix fraction decomposition <-
            
            if compute_derivatives:
                dA = np.zeros([n, n, grad_params_no])
                dQ = np.zeros([n, n, grad_params_no])
                
                #AA  = np.zeros([2*n, 2*n, nparam])
                FF  = np.zeros([2*n, 2*n])
                AA = np.zeros([2*n, 2*n, grad_params_no])
                
                for p in range(0, grad_params_no):
                    
                    FF[:n,:n] = F
                    FF[n:,:n] = dF[:,:,p]
                    FF[n:,n:] = F

                    # Solve the matrix exponential
                    AA[:,:,p] = linalg.expm(FF*dt)

                    # Solve the differential equation
                    #foo         = AA[:,:,p].dot(np.vstack([m, dm[:,p]]))
                    #mm          = foo[:n,:]
                    #dm[:,p] = foo[n:,:]

                    # The discrete-time dynamical model*
                    if p==0:
                        A  = AA[:n,:n,p]
                        Q_noise_2  = P_inf - A.dot(P_inf).dot(A.T)
                        Q_noise = Q_noise_2
                        #PP = A.dot(P).dot(A.T) + Q_noise_2
    
                    # The derivatives of A and Q
                    dA[:,:,p] = AA[n:,:n,p]
                    dQ[:,:,p] = dP_inf[:,:,p] - dA[:,:,p].dot(P_inf).dot(A.T) \
                       - A.dot(dP_inf[:,:,p]).dot(A.T) - A.dot(P_inf).dot(dA[:,:,p].T) # Rewrite not ro multiply two times
    
                    # The derivatives of P
                    #dP[:,:,p] = dA.dot(P).dot(A.T) + A.dot(dP_prev[:,:,p]).dot(A.T) \
                    #   + A.dot(P).dot(dA.T) + dQ
                    #Q_noise = Q_noise_2
            else: 
              dA = None
              dQ = None
              Q_noise = Q_noise_1


            # Return
            return A, Q_noise,None, dA, dQ

        else: # iterable, array

            # Time discretizations (round to 14 decimals to avoid problems)
            dt_unique, tmp, reconstruct_index = np.unique(np.round(dt,8),
                                        return_index=True,return_inverse=True)
            del tmp
            # Allocate space for A and Q
            A = np.empty((n,n,dt_unique.shape[0]))
            Q_noise = np.empty((n,n,dt_unique.shape[0]))

            if compute_derivatives:
                dA = np.empty((n,n,grad_params_no,dt_unique.shape[0]))                 
                dQ = np.empty((n,n,grad_params_no,dt_unique.shape[0])) 
                
            # Call this function for each unique dt
            for j in range(0,dt_unique.shape[0]):
                A[:,:,j], Q_noise[:,:,j], tmp1, dA[:,:,:,j], dQ[:,:,:,j] = ContDescrStateSpace.lti_sde_to_descrete(F,L,Qc,dt_unique[j],
                    compute_derivatives=compute_derivatives, grad_params_no=grad_params_no, P_inf=P_inf, dP_inf=dP_inf, dF = dF, dQc=dQc)

            # Return
            return A, Q_noise, reconstruct_index, dA, dQ