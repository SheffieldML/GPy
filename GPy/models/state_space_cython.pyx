# -*- coding: utf-8 -*-
"""
Contains some cython code for state space modelling.
"""
import numpy as np
cimport numpy as np
import scipy as sp
cimport cython

#from libc.math cimport isnan # for nan checking in kalman filter cycle
cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

DTYPE = float
DTYPE_int = np.int64

ctypedef float_t DTYPE_t
ctypedef np.int64_t DTYPE_int_t

# Template class for dynamic callables
cdef class Dynamic_Callables_Cython:
    cpdef f_a(self, int k, np.ndarray[DTYPE_t, ndim=2] m, np.ndarray[DTYPE_t, ndim=2] A):
        raise NotImplemented("(cython) f_a is not implemented!")

    cpdef Ak(self, int k, np.ndarray[DTYPE_t, ndim=2] m, np.ndarray[DTYPE_t, ndim=2] P): # returns state iteration matrix
        raise NotImplemented("(cython) Ak is not implemented!")
        
    cpdef Qk(self, int k):
        raise NotImplemented("(cython) Qk is not implemented!")

    cpdef Q_srk(self, int k):
        raise NotImplemented("(cython) Q_srk is not implemented!")

    cpdef dAk(self, int k):
        raise NotImplemented("(cython) dAk is not implemented!")

    cpdef dQk(self, int k):
        raise NotImplemented("(cython) dQk is not implemented!")    
    
    cpdef reset(self, bint compute_derivatives = False):
        raise NotImplemented("(cython) reset is not implemented!") 

# Template class for measurement callables
cdef class Measurement_Callables_Cython:
    cpdef f_h(self, int k, np.ndarray[DTYPE_t, ndim=2] m_pred, np.ndarray[DTYPE_t, ndim=2] Hk):
        raise NotImplemented("(cython) f_a is not implemented!")

    cpdef Hk(self, int k, np.ndarray[DTYPE_t, ndim=2] m_pred, np.ndarray[DTYPE_t, ndim=2] P_pred): # returns state iteration matrix
        raise NotImplemented("(cython) Hk is not implemented!")
        
    cpdef Rk(self, int k):
        raise NotImplemented("(cython) Rk is not implemented!")
        
    cpdef R_isrk(self, int k):
        raise NotImplemented("(cython) Q_srk is not implemented!")

    cpdef dHk(self, int k):
        raise NotImplemented("(cython) dAk is not implemented!")

    cpdef dRk(self, int k):
        raise NotImplemented("(cython) dQk is not implemented!")

    cpdef reset(self,compute_derivatives = False):
        raise NotImplemented("(cython) reset is not implemented!")

cdef class R_handling_Cython(Measurement_Callables_Cython):
    """
    The calss handles noise matrix R.
    """
    cdef:    
        np.ndarray R
        np.ndarray index
        int R_time_var_index
        np.ndarray dR
        bint svd_each_time         
        dict R_square_root
        
    def __init__(self, np.ndarray[DTYPE_t, ndim=3] R, np.ndarray[DTYPE_t, ndim=2] index, 
                   int R_time_var_index, int p_unique_R_number, np.ndarray[DTYPE_t, ndim=3] dR = None):
        """
        Input:        
        ---------------
        R - array with noise on various steps. The result of preprocessing
            the noise input.
        
        index - for each step of Kalman filter contains the corresponding index
                in the array.
        
        R_time_var_index - another index in the array R. Computed earlier and passed here.
        
        unique_R_number - number of unique noise matrices below which square roots
            are cached and above which they are computed each time.
        
        dR: 3D array[:, :, param_num]
            derivative of R. Derivative is supported only when R do not change over time
             
        Output:
        --------------
        Object which has two necessary functions:
            f_R(k)
            inv_R_square_root(k)
        """
        
        self.R = R
        self.index = index
        self.R_time_var_index = R_time_var_index
        self.dR = dR
        
        cdef int unique_len = len(np.unique(index))
        
        if (unique_len > p_unique_R_number):
            self.svd_each_time = True
        else:
            self.svd_each_time = False
            
        self.R_square_root = {}
        
    cpdef Rk(self, int k):
        return self.R[:,:, <int>self.index[self.R_time_var_index, k]]
    
    
    cpdef dRk(self,int k):
        if self.dR is None:
            raise ValueError("dR derivative is None")
            
        return self.dR # the same dirivative on each iteration
        
    cpdef R_isrk(self, int k):
        """
        Function returns the inverse square root of R matrix on step k.
        """
        cdef int ind = <int>self.index[self.R_time_var_index, k]
        cdef np.ndarray[DTYPE_t, ndim=2] R = self.R[:,:, ind ]
        
        cdef np.ndarray[DTYPE_t, ndim=2] inv_square_root
        
        cdef np.ndarray[DTYPE_t, ndim=2] U
        cdef np.ndarray[DTYPE_t, ndim=1] S           
        cdef np.ndarray[DTYPE_t, ndim=2] Vh
        
        if (R.shape[0] == 1): # 1-D case handle simplier. No storage
        # of the result, just compute it each time.
            inv_square_root = np.sqrt( 1.0/R )
        else:
            if self.svd_each_time:
                
                U,S,Vh = sp.linalg.svd( R,full_matrices=False, compute_uv=True, 
                          overwrite_a=False,check_finite=True)
                
                inv_square_root = U * 1.0/np.sqrt(S)
            else:
                if ind in self.R_square_root:
                    inv_square_root = self.R_square_root[ind]
                else:
                    U,S,Vh = sp.linalg.svd( R,full_matrices=False, compute_uv=True, 
                              overwrite_a=False,check_finite=True)
                              
                    inv_square_root = U * 1.0/np.sqrt(S)
                    
                    self.R_square_root[ind] = inv_square_root
                
        return inv_square_root


cdef class Std_Measurement_Callables_Cython(R_handling_Cython):
    
    cdef:    
        np.ndarray H
        int H_time_var_index
        np.ndarray dH

    def __init__(self, np.ndarray[DTYPE_t, ndim=3] H, int H_time_var_index, 
                 np.ndarray[DTYPE_t, ndim=3] R, np.ndarray[DTYPE_t, ndim=2] index, int R_time_var_index, 
                 int unique_R_number, np.ndarray[DTYPE_t, ndim=3] dH = None, 
                 np.ndarray[DTYPE_t, ndim=3] dR=None):
                     
        super(Std_Measurement_Callables_Cython,self).__init__(R, index, R_time_var_index, unique_R_number,dR)
    
        self.H = H
        self.H_time_var_index = H_time_var_index
        self.dH = dH
        
    cpdef f_h(self, int k, np.ndarray[DTYPE_t, ndim=2] m, np.ndarray[DTYPE_t, ndim=2] H):
        """
        function (k, x_{k}, H_{k}). Measurement function.
            k (iteration number), starts at 0
            x_{k} state 
            H_{k} Jacobian matrices of f_h. In the linear case it is exactly H_{k}.
        """

        return np.dot(H, m)

    cpdef Hk(self, int k, np.ndarray[DTYPE_t, ndim=2] m_pred, np.ndarray[DTYPE_t, ndim=2] P_pred): # returns state iteration matrix
        """
        function (k, m, P) return Jacobian of measurement function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        """

        return self.H[:,:, <int>self.index[self.H_time_var_index, k]]
        
    cpdef dHk(self,int k):
        if self.dH is None:
            raise ValueError("dH derivative is None")
    
        return self.dH # the same dirivative on each iteration



cdef class Q_handling_Cython(Dynamic_Callables_Cython):
    
    cdef:    
        np.ndarray Q
        np.ndarray index
        int Q_time_var_index
        np.ndarray dQ
        dict Q_square_root
        bint svd_each_time
        
    def __init__(self, np.ndarray[DTYPE_t, ndim=3] Q, np.ndarray[DTYPE_t, ndim=2] index, 
                 int Q_time_var_index, int p_unique_Q_number, np.ndarray[DTYPE_t, ndim=3] dQ = None):
        """
        Input:        
        ---------------
        Q - array with noise on various steps. The result of preprocessing
            the noise input.
        
        index - for each step of Kalman filter contains the corresponding index
                in the array.
        
        Q_time_var_index - another index in the array R. Computed earlier and passed here.
        
        unique_Q_number - number of unique noise matrices below which square roots
            are cached and above which they are computed each time.
            
        dQ: 3D array[:, :, param_num]
            derivative of Q. Derivative is supported only when Q do not change over time
            
        Output:
        --------------
        Object which has three necessary functions:
            Qk(k)
            dQk(k)
            Q_srkt(k)
        """
    
        self.Q = Q
        self.index = index
        self.Q_time_var_index = Q_time_var_index
        self.dQ = dQ         
        
        cdef int unique_len = len(np.unique(index))
        
        if (unique_len > p_unique_Q_number):
            self.svd_each_time = True
        else:
            self.svd_each_time = False
            
        self.Q_square_root = {}
    
        
    cpdef Qk(self, int k):
        """
        function (k). Returns noise matrix of dynamic model on iteration k.
                k (iteration number). starts at 0
        """
        return self.Q[:,:, <int>self.index[self.Q_time_var_index, k]]
    
    cpdef dQk(self, int k):
        if self.dQ is None:
            raise ValueError("dQ derivative is None")
            
        return self.dQ # the same dirivative on each iteration
        
    cpdef Q_srk(self, int k):
        """
        function (k). Returns the square root of noise matrix of dynamic model on iteration k.
                k (iteration number). starts at 0
                
        This function is implemented to use SVD prediction step.
        """
        cdef int ind = <int>self.index[self.Q_time_var_index, k]
        cdef np.ndarray[DTYPE_t, ndim=2] Q = self.Q[:,:, ind]
        
        
        cdef np.ndarray[DTYPE_t, ndim=2] square_root
        
        cdef np.ndarray[DTYPE_t, ndim=2] U
        cdef np.ndarray[DTYPE_t, ndim=1] S           
        cdef np.ndarray[DTYPE_t, ndim=2] Vh
        
        if (Q.shape[0] == 1): # 1-D case handle simplier. No storage
        # of the result, just compute it each time.
            square_root = np.sqrt( Q )
        else:
            if self.svd_each_time:
                
                U,S,Vh = sp.linalg.svd( Q,full_matrices=False, compute_uv=True, 
                          overwrite_a=False,check_finite=True)
                
                square_root = U * np.sqrt(S)
            else:
                
                if ind in self.Q_square_root:
                    square_root = self.Q_square_root[ind]
                else:
                    U,S,Vh = sp.linalg.svd( Q,full_matrices=False, compute_uv=True, 
                              overwrite_a=False,check_finite=True)
                    
                    square_root = U * np.sqrt(S)
                    
                    self.Q_square_root[ind] = square_root
            
        return square_root

cdef class Std_Dynamic_Callables_Cython(Q_handling_Cython):
    cdef:
        np.ndarray A
        int A_time_var_index
        np.ndarray  dA
        
    def __init__(self, np.ndarray[DTYPE_t, ndim=3] A, int A_time_var_index, 
                 np.ndarray[DTYPE_t, ndim=3] Q, 
                 np.ndarray[DTYPE_t, ndim=2] index, 
                 int Q_time_var_index, int unique_Q_number, 
                 np.ndarray[DTYPE_t, ndim=3] dA = None, 
                 np.ndarray[DTYPE_t, ndim=3] dQ=None):
                     
        super(Std_Dynamic_Callables_Cython,self).__init__(Q, index, Q_time_var_index, unique_Q_number,dQ)
    
        self.A = A
        self.A_time_var_index = A_time_var_index
        self.dA = dA
        
    cpdef f_a(self, int k, np.ndarray[DTYPE_t, ndim=2] m, np.ndarray[DTYPE_t, ndim=2] A):
        """
            f_a: function (k, x_{k-1}, A_{k}). Dynamic function.        
            k (iteration number), starts at 0
            x_{k-1} State from the previous step
            A_{k} Jacobian matrices of f_a. In the linear case it is exactly A_{k}.
        """

        return np.dot(A,m)

    cpdef Ak(self, int k, np.ndarray[DTYPE_t, ndim=2] m_pred, np.ndarray[DTYPE_t, ndim=2] P_pred): # returns state iteration matrix
        """
        function (k, m, P) return Jacobian of measurement function, it is
            passed into p_h.
            k (iteration number), starts at 0
            m: point where Jacobian is evaluated
            P: parameter for Jacobian, usually covariance matrix.
        """

        return self.A[:,:, <int>self.index[self.A_time_var_index, k]]
        
    cpdef dAk(self, int k):
        if self.dA is None:
            raise ValueError("dA derivative is None")
    
        return self.dA # the same dirivative on each iteration
        
        
    cpdef reset(self, bint compute_derivatives=False):
        """
        For reusing this object e.g. in smoother computation. It makes sence
        because necessary matrices have been already computed for all
        time steps.
        """
        return self
        
cdef class AQcompute_batch_Cython(Q_handling_Cython):
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
        #def __init__(self, F,L,Qc,dt,compute_derivatives=False, grad_params_no=None, P_inf=None, dP_inf=None, dF = None, dQc=None):
        cdef:
            np.ndarray As
            np.ndarray Qs
            np.ndarray dAs
            np.ndarray dQs
            np.ndarray reconstruct_indices
            #long total_size_of_data
            dict Q_svd_dict
            int last_k
            
        def __init__(self, np.ndarray[DTYPE_t, ndim=3] As, np.ndarray[DTYPE_t, ndim=3] Qs, 
                     np.ndarray[DTYPE_int_t, ndim=1] reconstruct_indices, 
                     np.ndarray[DTYPE_t, ndim=4] dAs=None, 
                     np.ndarray[DTYPE_t, ndim=4] dQs=None):
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
            
            """
            
            self.As = As
            self.Qs = Qs
            self.dAs = dAs
            self.dQs = dQs
            self.reconstruct_indices = reconstruct_indices
            self.total_size_of_data = self.As.nbytes + self.Qs.nbytes +\
                            (self.dAs.nbytes if (self.dAs is not None) else 0) +\
                            (self.dQs.nbytes if (self.dQs is not None) else 0) +\
                            (self.reconstruct_indices.nbytes if (self.reconstruct_indices is not None) else 0)
                            
            self.Q_svd_dict = {}
            self.Q_square_root_dict = {}
            self.Q_inverse_dict = {}
            self.last_k = 0
             # !!!Print statistics! Which object is created
            # !!!Print statistics! Print sizes of matrices
        cpdef f_a(self, int k, np.ndarray[DTYPE_t, ndim=2] m, np.ndarray[DTYPE_t, ndim=2] A):
            """
            Dynamic model
            """
            return np.dot(A, m) # default dynamic model
            
        cpdef reset(self, bint compute_derivatives=False):
            """
            For reusing this object e.g. in smoother computation. It makes sence
            because necessary matrices have been already computed for all
            time steps.
            """
            return self
            
        cpdef Ak(self,int k, np.ndarray[DTYPE_t, ndim=2] m, np.ndarray[DTYPE_t, ndim=2] P):
            self.last_k = k
            return self.As[:,:, <int>self.reconstruct_indices[k]]
            
        cpdef Qk(self,int k):
            self.last_k = k
            return self.Qs[:,:, <int>self.reconstruct_indices[k]]
        
        cpdef dAk(self, int k):
            self.last_k = k
            return self.dAs[:,:, :, <int>self.reconstruct_indices[k]]
        
        cpdef dQk(self, int k):
            self.last_k = k
            return self.dQs[:,:, :, <int>self.reconstruct_indices[k]]
        
        
        cpdef Q_srk(self, int k):
            """
            Square root of the noise matrix Q
            """
            
            cdef int matrix_index = <int>self.reconstruct_indices[k]
            cdef np.ndarray[DTYPE_t, ndim=2] square_root
            
            cdef np.ndarray[DTYPE_t, ndim=2] U
            cdef np.ndarray[DTYPE_t, ndim=1] S           
            cdef np.ndarray[DTYPE_t, ndim=2] Vh
            
        
            if matrix_index in self.Q_square_root_dict:
                square_root = self.Q_square_root_dict[matrix_index]
            else:
                if matrix_index not in self.Q_svd_dict:
                    U,S,Vh = sp.linalg.svd( self.Qs[:,:, matrix_index], 
                                        full_matrices=False, compute_uv=True, 
                                        overwrite_a=False, check_finite=False)
                    self.Q_svd_dict[matrix_index] = (U,S,Vh)
                else:
                    U,S,Vh = self.Q_svd_dict[matrix_index]
                       
                square_root = U * np.sqrt(S)
                self.Q_square_root_dict[matrix_index] = square_root
            
            return square_root
            
            
        cpdef Q_inverse(self, int k, float jitter=0.0):
            """
            Square root of the noise matrix Q
            """
            
            cdef int matrix_index = <int>self.reconstruct_indices[k]
            cdef np.ndarray[DTYPE_t, ndim=2] square_root
            
            cdef np.ndarray[DTYPE_t, ndim=2] U
            cdef np.ndarray[DTYPE_t, ndim=1] S           
            cdef np.ndarray[DTYPE_t, ndim=2] Vh
            
        
            if matrix_index in self.Q_inverse_dict:
                Q_inverse = self.Q_inverse_dict[matrix_index]
            else:
                if matrix_index not in self.Q_svd_dict:
                    U,S,Vh = sp.linalg.svd( self.Qs[:,:, matrix_index], 
                                        full_matrices=False, compute_uv=True, 
                                        overwrite_a=False, check_finite=False)
                    self.Q_svd_dict[matrix_index] = (U,S,Vh)
                else:
                    U,S,Vh = self.Q_svd_dict[matrix_index]
                       
                Q_inverse = Q_inverse = np.dot( Vh.T * ( 1.0/(S + jitter)) , U.T )
                self.Q_inverse_dict[matrix_index] = Q_inverse
            
            return Q_inverse
            
#        def return_last(self):
#            """
#            Function returns last available matrices.
#            """
#            
#            if (self.last_k is None):
#                raise ValueError("Matrices are not computed.")
#            else:
#                ind = self.reconstruct_indices[self.last_k]
#                A = self.As[:,:, ind]
#                Q = self.Qs[:,:, ind]
#                dA = self.dAs[:,:, :, ind]
#                dQ = self.dQs[:,:, :, ind]
#                
#            return self.last_k, A, Q, dA, dQ

@cython.boundscheck(False)
def _kalman_prediction_step_SVD_Cython(long k, np.ndarray[DTYPE_t, ndim=2] p_m , tuple p_P, 
                                Dynamic_Callables_Cython p_dynamic_callables, 
                                bint calc_grad_log_likelihood=False, 
                                np.ndarray[DTYPE_t, ndim=3] p_dm = None, 
                                np.ndarray[DTYPE_t, ndim=3] p_dP = None):
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
        
        p_f_Qsr: function (k). Returns square root of noise matrix of the 
            dynamic model on iteration k. k (iteration number). starts at 0
            
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
            
        grad_calc_params_1: List or None
        List with derivatives. The first component is 'f_dA' - function(k)
        which returns the derivative of A. The second element is 'f_dQ'
         - function(k). Function which returns the derivative of Q.
         
    Output:
    ----------------------------
    m_pred, P_pred, dm_pred, dP_pred: metrices, 3D objects
        Results of the prediction steps.        
        
    """
    
    # covariance from the previous step# p_prev_cov = v * S * V.T
    cdef np.ndarray[DTYPE_t, ndim=2] Prev_cov = p_P[0]
    cdef np.ndarray[DTYPE_t, ndim=1] S_old = p_P[1]
    cdef np.ndarray[DTYPE_t, ndim=2] V_old = p_P[2]     
    #p_prev_cov_tst = np.dot(p_V, (p_S * p_V).T) # reconstructed covariance from the previous step        
    
    # index correspond to values from previous iteration.
    cdef np.ndarray[DTYPE_t, ndim=2] A = p_dynamic_callables.Ak(k,p_m,Prev_cov) # state transition matrix (or Jacobian)
    cdef np.ndarray[DTYPE_t, ndim=2] Q = p_dynamic_callables.Qk(k) # state noise matrx. This is necessary for the square root calculation (next step)
    cdef np.ndarray[DTYPE_t, ndim=2] Q_sr = p_dynamic_callables.Q_srk(k)            
    # Prediction step ->
    cdef np.ndarray[DTYPE_t, ndim=2] m_pred = p_dynamic_callables.f_a(k, p_m, A) # predicted mean
    
    # coavariance prediction have changed:
    cdef np.ndarray[DTYPE_t, ndim=2] svd_1_matr = np.vstack( ( (np.sqrt(S_old)* np.dot(A,V_old)).T , Q_sr.T) )
    res = sp.linalg.svd( svd_1_matr,full_matrices=False, compute_uv=True, 
                  overwrite_a=False,check_finite=True)
    # (U,S,Vh)  
    cdef np.ndarray[DTYPE_t, ndim=2] U = res[0]
    cdef np.ndarray[DTYPE_t, ndim=1] S = res[1]             
    cdef np.ndarray[DTYPE_t, ndim=2] Vh = res[2]             
    # predicted variance computed by the regular method. For testing
    #P_pred_tst = A.dot(Prev_cov).dot(A.T) + Q
    cdef np.ndarray[DTYPE_t, ndim=2]  V_new = Vh.T
    cdef np.ndarray[DTYPE_t, ndim=1]  S_new = S**2
    
    cdef np.ndarray[DTYPE_t, ndim=2] P_pred = np.dot(V_new * S_new, V_new.T) # prediction covariance
    #tuple P_pred = (P_pred, S_new, Vh.T)
    # Prediction step <-
    
    # derivatives
    cdef np.ndarray[DTYPE_t, ndim=3] dA_all_params
    cdef np.ndarray[DTYPE_t, ndim=3] dQ_all_params
    
    cdef np.ndarray[DTYPE_t, ndim=3] dm_pred
    cdef np.ndarray[DTYPE_t, ndim=3] dP_pred
    
    cdef int param_number
    cdef int j
    cdef tuple ret    
    
    cdef np.ndarray[DTYPE_t, ndim=2] dA
    cdef np.ndarray[DTYPE_t, ndim=2] dQ
    if calc_grad_log_likelihood:
        dA_all_params = p_dynamic_callables.dAk(k) # derivatives of A wrt parameters 
        dQ_all_params = p_dynamic_callables.dQk(k) # derivatives of Q wrt parameters
        
        param_number = p_dP.shape[2]
        
        # p_dm, p_dP - derivatives form the previoius step
        dm_pred = np.empty((p_dm.shape[0], p_dm.shape[1], p_dm.shape[2]), dtype = DTYPE)
        dP_pred = np.empty((p_dP.shape[0], p_dP.shape[1], p_dP.shape[2]), dtype = DTYPE)
        
        for j in range(param_number):
            dA = dA_all_params[:,:,j]
            dQ = dQ_all_params[:,:,j]
            
            dm_pred[:,:,j] = np.dot(dA, p_m) + np.dot(A, p_dm[:,:,j])
            # prediction step derivatives for current parameter:
            
            dP_pred[:,:,j] = np.dot( dA ,np.dot(Prev_cov, A.T))
            dP_pred[:,:,j] += dP_pred[:,:,j].T            
            dP_pred[:,:,j] += np.dot( A ,np.dot( p_dP[:,:,j] , A.T)) + dQ
            
            dP_pred[:,:,j] = 0.5*(dP_pred[:,:,j] + dP_pred[:,:,j].T) #symmetrize
    else:
        dm_pred = None
        dP_pred = None
    
    ret = (P_pred, S_new, Vh.T)
    return m_pred, ret, dm_pred, dP_pred
    
    
    
@cython.boundscheck(False)
def _kalman_update_step_SVD_Cython(long k, np.ndarray[DTYPE_t, ndim=2] p_m, tuple p_P, 
                            Measurement_Callables_Cython p_measurement_callables, 
                            np.ndarray[DTYPE_t, ndim=2] measurement, 
                            bint calc_log_likelihood= False, 
                            bint calc_grad_log_likelihood=False, 
                            np.ndarray[DTYPE_t, ndim=3] p_dm = None, 
                            np.ndarray[DTYPE_t, ndim=3] p_dP = None):
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
    
    p_f_H: function (k, m, P) return Jacobian of dynamic function, it is
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
    
    cdef np.ndarray[DTYPE_t, ndim=2] m_pred = p_m # from prediction step
    #P_pred,S_pred,V_pred = p_P # from prediction step      
    cdef np.ndarray[DTYPE_t, ndim=2] P_pred = p_P[0]
    cdef np.ndarray[DTYPE_t, ndim=1] S_pred = p_P[1]
    cdef np.ndarray[DTYPE_t, ndim=2] V_pred = p_P[2]    
    
    cdef np.ndarray[DTYPE_t, ndim=2] H = p_measurement_callables.Hk(k, m_pred, P_pred)
    cdef np.ndarray[DTYPE_t, ndim=2] R = p_measurement_callables.Rk(k)
    cdef np.ndarray[DTYPE_t, ndim=2] R_isr =p_measurement_callables.R_isrk(k) # square root of the inverse of R matrix       
    
    cdef int time_series_no = p_m.shape[1] # number of time serieses
    
    cdef np.ndarray[DTYPE_t, ndim=2] log_likelihood_update # log_likelihood_update=None;
    # Update step (only if there is data)
    #if not np.any(np.isnan(measurement)): # TODO: if some dimensions are missing, do properly computations for other.
    cdef np.ndarray[DTYPE_t, ndim=2] v = measurement-p_measurement_callables.f_h(k, m_pred, H)
     
    cdef np.ndarray[DTYPE_t, ndim=2] svd_2_matr = np.vstack( ( np.dot( R_isr.T, np.dot(H, V_pred)) , np.diag( 1.0/np.sqrt(S_pred) ) ) )
        
    res = sp.linalg.svd( svd_2_matr,full_matrices=False, compute_uv=True, 
                 overwrite_a=False,check_finite=True)
                 
    #(U,S,Vh)    
    cdef np.ndarray[DTYPE_t, ndim=2] U = res[0]
    cdef np.ndarray[DTYPE_t, ndim=1] S_svd = res[1]
    cdef np.ndarray[DTYPE_t, ndim=2] Vh = res[2]
    
     # P_upd = U_upd S_upd**2 U_upd.T
    cdef np.ndarray[DTYPE_t, ndim=2] U_upd = np.dot(V_pred, Vh.T)             
    cdef np.ndarray[DTYPE_t, ndim=1] S_upd = (1.0/S_svd)**2
     
    cdef np.ndarray[DTYPE_t, ndim=2] P_upd = np.dot(U_upd * S_upd, U_upd.T) # update covariance
    #P_upd = (P_upd,S_upd,U_upd) # tuple to pass to the next step
    
     # stil need to compute S and K for derivative computation
    cdef np.ndarray[DTYPE_t, ndim=2] S = H.dot(P_pred).dot(H.T) + R
    cdef np.ndarray[DTYPE_t, ndim=2] K
    cdef bint measurement_dim_gt_one = False
    if measurement.shape[0]==1: # measurements are one dimensional
        if (S < 0):
            raise ValueError("Kalman Filter Update SVD: S is negative step %i" % k )
            #import pdb; pdb.set_trace()
             
        K = P_pred.dot(H.T) / S
        if calc_log_likelihood:
            log_likelihood_update = -0.5 * ( np.log(2*np.pi) + np.log(S) +
                                 v*v / S)
            #log_likelihood_update = log_likelihood_update[0,0] # to make int
            if np.any(np.isnan(log_likelihood_update)): # some member in P_pred is None.
                raise ValueError("Nan values in likelihood update!")
        else:
            log_likelihood_update = None
        #LL = None; islower = None
    else:
        measurement_dim_gt_one = True
        raise ValueError("""Measurement dimension larger then 1 is currently not supported""")
     
    # Old  method of computing updated covariance (for testing) ->
    #P_upd_tst = K.dot(S).dot(K.T)
    #P_upd_tst = 0.5*(P_upd_tst + P_upd_tst.T)
    #P_upd_tst =  P_pred - P_upd_tst# this update matrix is symmetric
    # Old  method of computing updated covariance (for testing) <-
    cdef np.ndarray[DTYPE_t, ndim=3] dm_upd # dm_upd=None;
    cdef np.ndarray[DTYPE_t, ndim=3] dP_upd # dP_upd=None;
    cdef np.ndarray[DTYPE_t, ndim=2] d_log_likelihood_update # d_log_likelihood_update=None     
     
    cdef np.ndarray[DTYPE_t, ndim=3] dm_pred_all_params
    cdef np.ndarray[DTYPE_t, ndim=3] dP_pred_all_params
    cdef int param_number
    
    cdef np.ndarray[DTYPE_t, ndim=3] dH_all_params
    cdef np.ndarray[DTYPE_t, ndim=3] dR_all_params
    
    cdef int param
    
    cdef np.ndarray[DTYPE_t, ndim=2] dH, dR, dm_pred, dP_pred, dv, dS, tmp1, tmp2, tmp3, dK, tmp5
    cdef tuple ret
    
    if calc_grad_log_likelihood:
        dm_pred_all_params = p_dm # derivativas of the prediction phase 
        dP_pred_all_params = p_dP
         
        param_number = p_dP.shape[2]
         
        dH_all_params = p_measurement_callables.dHk(k)
        dR_all_params = p_measurement_callables.dRk(k)
         
        dm_upd = np.empty((dm_pred_all_params.shape[0], dm_pred_all_params.shape[1], dm_pred_all_params.shape[2]), dtype = DTYPE)
        dP_upd = np.empty((dP_pred_all_params.shape[0], dP_pred_all_params.shape[1], dP_pred_all_params.shape[2]), dtype = DTYPE)
         
         # firts dimension parameter_no, second - time series number
        d_log_likelihood_update = np.empty((param_number,time_series_no), dtype = DTYPE)
        for param in range(param_number):
        
           dH = dH_all_params[:,:,param]
           dR = dR_all_params[:,:,param]

           dm_pred = dm_pred_all_params[:,:,param]
           dP_pred = dP_pred_all_params[:,:,param]
        
            # Terms in the likelihood derivatives
           dv = - np.dot( dH, m_pred) -  np.dot( H, dm_pred)           
           dS = np.dot(dH, np.dot( P_pred, H.T))
           dS += dS.T
           dS += np.dot(H, np.dot( dP_pred, H.T)) + dR
        
            # TODO: maybe symmetrize dS
        
           tmp1 = H.T / S
           tmp2 = dH.T / S
           tmp3 = dS.T / S
                
           dK = np.dot( dP_pred, tmp1) + np.dot( P_pred, tmp2) - \
                np.dot( P_pred, np.dot( tmp1, tmp3 ) )
            
            # terms required for the next step, save this for each parameter
           dm_upd[:,:,param] = dm_pred + np.dot(dK, v) + np.dot(K, dv)
                
           dP_upd[:,:,param] = -np.dot(dK, np.dot(S, K.T))      
           dP_upd[:,:,param] += dP_upd[:,:,param].T
           dP_upd[:,:,param] += dP_pred - np.dot(K , np.dot( dS, K.T))
            
           dP_upd[:,:,param] = 0.5*(dP_upd[:,:,param] + dP_upd[:,:,param].T) #symmetrize
           # computing the likelihood change for each parameter:
           tmp5 = v / S
                
            
           d_log_likelihood_update[param,:] = -(0.5*np.sum(np.diag(tmp3)) + \
               np.sum(tmp5*dv, axis=0) - 0.5 * np.sum(tmp5 * np.dot(dS, tmp5), axis=0) ) 
          
        # Compute the actual updates for mean of the states. Variance update
        # is computed earlier.
    else:
        dm_upd = None 
        dP_upd = None
        d_log_likelihood_update = None
        
    m_upd = m_pred + K.dot( v )
    
    ret = (P_upd,S_upd,U_upd)
    return m_upd, ret, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update
    
    
@cython.boundscheck(False)
def _cont_discr_kalman_filter_raw_Cython(int state_dim, Dynamic_Callables_Cython p_dynamic_callables, 
                                  Measurement_Callables_Cython p_measurement_callables, X, Y, 
                                  np.ndarray[DTYPE_t, ndim=2] m_init=None, np.ndarray[DTYPE_t, ndim=2] P_init=None, 
                                  p_kalman_filter_type='regular',
                                  bint calc_log_likelihood=False, 
                                  bint calc_grad_log_likelihood=False, 
                                  int grad_params_no=0, 
                                  np.ndarray[DTYPE_t, ndim=3] dm_init=None, 
                                  np.ndarray[DTYPE_t, ndim=3] dP_init=None):
    
    cdef int steps_no = Y.shape[0] # number of steps in the Kalman Filter
    cdef int time_series_no = Y.shape[2] # multiple time series mode
        
    # Allocate space for results
    # Mean estimations. Initial values will be included
    cdef np.ndarray[DTYPE_t, ndim=3] M = np.empty(((steps_no+1),state_dim,time_series_no), dtype=DTYPE)
    M[0,:,:] = m_init # Initialize mean values
    # Variance estimations. Initial values will be included
    cdef np.ndarray[DTYPE_t, ndim=3] P = np.empty(((steps_no+1),state_dim,state_dim))
    P_init = 0.5*( P_init + P_init.T) # symmetrize initial covariance. In some ustable cases this is uiseful
    P[0,:,:] = P_init # Initialize initial covariance matrix
    
    cdef np.ndarray[DTYPE_t, ndim=2] U
    cdef np.ndarray[DTYPE_t, ndim=1] S
    cdef np.ndarray[DTYPE_t, ndim=2] Vh
    
    U,S,Vh = sp.linalg.svd( P_init,full_matrices=False, compute_uv=True, 
              overwrite_a=False,check_finite=True)
    S[ (S==0) ] = 1e-17 # allows to run algorithm for singular initial variance
    cdef tuple P_upd = (P_init, S,U)
    #log_likelihood = 0
    #grad_log_likelihood = np.zeros((grad_params_no,1))
    cdef np.ndarray[DTYPE_t, ndim=2] log_likelihood = np.zeros((1, time_series_no), dtype = DTYPE) #if calc_log_likelihood else None
    cdef np.ndarray[DTYPE_t, ndim=2] grad_log_likelihood = np.zeros((grad_params_no, time_series_no), dtype = DTYPE) #if calc_grad_log_likelihood else None
    
    #setting initial values for derivatives update
    cdef np.ndarray[DTYPE_t, ndim=3] dm_upd = dm_init
    cdef np.ndarray[DTYPE_t, ndim=3] dP_upd = dP_init
    # Main loop of the Kalman filter
    cdef np.ndarray[DTYPE_t, ndim=2] prev_mean, k_measurment
    cdef np.ndarray[DTYPE_t, ndim=2] m_pred, m_upd
    cdef tuple P_pred
    cdef np.ndarray[DTYPE_t, ndim=3] dm_pred, dP_pred
    cdef np.ndarray[DTYPE_t, ndim=2] log_likelihood_update, d_log_likelihood_update
    cdef int k
    
    #print "Hi I am cython"
    for k in range(0,steps_no):
        # In this loop index for new estimations is (k+1), old - (k)
        # This happened because initial values are stored at 0-th index.                 
        #import pdb; pdb.set_trace()
    
        prev_mean = M[k,:,:] # mean from the previous step
        
        m_pred, P_pred, dm_pred, dP_pred = \
        _kalman_prediction_step_SVD_Cython(k, prev_mean ,P_upd, p_dynamic_callables,
            calc_grad_log_likelihood, dm_upd, dP_upd)
        
        k_measurment = Y[k,:,:]
        if (np.any(np.isnan(k_measurment)) == False):
#        if np.any(np.isnan(k_measurment)):
#            raise ValueError("Nan measurements are currently not supported")
             
            m_upd, P_upd, log_likelihood_update, dm_upd, dP_upd, d_log_likelihood_update = \
            _kalman_update_step_SVD_Cython(k,  m_pred , P_pred, p_measurement_callables, 
                    k_measurment, calc_log_likelihood=calc_log_likelihood, 
                    calc_grad_log_likelihood=calc_grad_log_likelihood, 
                    p_dm = dm_pred, p_dP = dP_pred)
        else:
            if not np.all(np.isnan(k_measurment)):
                    raise ValueError("""Nan measurements are currently not supported if
                                     they are intermixed with not NaN measurements""")
            else:
                m_upd = m_pred; P_upd = P_pred; dm_upd = dm_pred; dP_upd = dP_pred
                if calc_log_likelihood:
                    log_likelihood_update = np.zeros((1,time_series_no))
                if calc_grad_log_likelihood:
                    d_log_likelihood_update = np.zeros((grad_params_no,time_series_no))
                    
            
        if calc_log_likelihood:
            log_likelihood += log_likelihood_update
        
        if calc_grad_log_likelihood:
            grad_log_likelihood += d_log_likelihood_update
        
        M[k+1,:,:] = m_upd # separate mean value for each time series
        P[k+1,:,:] = P_upd[0]
        
    return (M, P, log_likelihood, grad_log_likelihood, p_dynamic_callables.reset(False))
