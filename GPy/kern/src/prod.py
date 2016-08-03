# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import CombinationKernel
from paramz.caching import Cache_this
import itertools
from functools import reduce


def numpy_invalid_op_as_exception(func):
    """
    A decorator that allows catching numpy invalid operations
    as exceptions (the default behaviour is raising warnings).
    """
    def func_wrapper(*args, **kwargs):
        np.seterr(invalid='raise')
        result = func(*args, **kwargs)
        np.seterr(invalid='warn')
        return result
    return func_wrapper


class Prod(CombinationKernel):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kern
    :rtype: kernel object

    """
    def __init__(self, kernels, name='mul'):
        for i, kern in enumerate(kernels[:]):
            if isinstance(kern, Prod):
                del kernels[i]
                for part in kern.parts[::-1]:
                    kern.unlink_parameter(part)
                    kernels.insert(i, part)
        super(Prod, self).__init__(kernels, name)

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def K(self, X, X2=None, which_parts=None):
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.multiply, (p.K(X, X2) for p in which_parts))

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def Kdiag(self, X, which_parts=None):
        if which_parts is None:
            which_parts = self.parts
        return reduce(np.multiply, (p.Kdiag(X) for p in which_parts))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if len(self.parts)==2:
            self.parts[0].update_gradients_full(dL_dK*self.parts[1].K(X,X2), X, X2)
            self.parts[1].update_gradients_full(dL_dK*self.parts[0].K(X,X2), X, X2)
        else:
            for combination in itertools.combinations(self.parts, len(self.parts) - 1):
                prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
                to_update = list(set(self.parts) - set(combination))[0]
                to_update.update_gradients_full(dL_dK * prod, X, X2)

    def update_gradients_diag(self, dL_dKdiag, X):
        if len(self.parts)==2:
            self.parts[0].update_gradients_diag(dL_dKdiag*self.parts[1].Kdiag(X), X)
            self.parts[1].update_gradients_diag(dL_dKdiag*self.parts[0].Kdiag(X), X)
        else:
            for combination in itertools.combinations(self.parts, len(self.parts) - 1):
                prod = reduce(np.multiply, [p.Kdiag(X) for p in combination])
                to_update = list(set(self.parts) - set(combination))[0]
                to_update.update_gradients_diag(dL_dKdiag * prod, X)

    def gradients_X(self, dL_dK, X, X2=None):
        target = np.zeros(X.shape)
        if len(self.parts)==2:
            target += self.parts[0].gradients_X(dL_dK*self.parts[1].K(X, X2), X, X2)
            target += self.parts[1].gradients_X(dL_dK*self.parts[0].K(X, X2), X, X2)
        else:
            for combination in itertools.combinations(self.parts, len(self.parts) - 1):
                prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
                to_update = list(set(self.parts) - set(combination))[0]
                target += to_update.gradients_X(dL_dK * prod, X, X2)
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape)
        if len(self.parts)==2:
            target += self.parts[0].gradients_X_diag(dL_dKdiag*self.parts[1].Kdiag(X), X)
            target += self.parts[1].gradients_X_diag(dL_dKdiag*self.parts[0].Kdiag(X), X)
        else:
            k = self.Kdiag(X)*dL_dKdiag
            for p in self.parts:
                target += p.gradients_X_diag(k/p.Kdiag(X),X)
        return target

    def input_sensitivity(self, summarize=True):
        if summarize:
            i_s = np.ones((self.input_dim))
            for k in self.parts:
                i_s[k._all_dims_active] *= k.input_sensitivity(summarize)
            return i_s
        else:
            return super(Prod, self).input_sensitivity(summarize)

    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """
        part_start_param_index = 0
        for p in self.parts:
            if not p.is_fixed:
                part_param_num = len(p.param_array) # number of parameters in the part
                p.sde_update_gradient_full(gradients[part_start_param_index:(part_start_param_index+part_param_num)])
                part_start_param_index += part_param_num
                
    def sde(self):
        """
        """
        F      = np.array((0,), ndmin=2)
        L      = np.array((1,), ndmin=2)
        Qc     = np.array((1,), ndmin=2)
        H      = np.array((1,), ndmin=2)
        Pinf   = np.array((1,), ndmin=2)
        P0   = np.array((1,), ndmin=2)
        dF     = None
        dQc    = None
        dPinf  = None
        dP0  = None
        
         # Assign models
        for p in self.parts:
            (Ft,Lt,Qct,Ht,P_inft, P0t, dFt,dQct,dP_inft,dP0t) = p.sde()
            
            # check derivative dimensions ->
            number_of_parameters = len(p.param_array)            
            assert dFt.shape[2] == number_of_parameters, "Dynamic matrix derivative shape is wrong"
            assert dQct.shape[2] == number_of_parameters, "Diffusion matrix derivative shape is wrong"
            assert dP_inft.shape[2] == number_of_parameters, "Infinite covariance matrix derivative shape is wrong"
            # check derivative dimensions <-
            
            # exception for periodic kernel
            if (p.name == 'std_periodic'):
                Qct = P_inft  
                dQct = dP_inft                 
            
            dF    = dkron(F,dF,Ft,dFt,'sum')
            dQc   = dkron(Qc,dQc,Qct,dQct,'prod')
            dPinf = dkron(Pinf,dPinf,P_inft,dP_inft,'prod')
            dP0 = dkron(P0,dP0,P0t,dP0t,'prod')
            
            F    = np.kron(F,np.eye(Ft.shape[0])) + np.kron(np.eye(F.shape[0]),Ft)
            L    = np.kron(L,Lt)
            Qc   = np.kron(Qc,Qct)
            Pinf = np.kron(Pinf,P_inft)
            P0 = np.kron(P0,P_inft)
            H    = np.kron(H,Ht)
            
        return (F,L,Qc,H,Pinf,P0,dF,dQc,dPinf,dP0)

def dkron(A,dA,B,dB, operation='prod'):
    """
    Function computes the derivative of Kronecker product A*B 
    (or Kronecker sum A+B).
    
    Input:
    -----------------------
    
    A: 2D matrix
        Some matrix 
    dA: 3D (or 2D matrix)
        Derivarives of A
    B: 2D matrix
        Some matrix 
    dB: 3D (or 2D matrix)
        Derivarives of B    
    
    operation: str 'prod' or 'sum'
        Which operation is considered. If the operation is 'sum' it is assumed
        that A and are square matrices.s
    
    Output:
        dC: 3D matrix
        Derivative of Kronecker product A*B (or Kronecker sum A+B)
    """
    
    if dA is None:
        dA_param_num = 0
        dA = np.zeros((A.shape[0], A.shape[1],1))
    else:
        dA_param_num = dA.shape[2]
    
    if dB is None:
        dB_param_num = 0
        dB = np.zeros((B.shape[0], B.shape[1],1))    
    else:
        dB_param_num = dB.shape[2]

    # Space allocation for derivative matrix
    dC = np.zeros((A.shape[0]*B.shape[0], A.shape[1]*B.shape[1], dA_param_num +  dB_param_num))    
    
    for k in range(dA_param_num):
        if operation == 'prod':
            dC[:,:,k] = np.kron(dA[:,:,k],B);
        else:
            dC[:,:,k] = np.kron(dA[:,:,k],np.eye( B.shape[0] ))
            
    for k in range(dB_param_num):
        if operation == 'prod':
            dC[:,:,dA_param_num+k] = np.kron(A,dB[:,:,k])
        else:
            dC[:,:,dA_param_num+k] = np.kron(np.eye( A.shape[0] ),dB[:,:,k])
            
    return dC
