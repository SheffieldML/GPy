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
        _newkerns = []
        for kern in kernels:
            if isinstance(kern, Prod):
                for part in kern.parts:
                    #kern.unlink_parameter(part)
                    _newkerns.append(part.copy())
            else:
                _newkerns.append(kern.copy())

        super(Prod, self).__init__(_newkerns, name)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Prod, self)._save_to_input_dict()
        input_dict["class"] = str("GPy.kern.Prod")
        return input_dict

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

    def reset_gradients(self):
        for part in self.parts:
            part.reset_gradients()

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK_dX(self, X, X2, dimX, which_parts=None):
        """
        Compute the derivative of K with respect to:
            dimension dimX of set X.
        """
        prod_sum = np.zeros((X.shape[0], X2.shape[0]))
        for combination in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
            to_update = list(set(self.parts) - set(combination))[0]
            prod_sum += prod*to_update.dK_dX(X, X2, dimX)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK_dXdiag(self, X, dimX, which_parts=None):
        """
        Compute the derivative of K with respect to:
            dimension dimX of set X.

        Returns only diagonal elements.
        """
        prod_sum = np.zeros(X.shape[0])
        for combination in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.Kdiag(X) for p in combination])
            to_update = list(set(self.parts) - set(combination))[0]
            prod_sum += prod*to_update.dK_dXdiag(X, dimX)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK_dX2(self, X, X2, dimX2, which_parts=None):
        """
        Compute the derivative of K with respect to:
            dimension dimX2 of set X2.
        """
        prod_sum = np.zeros((X.shape[0], X2.shape[0]))
        for combination in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
            to_update = list(set(self.parts) - set(combination))[0]
            prod_sum += prod*to_update.dK_dX2(X, X2, dimX2)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK2_dXdX2(self, X, X2, dimX, dimX2, which_parts=None):
        """
        Compute the second derivative of K with respect to:
            dimension dimX of set X, and
            dimension dimX2 of set X2.
        """
        prod_sum = np.zeros((X.shape[0], X2.shape[0]))
        for combination1 in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.K(X, X2) for p in combination1])
            to_update1 = list(set(self.parts) - set(combination1))[0]
            prod_sum += prod*to_update1.dK2_dXdX2(X, X2, dimX, dimX2)
            for combination2 in itertools.combinations(combination1, len(combination1) - 1):
                if len(combination2) > 0:
                    prod = reduce(np.multiply, [p.K(X, X2) for p in combination2])
                else:
                    prod = np.ones(prod_sum.shape)
                to_update2 = list(set(combination1) - set(combination2))[0]
                prod_sum += prod*to_update1.dK_dX(X, X2, dimX)*to_update2.dK_dX2(X, X2, dimX2)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK2_dXdX2diag(self, X, dimX, dimX2, which_parts=None):
        """
        Compute the second derivative of K with respect to:
            dimension dimX of set X, and
            dimension dimX2 of set X2.

        Returns only diagonal elements.
        """
        prod_sum = np.zeros(X.shape[0])
        for combination1 in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.Kdiag(X) for p in combination1])
            to_update1 = list(set(self.parts) - set(combination1))[0]
            prod_sum += prod*to_update1.dK2_dXdX2diag(X, dimX, dimX2)
            for combination2 in itertools.combinations(combination1, len(combination1) - 1):
                if len(combination2) > 0:
                    prod = reduce(np.multiply, [p.Kdiag(X) for p in combination2])
                else:
                    prod = np.ones(prod_sum.shape)
                to_update2 = list(set(combination1) - set(combination2))[0]
                prod_sum += prod*to_update1.dK_dXdiag(X, dimX)*to_update2.dK_dX2diag(X, dimX)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK2_dXdX(self, X, X2, dimX_0, dimX_1, which_parts=None):
        """
        Compute the second derivative of K with respect to:
            dimension dimX_0 of set X, and
            dimension dimX_1 of set X.
        """
        prod_sum = np.zeros((X.shape[0], X2.shape[0]))
        for combination1 in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.K(X, X2) for p in combination1])
            to_update1 = list(set(self.parts) - set(combination1))[0]
            prod_sum += prod*to_update1.dK2_dXdX(X, X2, dimX_0, dimX_1)
            for combination2 in itertools.combinations(combination1, len(combination1) - 1):
                if len(combination2) > 0:
                    prod = reduce(np.multiply, [p.K(X, X2) for p in combination2])
                else:
                    prod = np.ones(prod_sum.shape)
                to_update2 = list(set(combination1) - set(combination2))[0]
                prod_sum += prod*to_update1.dK_dX(X, X2, dimX_0)*to_update2.dK_dX(X, X2, dimX_1)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK3_dXdXdX2(self, X, X2, dimX_0, dimX_1, dimX2, which_parts=None):
        """
        Compute the third derivative of K with respect to:
            dimension dimX_0 of set X,
            dimension dimX_1 of set X, and
            dimension dimX2 of set X2.
        """
        prod_sum = np.zeros((X.shape[0], X2.shape[0]))
        for combination1 in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.K(X, X2) for p in combination1])
            to_update1 = list(set(self.parts) - set(combination1))[0]
            prod_sum += prod*to_update1.dK3_dXdXdX2(X, X2, dimX_0, dimX_1, dimX2)
            for combination2 in itertools.combinations(combination1, len(combination1) - 1):
                if len(combination2) > 0:
                    prod = reduce(np.multiply, [p.K(X, X2) for p in combination2])
                else:
                    prod = np.ones(prod_sum.shape)
                to_update2 = list(set(combination1) - set(combination2))[0]
                prod_sum += prod*to_update1.dK2_dXdX2(X, X2, dimX_0, dimX2)*to_update2.dK_dX(X, X2, dimX_1)
                prod_sum += prod*to_update1.dK2_dXdX(X, X2, dimX_0, dimX_1)*to_update2.dK_dX2(X, X2, dimX2)
                prod_sum += prod*to_update1.dK_dX(X, X2, dimX_0)*to_update2.dK2_dXdX2(X, X2, dimX_1, dimX2)
                if len(self.parts) > 2:
                    for combination3 in itertools.combinations(combination2, len(combination2) - 1):
                        if len(combination3) > 0:
                            prod = reduce(np.multiply, [p.K(X, X2) for p in combination3])
                        else:
                            prod = np.ones(prod_sum.shape)
                        to_update3 = list(set(combination2) - set(combination3))[0]
                        prod_sum += prod*to_update1.dK_dX(X, X2, dimX_0)*to_update2.dK_dX2(X, X2, dimX2)*to_update3.dK_dX(X, X2, dimX_1)
        return prod_sum

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def dK3_dXdXdX2diag(self, X, dimX_0, dimX_1, dimX2, which_parts=None):
        """
        Compute the third derivative of K with respect to:
            dimension dimX_0 of set X,
            dimension dimX_1 of set X, and
            dimension dimX2 of set X2.

        Returns only diagonal elements of the covariance matrix.
        """
        prod_sum = np.zeros(X.shape[0])
        for combination1 in itertools.combinations(self.parts, len(self.parts) - 1):
            prod = reduce(np.multiply, [p.Kdiag(X) for p in combination1])
            to_update1 = list(set(self.parts) - set(combination1))[0]
            prod_sum += prod*to_update1.dK3_dXdXdX2diag(X, dimX_0, dimX_1, dimX2)
            for combination2 in itertools.combinations(combination1, len(combination1) - 1):
                if len(combination2) > 0:
                    prod = reduce(np.multiply, [p.Kdiag(X) for p in combination2])
                else:
                    prod = np.ones(prod_sum.shape)
                to_update2 = list(set(combination1) - set(combination2))[0]
                prod_sum += prod*to_update1.dK2_dXdX2diag(X, dimX_0, dimX2)*to_update2.dK_dXdiag(X, dimX_1)
                prod_sum += prod*to_update1.dK2_dXdXdiag(X, dimX_0, dimX_1)*to_update2.dK_dX2diag(X, dimX2)
                prod_sum += prod*to_update1.dK_dXdiag(X, dimX_0)*to_update2.dK2_dXdX2diag(X, dimX_1, dimX2)
                if len(self.parts) > 2:
                    for combination3 in itertools.combinations(combination2, len(combination2) - 1):
                        if len(combination3) > 0:
                            prod = reduce(np.multiply, [p.Kdiag(X) for p in combination3])
                        else:
                            prod = np.ones(prod_sum.shape)
                        to_update3 = list(set(combination2) - set(combination3))[0]
                        prod_sum += prod*to_update1.dK_dXdiag(X, dimX_0)*to_update2.dK_dX2diag(X, dimX2)*to_update3.dK_dXdiag(X, dimX_1)
        return prod_sum

    def update_gradients_direct(self, *args):
        for i, (g,p) in enumerate(zip(args, self.parts)):
            p.update_gradients_direct(*g)

    def dgradients_dX(self, X, X2, dimX, parts=None):
        if parts is None:
            parts=self.parts
        gradients = []
        for part in parts:
            dgradients_i = part.dgradients(X, X2)
            dgradients_dX_i = part.dgradients_dX(X, X2, dimX)
            neq_parts = [p for p in parts if p is not part]
            if len(neq_parts)>0:
                K_rest = self.K(X,X2, which_parts = neq_parts)
                K_rest_dX = self.dK_dX(X,X2,dimX, which_parts=neq_parts)
            else:
                K_rest = np.ones((X.shape[0], X2.shape[0]))
                K_res_dX = np.zeros((X.shape[0], X2.shape[0]))
            gradients += [ [ g_dX*K_rest + g*K_rest_dX   for i, (g,g_dX) in enumerate(zip(dgradients_i, dgradients_dX_i))] ]
            
        return gradients

    def dgradients_dX2(self, X, X2, dimX2, parts=None):
        if parts is None:
            parts=self.parts
        gradients = []
        for part in parts:
            dgradients_i = part.dgradients(X, X2)
            dgradients_dX_i = part.dgradients_dX2(X, X2, dimX2)
            neq_parts = [p for p in parts if p is not part]
            if len(neq_parts)>0:
                K_rest = self.K(X,X2, which_parts = neq_parts)
                K_rest_dX = self.dK_dX(X,X2,dimX2, which_parts=neq_parts)
            else:
                K_rest = np.ones((X.shape[0], X2.shape[0]))
                K_res_dX = np.zeros((X.shape[0], X2.shape[0]))
            gradients += [ [ g_dX*K_rest + g*K_rest_dX   for g, g_dX in zip(dgradients_i, dgradients_dX_i)] ]
            
        return gradients

    def dgradients2_dXdX2(self, X, X2, dimX, dimX2, parts=None):
        if parts is None:
            parts=self.parts
        gradients = []
        for part in parts:
            g_dxdx2 = part.dgradients2_dXdX2(X,X2,dimX,dimX2)
            g_dx = part.dgradients_dX(X,X2,dimX)
            g_dx2 = part.dgradients_dX2(X,X2,dimX2)
            g = part.dgradients(X, X2)
            neq_parts = [p for p in self.parts if p is not part]
            K_dxdx2 = self.dK2_dXdX2(X, X2, dimX, dimX2, which_parts=neq_parts)
            K_dx = self.dK_dX(X,X2,dimX, which_parts=neq_parts)
            K_dx2 = self.dK_dX2(X,X2,dimX2, which_parts=neq_parts)
            K = self.K(X, X2)
            gradients += [[ g_i*K_dxdx2 + g_dx_i*K_dx2 + g_dx2_i*K_dx + g_dxdx2_i*K for g_i, g_dx_i, g_dx2_i, g_dxdx2_i in zip(g, g_dx, g_dx2, g_dxdx2)]]
        return gradients

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
