'''
Created on 11 Mar 2014

@author: @mzwiessele

This module provides a meta class for the kernels. The meta class is for
slicing the inputs (X, X2) for the kernels, before K (or any other method involving X)
gets calls. The `_all_dims_active` of a kernel decide which dimensions the kernel works on.
'''
import numpy as np
from functools import wraps
from paramz.parameterized import ParametersChangedMeta

def put_clean(dct, name, func):
    if name in dct:
        dct['_clean_{}'.format(name)] = dct[name]
        dct[name] = func(dct[name])

class KernCallsViaSlicerMeta(ParametersChangedMeta):
    def __new__(cls, name, bases, dct):
        put_clean(dct, 'K', _slice_K)
        put_clean(dct, 'dK_dX', _slice_dK_dX)
        put_clean(dct, 'dK_dX2', _slice_dK_dX)
        put_clean(dct, 'dK2_dXdX2', _slice_dK2_dXdX2)
        put_clean(dct, 'Kdiag', _slice_Kdiag)
        put_clean(dct, 'phi', _slice_Kdiag)
        put_clean(dct, 'update_gradients_full', _slice_update_gradients_full)
        put_clean(dct, 'update_gradients_diag', _slice_update_gradients_diag)
        put_clean(dct, 'update_gradients_dK_dX', _slice_update_gradients_full)
        put_clean(dct, 'update_gradients_dK_dX2', _slice_update_gradients_full)
        put_clean(dct, 'gradients_X', _slice_gradients_X)
        put_clean(dct, 'gradients_X2', _slice_gradients_X)
        put_clean(dct, 'gradients_X_X2', _slice_gradients_X)
        put_clean(dct, 'gradients_XX', _slice_gradients_XX)
        put_clean(dct, 'gradients_XX_diag', _slice_gradients_XX_diag)
        put_clean(dct, 'gradients_X_diag', _slice_gradients_X_diag)

        put_clean(dct, 'dgradients_dX',_slice_partial_gradients_list_X)
        put_clean(dct, 'dgradients_dX2',_slice_partial_gradients_list_X)
        put_clean(dct, 'dgradients2_dXdX2',_slice_partial_gradients_list_XX)

        put_clean(dct, 'psi0', _slice_psi)
        put_clean(dct, 'psi1', _slice_psi)
        put_clean(dct, 'psi2', _slice_psi)
        put_clean(dct, 'psi2n', _slice_psi)
        put_clean(dct, 'update_gradients_expectations', _slice_update_gradients_expectations)
        put_clean(dct, 'gradients_Z_expectations', _slice_gradients_Z_expectations)
        put_clean(dct, 'gradients_qX_expectations', _slice_gradients_qX_expectations)
        return super(KernCallsViaSlicerMeta, cls).__new__(cls, name, bases, dct)

class _Slice_wrap(object):
    def __init__(self, k, X, X2=None, diag=False, ret_shape=None):
        self.k = k
        self.diag = diag
        if ret_shape is None:
            self.shape = X.shape
        else:
            self.shape = ret_shape
        assert X.ndim == 2, "need at least column vectors as inputs to kernels for now, given X.shape={!s}".format(X.shape)
        if X2 is not None:
            assert X2.ndim == 2, "need at least column vectors as inputs to kernels for now, given X2.shape={!s}".format(X2.shape)
        if (self.k._all_dims_active is not None) and (self.k._sliced_X == 0):
            self.k._check_active_dims(X)
            self.X = self.k._slice_X(X)
            self.X2 = self.k._slice_X(X2) if X2 is not None else X2
            self.ret = True
        else:
            self.k._check_input_dim(X)
            self.X = X
            self.X2 = X2
            self.ret = False
    def __enter__(self):
        self.k._sliced_X += 1
        return self
    def __exit__(self, *a):
        self.k._sliced_X -= 1
    def handle_return_array(self, return_val):
        if self.ret:
            ret = np.zeros(self.shape)
            if len(self.shape) == 2:
                ret[:, self.k._all_dims_active] = return_val
            elif len(self.shape) == 3: # derivative for X2!=None
                if self.diag:
                    ret.T[np.ix_(self.k._all_dims_active, self.k._all_dims_active)] = return_val.T
                else:
                    ret[:, :, self.k._all_dims_active] = return_val
            elif len(self.shape) == 4: # second order derivative
                ret.T[np.ix_(self.k._all_dims_active, self.k._all_dims_active)] = return_val.T
            return ret
        return return_val

    def handle_return_list(self, return_val):
        if self.ret:
            ret = [np.zeros(self.shape) for _ in range(len(return_val))]
            if len(self.shape) == 3:
                for i in range(len(return_val)):
                    ret[i][self.k._all_dims_active, :, :] = return_val[i]
                #[ret[i].__setitem__((:, :, self.k._all_dims_active), return_val[i])  for i in range(len(return_val))]             
            elif len(self.shape) == 4:
                for i in range(len(return_val)):
                    ret[i][np.ix_(self.k._all_dims_active, self.k._all_dims_active)] = return_val[i]
                #[ret[i].__setitem__((:, :, self.k._all_dims_active, self.k._all_dims_active), return_val[i])  for i in range(len(return_val))]             
            return ret
        return return_val

def _slice_K(f):
    @wraps(f)
    def wrap(self, X, X2 = None, *a, **kw):
        with _Slice_wrap(self, X, X2) as s:
            ret = f(self, s.X, s.X2, *a, **kw)
        return ret
    return wrap

def _slice_Kdiag(f):
    @wraps(f)
    def wrap(self, X, *a, **kw):
        with _Slice_wrap(self, X, None) as s:
            ret = f(self, s.X, *a, **kw)
        return ret
    return wrap

def _slice_update_gradients_full(f):
    @wraps(f)
    def wrap(self, dL_dK, X, X2=None, *a, **kw):
        with _Slice_wrap(self, X, X2) as s:
            ret = f(self, dL_dK, s.X, s.X2, *a, **kw)
        return ret
    return wrap

def _slice_update_gradients_diag(f):
    @wraps(f)
    def wrap(self, dL_dKdiag, X, *a, **kw):
        with _Slice_wrap(self, X, None) as s:
            ret = f(self, dL_dKdiag, s.X)
        return ret
    return wrap

def _slice_gradients_X(f):
    @wraps(f)
    def wrap(self, dL_dK, X, X2=None):
        with _Slice_wrap(self, X, X2) as s:
            ret = s.handle_return_array(f(self, dL_dK, s.X, s.X2))
        return ret
    return wrap

def _slice_dK_dX(f):
    @wraps(f)
    def wrap(self, X, X2, dim, *a, **kw):
        with _Slice_wrap(self, X, X2) as s:
            d = s.k._project_dim(dim)
            if d is None:
                ret = np.zeros((X.shape[0], X2.shape[0]))
            else:
                ret = f(self, s.X, s.X2, d, *a, **kw)
        return ret
    return wrap

def _slice_dK2_dXdX2(f):
    @wraps(f)
    def wrap(self, X, X2, dimX, dimX2, *a, **kw):
        with _Slice_wrap(self, X, X2) as s:
            d = s.k._project_dim(dimX)
            d2 = s.k._project_dim(dimX2)
            if (d is None) or (d2 is None):
                ret = np.zeros((X.shape[0], X2.shape[0]))
            else:
                ret = f(self, s.X, s.X2, d, d2, *a, **kw)
        return ret
    return wrap

def _slice_partial_gradients_X(f):
    @wraps(f)
    def wrap(self, X, X2, dim):
        if X2 is None:
            N, M = X.shape[0], X.shape[0]
        else:
            N, M = X.shape[0], X2.shape[0]
        Q1 = X.shape[1]
        with _Slice_wrap(self, X, X2, ret_shape=(N, M, Q1)) as s:
            ret = s.handle_return_array(f(self, s.X, s.X2, dim))
        return ret
    return wrap

def _slice_partial_gradients_list_X(f):
    @wraps(f)
    def wrap(self, X, X2, dim):
        if X2 is None:
            N, M = X.shape[0], X.shape[0]
        else:
            N, M = X.shape[0], X2.shape[0]
        with _Slice_wrap(self, X, X2, ret_shape=(N, M)) as s:
            d = s.k._project_dim(dim)
            if d is None:
                ret = [np.zeros((N, M)) for i in range(s.k.size)]
            else:            
                ret = f(self, s.X, s.X2, d)
        return ret
    return wrap

def _slice_partial_gradients_XX(f):
    @wraps(f)
    def wrap(self, X, X2, dim, dimX2):
        if X2 is None:
            N, M = X.shape[0], X.shape[0]
            Q1 = X.shape[1]
        else:
            N, M = X.shape[0], X2.shape[0]
            Q1, Q2 = X.shape[1], X2.shape[1]
        with _Slice_wrap(self, X, X2, ret_shape=(N, M, Q1, Q2)) as s:
            ret = s.handle_return_array(f(self, s.X, s.X2, dim, dimX2))
        return ret
    return wrap

def _slice_partial_gradients_list_XX(f):
    @wraps(f)
    def wrap(self, X, X2, dimX, dimX2):
        if X2 is None:
            N, M = X.shape[0], X.shape[0]
        else:
            N, M = X.shape[0], X2.shape[0]
        with _Slice_wrap(self, X, X2, ret_shape=(N, M)) as s:
            d = s.k._project_dim(dimX)
            d2 = s.k._project_dim(dimX2)
            if (d is None) or (d2 is None):
                ret = [np.zeros((N, M)) for i in range(s.k.size)]
            else:
                ret = f(self, s.X, s.X2, d, d2)
        return ret
    return wrap

def _slice_gradient_derivatives_X(f):
    @wraps(f)
    def wrap(self, dL_dK, X, X2=None):
        with _Slice_wrap(self, X, X2) as s:
            ret = s.handle_return_array(f(self, dL_dK, s.X, s.X2))
        return ret
    return wrap

def _slice_gradients_X_diag(f):
    @wraps(f)
    def wrap(self, dL_dKdiag, X):
        with _Slice_wrap(self, X, None) as s:
            ret = s.handle_return_array(f(self, dL_dKdiag, s.X))
        return ret
    return wrap

def _slice_gradients_XX(f):
    @wraps(f)
    def wrap(self, dL_dK, X, X2=None):
        if X2 is None:
            N, M = X.shape[0], X.shape[0]
            Q1 = Q2 = X.shape[1]
        else:
            N, M = X.shape[0], X2.shape[0]
            Q1, Q2 = X.shape[1], X2.shape[1]
        #with _Slice_wrap(self, X, X2, ret_shape=None) as s:
        with _Slice_wrap(self, X, X2, ret_shape=(N, M, Q1, Q2)) as s:
            ret = s.handle_return_array(f(self, dL_dK, s.X, s.X2))
        return ret
    return wrap

def _slice_gradients_XX_diag(f):
    @wraps(f)
    def wrap(self, dL_dKdiag, X):
        N, Q = X.shape
        with _Slice_wrap(self, X, None, diag=True, ret_shape=(N, Q, Q)) as s:
            ret = s.handle_return_array(f(self, dL_dKdiag, s.X))
        return ret
    return wrap

def _slice_psi(f):
    @wraps(f)
    def wrap(self, Z, variational_posterior):
        with _Slice_wrap(self, Z, variational_posterior) as s:
            ret = f(self, s.X, s.X2)
        return ret
    return wrap

def _slice_update_gradients_expectations(f):
    @wraps(f)
    def wrap(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        with _Slice_wrap(self, Z, variational_posterior) as s:
            ret = f(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, s.X, s.X2)
        return ret
    return wrap

def _slice_gradients_Z_expectations(f):
    @wraps(f)
    def wrap(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior,
             psi0=None, psi1=None, psi2=None, Lpsi0=None, Lpsi1=None, Lpsi2=None):
        with _Slice_wrap(self, Z, variational_posterior) as s:
            ret = s.handle_return_array(f(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, s.X, s.X2))
        return ret
    return wrap

def _slice_gradients_qX_expectations(f):
    @wraps(f)
    def wrap(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior,
             psi0=None, psi1=None, psi2=None, Lpsi0=None, Lpsi1=None, Lpsi2=None):
        with _Slice_wrap(self, variational_posterior, Z) as s:
            ret = list(f(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, s.X2, s.X))
            r2 = ret[:2]
            ret[0] = s.handle_return_array(r2[0])
            ret[1] = s.handle_return_array(r2[1])
            del r2
        return ret
    return wrap
