'''
Created on 11 Mar 2014

@author: maxz
'''
from ...core.parameterization.parameterized import ParametersChangedMeta
import numpy as np

def put_clean(dct, name, *args, **kw):
    if name in dct:
        dct['_clean_{}'.format(name)] = dct[name]
        dct[name] = _slice_wrapper(None, dct[name], *args, **kw)
    
class KernCallsViaSlicerMeta(ParametersChangedMeta):
    def __new__(cls, name, bases, dct):
        put_clean(dct, 'K')
        put_clean(dct, 'Kdiag', diag=True)
        put_clean(dct, 'update_gradients_full', diag=False, derivative=True)
        put_clean(dct, 'gradients_X', diag=False, derivative=True, ret_X=True)
        put_clean(dct, 'gradients_X_diag', diag=True, derivative=True, ret_X=True)
        put_clean(dct, 'psi0', diag=False, derivative=False)
        put_clean(dct, 'psi1', diag=False, derivative=False)
        put_clean(dct, 'psi2', diag=False, derivative=False)
        put_clean(dct, 'update_gradients_expectations', derivative=True, psi_stat=True)
        put_clean(dct, 'gradients_Z_expectations', derivative=True, psi_stat_Z=True, ret_X=True)
        put_clean(dct, 'gradients_qX_expectations', derivative=True, psi_stat=True, ret_X=True)
        return super(KernCallsViaSlicerMeta, cls).__new__(cls, name, bases, dct)
    
class _Slice_wrap(object):
    def __init__(self, instance, f):
        self.k = instance
        self.f = f
    def copy_to(self, new_instance):
        return self.__class__(new_instance, self.f)
    def _slice_X(self, X):
        return self.k._slice_X(X) if not self.k._sliced_X else X
    def _slice_X_X2(self, X, X2):
        return self.k._slice_X(X) if not self.k._sliced_X else X, self.k._slice_X(X2) if X2 is not None and not self.k._sliced_X else X2
    def __enter__(self):
        self.k._sliced_X += 1
        return self
    def __exit__(self, *a):
        self.k._sliced_X -= 1

class _Slice_wrapper(_Slice_wrap):
    def __call__(self, X, X2 = None, *a, **kw):
        X, X2 = self._slice_X_X2(X, X2)
        with self:
            ret = self.f(X, X2, *a, **kw)
        return ret

class _Slice_wrapper_diag(_Slice_wrap):
    def __call__(self, X, *a, **kw):
        X = self._slice_X(X)
        with self:
            ret = self.f(X, *a, **kw)
        return ret

class _Slice_wrapper_derivative(_Slice_wrap):
    def __call__(self, dL_dK, X, X2=None):
        self._slice_X(X)
        with self:
            ret = self.f(dL_dK, X, X2)
        return ret

class _Slice_wrapper_diag_derivative(_Slice_wrap):
    def __call__(self, dL_dKdiag, X):
        X = self._slice_X(X)
        with self:
            ret = self.f(dL_dKdiag, X)
        return ret

class _Slice_wrapper_grad_X(_Slice_wrap):
    def __call__(self, dL_dK, X, X2=None):
        ret = np.zeros(X.shape)
        X, X2 = self._slice_X_X2(X, X2)
        with self:
            ret[:, self.k.active_dims] = self.f(dL_dK, X, X2)
        return ret

class _Slice_wrapper_grad_X_diag(_Slice_wrap):
    def __call__(self, dL_dKdiag, X):
        ret = np.zeros(X.shape)
        X = self._slice_X(X)
        with self:
            ret[:, self.k.active_dims] = self.f(dL_dKdiag, X)
        return ret

class _Slice_wrapper_psi_stat_derivative_no_ret(_Slice_wrap):
    def __call__(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        Z, variational_posterior = self._slice_X_X2(Z, variational_posterior)
        with self:
            ret = self.f(dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        return ret

class _Slice_wrapper_psi_stat_derivative(_Slice_wrap):
    def __call__(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        ret1, ret2 = np.zeros(variational_posterior.shape), np.zeros(variational_posterior.shape)
        Z, variational_posterior = self._slice_X_X2(Z, variational_posterior)
        with self:
            ret = list(self.f(dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior))
            r2 = ret[:2]
            ret[0] = ret1
            ret[1] = ret2
            ret[0][:, self.k.active_dims] = r2[0]
            ret[1][:, self.k.active_dims] = r2[1]
            del r2
        return ret

class _Slice_wrapper_psi_stat_derivative_Z(_Slice_wrap):
    def __call__(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        ret1, ret2 = np.zeros(variational_posterior.shape), np.zeros(variational_posterior.shape)
        Z, variational_posterior = self._slice_X_X2(Z, variational_posterior)
        with self:
            ret = list(self.f(dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior))
            r2 = ret[:2]
            ret[0] = ret1
            ret[1] = ret2
            ret[0][:, self.k.active_dims] = r2[0]
            ret[1][:, self.k.active_dims] = r2[1]
            del r2
        return ret
