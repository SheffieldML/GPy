'''
Created on 11 Mar 2014

@author: maxz
'''
from ...core.parameterization.parameterized import ParametersChangedMeta
import numpy as np
import functools

class KernCallsViaSlicerMeta(ParametersChangedMeta):
    def __call__(self, *args, **kw):
        instance = super(ParametersChangedMeta, self).__call__(*args, **kw)
        instance.K = _Slice_wrapper(instance, instance.K)
        instance.Kdiag = _Slice_wrapper_diag(instance, instance.Kdiag)

        instance.update_gradients_full = _Slice_wrapper_derivative(instance, instance.update_gradients_full)
        instance.update_gradients_diag = _Slice_wrapper_diag_derivative(instance, instance.update_gradients_diag)

        instance.gradients_X = _Slice_wrapper_grad_X(instance, instance.gradients_X)
        instance.gradients_X_diag = _Slice_wrapper_grad_X_diag(instance, instance.gradients_X_diag)

        instance.psi0 = _Slice_wrapper(instance, instance.psi0)
        instance.psi1 = _Slice_wrapper(instance, instance.psi1)
        instance.psi2 = _Slice_wrapper(instance, instance.psi2)

        instance.update_gradients_expectations = _Slice_wrapper_psi_stat_derivative_no_ret(instance, instance.update_gradients_expectations)
        instance.gradients_Z_expectations = _Slice_wrapper_psi_stat_derivative_Z(instance, instance.gradients_Z_expectations)
        instance.gradients_qX_expectations = _Slice_wrapper_psi_stat_derivative(instance, instance.gradients_qX_expectations)
        instance.parameters_changed()
        return instance

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
