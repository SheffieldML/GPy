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
    
def _slice_wrapper(kern, operation, diag=False, derivative=False, psi_stat=False, psi_stat_Z=False, ret_X=False):
    """
    This method wraps the functions in kernel to make sure all kernels allways see their respective input dimension.
    The different switches are:
        diag: if X2 exists
        derivative: if first arg is dL_dK
        psi_stat: if first 3 args are dL_dpsi0..2
        psi_stat_Z: if first 2 args are dL_dpsi1..2
    """
    if derivative:
        if diag:
            def x_slice_wrapper(kern, dL_dKdiag, X):
                ret_X_not_sliced = ret_X and kern._sliced_X == 0
                if ret_X_not_sliced:
                    ret = np.zeros(X.shape)
                X = kern._slice_X(X) if not kern._sliced_X else X
                # if the return value is of shape X.shape, we need to make sure to return the right shape
                kern._sliced_X += 1
                try:
                    if ret_X_not_sliced: ret[:, kern.active_dims] = operation(kern, dL_dKdiag, X)
                    else: ret = operation(kern, dL_dKdiag, X)
                except:
                    raise
                finally:
                    kern._sliced_X -= 1
                return ret
        elif psi_stat:
            def x_slice_wrapper(kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
                ret_X_not_sliced = ret_X and kern._sliced_X == 0
                if ret_X_not_sliced:
                    ret1, ret2 = np.zeros(variational_posterior.shape), np.zeros(variational_posterior.shape)
                Z, variational_posterior = kern._slice_X(Z) if not kern._sliced_X else Z, kern._slice_X(variational_posterior) if not kern._sliced_X else variational_posterior
                kern._sliced_X += 1
                # if the return value is of shape X.shape, we need to make sure to return the right shape
                try:
                    if ret_X_not_sliced:
                        ret = list(operation(kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior))
                        r2 = ret[:2]
                        ret[0] = ret1
                        ret[1] = ret2
                        ret[0][:, kern.active_dims] = r2[0]
                        ret[1][:, kern.active_dims] = r2[1]
                        del r2
                    else: ret = operation(kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
                except:
                    raise
                finally:
                    kern._sliced_X -= 1
                return ret
        elif psi_stat_Z:
            def x_slice_wrapper(kern, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
                ret_X_not_sliced = ret_X and kern._sliced_X == 0
                if ret_X_not_sliced: ret = np.zeros(Z.shape)
                Z, variational_posterior = kern._slice_X(Z) if not kern._sliced_X else Z, kern._slice_X(variational_posterior) if not kern._sliced_X else variational_posterior
                kern._sliced_X += 1
                try:
                    if ret_X_not_sliced:
                        ret[:, kern.active_dims] = operation(kern, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
                    else: ret = operation(kern, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
                except:
                    raise
                finally:
                    kern._sliced_X -= 1
                return ret
        else:
            def x_slice_wrapper(kern, dL_dK, X, X2=None):
                ret_X_not_sliced = ret_X and kern._sliced_X == 0
                if ret_X_not_sliced:
                    ret = np.zeros(X.shape)
                X, X2 = kern._slice_X(X) if not kern._sliced_X else X, kern._slice_X(X2) if X2 is not None and not kern._sliced_X else X2
                kern._sliced_X += 1
                try:
                    if ret_X_not_sliced: ret[:, kern.active_dims] = operation(kern, dL_dK, X, X2)
                    else: ret = operation(kern, dL_dK, X, X2)
                except:
                    raise
                finally:
                    kern._sliced_X -= 1
                return ret
    else:
        if diag:
            def x_slice_wrapper(kern, X, *args, **kw):
                X = kern._slice_X(X) if not kern._sliced_X else X
                kern._sliced_X += 1
                try:
                    ret = operation(kern, X, *args, **kw)
                except:
                    raise
                finally:
                    kern._sliced_X -= 1
                return ret
        else: 
            def x_slice_wrapper(kern, X, X2=None, *args, **kw):
                X, X2 = kern._slice_X(X) if not kern._sliced_X else X, kern._slice_X(X2) if X2 is not None and not kern._sliced_X else X2
                kern._sliced_X += 1
                try:
                    ret = operation(kern, X, X2, *args, **kw)
                except: raise
                finally:
                    kern._sliced_X -= 1
                return ret
    x_slice_wrapper._operation = operation
    x_slice_wrapper.__name__ = ("slicer("+str(operation)
                                +(","+str('diag') if diag else'')
                                +(','+str('derivative') if derivative else '')
                                +')')
    x_slice_wrapper.__doc__ = "**sliced**\n" + (operation.__doc__ or "")
    return x_slice_wrapper