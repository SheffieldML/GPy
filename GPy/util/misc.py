# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import cbrt
from .config import *

_lim_val = np.finfo(np.float64).max
_lim_val_exp = np.log(_lim_val)
_lim_val_square = np.sqrt(_lim_val)
#_lim_val_cube = cbrt(_lim_val)
_lim_val_cube = np.nextafter(_lim_val**(1/3.0), -np.inf)
_lim_val_quad = np.nextafter(_lim_val**(1/4.0), -np.inf)
_lim_val_three_times = np.nextafter(_lim_val/3.0, -np.inf)

def safe_exp(f):
    clip_f = np.clip(f, -np.inf, _lim_val_exp)
    return np.exp(clip_f)

def safe_square(f):
    f = np.clip(f, -np.inf, _lim_val_square)
    return f**2

def safe_cube(f):
    f = np.clip(f, -np.inf, _lim_val_cube)
    return f**3

def safe_quad(f):
    f = np.clip(f, -np.inf, _lim_val_quad)
    return f**4

def safe_three_times(f):
    f = np.clip(f, -np.inf, _lim_val_three_times)
    return 3*f

def chain_1(df_dg, dg_dx):
    """
    Generic chaining function for first derivative

    .. math::
        \\frac{d(f . g)}{dx} = \\frac{df}{dg} \\frac{dg}{dx}
    """
    if np.all(dg_dx==1.):
        return df_dg
    return df_dg * dg_dx

def chain_2(d2f_dg2, dg_dx, df_dg, d2g_dx2):
    """
    Generic chaining function for second derivative

    .. math::
        \\frac{d^{2}(f . g)}{dx^{2}} = \\frac{d^{2}f}{dg^{2}}(\\frac{dg}{dx})^{2} + \\frac{df}{dg}\\frac{d^{2}g}{dx^{2}}
    """
    if np.all(dg_dx==1.) and np.all(d2g_dx2 == 0):
        return d2f_dg2
    dg_dx_2 = np.clip(dg_dx, -np.inf, _lim_val_square)**2
    #dg_dx_2 = dg_dx**2
    return d2f_dg2*(dg_dx_2) + df_dg*d2g_dx2

def chain_3(d3f_dg3, dg_dx, d2f_dg2, d2g_dx2, df_dg, d3g_dx3):
    """
    Generic chaining function for third derivative

    .. math::
        \\frac{d^{3}(f . g)}{dx^{3}} = \\frac{d^{3}f}{dg^{3}}(\\frac{dg}{dx})^{3} + 3\\frac{d^{2}f}{dg^{2}}\\frac{dg}{dx}\\frac{d^{2}g}{dx^{2}} + \\frac{df}{dg}\\frac{d^{3}g}{dx^{3}}
    """
    if np.all(dg_dx==1.) and np.all(d2g_dx2==0) and np.all(d3g_dx3==0):
        return d3f_dg3
    dg_dx_3 = np.clip(dg_dx, -np.inf, _lim_val_cube)**3
    return d3f_dg3*(dg_dx_3) + 3*d2f_dg2*dg_dx*d2g_dx2 + df_dg*d3g_dx3

def opt_wrapper(m, **kwargs):
    """
    Thit function just wraps the optimization procedure of a GPy
    object so that optimize() pickleable (necessary for multiprocessing).
    """
    m.optimize(**kwargs)
    return m.optimization_runs[-1]


def linear_grid(D, n = 100, min_max = (-100, 100)):
    """
    Creates a D-dimensional grid of n linearly spaced points

    :param D: dimension of the grid
    :param n: number of points
    :param min_max: (min, max) list

    """

    g = np.linspace(min_max[0], min_max[1], n)
    G = np.ones((n, D))

    return G*g[:,None]

def kmm_init(X, m = 10):
    """
    This is the same initialization algorithm that is used
    in Kmeans++. It's quite simple and very useful to initialize
    the locations of the inducing points in sparse GPs.

    :param X: data
    :param m: number of inducing points

    """

    # compute the distances
    XXT = np.dot(X, X.T)
    D = (-2.*XXT + np.diag(XXT)[:,np.newaxis] + np.diag(XXT)[np.newaxis,:])

    # select the first point
    s = np.random.permutation(X.shape[0])[0]
    inducing = [s]
    prob = D[s]/D[s].sum()

    for z in range(m-1):
        s = np.random.multinomial(1, prob.flatten()).argmax()
        inducing.append(s)
        prob = D[s]/D[s].sum()

    inducing = np.array(inducing)
    return X[inducing]

### make a parameter to its corresponding array:
def param_to_array(*param):
    """
    Convert an arbitrary number of parameters to :class:ndarray class objects.
    This is for converting parameter objects to numpy arrays, when using
    scipy.weave.inline routine.  In scipy.weave.blitz there is no automatic
    array detection (even when the array inherits from :class:ndarray)
    """
    import warnings
    warnings.warn("Please use param.values, as this function will be deprecated in the next release.", DeprecationWarning)
    assert len(param) > 0, "At least one parameter needed"
    if len(param) == 1:
        return param[0].view(np.ndarray)
    return [x.view(np.ndarray) for x in param]

def blockify_hessian(func):
    def wrapper_func(self, *args, **kwargs):
        # Invoke the wrapped function first
        retval = func(self, *args, **kwargs)
        # Now do something here with retval and/or action
        if self.not_block_really and (retval.shape[0] != retval.shape[1]):
            return np.diagflat(retval)
        else:
            return retval
    return wrapper_func

def blockify_third(func):
    def wrapper_func(self, *args, **kwargs):
        # Invoke the wrapped function first
        retval = func(self, *args, **kwargs)
        # Now do something here with retval and/or action
        if self.not_block_really and (len(retval.shape) < 3):
            num_data = retval.shape[0]
            d3_block_cache = np.zeros((num_data, num_data, num_data))
            diag_slice = range(num_data)
            d3_block_cache[diag_slice, diag_slice, diag_slice] = np.squeeze(retval)
            return d3_block_cache
        else:
            return retval
    return wrapper_func

def blockify_dhess_dtheta(func):
    def wrapper_func(self, *args, **kwargs):
        # Invoke the wrapped function first
        retval = func(self, *args, **kwargs)
        # Now do something here with retval and/or action
        if self.not_block_really and (len(retval.shape) < 3):
            num_data = retval.shape[0]
            num_params = retval.shape[-1]
            dhess_dtheta = np.zeros((num_data, num_data, num_params))
            diag_slice = range(num_data)
            for param_ind in range(num_params):
                dhess_dtheta[diag_slice, diag_slice, param_ind] = np.squeeze(retval[:,param_ind])
            return dhess_dtheta
        else:
            return retval
    return wrapper_func

