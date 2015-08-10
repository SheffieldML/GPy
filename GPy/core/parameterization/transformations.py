# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from .domains import _POSITIVE,_NEGATIVE, _BOUNDED
import weakref

import sys

_exp_lim_val = np.finfo(np.float64).max
_lim_val = 36.0
epsilon = np.finfo(np.float64).resolution

#===============================================================================
# Fixing constants
__fixed__ = "fixed"
FIXED = False
UNFIXED = True
#===============================================================================


class Transformation(object):
    domain = None
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance or cls._instance.__class__ is not cls:
            cls._instance = super(Transformation, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    def f(self, opt_param):
        raise NotImplementedError
    def finv(self, model_param):
        raise NotImplementedError
    def log_jacobian(self, model_param):
        """
        compute the log of the jacobian of f, evaluated at f(x)= model_param
        """
        raise NotImplementedError
    def log_jacobian_grad(self, model_param):
        """
        compute the drivative of the log of the jacobian of f, evaluated at f(x)= model_param
        """
        raise NotImplementedError
    def gradfactor(self, model_param, dL_dmodel_param):
        """ df(opt_param)_dopt_param evaluated at self.f(opt_param)=model_param, times the gradient dL_dmodel_param,

        i.e.:
        define

        .. math::

            \frac{\frac{\partial L}{\partial f}\left(\left.\partial f(x)}{\partial x}\right|_{x=f^{-1}(f)\right)}
        """
        raise NotImplementedError
    def gradfactor_non_natural(self, model_param, dL_dmodel_param):
        return self.gradfactor(model_param, dL_dmodel_param)
    def initialize(self, f):
        """ produce a sensible initial value for f(x)"""
        raise NotImplementedError
    def plot(self, xlabel=r'transformed $\theta$', ylabel=r'$\theta$', axes=None, *args,**kw):
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        import matplotlib.pyplot as plt
        from ...plotting.matplot_dep import base_plots
        x = np.linspace(-8,8)
        base_plots.meanplot(x, self.f(x), *args, ax=axes, **kw)
        axes = plt.gca()
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
    def __str__(self):
        raise NotImplementedError
    def __repr__(self):
        return self.__class__.__name__

class Logexp(Transformation):
    domain = _POSITIVE
    def f(self, x):
        return np.where(x>_lim_val, x, np.log1p(np.exp(np.clip(x, -_lim_val, _lim_val)))) + epsilon
        #raises overflow warning: return np.where(x>_lim_val, x, np.log(1. + np.exp(x)))
    def finv(self, f):
        return np.where(f>_lim_val, f, np.log(np.exp(f+1e-20) - 1.))
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, np.where(f>_lim_val, 1., 1. - np.exp(-f)))
    def initialize(self, f):
        if np.any(f < 0.):
            print("Warning: changing parameters to satisfy constraints")
        return np.abs(f)
    def log_jacobian(self, model_param):
        return np.where(model_param>_lim_val, model_param, np.log(np.exp(model_param+1e-20) - 1.)) - model_param
    def log_jacobian_grad(self, model_param):
        return 1./(np.exp(model_param)-1.)
    def __str__(self):
        return '+ve'

class Exponent(Transformation):
    domain = _POSITIVE
    def f(self, x):
        return np.where(x<_lim_val, np.where(x>-_lim_val, np.exp(x), np.exp(-_lim_val)), np.exp(_lim_val))
    def finv(self, x):
        return np.log(x)
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, f)
    def initialize(self, f):
        if np.any(f < 0.):
            print("Warning: changing parameters to satisfy constraints")
        return np.abs(f)
    def log_jacobian(self, model_param):
        return np.log(model_param)
    def log_jacobian_grad(self, model_param):
        return 1./model_param
    def __str__(self):
        return '+ve'



class NormalTheta(Transformation):
    "Do not use, not officially supported!"
    _instances = []
    def __new__(cls, mu_indices=None, var_indices=None):
        "Do not use, not officially supported!"
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu_indices==mu_indices, keepdims=False) and np.all(instance().var_indices==var_indices, keepdims=False):
                    return instance()
        o = super(Transformation, cls).__new__(cls, mu_indices, var_indices)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu_indices, var_indices):
        self.mu_indices = mu_indices
        self.var_indices = var_indices

    def f(self, theta):
        # In here abs is only a trick to make sure the numerics are ok.
        # The variance will never go below zero, but at initialization we need to make sure
        # that the values are ok
        # Before:
        theta[self.var_indices] = np.abs(-.5/theta[self.var_indices])
        #theta[self.var_indices] = np.exp(-.5/theta[self.var_indices])
        theta[self.mu_indices] *= theta[self.var_indices]
        return theta # which is now {mu, var}

    def finv(self, muvar):
        # before:
        varp = muvar[self.var_indices]
        muvar[self.mu_indices] /= varp
        muvar[self.var_indices] = -.5/varp
        #muvar[self.var_indices] = -.5/np.log(varp)

        return muvar # which is now {theta1, theta2}

    def gradfactor(self, muvar, dmuvar):
        mu = muvar[self.mu_indices]
        var = muvar[self.var_indices]
        #=======================================================================
        # theta gradients
        # This works and the gradient checks!
        dmuvar[self.mu_indices] *= var
        dmuvar[self.var_indices] *= 2*(var)**2
        dmuvar[self.var_indices] += 2*dmuvar[self.mu_indices]*mu
        #=======================================================================

        return dmuvar # which is now the gradient multiplicator for {theta1, theta2}

    def initialize(self, f):
        if np.any(f[self.var_indices] < 0.):
            print("Warning: changing parameters to satisfy constraints")
            f[self.var_indices] = np.abs(f[self.var_indices])
        return f

    def __str__(self):
        return "theta"

    def __getstate__(self):
        return [self.mu_indices, self.var_indices]

    def __setstate__(self, state):
        self.mu_indices = state[0]
        self.var_indices = state[1]

class NormalNaturalAntti(NormalTheta):
    "Do not use, not officially supported!"
    _instances = []
    def __new__(cls, mu_indices=None, var_indices=None):
        "Do not use, not officially supported!"
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu_indices==mu_indices, keepdims=False) and np.all(instance().var_indices==var_indices, keepdims=False):
                    return instance()
        o = super(Transformation, cls).__new__(cls, mu_indices, var_indices)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu_indices, var_indices):
        self.mu_indices = mu_indices
        self.var_indices = var_indices

    def gradfactor(self, muvar, dmuvar):
        mu = muvar[self.mu_indices]
        var = muvar[self.var_indices]

        #=======================================================================
        # theta gradients
        # This works and the gradient checks!
        dmuvar[self.mu_indices] *= var
        dmuvar[self.var_indices] *= 2*var**2#np.einsum('i,i,i,i->i', dmuvar[self.var_indices], [2], var, var)
        #=======================================================================

        return dmuvar # which is now the gradient multiplicator

    def initialize(self, f):
        if np.any(f[self.var_indices] < 0.):
            print("Warning: changing parameters to satisfy constraints")
            f[self.var_indices] = np.abs(f[self.var_indices])
        return f

    def __str__(self):
        return "natantti"

class NormalEta(Transformation):
    "Do not use, not officially supported!"
    _instances = []
    def __new__(cls, mu_indices=None, var_indices=None):
        "Do not use, not officially supported!"
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu_indices==mu_indices, keepdims=False) and np.all(instance().var_indices==var_indices, keepdims=False):
                    return instance()
        o = super(Transformation, cls).__new__(cls, mu_indices, var_indices)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu_indices, var_indices):
        self.mu_indices = mu_indices
        self.var_indices = var_indices

    def f(self, theta):
        theta[self.var_indices] = np.abs(theta[self.var_indices] - theta[self.mu_indices]**2)
        return theta # which is now {mu, var}

    def finv(self, muvar):
        muvar[self.var_indices] += muvar[self.mu_indices]**2
        return muvar # which is now {eta1, eta2}

    def gradfactor(self, muvar, dmuvar):
        mu = muvar[self.mu_indices]
        #=======================================================================
        # Lets try natural gradients instead: Not working with bfgs... try stochastic!
        dmuvar[self.mu_indices] -= 2*mu*dmuvar[self.var_indices]
        #=======================================================================
        return dmuvar # which is now the gradient multiplicator

    def initialize(self, f):
        if np.any(f[self.var_indices] < 0.):
            print("Warning: changing parameters to satisfy constraints")
            f[self.var_indices] = np.abs(f[self.var_indices])
        return f

    def __str__(self):
        return "eta"

class NormalNaturalThroughTheta(NormalTheta):
    "Do not use, not officially supported!"
    _instances = []
    def __new__(cls, mu_indices=None, var_indices=None):
        "Do not use, not officially supported!"
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu_indices==mu_indices, keepdims=False) and np.all(instance().var_indices==var_indices, keepdims=False):
                    return instance()
        o = super(Transformation, cls).__new__(cls, mu_indices, var_indices)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu_indices, var_indices):
        self.mu_indices = mu_indices
        self.var_indices = var_indices

    def gradfactor(self, muvar, dmuvar):
        mu = muvar[self.mu_indices]
        var = muvar[self.var_indices]

        #=======================================================================
        # This is just eta direction:
        dmuvar[self.mu_indices] -= 2*mu*dmuvar[self.var_indices]
        #=======================================================================

        #=======================================================================
        # This is by going through theta fully and then going into eta direction:
        #dmu = dmuvar[self.mu_indices]
        #dmuvar[self.var_indices] += dmu*mu*(var + 4/var)
        #=======================================================================
        return dmuvar # which is now the gradient multiplicator

    def gradfactor_non_natural(self, muvar, dmuvar):
        mu = muvar[self.mu_indices]
        var = muvar[self.var_indices]
        #=======================================================================
        # theta gradients
        # This works and the gradient checks!
        dmuvar[self.mu_indices] *= var
        dmuvar[self.var_indices] *= 2*(var)**2
        dmuvar[self.var_indices] += 2*dmuvar[self.mu_indices]*mu
        #=======================================================================

        return dmuvar # which is now the gradient multiplicator for {theta1, theta2}

    def __str__(self):
        return "natgrad"


class NormalNaturalWhooot(NormalTheta):
    "Do not use, not officially supported!"
    _instances = []
    def __new__(cls, mu_indices=None, var_indices=None):
        "Do not use, not officially supported!"
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu_indices==mu_indices, keepdims=False) and np.all(instance().var_indices==var_indices, keepdims=False):
                    return instance()
        o = super(Transformation, cls).__new__(cls, mu_indices, var_indices)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu_indices, var_indices):
        self.mu_indices = mu_indices
        self.var_indices = var_indices

    def gradfactor(self, muvar, dmuvar):
        #mu = muvar[self.mu_indices]
        #var = muvar[self.var_indices]

        #=======================================================================
        # This is just eta direction:
        #dmuvar[self.mu_indices] -= 2*mu*dmuvar[self.var_indices]
        #=======================================================================

        #=======================================================================
        # This is by going through theta fully and then going into eta direction:
        #dmu = dmuvar[self.mu_indices]
        #dmuvar[self.var_indices] += dmu*mu*(var + 4/var)
        #=======================================================================
        return dmuvar # which is now the gradient multiplicator

    def __str__(self):
        return "natgrad"

class NormalNaturalThroughEta(NormalEta):
    "Do not use, not officially supported!"
    _instances = []
    def __new__(cls, mu_indices=None, var_indices=None):
        "Do not use, not officially supported!"
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu_indices==mu_indices, keepdims=False) and np.all(instance().var_indices==var_indices, keepdims=False):
                    return instance()
        o = super(Transformation, cls).__new__(cls, mu_indices, var_indices)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu_indices, var_indices):
        self.mu_indices = mu_indices
        self.var_indices = var_indices

    def gradfactor(self, muvar, dmuvar):
        mu = muvar[self.mu_indices]
        var = muvar[self.var_indices]
        #=======================================================================
        # theta gradients
        # This works and the gradient checks!
        dmuvar[self.mu_indices] *= var
        dmuvar[self.var_indices] *= 2*(var)**2
        dmuvar[self.var_indices] += 2*dmuvar[self.mu_indices]*mu
        #=======================================================================
        return dmuvar

    def __str__(self):
        return "natgrad"


class LogexpNeg(Transformation):
    domain = _POSITIVE
    def f(self, x):
        return np.where(x>_lim_val, -x, -np.log(1. + np.exp(np.clip(x, -np.inf, _lim_val))))
        #raises overflow warning: return np.where(x>_lim_val, x, np.log(1. + np.exp(x)))
    def finv(self, f):
        return np.where(f>_lim_val, 0, np.log(np.exp(-f) - 1.))
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, np.where(f>_lim_val, -1, -1 + np.exp(-f)))
    def initialize(self, f):
        if np.any(f < 0.):
            print("Warning: changing parameters to satisfy constraints")
        return np.abs(f)
    def __str__(self):
        return '+ve'


class NegativeLogexp(Transformation):
    domain = _NEGATIVE
    logexp = Logexp()
    def f(self, x):
        return -self.logexp.f(x)  # np.log(1. + np.exp(x))
    def finv(self, f):
        return self.logexp.finv(-f)  # np.log(np.exp(-f) - 1.)
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, -self.logexp.gradfactor(-f))
    def initialize(self, f):
        return -self.logexp.initialize(f)  # np.abs(f)
    def __str__(self):
        return '-ve'

class LogexpClipped(Logexp):
    max_bound = 1e100
    min_bound = 1e-10
    log_max_bound = np.log(max_bound)
    log_min_bound = np.log(min_bound)
    domain = _POSITIVE
    _instances = []
    def __new__(cls, lower=1e-6, *args, **kwargs):
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().lower == lower:
                    return instance()
        o = super(Transformation, cls).__new__(cls, lower, *args, **kwargs)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, lower=1e-6):
        self.lower = lower
    def f(self, x):
        exp = np.exp(np.clip(x, self.log_min_bound, self.log_max_bound))
        f = np.log(1. + exp)
#         if np.isnan(f).any():
#             import ipdb;ipdb.set_trace()
        return np.clip(f, self.min_bound, self.max_bound)
    def finv(self, f):
        return np.log(np.exp(f - 1.))
    def gradfactor(self, f, df):
        ef = np.exp(f) # np.clip(f, self.min_bound, self.max_bound))
        gf = (ef - 1.) / ef
        return np.einsum('i,i->i', df, gf) # np.where(f < self.lower, 0, gf)
    def initialize(self, f):
        if np.any(f < 0.):
            print("Warning: changing parameters to satisfy constraints")
        return np.abs(f)
    def __str__(self):
        return '+ve_c'

class NegativeExponent(Exponent):
    domain = _NEGATIVE
    def f(self, x):
        return -Exponent.f(x)
    def finv(self, f):
        return Exponent.finv(-f)
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, f)
    def initialize(self, f):
        return -Exponent.initialize(f) #np.abs(f)
    def __str__(self):
        return '-ve'

class Square(Transformation):
    domain = _POSITIVE
    def f(self, x):
        return x ** 2
    def finv(self, x):
        return np.sqrt(x)
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, 2 * np.sqrt(f))
    def initialize(self, f):
        return np.abs(f)
    def __str__(self):
        return '+sq'

class Logistic(Transformation):
    domain = _BOUNDED
    _instances = []
    def __new__(cls, lower=1e-6, upper=1e-6, *args, **kwargs):
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().lower == lower and instance().upper == upper:
                    return instance()
        newfunc = super(Transformation, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)  
        else:
            o = newfunc(cls, lower, upper, *args, **kwargs)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, lower, upper):
        assert lower < upper
        self.lower, self.upper = float(lower), float(upper)
        self.difference = self.upper - self.lower
    def f(self, x):
        if (x<-300.).any():
            x = x.copy()
            x[x<-300.] = -300.
        return self.lower + self.difference / (1. + np.exp(-x))
    def finv(self, f):
        return np.log(np.clip(f - self.lower, 1e-10, np.inf) / np.clip(self.upper - f, 1e-10, np.inf))
    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, (f - self.lower) * (self.upper - f) / self.difference)
    def initialize(self, f):
        if np.any(np.logical_or(f < self.lower, f > self.upper)):
            print("Warning: changing parameters to satisfy constraints")
        #return np.where(np.logical_or(f < self.lower, f > self.upper), self.f(f * 0.), f)
        #FIXME: Max, zeros_like right?
        return np.where(np.logical_or(f < self.lower, f > self.upper), self.f(np.zeros_like(f)), f)
    def __str__(self):
        return '{},{}'.format(self.lower, self.upper)


