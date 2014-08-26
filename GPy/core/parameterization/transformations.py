# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from domains import _POSITIVE,_NEGATIVE, _BOUNDED
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
    def f(self, x):
        raise NotImplementedError
    def finv(self, x):
        raise NotImplementedError
    def gradfactor(self, f):
        """ df_dx evaluated at self.f(x)=f"""
        raise NotImplementedError
    def initialize(self, f):
        """ produce a sensible initial value for f(x)"""
        raise NotImplementedError
    def plot(self, xlabel=r'transformed $\theta$', ylabel=r'$\theta$', axes=None, *args,**kw):
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        import matplotlib.pyplot as plt
        from ...plotting.matplot_dep import base_plots
        x = np.linspace(-8,8)
        base_plots.meanplot(x, self.f(x),axes=axes*args,**kw)
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
    def gradfactor(self, f):
        return np.where(f>_lim_val, 1., 1. - np.exp(-f))
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
        return np.abs(f)
    def __str__(self):
        return '+ve'


class LogexpNeg(Transformation):
    domain = _POSITIVE
    def f(self, x):
        return np.where(x>_lim_val, -x, -np.log(1. + np.exp(np.clip(x, -np.inf, _lim_val))))
        #raises overflow warning: return np.where(x>_lim_val, x, np.log(1. + np.exp(x)))
    def finv(self, f):
        return np.where(f>_lim_val, 0, np.log(np.exp(-f) - 1.))
    def gradfactor(self, f):
        return np.where(f>_lim_val, -1, -1 + np.exp(-f))
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
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
    def gradfactor(self, f):
        return -self.logexp.gradfactor(-f)
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
    def gradfactor(self, f):
        ef = np.exp(f) # np.clip(f, self.min_bound, self.max_bound))
        gf = (ef - 1.) / ef
        return gf # np.where(f < self.lower, 0, gf)
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
        return np.abs(f)
    def __str__(self):
        return '+ve_c'

class Exponent(Transformation):
    # TODO: can't allow this to go to zero, need to set a lower bound. Similar with negative Exponent below. See old MATLAB code.
    domain = _POSITIVE
    def f(self, x):
        return np.where(x<_lim_val, np.where(x>-_lim_val, np.exp(x), np.exp(-_lim_val)), np.exp(_lim_val))
    def finv(self, x):
        return np.log(x)
    def gradfactor(self, f):
        return f
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
        return np.abs(f)
    def __str__(self):
        return '+ve'

class NegativeExponent(Exponent):
    domain = _NEGATIVE
    def f(self, x):
        return -Exponent.f(x)
    def finv(self, f):
        return Exponent.finv(-f)
    def gradfactor(self, f):
        return f
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
    def gradfactor(self, f):
        return 2 * np.sqrt(f)
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
        o = super(Transformation, cls).__new__(cls, lower, upper, *args, **kwargs)
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
    def gradfactor(self, f):
        return (f - self.lower) * (self.upper - f) / self.difference
    def initialize(self, f):
        if np.any(np.logical_or(f < self.lower, f > self.upper)):
            print "Warning: changing parameters to satisfy constraints"
        #return np.where(np.logical_or(f < self.lower, f > self.upper), self.f(f * 0.), f)
        #FIXME: Max, zeros_like right?
        return np.where(np.logical_or(f < self.lower, f > self.upper), self.f(np.zeros_like(f)), f)
    def __str__(self):
        return '{},{}'.format(self.lower, self.upper)



