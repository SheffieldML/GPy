# Code for testing functions written in sympy_helpers.cpp
from scipy import weave
import tempfile
import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
extra_compile_args = []

weave_kwargs = {
    'support_code': "",
    'include_dirs':[tempfile.gettempdir(), current_dir],
    'headers':['"parts/sympy_helpers.h"'],
    'sources':[os.path.join(current_dir,"parts/sympy_helpers.cpp")],
    'extra_compile_args':extra_compile_args,
    'extra_link_args':['-lgomp'],
    'verbose':True}

def erfcx(x):
    code = """
        // Code for computing scaled complementary erf
        int i;
        int dim;
        int elements = Ntarget[0];
        for (dim=1; dim<Dtarget; dim++)
          elements *= Ntarget[dim];
        for (i=0;i<elements;i++) 
            target[i] = erfcx(x[i]);
        """
    x = np.asarray(x)
    arg_names = ['target','x']
    target = np.zeros_like(x)
    weave.inline(code=code, arg_names=arg_names,**weave_kwargs)
    return target

def ln_diff_erf(x, y):
    code = """
        // Code for computing scaled complementary erf
        int i;
        int dim;
        int elements = Ntarget[0];
        for (dim=1; dim<Dtarget; dim++)
          elements *= Ntarget[dim];
        for (i=0;i<elements;i++) 
          target[i] = ln_diff_erf(x[i], y[i]);
        """
    x = np.asarray(x)
    y = np.asarray(y)
    assert(x.shape==y.shape)
    target = np.zeros_like(x)
    arg_names = ['target','x', 'y']
    weave.inline(code=code, arg_names=arg_names,**weave_kwargs)
    return target

def h(t, tprime, d_i, d_j, l):
    code = """
        // Code for computing the 1st order ODE h helper function.
        int i;
        int dim;
        int elements = Ntarget[0];
        for (dim=1; dim<Dtarget; dim++)
          elements *= Ntarget[dim];
        for (i=0;i<elements;i++) 
          target[i] = h(t[i], tprime[i], d_i, d_j, l);
        """
    t = np.asarray(t)
    tprime = np.asarray(tprime)
    assert(tprime.shape==t.shape)
    target = np.zeros_like(t)
    arg_names = ['target','t', 'tprime', 'd_i', 'd_j', 'l']
    weave.inline(code=code, arg_names=arg_names,**weave_kwargs)
    return target
