# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import weave
from config import *

def chain_1(df_dg, dg_dx):
    """
    Generic chaining function for first derivative

    .. math::
        \\frac{d(f . g)}{dx} = \\frac{df}{dg} \\frac{dg}{dx}
    """
    return df_dg * dg_dx

def chain_2(d2f_dg2, dg_dx, df_dg, d2g_dx2):
    """
    Generic chaining function for second derivative

    .. math::
        \\frac{d^{2}(f . g)}{dx^{2}} = \\frac{d^{2}f}{dg^{2}}(\\frac{dg}{dx})^{2} + \\frac{df}{dg}\\frac{d^{2}g}{dx^{2}}
    """
    return d2f_dg2*(dg_dx**2) + df_dg*d2g_dx2

def chain_3(d3f_dg3, dg_dx, d2f_dg2, d2g_dx2, df_dg, d3g_dx3):
    """
    Generic chaining function for third derivative

    .. math::
        \\frac{d^{3}(f . g)}{dx^{3}} = \\frac{d^{3}f}{dg^{3}}(\\frac{dg}{dx})^{3} + 3\\frac{d^{2}f}{dg^{2}}\\frac{dg}{dx}\\frac{d^{2}g}{dx^{2}} + \\frac{df}{dg}\\frac{d^{3}g}{dx^{3}}
    """
    return d3f_dg3*(dg_dx**3) + 3*d2f_dg2*dg_dx*d2g_dx2 + df_dg*d3g_dx3

def opt_wrapper(m, **kwargs):
    """
    This function just wraps the optimization procedure of a GPy
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

def fast_array_equal(A, B):

    if config.getboolean('parallel', 'openmp'):
        pragma_string = '#pragma omp parallel for private(i, j)'
    else:
        pragma_string = ''

    code2="""
    int i, j;
    return_val = 1;

    %s
    for(i=0;i<N;i++){
       for(j=0;j<D;j++){
          if(A(i, j) != B(i, j)){
              return_val = 0;
              break;
          }
       }
    }
    """ % pragma_string

    if config.getboolean('parallel', 'openmp'):
        pragma_string = '#pragma omp parallel for private(i, j, z)'
    else:
        pragma_string = ''

    code3="""
    int i, j, z;
    return_val = 1;

    %s
    for(i=0;i<N;i++){
       for(j=0;j<D;j++){
         for(z=0;z<Q;z++){
            if(A(i, j, z) != B(i, j, z)){
               return_val = 0;
               break;
            }
          }
       }
    }
    """ % pragma_string

    if config.getboolean('parallel', 'openmp'):
        pragma_string = '#include <omp.h>'
    else:
        pragma_string = ''

    support_code = """
    %s
    #include <math.h>
    """ % pragma_string


    weave_options_openmp = {'headers'           : ['<omp.h>'],
                            'extra_compile_args': ['-fopenmp -O3'],
                            'extra_link_args'   : ['-lgomp'],
                            'libraries': ['gomp']}
    weave_options_noopenmp = {'extra_compile_args': ['-O3']}

    if config.getboolean('parallel', 'openmp'):
        weave_options = weave_options_openmp
    else:
        weave_options = weave_options_noopenmp

    value = False


    if (A == None) and (B == None):
        return True
    elif ((A == None) and (B != None)) or ((A != None) and (B == None)):
        return False
    elif A.shape == B.shape:
        if A.ndim == 2:
            N, D = [int(i) for i in A.shape]
            value = weave.inline(code2, support_code=support_code,
                                 arg_names=['A', 'B', 'N', 'D'],
                                 type_converters=weave.converters.blitz, **weave_options)
        elif A.ndim == 3:
            N, D, Q = [int(i) for i in A.shape]
            value = weave.inline(code3, support_code=support_code,
                                 arg_names=['A', 'B', 'N', 'D', 'Q'],
                                 type_converters=weave.converters.blitz, **weave_options)
        else:
            value = np.array_equal(A,B)

    return value

def fast_array_equal2(A, B):
    if (A == None) and (B == None):
        return True
    elif ((A == None) and (B != None)) or ((A != None) and (B == None)):
        return False
    elif not (A.shape == B.shape):
        return False

    if config.getboolean('parallel', 'openmp'):
        pragma_string = '#include <omp.h>'
        weave_options = {'headers'           : ['<omp.h>'],
                         'extra_compile_args': ['-fopenmp -O3'],
                         'extra_link_args'   : ['-lgomp'],
                         'libraries'         : ['gomp']}
    else:
        weave_options = {'extra_compile_args': ['-O3']}
        pragma_string = ''

    support_code = """
    %s
    #include <math.h>
    """ % pragma_string

    code = """
    int i;
    return_val = 1;

    %s
    for(i=0;i<N;i++){
      if(A[i] != B[i]){
        return_val = 0;
        break;
      }
    }
    """ % pragma_string

    N = A.size
    value = weave.inline(code, support_code=support_code,
                         arg_names=['A', 'B', 'N'],
                         **weave_options)
    return bool(value)




if __name__ == '__main__':
    import pylab as plt
    X = np.linspace(1,10, 100)[:, None]
    X = X[np.random.permutation(X.shape[0])[:20]]
    inducing = kmm_init(X, m = 5)
    plt.figure()
    plt.plot(X.flatten(), np.ones((X.shape[0],)), 'x')
    plt.plot(inducing, 0.5* np.ones((len(inducing),)), 'o')
    plt.ylim((0.0, 10.0))
