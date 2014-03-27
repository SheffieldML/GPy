# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


#
# The utility functions for GPU computation
#
import numpy as np

try:
    import pycuda.autoinit
    from pycuda.reduction import ReductionKernel
    from pycuda.elementwise import ElementwiseKernel
    
    # log|A| for A is a low triangle matrix
    # logDiagSum(A, A.shape[0]+1)
    logDiagSum = ReductionKernel(np.float64, neutral="0", reduce_expr="a+b", map_expr="i%step==0?log(x[i]):0", arguments="double *x, int step")
    
    #=======================================================================================
    # Element-wise functions
    #=======================================================================================
    
    # log(X)
    log = ElementwiseKernel("double *in, double *out", "out[i] = log(in[i])", "log_element")
    
    # log(1.0-X)
    logOne = ElementwiseKernel("double *in, double *out", "out[i] = log(1.-in[i])", "logOne_element")
except:
    pass
