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
    
    strideSum = ReductionKernel(np.float64, neutral="0", reduce_expr="a+b", map_expr="i%step==0?x[i]:0", arguments="double *x, int step")
    
    #=======================================================================================
    # Element-wise functions
    #=======================================================================================
    
    # log(X)
    log = ElementwiseKernel("double *in, double *out", "out[i] = log(in[i])", "log_element")
    
    # log(1.0-X)
    logOne = ElementwiseKernel("double *in, double *out", "out[i] = log(1.-in[i])", "logOne_element")
    
    # multiplication with broadcast on the last dimension
    mul_bcast = ElementwiseKernel("double *out, double *shorter, double *longer, int shorter_size", "out[i] = longer[i]*shorter[i%shorter_size]", "mul_bcast")
    
    # sum through the middle dimension (size_2) of a 3D matrix (size_1, size_2, size_3) 
    sum_axis = ElementwiseKernel("double *out, double *in, int size_1, int size_2", "out[i] += sum_axis_element(in, size_1, size_2, i)", "sum_axis",preamble="""        
        __device__ double sum_axis_element(double *in, int size_1, int size_2, int idx)
        {
            int k = idx/size_1;
            int i = idx%size_1;
            double sum=0;
            for(int j=0;j<size_2;j++) {
                sum += in[(k*size_2+j)*size_1+i];
            }
            return sum;
        }
        """)
    
except:
    pass
