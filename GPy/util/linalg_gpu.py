# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


#
# The utility functions for GPU computation
#
import numpy as np

from ..util import gpu_init

try:
    from pycuda.reduction import ReductionKernel
    from pycuda.elementwise import ElementwiseKernel
    
    # log|A| for A is a low triangle matrix
    # logDiagSum(A, A.shape[0]+1)
    logDiagSum = ReductionKernel(np.float64, neutral="0", reduce_expr="a+b", map_expr="i%step==0?log(x[i]):0", arguments="double *x, int step")
    
    strideSum = ReductionKernel(np.float64, neutral="0", reduce_expr="a+b", map_expr="i%step==0?x[i]:0", arguments="double *x, int step")
    
    # np.trace(np.dot(A,B)) (also equivalent to (A*B.T).sum() ) A - a1 x a2, B - a2 x a1
    traceDot = ReductionKernel(np.float64, neutral="0", reduce_expr="a+b", map_expr="A[i]*B[(i%a1)*a2+i/a1]", arguments="double *A, double *B, int a1, int a2")
    
    #=======================================================================================
    # Element-wise functions
    #=======================================================================================
    
    # log(X)
    log = ElementwiseKernel("double *in, double *out", "out[i] = log(in[i])", "log_element")
    
    # log(1.0-X)
    logOne = ElementwiseKernel("double *in, double *out", "out[i] = log(1.-in[i])", "logOne_element")
    
    # multiplication with broadcast on the last dimension (out = shorter[:,None]*longer)
    mul_bcast = ElementwiseKernel("double *out, double *shorter, double *longer, int shorter_size", "out[i] = longer[i]*shorter[i%shorter_size]", "mul_bcast")
    
    # multiplication with broadcast on the first dimension (out = shorter[None,:]*longer)
    mul_bcast_first = ElementwiseKernel("double *out, double *shorter, double *longer, int first_dim", "out[i] = longer[i]*shorter[i/first_dim]", "mul_bcast")
    
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
    
    # the outer product between two vectors (out = np.dot(v1,v2.T))
    outer_prod = ElementwiseKernel("double *out, double *v1, double *v2, int v1_size", "out[i] = v1[i%v1_size]*v2[i/v1_size]", "outer_prod")

    # the outer product between two vectors (out = np.einsum('na,nb->nab',m1,m2) a=dim1, b=dim2 )
    join_prod = ElementwiseKernel("double *out, double *m1, double *m2, int dim1, int dim2", "out[i] = m1[(i%dim1)*dim1+(i%(dim1*dim2))/dim1]*m2[(i%dim1)*dim1+i/(dim1*dim2)]", "join_prod")

except:
    pass

try:
    import scikits.cuda.linalg as culinalg
    from scikits.cuda import cublas
    from scikits.cuda.cula import culaExceptions
except:
    pass

def jitchol(A, L, cublas_handle, maxtries=5):
    try:
        cublas.cublasDcopy(cublas_handle, A.size, A.gpudata, 1, L.gpudata, 1)
        culinalg.cho_factor(L,'L')
    except culaExceptions:
        
        
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError, "not pd: non-positive diagonal elements"
        jitter = diagA.mean() * 1e-6
        while maxtries > 0 and np.isfinite(jitter):
            print 'Warning: adding jitter of {:.10e}'.format(jitter)
            try:
                return linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise linalg.LinAlgError, "not positive definite, even with jitter."
    
    
    
