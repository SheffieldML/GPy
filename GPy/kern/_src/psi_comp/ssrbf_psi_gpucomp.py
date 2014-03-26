# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the psi statistics computation on GPU
"""

import numpy as np
from GPy.util.caching import Cache_this

try:
    import scikits.cuda.linalg as culinalg
    import pycuda.gpuarray as gpuarray
    from scikits.cuda import cublas
    import pycuda.autoinit
    from pycuda.reduction import ReductionKernel
    from ...util.linalg_gpu import logDiagSum
    
    from pycuda.elementwise import ElementwiseKernel
    
    # The kernel form computing psi1
    comp_psi1 = ElementwiseKernel(
        "double *psi1, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q",
        "psi1[i] = comp_psi1_element(var,l, Z, mu, S, logGamma, log1Gamma, psi1denom, N, M, Q, i)",
        "comp_psi1",
        preamble="""
        #define IDX_MQ(n,m,q) ((n*M+m)*Q+q)
        #define IDX_Q(n,q) (n*Q+q)
        
        __device__ double comp_psi1_element(double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q, int idx)
        {
            int n = idx/M;
            int m = idx%M;
            double psi1=0;
            for(int q=0;q<Q;q++){
                double muZ = mu[IDX_Q(n,q)]-Z[IDX_Q(m,q)];
                double exp1 = logGamma[IDX_Q(n,q)] - (logpsi1denom[IDX_Q(n,q)] + muZ*muZ/(S[IDX_Q(n,q)]+l[q]) )/2.0;
                double exp2 = log1Gamma[IDX_Q(n,q)] - (Z[IDX_Q(m,q)]*Z[IDX_Q(m,q)]/l[q])/2.0;
                psi1 += exp1>=exp2?exp1+log(1.0+exp(exp2-exp1)):exp2+log(1.0+exp(exp1-exp2));
            }
            return var*exp(psi1);
        }
        """)
except:
    pass

class PSICOMP_SSRBF(object):
    def __init__(self):
        pass

@Cache_this(limit=1)
def _Z_distances(Z):
    Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
    Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
    return Zhat, Zdist

def _psicomputations(variance, lengthscale, Z, mu, S, gamma):
    """
    """
    

@Cache_this(limit=1)
def _psi1computations(variance, lengthscale, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi1                NxM
    # _dpsi1_dvariance     NxM
    # _dpsi1_dlengthscale  NxMxQ
    # _dpsi1_dZ            NxMxQ
    # _dpsi1_dgamma        NxMxQ
    # _dpsi1_dmu           NxMxQ
    # _dpsi1_dS            NxMxQ
    
    lengthscale2 = np.square(lengthscale)

    # psi1
    _psi1_denom = S[:, None, :] / lengthscale2 + 1.  # Nx1xQ
    _psi1_denom_sqrt = np.sqrt(_psi1_denom) #Nx1xQ
    _psi1_dist = Z[None, :, :] - mu[:, None, :]  # NxMxQ
    _psi1_dist_sq = np.square(_psi1_dist) / (lengthscale2 * _psi1_denom) # NxMxQ
    _psi1_common = gamma[:,None,:] / (lengthscale2*_psi1_denom*_psi1_denom_sqrt) #Nx1xQ
    _psi1_exponent1 = np.log(gamma[:,None,:]) -0.5 * (_psi1_dist_sq + np.log(_psi1_denom)) # NxMxQ
    _psi1_exponent2 = np.log(1.-gamma[:,None,:]) -0.5 * (np.square(Z[None,:,:])/lengthscale2) # NxMxQ
    _psi1_exponent_max = np.maximum(_psi1_exponent1,_psi1_exponent2)
    _psi1_exponent = _psi1_exponent_max+np.log(np.exp(_psi1_exponent1-_psi1_exponent_max) + np.exp(_psi1_exponent2-_psi1_exponent_max)) #NxMxQ
    _psi1_exp_sum = _psi1_exponent.sum(axis=-1) #NxM
    _psi1_exp_dist_sq = np.exp(-0.5*_psi1_dist_sq) # NxMxQ
    _psi1_exp_Z = np.exp(-0.5*np.square(Z[None,:,:])/lengthscale2) # 1xMxQ
    _psi1_q = variance * np.exp(_psi1_exp_sum[:,:,None] - _psi1_exponent) # NxMxQ
    _psi1 = variance * np.exp(_psi1_exp_sum) # NxM
    _dpsi1_dvariance = _psi1 / variance # NxM
    _dpsi1_dgamma = _psi1_q * (_psi1_exp_dist_sq/_psi1_denom_sqrt-_psi1_exp_Z) # NxMxQ
    _dpsi1_dmu = _psi1_q * (_psi1_exp_dist_sq * _psi1_dist * _psi1_common) # NxMxQ
    _dpsi1_dS = _psi1_q * (_psi1_exp_dist_sq * _psi1_common * 0.5 * (_psi1_dist_sq - 1.)) # NxMxQ
    _dpsi1_dZ = _psi1_q * (- _psi1_common * _psi1_dist * _psi1_exp_dist_sq - (1-gamma[:,None,:])/lengthscale2*Z[None,:,:]*_psi1_exp_Z) # NxMxQ
    _dpsi1_dlengthscale = 2.*lengthscale*_psi1_q * (0.5*_psi1_common*(S[:,None,:]/lengthscale2+_psi1_dist_sq)*_psi1_exp_dist_sq + 0.5*(1-gamma[:,None,:])*np.square(Z[None,:,:]/lengthscale2)*_psi1_exp_Z) # NxMxQ

    N = mu.shape[0]
    M = Z.shape[0]
    Q = mu.shape[1]

    l_gpu = gpuarray.to_gpu(lengthscale2)
    Z_gpu = gpuarray.to_gpu(Z)
    mu_gpu = gpuarray.to_gpu(mu)
    S_gpu = gpuarray.to_gpu(S)
    #gamma_gpu = gpuarray.to_gpu(gamma)
    logGamma_gpu = gpuarray.to_gpu(np.log(gamma))
    log1Gamma_gpu = gpuarray.to_gpu(np.log(1.-gamma))
    logpsi1denom_gpu = gpuarray.to_gpu(np.log(S/lengthscale2+1.))
    psi1_gpu = gpuarray.empty((mu.shape[0],Z.shape[0]),np.float64)
    
    comp_psi1(psi1_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi1denom_gpu, N, M, Q)
    
    print np.abs(psi1_gpu.get()-_psi1).max()

    return _psi1, _dpsi1_dvariance, _dpsi1_dgamma, _dpsi1_dmu, _dpsi1_dS, _dpsi1_dZ, _dpsi1_dlengthscale

@Cache_this(limit=1)
def _psi2computations(variance, lengthscale, Z, mu, S, gamma):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1 and psi2
    # Produced intermediate results:
    # _psi2                NxMxM
    # _psi2_dvariance      NxMxM
    # _psi2_dlengthscale   NxMxMxQ
    # _psi2_dZ             NxMxMxQ
    # _psi2_dgamma         NxMxMxQ
    # _psi2_dmu            NxMxMxQ
    # _psi2_dS             NxMxMxQ
    
    lengthscale2 = np.square(lengthscale)
    
    _psi2_Zhat, _psi2_Zdist = _Z_distances(Z)
    _psi2_Zdist_sq = np.square(_psi2_Zdist / lengthscale) # M,M,Q
    _psi2_Z_sq_sum = (np.square(Z[:,None,:])+np.square(Z[None,:,:]))/lengthscale2 # MxMxQ

    # psi2
    _psi2_denom = 2.*S[:, None, None, :] / lengthscale2 + 1. # Nx1x1xQ
    _psi2_denom_sqrt = np.sqrt(_psi2_denom)
    _psi2_mudist = mu[:,None,None,:]-_psi2_Zhat #N,M,M,Q
    _psi2_mudist_sq = np.square(_psi2_mudist)/(lengthscale2*_psi2_denom)
    _psi2_common = gamma[:,None,None,:]/(lengthscale2 * _psi2_denom * _psi2_denom_sqrt) # Nx1x1xQ
    _psi2_exponent1 = -_psi2_Zdist_sq -_psi2_mudist_sq -0.5*np.log(_psi2_denom)+np.log(gamma[:,None,None,:]) #N,M,M,Q
    _psi2_exponent2 = np.log(1.-gamma[:,None,None,:]) - 0.5*(_psi2_Z_sq_sum) # NxMxMxQ
    _psi2_exponent_max = np.maximum(_psi2_exponent1, _psi2_exponent2)
    _psi2_exponent = _psi2_exponent_max+np.log(np.exp(_psi2_exponent1-_psi2_exponent_max) + np.exp(_psi2_exponent2-_psi2_exponent_max))
    _psi2_exp_sum = _psi2_exponent.sum(axis=-1) #NxM
    _psi2_q = np.square(variance) * np.exp(_psi2_exp_sum[:,:,:,None]-_psi2_exponent) # NxMxMxQ 
    _psi2_exp_dist_sq = np.exp(-_psi2_Zdist_sq -_psi2_mudist_sq) # NxMxMxQ
    _psi2_exp_Z = np.exp(-0.5*_psi2_Z_sq_sum) # MxMxQ
    _psi2 = np.square(variance) * np.exp(_psi2_exp_sum) # N,M,M
    _dpsi2_dvariance = 2. * _psi2/variance # NxMxM
    _dpsi2_dgamma = _psi2_q * (_psi2_exp_dist_sq/_psi2_denom_sqrt - _psi2_exp_Z) # NxMxMxQ
    _dpsi2_dmu = _psi2_q * (-2.*_psi2_common*_psi2_mudist * _psi2_exp_dist_sq) # NxMxMxQ
    _dpsi2_dS = _psi2_q * (_psi2_common * (2.*_psi2_mudist_sq - 1.) * _psi2_exp_dist_sq) # NxMxMxQ
    _dpsi2_dZ = 2.*_psi2_q * (_psi2_common*(-_psi2_Zdist*_psi2_denom+_psi2_mudist)*_psi2_exp_dist_sq - (1-gamma[:,None,None,:])*Z[:,None,:]/lengthscale2*_psi2_exp_Z) # NxMxMxQ
    _dpsi2_dlengthscale = 2.*lengthscale* _psi2_q * (_psi2_common*(S[:,None,None,:]/lengthscale2+_psi2_Zdist_sq*_psi2_denom+_psi2_mudist_sq)*_psi2_exp_dist_sq+(1-gamma[:,None,None,:])*_psi2_Z_sq_sum*0.5/lengthscale2*_psi2_exp_Z) # NxMxMxQ

    return _psi2, _dpsi2_dvariance, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _dpsi2_dZ, _dpsi2_dlengthscale
