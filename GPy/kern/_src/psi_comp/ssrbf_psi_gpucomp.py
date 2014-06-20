# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the psi statistics computation on GPU
"""

import numpy as np
from GPy.util.caching import Cache_this

from ....util import gpu_init

try:
    import pycuda.gpuarray as gpuarray
    from scikits.cuda import cublas
    from pycuda.reduction import ReductionKernel    
    from pycuda.elementwise import ElementwiseKernel
    from ....util import linalg_gpu
    
    
    # The kernel form computing psi1 het_noise
    comp_psi1 = ElementwiseKernel(
        "double *psi1, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q",
        "psi1[i] = comp_psi1_element(var, l, Z, mu, S, logGamma, log1Gamma, logpsi1denom, N, M, Q, i)",
        "comp_psi1",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_psi1_element(double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q, int idx)
        {
            int n = idx%N;
            int m = idx/N;
            double psi1_exp=0;
            for(int q=0;q<Q;q++){
                double muZ = mu[IDX_NQ(n,q)]-Z[IDX_MQ(m,q)];
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi1denom[IDX_NQ(n,q)] + muZ*muZ/(S[IDX_NQ(n,q)]+l[q]) )/2.0;
                double exp2 = log1Gamma[IDX_NQ(n,q)] - Z[IDX_MQ(m,q)]*Z[IDX_MQ(m,q)]/(l[q]*2.0);
                psi1_exp += LOGEXPSUM(exp1,exp2);
            }
            return var*exp(psi1_exp);
        }
        """)
        
    # The kernel form computing psi2 het_noise
    comp_psi2 = ElementwiseKernel(
        "double *psi2, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q",
        "psi2[i] = comp_psi2_element(var, l, Z, mu, S, logGamma, log1Gamma, logpsi2denom, N, M, Q, i)",
        "comp_psi2",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_psi2_element(double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q, int idx)
        {
            // psi2 (n,m1,m2)
            int m2 = idx/M;
            int m1 = idx%M;
            
            double psi2=0;            
            for(int n=0;n<N;n++){
                double psi2_exp=0;
                for(int q=0;q<Q;q++){ 
                    double dZ = Z[IDX_MQ(m1,q)]-Z[IDX_MQ(m2,q)];
                    double muZ = mu[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.0;
                    double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi2denom[IDX_NQ(n,q)])/2.0 - dZ*dZ/(l[q]*4.0) - muZ*muZ/(2*S[IDX_NQ(n,q)]+l[q]);
                    double exp2 = log1Gamma[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]*Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)]*Z[IDX_MQ(m2,q)])/(l[q]*2.0);
                    psi2_exp += LOGEXPSUM(exp1,exp2);
                }
                psi2 += exp(psi2_exp);
            }
            return var*var*psi2;
        }
        """) 
    
    # compute psidenom
    comp_logpsidenom = ElementwiseKernel(
        "double *out, double *S, double *l, double scale, int N",
        "out[i] = comp_logpsidenom_element(S, l, scale, N, i)",
        "comp_logpsidenom",
        preamble="""        
        __device__ double comp_logpsidenom_element(double *S, double *l, double scale, int N, int idx)
        {
            int q = idx/N;
            
            return log(scale*S[idx]/l[q]+1.0);
        }
        """)
    
    # The kernel form computing psi1 het_noise
    comp_dpsi1_dvar = ElementwiseKernel(
        "double *dpsi1_dvar, double *psi1_neq, double *psi1exp1, double *psi1exp2, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q",
        "dpsi1_dvar[i] = comp_dpsi1_dvar_element(psi1_neq, psi1exp1, psi1exp2, l, Z, mu, S, logGamma, log1Gamma, logpsi1denom, N, M, Q, i)",
        "comp_dpsi1_dvar",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_dpsi1_dvar_element(double *psi1_neq, double *psi1exp1, double *psi1exp2, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi1denom, int N, int M, int Q, int idx)
        {
            int n = idx%N;
            int m = idx/N;
            
            double psi1_sum = 0;
            for(int q=0;q<Q;q++){            
                double muZ = mu[IDX_NQ(n,q)]-Z[IDX_MQ(m,q)];
                double exp1_e = -(muZ*muZ/(S[IDX_NQ(n,q)]+l[q]) )/2.0;
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi1denom[IDX_NQ(n,q)])/2.0 + exp1_e;
                double exp2_e = - Z[IDX_MQ(m,q)]*Z[IDX_MQ(m,q)]/(l[q]*2.0);
                double exp2 = log1Gamma[IDX_NQ(n,q)] + exp2_e;
                double psi1_q = LOGEXPSUM(exp1,exp2);
                psi1_neq[IDX_NMQ(n,m,q)] = -psi1_q;
                psi1exp1[IDX_NMQ(n,m,q)] = exp(exp1_e);
                psi1exp2[IDX_MQ(m,q)] = exp(exp2_e);
                psi1_sum += psi1_q;
            }
            for(int q=0;q<Q;q++) {
                psi1_neq[IDX_NMQ(n,m,q)] = exp(psi1_neq[IDX_NMQ(n,m,q)]+psi1_sum);
            }
            return exp(psi1_sum);
        }
        """)
    
    # The kernel form computing psi1 het_noise
    comp_psi1_der = ElementwiseKernel(
        "double *dpsi1_dl, double *dpsi1_dmu, double *dpsi1_dS, double *dpsi1_dgamma, double *dpsi1_dZ, double *psi1_neq, double *psi1exp1, double *psi1exp2, double var, double *l, double *Z, double *mu, double *S, double *gamma, int N, int M, int Q",
        "dpsi1_dl[i] = comp_psi1_der_element(dpsi1_dmu, dpsi1_dS, dpsi1_dgamma, dpsi1_dZ, psi1_neq, psi1exp1, psi1exp2, var, l, Z, mu, S, gamma, N, M, Q, i)",
        "comp_psi1_der",
        preamble="""
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        
        __device__ double comp_psi1_der_element(double *dpsi1_dmu, double *dpsi1_dS, double *dpsi1_dgamma, double *dpsi1_dZ, double *psi1_neq, double *psi1exp1, double *psi1exp2, double var, double *l, double *Z, double *mu, double *S, double *gamma, int N, int M, int Q, int idx)
        {
            int q = idx/(M*N);
            int m = (idx%(M*N))/N;
            int n = idx%N;
            
            double neq = psi1_neq[IDX_NMQ(n,m,q)];
            double gamma_c = gamma[IDX_NQ(n,q)];
            double Z_c = Z[IDX_MQ(m,q)];
            double S_c = S[IDX_NQ(n,q)];
            double l_c = l[q];
            double l_sqrt_c = sqrt(l[q]);
            double psi1exp1_c = psi1exp1[IDX_NMQ(n,m,q)];
            double psi1exp2_c = psi1exp2[IDX_MQ(m,q)];

            double denom = S_c/l_c+1.0;
            double denom_sqrt = sqrt(denom);
            double Zmu = Z_c-mu[IDX_NQ(n,q)];
            double psi1_common = gamma_c/(denom_sqrt*denom*l_c);
            double gamma1 = 1-gamma_c;
            
            dpsi1_dgamma[IDX_NMQ(n,m,q)] = var*neq*(psi1exp1_c/denom_sqrt - psi1exp2_c);
            dpsi1_dmu[IDX_NMQ(n,m,q)] = var*neq*(psi1_common*Zmu*psi1exp1_c);
            dpsi1_dS[IDX_NMQ(n,m,q)] = var*neq*(psi1_common*(Zmu*Zmu/(S_c+l_c)-1.0)*psi1exp1_c)/2.0;
            dpsi1_dZ[IDX_NMQ(n,m,q)] = var*neq*(-psi1_common*Zmu*psi1exp1_c-gamma1*Z_c/l_c*psi1exp2_c);
            return var*neq*(psi1_common*(S_c/l_c+Zmu*Zmu/(S_c+l_c))*psi1exp1_c+gamma1*Z_c*Z_c/l_c*psi1exp2_c)*l_sqrt_c;    
        }
        """)
    
    # The kernel form computing psi1 het_noise
    comp_dpsi2_dvar = ElementwiseKernel(
        "double *dpsi2_dvar, double *psi2_neq, double *psi2exp1, double *psi2exp2, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q",
        "dpsi2_dvar[i] = comp_dpsi2_dvar_element(psi2_neq, psi2exp1, psi2exp2, var, l, Z, mu, S, logGamma, log1Gamma, logpsi2denom, N, M, Q, i)",
        "comp_dpsi2_dvar",
        preamble="""
        #define IDX_NMMQ(n,m1,m2,q) (((q*M+m2)*M+m1)*N+n)
        #define IDX_MMQ(m1,m2,q) ((q*M+m2)*M+m1)
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        #define LOGEXPSUM(a,b) (a>=b?a+log(1.0+exp(b-a)):b+log(1.0+exp(a-b)))
        
        __device__ double comp_dpsi2_dvar_element(double *psi2_neq, double *psi2exp1, double *psi2exp2, double var, double *l, double *Z, double *mu, double *S, double *logGamma, double *log1Gamma, double *logpsi2denom, int N, int M, int Q, int idx)
        {
            // psi2 (n,m1,m2)
            int m2 = idx/(M*N);
            int m1 = (idx%(M*N))/N;
            int n = idx%N;

            double psi2_sum=0;
            for(int q=0;q<Q;q++){ 
                double dZ = Z[IDX_MQ(m1,q)]-Z[IDX_MQ(m2,q)];
                double muZ = mu[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.0;
                double exp1_e = - dZ*dZ/(l[q]*4.0) - muZ*muZ/(2*S[IDX_NQ(n,q)]+l[q]);
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi2denom[IDX_NQ(n,q)])/2.0 +exp1_e;
                double exp2_e = - (Z[IDX_MQ(m1,q)]*Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)]*Z[IDX_MQ(m2,q)])/(l[q]*2.0);
                double exp2 = log1Gamma[IDX_NQ(n,q)] + exp2_e;
                double psi2_q = LOGEXPSUM(exp1,exp2);
                psi2_neq[IDX_NMMQ(n,m1,m2,q)] = -psi2_q;
                psi2exp1[IDX_NMMQ(n,m1,m2,q)] = exp(exp1_e);
                psi2exp2[IDX_MMQ(m1,m2,q)] = exp(exp2_e);
                psi2_sum += psi2_q;
            }
            for(int q=0;q<Q;q++) {
                psi2_neq[IDX_NMMQ(n,m1,m2,q)] = exp(psi2_neq[IDX_NMMQ(n,m1,m2,q)]+psi2_sum);
            }
            return 2*var*exp(psi2_sum);            
        }
        """)
    
    # The kernel form computing psi1 het_noise
    comp_psi2_der = ElementwiseKernel(
        "double *dpsi2_dl, double *dpsi2_dmu, double *dpsi2_dS, double *dpsi2_dgamma, double *dpsi2_dZ, double *psi2_neq, double *psi2exp1, double *psi2exp2, double var, double *l, double *Z, double *mu, double *S, double *gamma, int N, int M, int Q",
        "dpsi2_dl[i] = comp_psi2_der_element(dpsi2_dmu, dpsi2_dS, dpsi2_dgamma, dpsi2_dZ, psi2_neq, psi2exp1, psi2exp2, var, l, Z, mu, S, gamma, N, M, Q, i)",
        "comp_psi2_der",
        preamble="""
        #define IDX_NMMQ(n,m1,m2,q) (((q*M+m2)*M+m1)*N+n)
        #define IDX_MMQ(m1,m2,q) ((q*M+m2)*M+m1)
        #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
        #define IDX_NQ(n,q) (q*N+n)
        #define IDX_MQ(m,q) (q*M+m)
        
        __device__ double comp_psi2_der_element(double *dpsi2_dmu, double *dpsi2_dS, double *dpsi2_dgamma, double *dpsi2_dZ, double *psi2_neq, double *psi2exp1, double *psi2exp2, double var, double *l, double *Z, double *mu, double *S, double *gamma, int N, int M, int Q, int idx)
        {
            // dpsi2 (n,m1,m2,q)
            int q = idx/(M*M*N);
            int m2 = (idx%(M*M*N))/(M*N);
            int m1 = (idx%(M*N))/N;
            int n = idx%N;
            
            double neq = psi2_neq[IDX_NMMQ(n,m1,m2,q)];
            double gamma_c = gamma[IDX_NQ(n,q)];
            double Z1_c = Z[IDX_MQ(m1,q)];
            double Z2_c = Z[IDX_MQ(m2,q)];
            double S_c = S[IDX_NQ(n,q)];
            double l_c = l[q];
            double l_sqrt_c = sqrt(l[q]);
            double psi2exp1_c = psi2exp1[IDX_NMMQ(n,m1,m2,q)];
            double psi2exp2_c = psi2exp2[IDX_MMQ(m1,m2,q)];

            double dZ = Z2_c - Z1_c;
            double muZ = mu[IDX_NQ(n,q)] - (Z1_c+Z2_c)/2.0;
            double Z2 = Z1_c*Z1_c+Z2_c*Z2_c;
            double denom = 2.0*S_c/l_c+1.0;
            double denom_sqrt = sqrt(denom);
            double psi2_common = gamma_c/(denom_sqrt*denom*l_c);
            double gamma1 = 1-gamma_c;
            double var2 = var*var;
            
            dpsi2_dgamma[IDX_NMMQ(n,m1,m2,q)] = var2*neq*(psi2exp1_c/denom_sqrt - psi2exp2_c);
            dpsi2_dmu[IDX_NMMQ(n,m1,m2,q)] = var2*neq*(-2.0*psi2_common*muZ*psi2exp1_c);
            dpsi2_dS[IDX_NMMQ(n,m1,m2,q)] = var2*neq*(psi2_common*(2.0*muZ*muZ/(2.0*S_c+l_c)-1.0)*psi2exp1_c);
            dpsi2_dZ[IDX_NMMQ(n,m1,m2,q)] = var2*neq*(psi2_common*(dZ*denom/-2.0+muZ)*psi2exp1_c-gamma1*Z2_c/l_c*psi2exp2_c)*2.0;
            return var2*neq*(psi2_common*(S_c/l_c+dZ*dZ*denom/(4.0*l_c)+muZ*muZ/(2.0*S_c+l_c))*psi2exp1_c+gamma1*Z2/(2.0*l_c)*psi2exp2_c)*l_sqrt_c*2.0;    
        }
        """)
        
except:
    pass

class PSICOMP_SSRBF(object):
    def __init__(self):
        assert gpu_init.initSuccess, "GPU initialization failed!"
        self.cublas_handle = gpu_init.cublas_handle
        self.gpuCache = None
        self.gpuCacheAll = None
        
    def _initGPUCache(self, N, M, Q):
        if self.gpuCache!=None and self.gpuCache['mu_gpu'].shape[0] == N:
            return
        
        if self.gpuCacheAll!=None and self.gpuCacheAll['mu_gpu'].shape[0]<N: # Too small cache -> reallocate
            self._releaseMemory()
            
        if self.gpuCacheAll == None:
            self.gpuCacheAll = {
                             'l_gpu'                :gpuarray.empty((Q,),np.float64,order='F'),
                             'Z_gpu'                :gpuarray.empty((M,Q),np.float64,order='F'),
                             'mu_gpu'               :gpuarray.empty((N,Q),np.float64,order='F'),
                             'S_gpu'                :gpuarray.empty((N,Q),np.float64,order='F'),
                             'gamma_gpu'            :gpuarray.empty((N,Q),np.float64,order='F'),
                             'logGamma_gpu'         :gpuarray.empty((N,Q),np.float64,order='F'),
                             'log1Gamma_gpu'        :gpuarray.empty((N,Q),np.float64,order='F'),
                             'logpsi1denom_gpu'      :gpuarray.empty((N,Q),np.float64,order='F'),
                             'logpsi2denom_gpu'      :gpuarray.empty((N,Q),np.float64,order='F'),
                             'psi0_gpu'             :gpuarray.empty((N,),np.float64,order='F'),
                             'psi1_gpu'             :gpuarray.empty((N,M),np.float64,order='F'),
                             'psi2_gpu'             :gpuarray.empty((M,M),np.float64,order='F'),
                             # derivatives psi1
                             'psi1_neq_gpu'         :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'psi1exp1_gpu'         :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'psi1exp2_gpu'         :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'dpsi1_dvar_gpu'       :gpuarray.empty((N,M),np.float64, order='F'),
                             'dpsi1_dl_gpu'         :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'dpsi1_dZ_gpu'         :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'dpsi1_dgamma_gpu'     :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'dpsi1_dmu_gpu'        :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             'dpsi1_dS_gpu'         :gpuarray.empty((N,M,Q),np.float64, order='F'),
                             # derivatives psi2
                             'psi2_neq_gpu'         :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             'psi2exp1_gpu'         :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             'psi2exp2_gpu'         :gpuarray.empty((M,M,Q),np.float64, order='F'),
                             'dpsi2_dvar_gpu'       :gpuarray.empty((N,M,M),np.float64, order='F'),
                             'dpsi2_dl_gpu'         :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             'dpsi2_dZ_gpu'         :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             'dpsi2_dgamma_gpu'     :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             'dpsi2_dmu_gpu'        :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             'dpsi2_dS_gpu'         :gpuarray.empty((N,M,M,Q),np.float64, order='F'),
                             # gradients
                             'grad_l_gpu'           :gpuarray.empty((Q,),np.float64,order='F'),
                             'grad_Z_gpu'           :gpuarray.empty((M,Q),np.float64,order='F'),
                             'grad_mu_gpu'          :gpuarray.empty((N,Q),np.float64,order='F'),
                             'grad_S_gpu'           :gpuarray.empty((N,Q),np.float64,order='F'),
                             'grad_gamma_gpu'       :gpuarray.empty((N,Q),np.float64,order='F'),
                             }
            self.gpuCache = self.gpuCacheAll
        elif self.gpuCacheAll['mu_gpu'].shape[0]==N:
            self.gpuCache = self.gpuCacheAll
        else:
            # remap to a smaller cache
            self.gpuCache = self.gpuCacheAll.copy()
            Nlist=['mu_gpu','S_gpu','gamma_gpu','logGamma_gpu','log1Gamma_gpu','logpsi1denom_gpu','logpsi2denom_gpu','psi0_gpu','psi1_gpu','psi2_gpu',
                   'psi1_neq_gpu','psi1exp1_gpu','psi1exp2_gpu','dpsi1_dvar_gpu','dpsi1_dl_gpu','dpsi1_dZ_gpu','dpsi1_dgamma_gpu','dpsi1_dmu_gpu',
                   'dpsi1_dS_gpu','psi2_neq_gpu','psi2exp1_gpu','dpsi2_dvar_gpu','dpsi2_dl_gpu','dpsi2_dZ_gpu','dpsi2_dgamma_gpu','dpsi2_dmu_gpu','dpsi2_dS_gpu','grad_mu_gpu','grad_S_gpu','grad_gamma_gpu',]
            oldN = self.gpuCacheAll['mu_gpu'].shape[0]
            for v in Nlist:
                u = self.gpuCacheAll[v]
                self.gpuCache[v] = u.ravel()[:u.size/oldN*N].reshape(*((N,)+u.shape[1:]))
    
    def _releaseMemory(self):
        if self.gpuCacheAll!=None:
            [v.gpudata.free() for v in self.gpuCacheAll.values()]
            self.gpuCacheAll = None
            self.gpuCache = None
    
    def estimateMemoryOccupation(self, N, M, Q):
        """
        Estimate the best batch size.
        N - the number of total datapoints
        M - the number of inducing points
        Q - the number of hidden (input) dimensions
        return: the constant memory size, the memory occupation of batchsize=1
        unit: GB
        """
        return (2.*Q+2.*M*Q+M*M*Q)*8./1024./1024./1024., (1.+2.*M+10.*Q+2.*M*M+8.*M*Q+7.*M*M*Q)*8./1024./1024./1024.

    @Cache_this(limit=1,ignore_args=(0,))
    def psicomputations(self, variance, lengthscale, Z, mu, S, gamma):
        """Compute Psi statitsitcs"""
        if isinstance(lengthscale, np.ndarray) and len(lengthscale)>1:
            ARD = True
        else:
            ARD = False
        
        N = mu.shape[0]
        M = Z.shape[0]
        Q = mu.shape[1]
        
        self._initGPUCache(N,M,Q)
        l_gpu = self.gpuCache['l_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        gamma_gpu = self.gpuCache['gamma_gpu']
        logGamma_gpu = self.gpuCache['logGamma_gpu']
        log1Gamma_gpu = self.gpuCache['log1Gamma_gpu']
        logpsi1denom_gpu = self.gpuCache['logpsi1denom_gpu']
        logpsi2denom_gpu = self.gpuCache['logpsi2denom_gpu']
        psi0_gpu = self.gpuCache['psi0_gpu']
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']

        if ARD:
            l_gpu.set(np.asfortranarray(lengthscale**2))
        else:
            l_gpu.fill(lengthscale*lengthscale)
        Z_gpu.set(np.asfortranarray(Z))
        mu_gpu.set(np.asfortranarray(mu))
        S_gpu.set(np.asfortranarray(S))
        gamma_gpu.set(np.asfortranarray(gamma))
        linalg_gpu.log(gamma_gpu,logGamma_gpu)
        linalg_gpu.logOne(gamma_gpu,log1Gamma_gpu)
        comp_logpsidenom(logpsi1denom_gpu, S_gpu,l_gpu,1.0,N)
        comp_logpsidenom(logpsi2denom_gpu, S_gpu,l_gpu,2.0,N)
        
        psi0_gpu.fill(variance)
        comp_psi1(psi1_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi1denom_gpu, N, M, Q)
        comp_psi2(psi2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi2denom_gpu, N, M, Q)
        
        return psi0_gpu, psi1_gpu, psi2_gpu
    
    @Cache_this(limit=1,ignore_args=(0,))
    def _psiDercomputations(self, variance, lengthscale, Z, mu, S, gamma):
        """Compute the derivatives w.r.t. Psi statistics"""        
        N, M, Q = mu.shape[0],Z.shape[0], mu.shape[1]
        
        self._initGPUCache(N,M,Q)
        l_gpu = self.gpuCache['l_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        gamma_gpu = self.gpuCache['gamma_gpu']
        logGamma_gpu = self.gpuCache['logGamma_gpu']
        log1Gamma_gpu = self.gpuCache['log1Gamma_gpu']
        logpsi1denom_gpu = self.gpuCache['logpsi1denom_gpu']
        logpsi2denom_gpu = self.gpuCache['logpsi2denom_gpu']

        psi1_neq_gpu = self.gpuCache['psi1_neq_gpu']
        psi1exp1_gpu = self.gpuCache['psi1exp1_gpu']
        psi1exp2_gpu = self.gpuCache['psi1exp2_gpu']
        dpsi1_dvar_gpu = self.gpuCache['dpsi1_dvar_gpu']
        dpsi1_dl_gpu = self.gpuCache['dpsi1_dl_gpu']
        dpsi1_dZ_gpu = self.gpuCache['dpsi1_dZ_gpu']
        dpsi1_dgamma_gpu = self.gpuCache['dpsi1_dgamma_gpu']
        dpsi1_dmu_gpu = self.gpuCache['dpsi1_dmu_gpu']
        dpsi1_dS_gpu = self.gpuCache['dpsi1_dS_gpu']

        psi2_neq_gpu = self.gpuCache['psi2_neq_gpu']
        psi2exp1_gpu = self.gpuCache['psi2exp1_gpu']
        psi2exp2_gpu = self.gpuCache['psi2exp2_gpu']
        dpsi2_dvar_gpu = self.gpuCache['dpsi2_dvar_gpu']
        dpsi2_dl_gpu = self.gpuCache['dpsi2_dl_gpu']
        dpsi2_dZ_gpu = self.gpuCache['dpsi2_dZ_gpu']
        dpsi2_dgamma_gpu = self.gpuCache['dpsi2_dgamma_gpu']
        dpsi2_dmu_gpu = self.gpuCache['dpsi2_dmu_gpu']
        dpsi2_dS_gpu = self.gpuCache['dpsi2_dS_gpu']

        #==========================================================================================================
        # Assuming the l_gpu, Z_gpu, mu_gpu, S_gpu, gamma_gpu, logGamma_gpu, log1Gamma_gpu, 
        # logpsi1denom_gpu, logpsi2denom_gpu has been synchonized.
        #==========================================================================================================
        
        # psi1 derivatives
        comp_dpsi1_dvar(dpsi1_dvar_gpu, psi1_neq_gpu, psi1exp1_gpu,psi1exp2_gpu, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi1denom_gpu, N, M, Q)
        comp_psi1_der(dpsi1_dl_gpu,dpsi1_dmu_gpu,dpsi1_dS_gpu,dpsi1_dgamma_gpu, dpsi1_dZ_gpu, psi1_neq_gpu,psi1exp1_gpu,psi1exp2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, gamma_gpu, N, M, Q)

        # psi2 derivatives
        comp_dpsi2_dvar(dpsi2_dvar_gpu, psi2_neq_gpu, psi2exp1_gpu,psi2exp2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi2denom_gpu, N, M, Q)
        comp_psi2_der(dpsi2_dl_gpu,dpsi2_dmu_gpu,dpsi2_dS_gpu,dpsi2_dgamma_gpu, dpsi2_dZ_gpu, psi2_neq_gpu,psi2exp1_gpu,psi2exp2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, gamma_gpu, N, M, Q)

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        self._psiDercomputations(variance, lengthscale, Z, mu, S, gamma)
        N, M, Q = mu.shape[0],Z.shape[0], mu.shape[1]
        
        if isinstance(lengthscale, np.ndarray) and len(lengthscale)>1:
            ARD = True
        else:
            ARD = False
            
        dpsi1_dvar_gpu = self.gpuCache['dpsi1_dvar_gpu']
        dpsi2_dvar_gpu = self.gpuCache['dpsi2_dvar_gpu']
        dpsi1_dl_gpu = self.gpuCache['dpsi1_dl_gpu']
        dpsi2_dl_gpu = self.gpuCache['dpsi2_dl_gpu']
        psi1_comb_gpu = self.gpuCache['psi1_neq_gpu']
        psi2_comb_gpu = self.gpuCache['psi2_neq_gpu']
        grad_l_gpu = self.gpuCache['grad_l_gpu']
        
        # variance
        variance.gradient = gpuarray.sum(dL_dpsi0).get() \
                            + cublas.cublasDdot(self.cublas_handle, dL_dpsi1.size, dL_dpsi1.gpudata, 1, dpsi1_dvar_gpu.gpudata, 1) \
                            + cublas.cublasDdot(self.cublas_handle, dL_dpsi2.size, dL_dpsi2.gpudata, 1, dpsi2_dvar_gpu.gpudata, 1)

        # lengscale
        if ARD:
            grad_l_gpu.fill(0.)
            linalg_gpu.mul_bcast(psi1_comb_gpu, dL_dpsi1, dpsi1_dl_gpu, dL_dpsi1.size)
            linalg_gpu.sum_axis(grad_l_gpu, psi1_comb_gpu, 1, N*M)
            linalg_gpu.mul_bcast(psi2_comb_gpu, dL_dpsi2, dpsi2_dl_gpu, dL_dpsi2.size)
            linalg_gpu.sum_axis(grad_l_gpu, psi2_comb_gpu, 1, N*M*M)            
            lengthscale.gradient = grad_l_gpu.get()
        else:
            linalg_gpu.mul_bcast(psi1_comb_gpu, dL_dpsi1, dpsi1_dl_gpu, dL_dpsi1.size)
            linalg_gpu.mul_bcast(psi2_comb_gpu, dL_dpsi2, dpsi2_dl_gpu, dL_dpsi2.size)
            lengthscale.gradient = gpuarray.sum(psi1_comb_gpu).get() + gpuarray.sum(psi2_comb_gpu).get()
                
    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        self._psiDercomputations(variance, lengthscale, Z, mu, S, gamma)
        N, M, Q = mu.shape[0],Z.shape[0], mu.shape[1]

        dpsi1_dZ_gpu = self.gpuCache['dpsi1_dZ_gpu']
        dpsi2_dZ_gpu = self.gpuCache['dpsi2_dZ_gpu']
        psi1_comb_gpu = self.gpuCache['psi1_neq_gpu']
        psi2_comb_gpu = self.gpuCache['psi2_neq_gpu']
        grad_Z_gpu = self.gpuCache['grad_Z_gpu']

        grad_Z_gpu.fill(0.)
        linalg_gpu.mul_bcast(psi1_comb_gpu, dL_dpsi1, dpsi1_dZ_gpu, dL_dpsi1.size)
        linalg_gpu.sum_axis(grad_Z_gpu, psi1_comb_gpu, 1, N)
        linalg_gpu.mul_bcast(psi2_comb_gpu, dL_dpsi2, dpsi2_dZ_gpu, dL_dpsi2.size)
        linalg_gpu.sum_axis(grad_Z_gpu, psi2_comb_gpu, 1, N*M)
        return grad_Z_gpu.get()
        
    def gradients_qX_expectations(self, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        self._psiDercomputations(variance, lengthscale, Z, mu, S, gamma)
        N, M, Q = mu.shape[0],Z.shape[0], mu.shape[1]

        dpsi1_dmu_gpu = self.gpuCache['dpsi1_dmu_gpu']
        dpsi2_dmu_gpu = self.gpuCache['dpsi2_dmu_gpu']
        dpsi1_dS_gpu = self.gpuCache['dpsi1_dS_gpu']
        dpsi2_dS_gpu = self.gpuCache['dpsi2_dS_gpu']
        dpsi1_dgamma_gpu = self.gpuCache['dpsi1_dgamma_gpu']
        dpsi2_dgamma_gpu = self.gpuCache['dpsi2_dgamma_gpu']
        psi1_comb_gpu = self.gpuCache['psi1_neq_gpu']
        psi2_comb_gpu = self.gpuCache['psi2_neq_gpu']
        grad_mu_gpu = self.gpuCache['grad_mu_gpu']
        grad_S_gpu = self.gpuCache['grad_S_gpu']
        grad_gamma_gpu = self.gpuCache['grad_gamma_gpu']
        
        # mu gradients
        grad_mu_gpu.fill(0.)
        linalg_gpu.mul_bcast(psi1_comb_gpu, dL_dpsi1, dpsi1_dmu_gpu, dL_dpsi1.size)
        linalg_gpu.sum_axis(grad_mu_gpu, psi1_comb_gpu, N, M)
        linalg_gpu.mul_bcast(psi2_comb_gpu, dL_dpsi2, dpsi2_dmu_gpu, dL_dpsi2.size)
        linalg_gpu.sum_axis(grad_mu_gpu, psi2_comb_gpu, N, M*M)

        # S gradients
        grad_S_gpu.fill(0.)
        linalg_gpu.mul_bcast(psi1_comb_gpu, dL_dpsi1, dpsi1_dS_gpu, dL_dpsi1.size)
        linalg_gpu.sum_axis(grad_S_gpu, psi1_comb_gpu, N, M)
        linalg_gpu.mul_bcast(psi2_comb_gpu, dL_dpsi2, dpsi2_dS_gpu, dL_dpsi2.size)
        linalg_gpu.sum_axis(grad_S_gpu, psi2_comb_gpu, N, M*M)

        # gamma gradients
        grad_gamma_gpu.fill(0.)
        linalg_gpu.mul_bcast(psi1_comb_gpu, dL_dpsi1, dpsi1_dgamma_gpu, dL_dpsi1.size)
        linalg_gpu.sum_axis(grad_gamma_gpu, psi1_comb_gpu, N, M)
        linalg_gpu.mul_bcast(psi2_comb_gpu, dL_dpsi2, dpsi2_dgamma_gpu, dL_dpsi2.size)
        linalg_gpu.sum_axis(grad_gamma_gpu, psi2_comb_gpu, N, M*M)
        
        return grad_mu_gpu.get(), grad_S_gpu.get(), grad_gamma_gpu.get()
