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
            int m2 = idx/(M*N);
            int m1 = (idx%(M*N))/N;
            int n = idx%N;

            double psi2_exp=0;
            for(int q=0;q<Q;q++){ 
                double dZ = Z[IDX_MQ(m1,q)]-Z[IDX_MQ(m2,q)];
                double muZ = mu[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.0;
                double exp1 = logGamma[IDX_NQ(n,q)] - (logpsi2denom[IDX_NQ(n,q)])/2.0 - dZ*dZ/(l[q]*4.0) - muZ*muZ/(2*S[IDX_NQ(n,q)]+l[q]);
                double exp2 = log1Gamma[IDX_NQ(n,q)] - (Z[IDX_MQ(m1,q)]*Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)]*Z[IDX_MQ(m2,q)])/(l[q]*2.0);
                psi2_exp += LOGEXPSUM(exp1,exp2);
            }
            return var*var*exp(psi2_exp);
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

            double dZ = Z1_c - Z2_c;
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
        self.cublas_handle = cublas.cublasCreate()
        self.gpuCache = None
    
    def _initGPUCache(self, N, M, Q):
        if self.gpuCache == None:
            self.gpuCache = {
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
                             'psi2_gpu'             :gpuarray.empty((N,M,M),np.float64,order='F'),
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
        
        comp_psi1(psi1_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi1denom_gpu, N, M, Q)
        comp_psi2(psi2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi2denom_gpu, N, M, Q)
        
        return psi0_gpu.get(), psi1_gpu.get(), psi2_gpu.get()

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

    l_gpu = gpuarray.empty((Q,),np.float64, order='F')
    l_gpu.fill(lengthscale2)
    Z_gpu = gpuarray.to_gpu(np.asfortranarray(Z))
    mu_gpu = gpuarray.to_gpu(np.asfortranarray(mu))
    S_gpu = gpuarray.to_gpu(np.asfortranarray(S))
    gamma_gpu = gpuarray.to_gpu(np.asfortranarray(gamma))
    logGamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(gamma)))
    log1Gamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(1.-gamma)))
    logpsi1denom_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(S/lengthscale2+1.)))
    psi1_gpu = gpuarray.empty((mu.shape[0],Z.shape[0]),np.float64, order='F')
    psi1_neq_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    psi1exp1_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    psi1exp2_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    dpsi1_dvar_gpu = gpuarray.empty((N,M),np.float64, order='F')
    dpsi1_dl_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    dpsi1_dZ_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    dpsi1_dgamma_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    dpsi1_dmu_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    dpsi1_dS_gpu = gpuarray.empty((N,M,Q),np.float64, order='F')
    
    comp_dpsi1_dvar(dpsi1_dvar_gpu,psi1_neq_gpu,psi1exp1_gpu,psi1exp2_gpu, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi1denom_gpu, N, M, Q)
    comp_psi1_der(dpsi1_dl_gpu,dpsi1_dmu_gpu,dpsi1_dS_gpu,dpsi1_dgamma_gpu, dpsi1_dZ_gpu, psi1_neq_gpu,psi1exp1_gpu,psi1exp2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, gamma_gpu, N, M, Q)
    
#     print np.abs(dpsi1_dmu_gpu.get()-_dpsi1_dmu).max()

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

    N = mu.shape[0]
    M = Z.shape[0]
    Q = mu.shape[1]

    l_gpu = gpuarray.empty((Q,),np.float64, order='F')
    l_gpu.fill(lengthscale2)
    Z_gpu = gpuarray.to_gpu(np.asfortranarray(Z))
    mu_gpu = gpuarray.to_gpu(np.asfortranarray(mu))
    S_gpu = gpuarray.to_gpu(np.asfortranarray(S))
    gamma_gpu = gpuarray.to_gpu(np.asfortranarray(gamma))
    logGamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(gamma)))
    log1Gamma_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(1.-gamma)))
    logpsi2denom_gpu = gpuarray.to_gpu(np.asfortranarray(np.log(2.*S/lengthscale2+1.)))
    psi2_gpu = gpuarray.empty((mu.shape[0],Z.shape[0],Z.shape[0]),np.float64, order='F')
    psi2_neq_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    psi2exp1_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    psi2exp2_gpu = gpuarray.empty((M,M,Q),np.float64, order='F')
    dpsi2_dvar_gpu = gpuarray.empty((N,M,M),np.float64, order='F')
    dpsi2_dl_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    dpsi2_dZ_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    dpsi2_dgamma_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    dpsi2_dmu_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    dpsi2_dS_gpu = gpuarray.empty((N,M,M,Q),np.float64, order='F')
    
    #comp_psi2(psi2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi2denom_gpu, N, M, Q)

    comp_dpsi2_dvar(dpsi2_dvar_gpu,psi2_neq_gpu,psi2exp1_gpu,psi2exp2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, logGamma_gpu, log1Gamma_gpu, logpsi2denom_gpu, N, M, Q)
    comp_psi2_der(dpsi2_dl_gpu,dpsi2_dmu_gpu,dpsi2_dS_gpu,dpsi2_dgamma_gpu, dpsi2_dZ_gpu, psi2_neq_gpu,psi2exp1_gpu,psi2exp2_gpu, variance, l_gpu, Z_gpu, mu_gpu, S_gpu, gamma_gpu, N, M, Q)
    
#     print np.abs(dpsi2_dvar_gpu.get()-_dpsi2_dvariance).max()

    return _psi2, _dpsi2_dvariance, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _dpsi2_dZ, _dpsi2_dlengthscale
