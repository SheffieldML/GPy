"""
The module for psi-statistics for RBF kernel
"""

import numpy as np
from GPy.util.caching import Cacher
from . import PSICOMP_RBF
from ....util import gpu_init

try:
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
except:
    pass    

gpu_code = """
    // define THREADNUM

    #define IDX_NMQ(n,m,q) ((q*M+m)*N+n)
    #define IDX_NMM(n,m1,m2) ((m2*M+m1)*N+n)
    #define IDX_NQ(n,q) (q*N+n)
    #define IDX_NM(n,m) (m*N+n)
    #define IDX_MQ(m,q) (q*M+m)
    #define IDX_MM(m1,m2) (m2*M+m1)
    #define IDX_NQB(n,q,b) ((b*Q+q)*N+n)
    #define IDX_QB(q,b) (b*Q+q)

    // Divide data evenly
    __device__ void divide_data(int total_data, int psize, int pidx, int *start, int *end) {
        int residue = (total_data)%psize;
        if(pidx<residue) {
            int size = total_data/psize+1;
            *start = size*pidx;
            *end = *start+size;
        } else {
            int size = total_data/psize;
            *start = size*pidx+residue;
            *end = *start+size;
        }
    }
    
    __device__ void reduce_sum(double* array, int array_size) {
        int s;
        if(array_size >= blockDim.x) {
            for(int i=blockDim.x+threadIdx.x; i<array_size; i+= blockDim.x) {
                array[threadIdx.x] += array[i];
            }
            array_size = blockDim.x;
        }
        __syncthreads();
        for(int i=1; i<=array_size;i*=2) {s=i;}
        if(threadIdx.x < array_size-s) {array[threadIdx.x] += array[s+threadIdx.x];}
        __syncthreads();
        for(s=s/2;s>=1;s=s/2) {
            if(threadIdx.x < s) {array[threadIdx.x] += array[s+threadIdx.x];}
            __syncthreads();
        }
    }

    __global__ void psi1computations(double *psi1, double var, double *l, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int m_start, m_end;
        divide_data(M, gridDim.x, blockIdx.x, &m_start, &m_end);
        
        for(int m=m_start; m<m_end; m++) {
            for(int n=threadIdx.x; n<N; n+= blockDim.x) {            
                double log_psi1 = 0;
                for(int q=0;q<Q;q++) {
                    double muZ = mu[IDX_NQ(n,q)]-Z[IDX_MQ(m,q)];
                    double Snq = S[IDX_NQ(n,q)];
                    double lq = l[q]*l[q];
                    log_psi1 += (muZ*muZ/(Snq+lq))/(-2.);
                    log_psi1 += log(Snq/lq+1)/(-2.);
                }
                psi1[IDX_NM(n,m)] = var*exp(log_psi1);
            }
        }
    }
    
    __global__ void psi2computations(double *psi2, double *psi2n, double var, double *l, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int psi2_idx_start, psi2_idx_end;
        __shared__ double psi2_local[THREADNUM];
        divide_data((M+1)*M/2, gridDim.x, blockIdx.x, &psi2_idx_start, &psi2_idx_end);
        
        for(int psi2_idx=psi2_idx_start; psi2_idx<psi2_idx_end; psi2_idx++) {
            int m1 = int((sqrt(8.*psi2_idx+1.)-1.)/2.);
            int m2 = psi2_idx - (m1+1)*m1/2;
            
            psi2_local[threadIdx.x] = 0;
            for(int n=threadIdx.x;n<N;n+=blockDim.x) {
                double log_psi2_n = 0;
                for(int q=0;q<Q;q++) {
                    double dZ = Z[IDX_MQ(m1,q)] - Z[IDX_MQ(m2,q)];
                    double muZhat = mu[IDX_NQ(n,q)]- (Z[IDX_MQ(m1,q)]+Z[IDX_MQ(m2,q)])/2.;
                    double Snq = S[IDX_NQ(n,q)];
                    double lq = l[q]*l[q];
                    log_psi2_n += dZ*dZ/(-4.*lq)-muZhat*muZhat/(2.*Snq+lq);
                    log_psi2_n += log(2.*Snq/lq+1)/(-2.);
                }
                double exp_psi2_n = exp(log_psi2_n);
                psi2n[IDX_NMM(n,m1,m2)] = var*var*exp_psi2_n;
                if(m1!=m2) { psi2n[IDX_NMM(n,m2,m1)] = var*var*exp_psi2_n;}
                psi2_local[threadIdx.x] += exp_psi2_n;
            }
            __syncthreads();
            reduce_sum(psi2_local, THREADNUM);
            if(threadIdx.x==0) {
                psi2[IDX_MM(m1,m2)] = var*var*psi2_local[0];
                if(m1!=m2) { psi2[IDX_MM(m2,m1)] = var*var*psi2_local[0]; }
            }
            __syncthreads();
        }
    }
    
    __global__ void psi1compDer(double *dvar, double *dl, double *dZ, double *dmu, double *dS, double *dL_dpsi1, double *psi1, double var, double *l, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int m_start, m_end;
        __shared__ double g_local[THREADNUM];
        divide_data(M, gridDim.x, blockIdx.x, &m_start, &m_end);
        int P = int(ceil(double(N)/THREADNUM));

        double dvar_local = 0;
        for(int q=0;q<Q;q++) {
            double lq_sqrt = l[q];
            double lq = lq_sqrt*lq_sqrt;
            double dl_local = 0;
            for(int p=0;p<P;p++) {
                int n = p*THREADNUM + threadIdx.x;
                double dmu_local = 0;
                double dS_local = 0;
                double Snq,mu_nq;
                if(n<N) {Snq = S[IDX_NQ(n,q)]; mu_nq=mu[IDX_NQ(n,q)];}
                for(int m=m_start; m<m_end; m++) {
                    if(n<N) {
                        double lpsi1 = psi1[IDX_NM(n,m)]*dL_dpsi1[IDX_NM(n,m)];
                        if(q==0) {dvar_local += lpsi1;}
                        
                        double Zmu = Z[IDX_MQ(m,q)] - mu_nq;
                        double denom = Snq+lq;
                        double Zmu2_denom = Zmu*Zmu/denom;
                        
                        dmu_local += lpsi1*Zmu/denom;
                        dS_local += lpsi1*(Zmu2_denom-1.)/denom;
                        dl_local += lpsi1*(Zmu2_denom+Snq/lq)/denom;
                        g_local[threadIdx.x] = -lpsi1*Zmu/denom;
                    }
                    __syncthreads();
                    reduce_sum(g_local, p<P-1?THREADNUM:N-(P-1)*THREADNUM);
                    if(threadIdx.x==0) {dZ[IDX_MQ(m,q)] += g_local[0];}
                }
                if(n<N) {
                    dmu[IDX_NQB(n,q,blockIdx.x)] += dmu_local;
                    dS[IDX_NQB(n,q,blockIdx.x)] += dS_local/2.;
                }
                __threadfence_block();
            }
            g_local[threadIdx.x] = dl_local*lq_sqrt;
            __syncthreads();
            reduce_sum(g_local, THREADNUM);
            if(threadIdx.x==0) {dl[IDX_QB(q,blockIdx.x)] += g_local[0];}
        }
        g_local[threadIdx.x] = dvar_local;
        __syncthreads();
        reduce_sum(g_local, THREADNUM);
        if(threadIdx.x==0) {dvar[blockIdx.x] += g_local[0]/var;}        
    }
    
    __global__ void psi2compDer(double *dvar, double *dl, double *dZ, double *dmu, double *dS, double *dL_dpsi2, double *psi2n, double var, double *l, double *Z, double *mu, double *S, int N, int M, int Q)
    {
        int m_start, m_end;
        __shared__ double g_local[THREADNUM];
        divide_data(M, gridDim.x, blockIdx.x, &m_start, &m_end);
        int P = int(ceil(double(N)/THREADNUM));

        double dvar_local = 0;
        for(int q=0;q<Q;q++) {
            double lq_sqrt = l[q];
            double lq = lq_sqrt*lq_sqrt;
            double dl_local = 0;
            for(int p=0;p<P;p++) {
                int n = p*THREADNUM + threadIdx.x;
                double dmu_local = 0;
                double dS_local = 0;
                double Snq,mu_nq;
                if(n<N) {Snq = S[IDX_NQ(n,q)]; mu_nq=mu[IDX_NQ(n,q)];}
                for(int m1=m_start; m1<m_end; m1++) {
                    g_local[threadIdx.x] = 0;
                    for(int m2=0;m2<M;m2++) {
                        if(n<N) {
                            double lpsi2 = psi2n[IDX_NMM(n,m1,m2)]*dL_dpsi2[IDX_MM(m1,m2)];
                            if(q==0) {dvar_local += lpsi2;}
                            
                            double dZ = Z[IDX_MQ(m1,q)] - Z[IDX_MQ(m2,q)];
                            double muZhat =  mu_nq - (Z[IDX_MQ(m1,q)] + Z[IDX_MQ(m2,q)])/2.;
                            double denom = 2.*Snq+lq;
                            double muZhat2_denom = muZhat*muZhat/denom;
                            
                            dmu_local += lpsi2*muZhat/denom;
                            dS_local += lpsi2*(2.*muZhat2_denom-1.)/denom;
                            dl_local += lpsi2*((Snq/lq+muZhat2_denom)/denom+dZ*dZ/(4.*lq*lq));
                            g_local[threadIdx.x] += 2.*lpsi2*(muZhat/denom-dZ/(2*lq));
                        }
                    }
                    __syncthreads();
                    reduce_sum(g_local, p<P-1?THREADNUM:N-(P-1)*THREADNUM);
                    if(threadIdx.x==0) {dZ[IDX_MQ(m1,q)] += g_local[0];}
                }
                if(n<N) {
                    dmu[IDX_NQB(n,q,blockIdx.x)] += -2.*dmu_local;
                    dS[IDX_NQB(n,q,blockIdx.x)] += dS_local;
                }
                __threadfence_block();
            }
            g_local[threadIdx.x] = dl_local*2.*lq_sqrt;
            __syncthreads();
            reduce_sum(g_local, THREADNUM);
            if(threadIdx.x==0) {dl[IDX_QB(q,blockIdx.x)] += g_local[0];}
        }
        g_local[threadIdx.x] = dvar_local;
        __syncthreads();
        reduce_sum(g_local, THREADNUM);
        if(threadIdx.x==0) {dvar[blockIdx.x] += g_local[0]*2/var;}
    }
    """

class PSICOMP_RBF_GPU(PSICOMP_RBF):

    def __init__(self, GPU_direct=False):
        assert gpu_init.initSuccess, "GPU initialization failed!"
        self.GPU_direct = GPU_direct
        self.cublas_handle = gpu_init.cublas_handle
        self.gpuCache = None
        
        self.threadnum = 128
        self.blocknum = 15
        module = SourceModule("#define THREADNUM "+str(self.threadnum)+"\n"+gpu_code)
        self.g_psi1computations = module.get_function('psi1computations')
        self.g_psi2computations = module.get_function('psi2computations')
        self.g_psi1compDer = module.get_function('psi1compDer')
        self.g_psi2compDer = module.get_function('psi2compDer')
    
    def _initGPUCache(self, N, M, Q):            
        if self.gpuCache == None:
            self.gpuCache = {
                             'l_gpu'                :gpuarray.empty((Q,),np.float64,order='F'),
                             'Z_gpu'                :gpuarray.empty((M,Q),np.float64,order='F'),
                             'mu_gpu'               :gpuarray.empty((N,Q),np.float64,order='F'),
                             'S_gpu'                :gpuarray.empty((N,Q),np.float64,order='F'),
                             'psi0_gpu'             :gpuarray.empty((N,),np.float64,order='F'),
                             'psi1_gpu'             :gpuarray.empty((N,M),np.float64,order='F'),
                             'psi2_gpu'             :gpuarray.empty((M,M),np.float64,order='F'),
                             'psi2n_gpu'            :gpuarray.empty((N,M,M),np.float64,order='F'),
                             # derivatives
                             'dvar_gpu'             :gpuarray.empty((self.blocknum,),np.float64, order='F'),
                             'dl_gpu'               :gpuarray.empty((Q,self.blocknum),np.float64, order='F'),
                             'dZ_gpu'               :gpuarray.empty((M,Q),np.float64, order='F'),
                             'dmu_gpu'              :gpuarray.empty((N,Q,self.blocknum),np.float64, order='F'),
                             'dS_gpu'               :gpuarray.empty((N,Q,self.blocknum),np.float64, order='F'),
                             # gradients
                             'grad_l_gpu'           :gpuarray.empty((Q,),np.float64,order='F'),
                             'grad_Z_gpu'           :gpuarray.empty((M,Q),np.float64,order='F'),
                             }
    
    def sync_params(self, lengthscale, Z, mu, S):
        self.gpuCache['l_gpu'].set(np.asfortranarray(lengthscale))
        self.gpuCache['Z_gpu'].set(np.asfortranarray(Z))
        self.gpuCache['mu_gpu'].set(np.asfortranarray(mu))
        self.gpuCache['S_gpu'].set(np.asfortranarray(S))
        
    def reset_derivative(self):
        self.gpuCache['dvar_gpu'].fill(0.)
        self.gpuCache['dl_gpu'].fill(0.)
        self.gpuCache['dZ_gpu'].fill(0.)
        self.gpuCache['dmu_gpu'].fill(0.)
        self.gpuCache['dS_gpu'].fill(0.)

#     @Cache_this(limit=1, ignore_args=(0,))
    def psicomputations(self, variance, lengthscale, Z, variational_posterior):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi0, psi1 and psi2
        # Produced intermediate results:
        # _psi1                NxM
        mu = variational_posterior.mean
        S = variational_posterior.variance
        N = mu.shape[0]
        M = Z.shape[0]
        Q = Z.shape[1]
        self._initGPUCache(N,M,Q)
        self.sync_params(lengthscale, Z, variational_posterior.mean, variational_posterior.variance)
        
        psi0_gpu = self.gpuCache['psi0_gpu']
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']
        psi2n_gpu = self.gpuCache['psi2n_gpu']
        l_gpu = self.gpuCache['l_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        
        psi0_gpu.fill(variance)
        self.g_psi1computations(psi1_gpu, np.float64(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1))
        self.g_psi2computations(psi2_gpu, psi2n_gpu, np.float64(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1))
         
        if self.GPU_direct:
            return psi0_gpu, psi1_gpu, psi2_gpu
        else:
            return psi0_gpu.get(), psi1_gpu.get(), psi2_gpu.get()

        psi0 = np.empty(mu.shape[0])
        psi0[:] = variance
        
        psi1 = _psi1computations(variance, lengthscale, Z, mu, S)
        self.g_psi1computations(psi1_gpu, np.float64(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1))
        psi1g = psi1_gpu.get()
        print np.abs(psi1-psi1g).max()

        psi2 = _psi2computations(variance, lengthscale, Z, mu, S).sum(axis=0)
        self.g_psi2computations(psi2_gpu, psi2n_gpu, np.float64(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1))
        psi2g = psi2_gpu.get()
        print np.abs(psi2-psi2g).max()

        return psi0, psi1, psi2

#     @Cache_this(limit=1, ignore_args=(0,1,2,3))
    def psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        ARD = (len(lengthscale)!=1)
        
        dvar_psi1, dl_psi1, dZ_psi1, dmu_psi1, dS_psi1 = _psi1compDer(dL_dpsi1, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance)
        dvar_psi2, dl_psi2, dZ_psi2, dmu_psi2, dS_psi2 = _psi2compDer(dL_dpsi2, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance)

        mu = variational_posterior.mean
        S = variational_posterior.variance
        N = mu.shape[0]
        M = Z.shape[0]
        Q = Z.shape[1]
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2n_gpu = self.gpuCache['psi2n_gpu']
        l_gpu = self.gpuCache['l_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        dvar_gpu = self.gpuCache['dvar_gpu']
        dl_gpu = self.gpuCache['dl_gpu']
        dZ_gpu = self.gpuCache['dZ_gpu']
        dmu_gpu = self.gpuCache['dmu_gpu']
        dS_gpu = self.gpuCache['dS_gpu']
        
        if self.GPU_direct:
            dL_dpsi1_gpu = dL_dpsi1
            dL_dpsi2_gpu = dL_dpsi2
        else:
            dL_dpsi1_gpu = gpuarray.to_gpu(np.asfortranarray(dL_dpsi1))
            dL_dpsi2_gpu = gpuarray.to_gpu(np.asfortranarray(dL_dpsi2))

        self.reset_derivative()
        self.g_psi1compDer(dvar_gpu,dl_gpu,dZ_gpu,dmu_gpu,dS_gpu,dL_dpsi1_gpu,psi1_gpu, np.float64(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1))
#         print np.abs(dvar_psi1-dvar_gpu.get().sum(axis=-1)).max()
#         print np.abs(dl_psi1-dl_gpu.get().sum(axis=-1)).max()
#         print np.abs(dmu_psi1-dmu_gpu.get().sum(axis=-1)).max()
#         print np.abs(dS_psi1-dS_gpu.get().sum(axis=-1)).max()
#         print np.abs(dZ_psi1-dZ_gpu.get()).max()

#         self.reset_derivative()
        self.g_psi2compDer(dvar_gpu,dl_gpu,dZ_gpu,dmu_gpu,dS_gpu,dL_dpsi2_gpu,psi2n_gpu, np.float64(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1))
#         print np.abs(dvar_psi2-dvar_gpu.get().sum(axis=-1)).max()
#         print np.abs(dl_psi2-dl_gpu.get().sum(axis=-1)).max()
#         print np.abs(dmu_psi2-dmu_gpu.get().sum(axis=-1)).max()
#         print np.abs(dS_psi2-dS_gpu.get().sum(axis=-1)).max()
#         print np.abs(dZ_psi2-dZ_gpu.get()).max()

        dL_dvar = np.sum(dL_dpsi0) + dvar_gpu.get().sum()
        dL_dmu = dmu_gpu.get().sum(axis=-1)
        dL_dS = dS_gpu.get().sum(axis=-1)
        dL_dZ = dZ_gpu.get()
        if ARD:
            dL_dlengscale = dl_gpu.get().sum(axis=-1)
        else:
            dL_dlengscale = dl_gpu.get().sum()
            
#         print np.abs(dL_dlengscale - dl_psi1-dl_psi2).max()
        
#     
#         dL_dvar = np.sum(dL_dpsi0) + dvar_psi1 + dvar_psi2
#         
#         dL_dlengscale = dl_psi1 + dl_psi2
#         if not ARD:
#             dL_dlengscale = dL_dlengscale.sum()
#     
#         dL_dmu = dmu_psi1 + dmu_psi2
#         dL_dS = dS_psi1 + dS_psi2
#         dL_dZ = dZ_psi1 + dZ_psi2
        
        return dL_dvar, dL_dlengscale, dL_dZ, dL_dmu, dL_dS
    

def psicomputations(variance, lengthscale, Z, variational_posterior):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi0, psi1 and psi2
    # Produced intermediate results:
    # _psi1                NxM
    mu = variational_posterior.mean
    S = variational_posterior.variance
    
    psi0 = np.empty(mu.shape[0])
    psi0[:] = variance
    psi1 = _psi1computations(variance, lengthscale, Z, mu, S)
    psi2 = _psi2computations(variance, lengthscale, Z, mu, S).sum(axis=0)
    return psi0, psi1, psi2

def __psi1computations(variance, lengthscale, Z, mu, S):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1
    # Produced intermediate results:
    # _psi1                NxM

    lengthscale2 = np.square(lengthscale)

    # psi1
    _psi1_logdenom = np.log(S/lengthscale2+1.).sum(axis=-1) # N
    _psi1_log = (_psi1_logdenom[:,None]+np.einsum('nmq,nq->nm',np.square(mu[:,None,:]-Z[None,:,:]),1./(S+lengthscale2)))/(-2.)
    _psi1 = variance*np.exp(_psi1_log)
        
    return _psi1

def __psi2computations(variance, lengthscale, Z, mu, S):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi2
    # Produced intermediate results:
    # _psi2                MxM
    
    lengthscale2 = np.square(lengthscale)
    
    _psi2_logdenom = np.log(2.*S/lengthscale2+1.).sum(axis=-1)/(-2.) # N
    _psi2_exp1 = (np.square(Z[:,None,:]-Z[None,:,:])/lengthscale2).sum(axis=-1)/(-4.) #MxM
    Z_hat = (Z[:,None,:]+Z[None,:,:])/2. #MxMxQ
    denom = 1./(2.*S+lengthscale2)
    _psi2_exp2 = -(np.square(mu)*denom).sum(axis=-1)[:,None,None]+2.*np.einsum('nq,moq,nq->nmo',mu,Z_hat,denom)-np.einsum('moq,nq->nmo',np.square(Z_hat),denom)
    _psi2 = variance*variance*np.exp(_psi2_logdenom[:,None,None]+_psi2_exp1[None,:,:]+_psi2_exp2)
    

    return _psi2

def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
    ARD = (len(lengthscale)!=1)
    
    dvar_psi1, dl_psi1, dZ_psi1, dmu_psi1, dS_psi1 = _psi1compDer(dL_dpsi1, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance)
    dvar_psi2, dl_psi2, dZ_psi2, dmu_psi2, dS_psi2 = _psi2compDer(dL_dpsi2, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance)

    dL_dvar = np.sum(dL_dpsi0) + dvar_psi1 + dvar_psi2
    
    dL_dlengscale = dl_psi1 + dl_psi2
    if not ARD:
        dL_dlengscale = dL_dlengscale.sum()

    dL_dmu = dmu_psi1 + dmu_psi2
    dL_dS = dS_psi1 + dS_psi2
    dL_dZ = dZ_psi1 + dZ_psi2
    
    return dL_dvar, dL_dlengscale, dL_dZ, dL_dmu, dL_dS

def _psi1compDer(dL_dpsi1, variance, lengthscale, Z, mu, S):
    """
    dL_dpsi1 - NxM
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    """
    # here are the "statistics" for psi1
    # Produced intermediate results: dL_dparams w.r.t. psi1
    # _dL_dvariance     1
    # _dL_dlengthscale  Q
    # _dL_dZ            MxQ
    # _dL_dgamma        NxQ
    # _dL_dmu           NxQ
    # _dL_dS            NxQ
    
    lengthscale2 = np.square(lengthscale)
    
    _psi1 = _psi1computations(variance, lengthscale, Z, mu, S)
    Lpsi1 = dL_dpsi1*_psi1
    Zmu = Z[None,:,:]-mu[:,None,:] # NxMxQ
    denom = 1./(S+lengthscale2)
    Zmu2_denom = np.square(Zmu)*denom[:,None,:] #NxMxQ
    _dL_dvar = Lpsi1.sum()/variance
    _dL_dmu = np.einsum('nm,nmq,nq->nq',Lpsi1,Zmu,denom)
    _dL_dS = np.einsum('nm,nmq,nq->nq',Lpsi1,(Zmu2_denom-1.),denom)/2.
    _dL_dZ = -np.einsum('nm,nmq,nq->mq',Lpsi1,Zmu,denom)
    _dL_dl = np.einsum('nm,nmq,nq->q',Lpsi1,(Zmu2_denom+(S/lengthscale2)[:,None,:]),denom*lengthscale)
    
    return _dL_dvar, _dL_dl, _dL_dZ, _dL_dmu, _dL_dS

def _psi2compDer(dL_dpsi2, variance, lengthscale, Z, mu, S):
    """
    Z - MxQ
    mu - NxQ
    S - NxQ
    gamma - NxQ
    dL_dpsi2 - MxM
    """
    # here are the "statistics" for psi2
    # Produced the derivatives w.r.t. psi2:
    # _dL_dvariance      1
    # _dL_dlengthscale   Q
    # _dL_dZ             MxQ
    # _dL_dgamma         NxQ
    # _dL_dmu            NxQ
    # _dL_dS             NxQ
    
    lengthscale2 = np.square(lengthscale)
    denom = 1./(2*S+lengthscale2)
    denom2 = np.square(denom)

    _psi2 = _psi2computations(variance, lengthscale, Z, mu, S) # NxMxM
    
    Lpsi2 = dL_dpsi2[None,:,:]*_psi2
    Lpsi2sum = np.einsum('nmo->n',Lpsi2) #N
    Lpsi2Z = np.einsum('nmo,oq->nq',Lpsi2,Z) #NxQ
    Lpsi2Z2 = np.einsum('nmo,oq,oq->nq',Lpsi2,Z,Z) #NxQ
    Lpsi2Z2p = np.einsum('nmo,mq,oq->nq',Lpsi2,Z,Z) #NxQ
    Lpsi2Zhat = Lpsi2Z
    Lpsi2Zhat2 = (Lpsi2Z2+Lpsi2Z2p)/2
    
    _dL_dvar = Lpsi2sum.sum()*2/variance
    _dL_dmu = (-2*denom) * (mu*Lpsi2sum[:,None]-Lpsi2Zhat)
    _dL_dS = (2*np.square(denom))*(np.square(mu)*Lpsi2sum[:,None]-2*mu*Lpsi2Zhat+Lpsi2Zhat2) - denom*Lpsi2sum[:,None]
    _dL_dZ = -np.einsum('nmo,oq->oq',Lpsi2,Z)/lengthscale2+np.einsum('nmo,oq->mq',Lpsi2,Z)/lengthscale2+ \
             2*np.einsum('nmo,nq,nq->mq',Lpsi2,mu,denom) - np.einsum('nmo,nq,mq->mq',Lpsi2,denom,Z) - np.einsum('nmo,oq,nq->mq',Lpsi2,Z,denom)
    _dL_dl = 2*lengthscale* ((S/lengthscale2*denom+np.square(mu*denom))*Lpsi2sum[:,None]+(Lpsi2Z2-Lpsi2Z2p)/(2*np.square(lengthscale2))-
                             (2*mu*denom2)*Lpsi2Zhat+denom2*Lpsi2Zhat2).sum(axis=0)

    return _dL_dvar, _dL_dl, _dL_dZ, _dL_dmu, _dL_dS
    
_psi1computations = Cacher(__psi1computations, limit=1)
_psi2computations = Cacher(__psi2computations, limit=1)
