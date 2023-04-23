"""
The module for psi-statistics for RBF kernel
"""

import numpy as np
from paramz.caching import Cache_this
from . import PSICOMP_RBF

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

    __global__ void compDenom(double *log_denom1, double *log_denom2, double *l, double *S, int N, int Q)
    {
        int n_start, n_end;
        divide_data(N, gridDim.x, blockIdx.x, &n_start, &n_end);
        
        for(int i=n_start*Q+threadIdx.x; i<n_end*Q; i+=blockDim.x) {
            int n=i/Q;
            int q=i%Q;

            double Snq = S[IDX_NQ(n,q)];
            double lq = l[q]*l[q];
            log_denom1[IDX_NQ(n,q)] = log(Snq/lq+1.);
            log_denom2[IDX_NQ(n,q)] = log(2.*Snq/lq+1.);
        }
    }

    __global__ void psi1computations(double *psi1, double *log_denom1, double var, double *l, double *Z, double *mu, double *S, int N, int M, int Q)
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
                    log_psi1 += (muZ*muZ/(Snq+lq)+log_denom1[IDX_NQ(n,q)])/(-2.);
                }
                psi1[IDX_NM(n,m)] = var*exp(log_psi1);
            }
        }
    }
    
    __global__ void psi2computations(double *psi2, double *psi2n, double *log_denom2, double var, double *l, double *Z, double *mu, double *S, int N, int M, int Q)
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
                    log_psi2_n += dZ*dZ/(-4.*lq)-muZhat*muZhat/(2.*Snq+lq) + log_denom2[IDX_NQ(n,q)]/(-2.);
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

    def __init__(self, threadnum=256, blocknum=30, GPU_direct=False):
        self.fall_back = PSICOMP_RBF()
        
        from pycuda.compiler import SourceModule
        import GPy.util.gpu_init
        
        self.GPU_direct = GPU_direct
        self.gpuCache = None
        
        self.threadnum = threadnum
        self.blocknum = blocknum
        module = SourceModule("#define THREADNUM "+str(self.threadnum)+"\n"+gpu_code)
        self.g_psi1computations = module.get_function('psi1computations')
        self.g_psi1computations.prepare('PPdPPPPiii')
        self.g_psi2computations = module.get_function('psi2computations')
        self.g_psi2computations.prepare('PPPdPPPPiii')
        self.g_psi1compDer = module.get_function('psi1compDer')
        self.g_psi1compDer.prepare('PPPPPPPdPPPPiii')
        self.g_psi2compDer = module.get_function('psi2compDer')
        self.g_psi2compDer.prepare('PPPPPPPdPPPPiii')
        self.g_compDenom = module.get_function('compDenom')
        self.g_compDenom.prepare('PPPPii')
        
    def __deepcopy__(self, memo):
        s = PSICOMP_RBF_GPU(threadnum=self.threadnum, blocknum=self.blocknum, GPU_direct=self.GPU_direct)
        memo[id(self)] = s 
        return s
    
    def _initGPUCache(self, N, M, Q):
        import pycuda.gpuarray as gpuarray
        if self.gpuCache == None:
            self.gpuCache = {
                             'l_gpu'                :gpuarray.empty((Q,),float,order='F'),
                             'Z_gpu'                :gpuarray.empty((M,Q),float,order='F'),
                             'mu_gpu'               :gpuarray.empty((N,Q),float,order='F'),
                             'S_gpu'                :gpuarray.empty((N,Q),float,order='F'),
                             'psi1_gpu'             :gpuarray.empty((N,M),float,order='F'),
                             'psi2_gpu'             :gpuarray.empty((M,M),float,order='F'),
                             'psi2n_gpu'            :gpuarray.empty((N,M,M),float,order='F'),
                             'dL_dpsi1_gpu'         :gpuarray.empty((N,M),float,order='F'),
                             'dL_dpsi2_gpu'         :gpuarray.empty((M,M),float,order='F'),
                             'log_denom1_gpu'       :gpuarray.empty((N,Q),float,order='F'),
                             'log_denom2_gpu'       :gpuarray.empty((N,Q),float,order='F'),
                             # derivatives
                             'dvar_gpu'             :gpuarray.empty((self.blocknum,),float, order='F'),
                             'dl_gpu'               :gpuarray.empty((Q,self.blocknum),float, order='F'),
                             'dZ_gpu'               :gpuarray.empty((M,Q),float, order='F'),
                             'dmu_gpu'              :gpuarray.empty((N,Q,self.blocknum),float, order='F'),
                             'dS_gpu'               :gpuarray.empty((N,Q,self.blocknum),float, order='F'),
                             # grad
                             'grad_l_gpu'               :gpuarray.empty((Q,),float, order='F'),
                             'grad_mu_gpu'              :gpuarray.empty((N,Q,),float, order='F'),
                             'grad_S_gpu'               :gpuarray.empty((N,Q,),float, order='F'),
                             }
        else:
            assert N==self.gpuCache['mu_gpu'].shape[0]
            assert M==self.gpuCache['Z_gpu'].shape[0]
            assert Q==self.gpuCache['l_gpu'].shape[0]
    
    def sync_params(self, lengthscale, Z, mu, S):
        if len(lengthscale)==1:
            self.gpuCache['l_gpu'].fill(lengthscale)
        else:
            self.gpuCache['l_gpu'].set(np.asfortranarray(lengthscale))
        self.gpuCache['Z_gpu'].set(np.asfortranarray(Z))
        self.gpuCache['mu_gpu'].set(np.asfortranarray(mu))
        self.gpuCache['S_gpu'].set(np.asfortranarray(S))
        N,Q = self.gpuCache['S_gpu'].shape
        # t=self.g_compDenom(self.gpuCache['log_denom1_gpu'],self.gpuCache['log_denom2_gpu'],self.gpuCache['l_gpu'],self.gpuCache['S_gpu'], np.int32(N), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1),time_kernel=True)
        # print 'g_compDenom '+str(t)
        self.g_compDenom.prepared_call((self.blocknum,1),(self.threadnum,1,1), self.gpuCache['log_denom1_gpu'].gpudata,self.gpuCache['log_denom2_gpu'].gpudata,self.gpuCache['l_gpu'].gpudata,self.gpuCache['S_gpu'].gpudata, np.int32(N), np.int32(Q))
        
    def reset_derivative(self):
        self.gpuCache['dvar_gpu'].fill(0.)
        self.gpuCache['dl_gpu'].fill(0.)
        self.gpuCache['dZ_gpu'].fill(0.)
        self.gpuCache['dmu_gpu'].fill(0.)
        self.gpuCache['dS_gpu'].fill(0.)
        self.gpuCache['grad_l_gpu'].fill(0.)
        self.gpuCache['grad_mu_gpu'].fill(0.)
        self.gpuCache['grad_S_gpu'].fill(0.)
    
    def get_dimensions(self, Z, variational_posterior):
        return variational_posterior.mean.shape[0], Z.shape[0], Z.shape[1]

    def psicomputations(self, kern, Z, variational_posterior, return_psi2_n=False):
        try:
            return self._psicomputations(kern, Z, variational_posterior, return_psi2_n)
        except:
            return self.fall_back.psicomputations(kern, Z, variational_posterior, return_psi2_n)

    @Cache_this(limit=3, ignore_args=(0,))
    def _psicomputations(self, kern, Z, variational_posterior, return_psi2_n=False):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        """
        variance, lengthscale = kern.variance, kern.lengthscale
        N,M,Q = self.get_dimensions(Z, variational_posterior)
        self._initGPUCache(N,M,Q)
        self.sync_params(lengthscale, Z, variational_posterior.mean, variational_posterior.variance)
        
        psi1_gpu = self.gpuCache['psi1_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']
        psi2n_gpu = self.gpuCache['psi2n_gpu']
        l_gpu = self.gpuCache['l_gpu']
        Z_gpu = self.gpuCache['Z_gpu']
        mu_gpu = self.gpuCache['mu_gpu']
        S_gpu = self.gpuCache['S_gpu']
        log_denom1_gpu = self.gpuCache['log_denom1_gpu']
        log_denom2_gpu = self.gpuCache['log_denom2_gpu']

        psi0 = np.empty((N,))
        psi0[:] = variance
        self.g_psi1computations.prepared_call((self.blocknum,1),(self.threadnum,1,1),psi1_gpu.gpudata, log_denom1_gpu.gpudata, float(variance),l_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
        self.g_psi2computations.prepared_call((self.blocknum,1),(self.threadnum,1,1),psi2_gpu.gpudata, psi2n_gpu.gpudata, log_denom2_gpu.gpudata, float(variance),l_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
        # t = self.g_psi1computations(psi1_gpu, log_denom1_gpu, float(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1),time_kernel=True)
        # print 'g_psi1computations '+str(t)
        # t = self.g_psi2computations(psi2_gpu, psi2n_gpu, log_denom2_gpu, float(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1),time_kernel=True)
        # print 'g_psi2computations '+str(t)
         
        if self.GPU_direct:
            return psi0, psi1_gpu, psi2_gpu
        else:
            if return_psi2_n:
                return psi0, psi1_gpu.get(), psi2n_gpu.get()
            else:
                return psi0, psi1_gpu.get(), psi2_gpu.get()
        
    def psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        try:
            return self._psiDerivativecomputations(kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        except:
            return self.fall_back.psiDerivativecomputations(kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)

    @Cache_this(limit=3, ignore_args=(0,2,3,4))
    def _psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        # resolve the requirement of dL_dpsi2 to be symmetric
        if len(dL_dpsi2.shape)==2: dL_dpsi2 = (dL_dpsi2+dL_dpsi2.T)/2
        else: dL_dpsi2  = (dL_dpsi2+ np.swapaxes(dL_dpsi2, 1,2))/2
    
        variance, lengthscale = kern.variance, kern.lengthscale
        from ....util.linalg_gpu import sum_axis
        ARD = (len(lengthscale)!=1)
        
        N,M,Q = self.get_dimensions(Z, variational_posterior)
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
        grad_l_gpu = self.gpuCache['grad_l_gpu']
        grad_mu_gpu = self.gpuCache['grad_mu_gpu']
        grad_S_gpu = self.gpuCache['grad_S_gpu']
        
        if self.GPU_direct:
            dL_dpsi1_gpu = dL_dpsi1
            dL_dpsi2_gpu = dL_dpsi2
            dL_dpsi0_sum = dL_dpsi0.get().sum() #gpuarray.sum(dL_dpsi0).get()
        else:
            dL_dpsi1_gpu = self.gpuCache['dL_dpsi1_gpu']
            dL_dpsi2_gpu = self.gpuCache['dL_dpsi2_gpu']
            dL_dpsi1_gpu.set(np.asfortranarray(dL_dpsi1))
            dL_dpsi2_gpu.set(np.asfortranarray(dL_dpsi2))
            dL_dpsi0_sum = dL_dpsi0.sum()

        self.reset_derivative()
        # t=self.g_psi1compDer(dvar_gpu,dl_gpu,dZ_gpu,dmu_gpu,dS_gpu,dL_dpsi1_gpu,psi1_gpu, float(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1),time_kernel=True)
        # print 'g_psi1compDer '+str(t)
        # t=self.g_psi2compDer(dvar_gpu,dl_gpu,dZ_gpu,dmu_gpu,dS_gpu,dL_dpsi2_gpu,psi2n_gpu, float(variance),l_gpu,Z_gpu,mu_gpu,S_gpu, np.int32(N), np.int32(M), np.int32(Q), block=(self.threadnum,1,1), grid=(self.blocknum,1),time_kernel=True)
        # print 'g_psi2compDer '+str(t)
        self.g_psi1compDer.prepared_call((self.blocknum,1),(self.threadnum,1,1),dvar_gpu.gpudata,dl_gpu.gpudata,dZ_gpu.gpudata,dmu_gpu.gpudata,dS_gpu.gpudata,dL_dpsi1_gpu.gpudata,psi1_gpu.gpudata, float(variance),l_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))
        self.g_psi2compDer.prepared_call((self.blocknum,1),(self.threadnum,1,1),dvar_gpu.gpudata,dl_gpu.gpudata,dZ_gpu.gpudata,dmu_gpu.gpudata,dS_gpu.gpudata,dL_dpsi2_gpu.gpudata,psi2n_gpu.gpudata, float(variance),l_gpu.gpudata,Z_gpu.gpudata,mu_gpu.gpudata,S_gpu.gpudata, np.int32(N), np.int32(M), np.int32(Q))

        dL_dvar = dL_dpsi0_sum + dvar_gpu.get().sum()#gpuarray.sum(dvar_gpu).get()
        sum_axis(grad_mu_gpu,dmu_gpu,N*Q,self.blocknum)
        dL_dmu = grad_mu_gpu.get()
        sum_axis(grad_S_gpu,dS_gpu,N*Q,self.blocknum)
        dL_dS = grad_S_gpu.get()
        dL_dZ = dZ_gpu.get()
        if ARD:
            sum_axis(grad_l_gpu,dl_gpu,Q,self.blocknum)
            dL_dlengscale = grad_l_gpu.get()
        else:
            dL_dlengscale = dl_gpu.get().sum() #gpuarray.sum(dl_gpu).get()
            
        return dL_dvar, dL_dlengscale, dL_dZ, dL_dmu, dL_dS
    

