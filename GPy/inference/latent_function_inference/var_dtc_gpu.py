# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from ...util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs
from ...util import diag
from ...core.parameterization.variational import VariationalPosterior
import numpy as np
from ...util.misc import param_to_array
log_2_pi = np.log(2*np.pi)

from ...util import gpu_init
assert gpu_init.initSuccess

try:
    import pycuda.gpuarray as gpuarray
    from scikits.cuda import cublas
    from ...util.linalg_gpu import logDiagSum, strideSum, mul_bcast, sum_axis, outer_prod, mul_bcast_first, join_prod
except:
    pass

class VarDTC_GPU(object):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = np.float64(1e-6)
    def __init__(self, batchsize=None, gpu_memory=4., limit=1):
        
        self.batchsize = batchsize
        self.gpu_memory = gpu_memory
                
        self.midRes = {}
        self.batch_pos = 0 # the starting position of the current mini-batch
        
        self.cublas_handle = gpu_init.cublas_handle
        
        # Initialize GPU caches
        self.gpuCache = None
        
    def _initGPUCache(self, kern, num_inducing, input_dim, output_dim, Y):
        ndata = Y.shape[0]
        if self.batchsize==None:
            self.batchsize = self._estimateBatchSize(kern, ndata, num_inducing, input_dim, output_dim)
        if self.gpuCache == None:
            self.gpuCache = {# inference_likelihood
                             'Kmm_gpu'              :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'Lm_gpu'               :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'ones_gpu'             :gpuarray.empty(num_inducing, np.float64,order='F'),
                             'LL_gpu'               :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'b_gpu'                :gpuarray.empty((num_inducing,output_dim),np.float64,order='F'),
                             'v_gpu'                :gpuarray.empty((num_inducing,output_dim),np.float64,order='F'),
                             'vvt_gpu'              :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'KmmInvPsi2LLInvT_gpu' :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'KmmInvPsi2P_gpu'      :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'dL_dpsi2R_gpu'        :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'dL_dKmm_gpu'          :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'psi1Y_gpu'            :gpuarray.empty((num_inducing,output_dim),np.float64,order='F'),
                             'psi2_gpu'             :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'beta_gpu'             :gpuarray.empty((ndata,),np.float64,order='F'),
                             'YT_gpu'               :gpuarray.to_gpu(np.asfortranarray(Y.T)), # DxN
                             'betaYT_gpu'           :gpuarray.empty(Y.T.shape,np.float64,order='F'), # DxN
                             'psi2_t_gpu'           :gpuarray.empty((num_inducing*num_inducing*self.batchsize),np.float64,order='F'),
                             # inference_minibatch
                             'dL_dpsi0_gpu'         :gpuarray.empty((self.batchsize,),np.float64,order='F'),
                             'dL_dpsi1_gpu'         :gpuarray.empty((self.batchsize,num_inducing),np.float64,order='F'),
                             'dL_dpsi2_gpu'         :gpuarray.empty((self.batchsize,num_inducing,num_inducing),np.float64,order='F'),
                             'dL_dthetaL_gpu'       :gpuarray.empty((self.batchsize,),np.float64,order='F'),
                             'betapsi1_gpu'         :gpuarray.empty((self.batchsize,num_inducing),np.float64,order='F'),
                             'thetaL_t_gpu'         :gpuarray.empty((self.batchsize,),np.float64,order='F'),
                             'betaYT2_gpu'          :gpuarray.empty((output_dim,self.batchsize),np.float64,order='F'),
                             'psi0p_gpu'            :gpuarray.empty((self.batchsize,),np.float64,order='F'),
                             'psi1p_gpu'            :gpuarray.empty((self.batchsize,num_inducing),np.float64,order='F'),
                             'psi2p_gpu'            :gpuarray.empty((self.batchsize,num_inducing,num_inducing),np.float64,order='F'),
                             }
            self.gpuCache['ones_gpu'].fill(1.0)
            
            YT_gpu = self.gpuCache['YT_gpu']
            self._trYYT = cublas.cublasDdot(self.cublas_handle, YT_gpu.size, YT_gpu.gpudata, 1, YT_gpu.gpudata, 1)
            
    def _estimateMemoryOccupation(self, N, M, D):
        """
        Estimate the best batch size.
        N - the number of total datapoints
        M - the number of inducing points
        D - the number of observed (output) dimensions
        return: the constant memory size, the memory occupation of batchsize=1
        unit: GB
        """
        return (M+9.*M*M+3*M*D+N+2.*N*D)*8./1024./1024./1024., (4.+3.*M+D+3.*M*M)*8./1024./1024./1024.
    
    def _estimateBatchSize(self, kern, N, M, Q, D):
        """
        Estimate the best batch size.
        N - the number of total datapoints
        M - the number of inducing points
        D - the number of observed (output) dimensions
        return: the constant memory size, the memory occupation of batchsize=1
        unit: GB
        """
        if kern.useGPU:
            x0,x1 = kern.psicomp.estimateMemoryOccupation(N,M,Q)
        else:
            x0, x1 = 0.,0.
        y0, y1 = self._estimateMemoryOccupation(N, M, D)
        
        return int((self.gpu_memory-y0-x0)/(x1+y1))
        
    def _get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT.

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>=D):
            return param_to_array(Y)
        else:
            return jitchol(tdot(Y))
        
    def inference_likelihood(self, kern, X, Z, likelihood, Y):
        """
        The first phase of inference:
        Compute: log-likelihood, dL_dKmm
        
        Cached intermediate results: Kmm, KmmInv,
        """
        
        num_inducing, input_dim = Z.shape[0], Z.shape[1]
        num_data, output_dim = Y.shape
        
        self._initGPUCache(kern, num_inducing, input_dim, output_dim, Y)

        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
        else:
            uncertain_inputs = False
        
        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        trYYT = self._trYYT
        
        psi1Y_gpu = self.gpuCache['psi1Y_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']
        beta_gpu = self.gpuCache['beta_gpu']
        YT_gpu = self.gpuCache['YT_gpu']
        betaYT_gpu = self.gpuCache['betaYT_gpu']
        psi2_t_gpu = self.gpuCache['psi2_t_gpu']
        
        if het_noise:
            beta_gpu.set(np.asfortranarray(beta))
            mul_bcast(betaYT_gpu,beta_gpu,YT_gpu,beta_gpu.size)
            YRY_full = cublas.cublasDdot(self.cublas_handle, YT_gpu.size, betaYT_gpu.gpudata, 1, YT_gpu.gpudata, 1)
        else:
            beta_gpu.fill(beta)
            betaYT_gpu.fill(0.)
            cublas.cublasDaxpy(self.cublas_handle, betaYT_gpu.size, beta, YT_gpu.gpudata, 1, betaYT_gpu.gpudata, 1)
            YRY_full = trYYT*beta

        if kern.useGPU:
            psi1Y_gpu.fill(0.)
            psi2_gpu.fill(0.)
            psi0_full = 0
            
            for n_start in xrange(0,num_data,self.batchsize):
                n_end = min(self.batchsize+n_start, num_data)
                ndata = n_end - n_start
                X_slice = X[n_start:n_end]
                beta_gpu_slice = beta_gpu[n_start:n_end]
                betaYT_gpu_slice = betaYT_gpu[:,n_start:n_end]
                if ndata==self.batchsize:
                    psi2_t_gpu_slice = psi2_t_gpu
                else:
                    psi2_t_gpu_slice = psi2_t_gpu[:num_inducing*num_inducing*ndata]
                if uncertain_inputs:
                    psi0p_gpu = kern.psi0(Z, X_slice)
                    psi1p_gpu = kern.psi1(Z, X_slice)
                    psi2p_gpu = kern.psi2(Z, X_slice)
                else:
                    psi0p_gpu = kern.Kdiag(X_slice)
                    psi1p_gpu = kern.K(X_slice, Z)

                cublas.cublasDgemm(self.cublas_handle, 'T', 'T', num_inducing, output_dim, ndata, 1.0, psi1p_gpu.gpudata, ndata, betaYT_gpu_slice.gpudata, output_dim, 1.0, psi1Y_gpu.gpudata, num_inducing)
                
                if het_noise:
                    psi0_full += cublas.cublasDdot(self.cublas_handle, psi0p_gpu.size, beta_gpu_slice.gpudata, 1, psi0p_gpu.gpudata, 1)
                else:
                    psi0_full += gpuarray.sum(psi0p_gpu).get()
                                    
                if uncertain_inputs:
                    if het_noise:
                        mul_bcast(psi2_t_gpu_slice,beta_gpu_slice,psi2p_gpu,beta_gpu_slice.size)
                        sum_axis(psi2_gpu,psi2_t_gpu_slice,1,ndata)
                    else:
                        sum_axis(psi2_gpu,psi2p_gpu,1,ndata)
                else:
                    if het_noise:
                        psi1_t_gpu = psi2_t_gpu_slice[:,num_inducing*ndata]
                        mul_bcast(psi1_t_gpu,beta_gpu_slice,psi1p_gpu,beta_gpu_slice.size)
                        cublas.cublasDgemm(self.cublas_handle, 'T', 'N', num_inducing, num_inducing, ndata, 1.0, psi1p_gpu.gpudata, ndata, psi1_t_gpu.gpudata, ndata, 1.0, psi2_gpu.gpudata, num_inducing)
                    else:
                        cublas.cublasDgemm(self.cublas_handle, 'T', 'N', num_inducing, num_inducing, ndata, beta, psi1p_gpu.gpudata, ndata, psi1p_gpu.gpudata, ndata, 1.0, psi2_gpu.gpudata, num_inducing)
                    
            if not het_noise:
                psi0_full *= beta
                if uncertain_inputs:
                    cublas.cublasDscal(self.cublas_handle, psi2_gpu.size, beta, psi2_gpu.gpudata, 1)
            
        else:
            psi2_full = np.zeros((num_inducing,num_inducing),order='F')
            psi1Y_full = np.zeros((num_inducing,output_dim),order='F') # MxD
            psi0_full = 0
#             YRY_full = 0
            
            for n_start in xrange(0,num_data,self.batchsize):
                n_end = min(self.batchsize+n_start, num_data)                
                Y_slice = Y[n_start:n_end]
                X_slice = X[n_start:n_end]
                if uncertain_inputs:
                    psi0 = kern.psi0(Z, X_slice)
                    psi1 = kern.psi1(Z, X_slice)
                    psi2 = kern.psi2(Z, X_slice)
                else:
                    psi0 = kern.Kdiag(X_slice)
                    psi1 = kern.K(X_slice, Z)
                    
                if het_noise:
                    beta_slice = beta[n_start:n_end]
                    psi0_full += (beta_slice*psi0).sum()
                    psi1Y_full += np.dot(psi1.T,beta_slice[:,None]*Y_slice) # MxD
#                     YRY_full += (beta_slice*np.square(Y_slice).sum(axis=-1)).sum()
                else:
                    psi0_full += psi0.sum()
                    psi1Y_full += np.dot(psi1.T,Y_slice) # MxD
                                    
                if uncertain_inputs:
                    if het_noise:
                        psi2_full += np.einsum('n,nmo->mo',beta_slice,psi2)
                    else:
                        psi2_full += psi2.sum(axis=0)
                else:
                    if het_noise:
                        psi2_full += np.einsum('n,nm,no->mo',beta_slice,psi1,psi1)
                    else:
                        psi2_full += tdot(psi1.T)
                    
            if not het_noise:
                psi0_full *= beta
                psi1Y_full *= beta
                psi2_full *= beta
#                 YRY_full = trYYT*beta
            
            psi1Y_gpu.set(psi1Y_full)
            psi2_gpu.set(psi2_full)
        
        #======================================================================
        # Compute Common Components
        #======================================================================
        
        Kmm = kern.K(Z).copy()
        Kmm_gpu = self.gpuCache['Kmm_gpu']
        Kmm_gpu.set(np.asfortranarray(Kmm))
        diag.add(Kmm, self.const_jitter)
        ones_gpu = self.gpuCache['ones_gpu']
        cublas.cublasDaxpy(self.cublas_handle, num_inducing, self.const_jitter, ones_gpu.gpudata, 1, Kmm_gpu.gpudata, num_inducing+1)
#         assert np.allclose(Kmm, Kmm_gpu.get())
        
#         Lm = jitchol(Kmm)
        #
        Lm_gpu = self.gpuCache['Lm_gpu']
        cublas.cublasDcopy(self.cublas_handle, Kmm_gpu.size, Kmm_gpu.gpudata, 1, Lm_gpu.gpudata, 1)
        culinalg.cho_factor(Lm_gpu,'L')
#         print np.abs(np.tril(Lm)-np.tril(Lm_gpu.get())).max()
                
#         Lambda = Kmm+psi2_full
#         LL = jitchol(Lambda)
        #
        Lambda_gpu = self.gpuCache['LL_gpu']
        cublas.cublasDcopy(self.cublas_handle, Kmm_gpu.size, Kmm_gpu.gpudata, 1, Lambda_gpu.gpudata, 1)
        cublas.cublasDaxpy(self.cublas_handle, psi2_gpu.size, np.float64(1.0), psi2_gpu.gpudata, 1, Lambda_gpu.gpudata, 1)
        LL_gpu = Lambda_gpu
        culinalg.cho_factor(LL_gpu,'L')
#         print np.abs(np.tril(LL)-np.tril(LL_gpu.get())).max()
        
#         b,_ = dtrtrs(LL, psi1Y_full)
#         bbt_cpu = np.square(b).sum()
        #
        b_gpu = self.gpuCache['b_gpu']
        cublas.cublasDcopy(self.cublas_handle, b_gpu.size, psi1Y_gpu.gpudata, 1, b_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'N', 'N', num_inducing, output_dim, np.float64(1.0), LL_gpu.gpudata, num_inducing, b_gpu.gpudata, num_inducing)
        bbt = cublas.cublasDdot(self.cublas_handle, b_gpu.size, b_gpu.gpudata, 1, b_gpu.gpudata, 1)
#         print np.abs(bbt-bbt_cpu)
        
#         v,_ = dtrtrs(LL.T,b,lower=False)
#         vvt = np.einsum('md,od->mo',v,v)
#         LmInvPsi2LmInvT = backsub_both_sides(Lm,psi2_full,transpose='right')
        #
        v_gpu = self.gpuCache['v_gpu']
        cublas.cublasDcopy(self.cublas_handle, v_gpu.size, b_gpu.gpudata, 1, v_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'T', 'N', num_inducing, output_dim, np.float64(1.0), LL_gpu.gpudata, num_inducing, v_gpu.gpudata, num_inducing)
        vvt_gpu = self.gpuCache['vvt_gpu']
        cublas.cublasDgemm(self.cublas_handle, 'N', 'T', num_inducing, num_inducing, output_dim, np.float64(1.0), v_gpu.gpudata, num_inducing, v_gpu.gpudata, num_inducing, np.float64(0.), vvt_gpu.gpudata, num_inducing)
        LmInvPsi2LmInvT_gpu = self.gpuCache['KmmInvPsi2LLInvT_gpu']
        cublas.cublasDcopy(self.cublas_handle, psi2_gpu.size, psi2_gpu.gpudata, 1, LmInvPsi2LmInvT_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'N', 'N', num_inducing, num_inducing, np.float64(1.0), Lm_gpu.gpudata, num_inducing, LmInvPsi2LmInvT_gpu.gpudata, num_inducing)
        cublas.cublasDtrsm(self.cublas_handle , 'r', 'L', 'T', 'N', num_inducing, num_inducing, np.float64(1.0), Lm_gpu.gpudata, num_inducing, LmInvPsi2LmInvT_gpu.gpudata, num_inducing)
        #tr_LmInvPsi2LmInvT = cublas.cublasDasum(self.cublas_handle, num_inducing, LmInvPsi2LmInvT_gpu.gpudata, num_inducing+1)
        tr_LmInvPsi2LmInvT = float(strideSum(LmInvPsi2LmInvT_gpu, num_inducing+1).get())
#         print np.abs(vvt-vvt_gpu.get()).max()
#         print np.abs(np.trace(LmInvPsi2LmInvT)-tr_LmInvPsi2LmInvT)
        
#         Psi2LLInvT = dtrtrs(LL,psi2_full)[0].T
#         LmInvPsi2LLInvT= dtrtrs(Lm,Psi2LLInvT)[0]
#         KmmInvPsi2LLInvT = dtrtrs(Lm,LmInvPsi2LLInvT,trans=True)[0]
#         KmmInvPsi2P = dtrtrs(LL,KmmInvPsi2LLInvT.T, trans=True)[0].T
        #
        KmmInvPsi2LLInvT_gpu = LmInvPsi2LmInvT_gpu # Reuse GPU memory (size:MxM)
        cublas.cublasDcopy(self.cublas_handle, psi2_gpu.size, psi2_gpu.gpudata, 1, KmmInvPsi2LLInvT_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'N', 'N', num_inducing, num_inducing, np.float64(1.0), Lm_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing)
        cublas.cublasDtrsm(self.cublas_handle , 'r', 'L', 'T', 'N', num_inducing, num_inducing, np.float64(1.0), LL_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'T', 'N', num_inducing, num_inducing, np.float64(1.0), Lm_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing)
        KmmInvPsi2P_gpu = self.gpuCache['KmmInvPsi2P_gpu']
        cublas.cublasDcopy(self.cublas_handle, KmmInvPsi2LLInvT_gpu.size, KmmInvPsi2LLInvT_gpu.gpudata, 1, KmmInvPsi2P_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'r', 'L', 'N', 'N', num_inducing, num_inducing, np.float64(1.0), LL_gpu.gpudata, num_inducing, KmmInvPsi2P_gpu.gpudata, num_inducing)
#         print np.abs(KmmInvPsi2P-KmmInvPsi2P_gpu.get()).max()
        
#         dL_dpsi2R = (output_dim*KmmInvPsi2P - vvt)/2. # dL_dpsi2 with R inside psi2
        #
        dL_dpsi2R_gpu = self.gpuCache['dL_dpsi2R_gpu']
        cublas.cublasDcopy(self.cublas_handle, vvt_gpu.size, vvt_gpu.gpudata, 1, dL_dpsi2R_gpu.gpudata, 1)
        cublas.cublasDaxpy(self.cublas_handle, KmmInvPsi2P_gpu.size, np.float64(-output_dim), KmmInvPsi2P_gpu.gpudata, 1, dL_dpsi2R_gpu.gpudata, 1)
        cublas.cublasDscal(self.cublas_handle, dL_dpsi2R_gpu.size, np.float64(-0.5), dL_dpsi2R_gpu.gpudata, 1)
#         print np.abs(dL_dpsi2R_gpu.get()-dL_dpsi2R).max()
                        
        #======================================================================
        # Compute log-likelihood
        #======================================================================
        if het_noise:
            logL_R = -np.log(beta).sum()
        else:
            logL_R = -num_data*np.log(beta)
#         logL_old = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-np.trace(LmInvPsi2LmInvT))+YRY_full-bbt)/2.-output_dim*(-np.log(np.diag(Lm)).sum()+np.log(np.diag(LL)).sum())
        
        logdetKmm = float(logDiagSum(Lm_gpu,num_inducing+1).get())
        logdetLambda = float(logDiagSum(LL_gpu,num_inducing+1).get())
        logL = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-tr_LmInvPsi2LmInvT)+YRY_full-bbt)/2.+output_dim*(logdetKmm-logdetLambda)
#         print np.abs(logL_old - logL)

        #======================================================================
        # Compute dL_dKmm
        #======================================================================
        
#         dL_dKmm =  -(output_dim*np.einsum('md,od->mo',KmmInvPsi2LLInvT,KmmInvPsi2LLInvT) + vvt)/2.
        #
        dL_dKmm_gpu = self.gpuCache['dL_dKmm_gpu']
        cublas.cublasDgemm(self.cublas_handle, 'N', 'T', num_inducing, num_inducing, num_inducing, np.float64(1.0), KmmInvPsi2LLInvT_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing, np.float64(0.), dL_dKmm_gpu.gpudata, num_inducing)
        cublas.cublasDaxpy(self.cublas_handle, dL_dKmm_gpu.size, np.float64(1./output_dim), vvt_gpu.gpudata, 1, dL_dKmm_gpu.gpudata, 1)
        cublas.cublasDscal(self.cublas_handle, dL_dKmm_gpu.size, np.float64(-output_dim/2.), dL_dKmm_gpu.gpudata, 1)
#         print np.abs(dL_dKmm - dL_dKmm_gpu.get()).max()

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================
                
        post = Posterior(woodbury_inv=KmmInvPsi2P_gpu.get(), woodbury_vector=v_gpu.get(), K=Kmm_gpu.get(), mean=None, cov=None, K_chol=Lm_gpu.get())

        return logL, dL_dKmm_gpu.get(), post

    def inference_minibatch(self, kern, X, Z, likelihood, Y):
        """
        The second phase of inference: Computing the derivatives over a minibatch of Y 
        Compute: dL_dpsi0, dL_dpsi1, dL_dpsi2, dL_dthetaL
        return a flag showing whether it reached the end of Y (isEnd)
        """

        num_data, output_dim = Y.shape
        num_inducing = Z.shape[0]

        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
        else:
            uncertain_inputs = False
        
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        
        n_start = self.batch_pos
        n_end = min(self.batchsize+n_start, num_data)
        if n_end==num_data:
            isEnd = True
            self.batch_pos = 0
        else:
            isEnd = False
            self.batch_pos = n_end
        
        nSlice = n_end-n_start
        X_slice = X[n_start:n_end]
        
        if kern.useGPU:
            if uncertain_inputs:
                psi0p_gpu = kern.psi0(Z, X_slice)
                psi1p_gpu = kern.psi1(Z, X_slice)
                psi2p_gpu = kern.psi2(Z, X_slice)
            else:
                psi0p_gpu = kern.Kdiag(X_slice)
                psi1p_gpu = kern.K(X_slice, Z)
                psi2p_gpu = self.gpuCache['psi2p_gpu']
                if psi2p_gpu.shape[0] > nSlice:
                    psi2p_gpu = psi2p_gpu.ravel()[:nSlice*num_inducing*num_inducing].reshape(nSlice,num_inducing,num_inducing)
        else:
            if uncertain_inputs:
                psi0 = kern.psi0(Z, X_slice)
                psi1 = kern.psi1(Z, X_slice)
                psi2 = kern.psi2(Z, X_slice)
            else:
                psi0 = kern.Kdiag(X_slice)
                psi1 = kern.K(X_slice, Z)

            psi0p_gpu = self.gpuCache['psi0p_gpu']
            psi1p_gpu = self.gpuCache['psi1p_gpu']
            psi2p_gpu = self.gpuCache['psi2p_gpu']
            if psi0p_gpu.shape[0] > nSlice:
                psi0p_gpu = psi0p_gpu[:nSlice]
                psi1p_gpu = psi1p_gpu.ravel()[:nSlice*num_inducing].reshape(nSlice,num_inducing)
                psi2p_gpu = psi2p_gpu.ravel()[:nSlice*num_inducing*num_inducing].reshape(nSlice,num_inducing,num_inducing)
            psi0p_gpu.set(np.asfortranarray(psi0))
            psi1p_gpu.set(np.asfortranarray(psi1))
            if uncertain_inputs:
                psi2p_gpu.set(np.asfortranarray(psi2))
                
        #======================================================================
        # Prepare gpu memory
        #======================================================================
        
        dL_dpsi2R_gpu = self.gpuCache['dL_dpsi2R_gpu']
        v_gpu = self.gpuCache['v_gpu']        
        betaYT_gpu = self.gpuCache['betaYT_gpu']
        beta_gpu = self.gpuCache['beta_gpu']
        dL_dpsi0_gpu = self.gpuCache['dL_dpsi0_gpu']
        dL_dpsi1_gpu = self.gpuCache['dL_dpsi1_gpu']
        dL_dpsi2_gpu = self.gpuCache['dL_dpsi2_gpu']
        dL_dthetaL_gpu = self.gpuCache['dL_dthetaL_gpu']
        psi2R_gpu = self.gpuCache['psi2_t_gpu'][:nSlice*num_inducing*num_inducing].reshape(nSlice,num_inducing,num_inducing)
        betapsi1_gpu = self.gpuCache['betapsi1_gpu']
        thetaL_t_gpu = self.gpuCache['thetaL_t_gpu']
        betaYT2_gpu = self.gpuCache['betaYT2_gpu']
        
        betaYT_gpu_slice = betaYT_gpu[:,n_start:n_end]
        beta_gpu_slice = beta_gpu[n_start:n_end]

        # Adjust to the batch size
        if dL_dpsi0_gpu.shape[0] > nSlice:
            betaYT2_gpu = betaYT2_gpu[:,:nSlice]
            dL_dpsi0_gpu = dL_dpsi0_gpu.ravel()[:nSlice]
            dL_dpsi1_gpu = dL_dpsi1_gpu.ravel()[:nSlice*num_inducing].reshape(nSlice,num_inducing)
            dL_dpsi2_gpu = dL_dpsi2_gpu.ravel()[:nSlice*num_inducing*num_inducing].reshape(nSlice,num_inducing,num_inducing)
            dL_dthetaL_gpu = dL_dthetaL_gpu.ravel()[:nSlice]
            psi2R_gpu = psi2R_gpu.ravel()[:nSlice*num_inducing*num_inducing].reshape(nSlice,num_inducing,num_inducing)
            thetaL_t_gpu = thetaL_t_gpu.ravel()[:nSlice]
            betapsi1_gpu = betapsi1_gpu.ravel()[:nSlice*num_inducing].reshape(nSlice,num_inducing)
        
        mul_bcast(betapsi1_gpu,beta_gpu_slice,psi1p_gpu,beta_gpu_slice.size)

        #======================================================================
        # Compute dL_dpsi
        #======================================================================
        
        dL_dpsi0_gpu.fill(0.)
        cublas.cublasDaxpy(self.cublas_handle, dL_dpsi0_gpu.size, output_dim/(-2.), beta_gpu_slice.gpudata, 1, dL_dpsi0_gpu.gpudata, 1)
        
        cublas.cublasDgemm(self.cublas_handle, 'T', 'T', nSlice, num_inducing, output_dim, 1.0, betaYT_gpu_slice.gpudata, output_dim, v_gpu.gpudata, num_inducing, 0., dL_dpsi1_gpu.gpudata, nSlice)
        
        if uncertain_inputs:
            outer_prod(dL_dpsi2_gpu,beta_gpu_slice,dL_dpsi2R_gpu,beta_gpu_slice.size)
        else:
            cublas.cublasDgemm(self.cublas_handle, 'N', 'N', nSlice, num_inducing, output_dim, 1.0, betapsi1_gpu.gpudata, nSlice, dL_dpsi2R_gpu.gpudata, num_inducing, 1.0, dL_dpsi1_gpu.gpudata, nSlice)
            
        #======================================================================
        # Compute dL_dthetaL
        #======================================================================
        
        if not uncertain_inputs:
            join_prod(psi2p_gpu,psi1p_gpu,psi1p_gpu,nSlice,num_inducing)

        mul_bcast_first(psi2R_gpu,dL_dpsi2R_gpu,psi2p_gpu,nSlice)
        

        dL_dthetaL_gpu.fill(0.)
        
        cublas.cublasDcopy(self.cublas_handle, betaYT_gpu_slice.size, betaYT_gpu_slice.gpudata, 1, betaYT2_gpu.gpudata, 1)
        mul_bcast(betaYT2_gpu,betaYT2_gpu,betaYT2_gpu,betaYT2_gpu.size)
        cublas.cublasDscal(self.cublas_handle, betaYT2_gpu.size, 0.5, betaYT2_gpu.gpudata, 1)
        sum_axis(dL_dthetaL_gpu, betaYT2_gpu, 1, output_dim)
        
        cublas.cublasDaxpy(self.cublas_handle, dL_dthetaL_gpu.size, output_dim/(-2.0), beta_gpu_slice.gpudata, 1, dL_dthetaL_gpu.gpudata, 1)
        cublas.cublasDcopy(self.cublas_handle, beta_gpu_slice.size, beta_gpu_slice.gpudata, 1, thetaL_t_gpu.gpudata, 1)
        mul_bcast(thetaL_t_gpu,thetaL_t_gpu,thetaL_t_gpu,thetaL_t_gpu.size)
        mul_bcast(thetaL_t_gpu,thetaL_t_gpu,psi0p_gpu,thetaL_t_gpu.size)
        cublas.cublasDaxpy(self.cublas_handle, dL_dthetaL_gpu.size, output_dim/2.0, thetaL_t_gpu.gpudata, 1, dL_dthetaL_gpu.gpudata, 1)
        
        thetaL_t_gpu.fill(0.)
        sum_axis(thetaL_t_gpu, psi2R_gpu, nSlice, num_inducing*num_inducing)
        mul_bcast(thetaL_t_gpu,thetaL_t_gpu,beta_gpu_slice,thetaL_t_gpu.size)
        mul_bcast(thetaL_t_gpu,thetaL_t_gpu,beta_gpu_slice,thetaL_t_gpu.size)
        cublas.cublasDaxpy(self.cublas_handle, dL_dthetaL_gpu.size, -1.0, thetaL_t_gpu.gpudata, 1, dL_dthetaL_gpu.gpudata, 1)
        
        cublas.cublasDgemm(self.cublas_handle, 'T', 'T', output_dim, nSlice, num_inducing, -1.0, v_gpu.gpudata, num_inducing, betapsi1_gpu.gpudata, nSlice, 0.0, betaYT2_gpu.gpudata, output_dim)
        mul_bcast(betaYT2_gpu,betaYT2_gpu,betaYT_gpu_slice,betaYT2_gpu.size)
        sum_axis(dL_dthetaL_gpu, betaYT2_gpu, 1, output_dim)

        if kern.useGPU:
            dL_dpsi0 = dL_dpsi0_gpu
            dL_dpsi1 = dL_dpsi1_gpu
        else:
            dL_dpsi0 = dL_dpsi0_gpu.get()
            dL_dpsi1 = dL_dpsi1_gpu.get()            
        if uncertain_inputs:
            if kern.useGPU:
                dL_dpsi2 = dL_dpsi2_gpu
            else:
                dL_dpsi2 = dL_dpsi2_gpu.get()
        if het_noise:
            dL_dthetaL = dL_dthetaL_gpu.get()
        else:
            dL_dthetaL = gpuarray.sum(dL_dthetaL_gpu).get()
        if uncertain_inputs:
            grad_dict = {'dL_dpsi0':dL_dpsi0,
                         'dL_dpsi1':dL_dpsi1,
                         'dL_dpsi2':dL_dpsi2,
                         'dL_dthetaL':dL_dthetaL}
        else:
            grad_dict = {'dL_dKdiag':dL_dpsi0,
                         'dL_dKnm':dL_dpsi1,
                         'dL_dthetaL':dL_dthetaL}
            
        return isEnd, (n_start,n_end), grad_dict
    
