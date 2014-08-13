# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from ...util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs
from ...util import diag
from ...core.parameterization.variational import VariationalPosterior
import numpy as np
from ...util.misc import param_to_array
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

from ...util import gpu_init

try:
    import scikits.cuda.linalg as culinalg
    import pycuda.gpuarray as gpuarray
    from scikits.cuda import cublas
    from ...util.linalg_gpu import logDiagSum, strideSum, mul_bcast, sum_axis, outer_prod, mul_bcast_first, join_prod, traceDot
except:
    pass

class VarDTC_GPU(LatentFunctionInference):
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
                             # inference_minibatch
                             'dL_dpsi0_gpu'         :gpuarray.empty((self.batchsize,),np.float64,order='F'),
                             'dL_dpsi1_gpu'         :gpuarray.empty((self.batchsize,num_inducing),np.float64,order='F'),
                             'dL_dpsi2_gpu'         :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
                             'psi0p_gpu'            :gpuarray.empty((self.batchsize,),np.float64,order='F'),
                             'psi1p_gpu'            :gpuarray.empty((self.batchsize,num_inducing),np.float64,order='F'),
                             'psi2p_gpu'            :gpuarray.empty((num_inducing,num_inducing),np.float64,order='F'),
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
        
        opt_batchsize = min(int((self.gpu_memory-y0-x0)/(x1+y1)), N)
        
        return opt_batchsize
        
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
        
    def gatherPsiStat(self, kern, X, Z, Y, beta, uncertain_inputs, het_noise):
        num_inducing, input_dim = Z.shape[0], Z.shape[1]
        num_data, output_dim = Y.shape
        trYYT = self._trYYT
        psi1Y_gpu = self.gpuCache['psi1Y_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']
        beta_gpu = self.gpuCache['beta_gpu']
        YT_gpu = self.gpuCache['YT_gpu']
        betaYT_gpu = self.gpuCache['betaYT_gpu']
        
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
                betaYT_gpu_slice = betaYT_gpu[:,n_start:n_end]

                if uncertain_inputs:
                    psi0 = kern.psi0(Z, X_slice)
                    psi1p_gpu = kern.psi1(Z, X_slice)
                    psi2p_gpu = kern.psi2(Z, X_slice)
                else:
                    psi0 = kern.Kdiag(X_slice)
                    psi1p_gpu = kern.K(X_slice, Z)

                cublas.cublasDgemm(self.cublas_handle, 'T', 'T', num_inducing, output_dim, ndata, 1.0, psi1p_gpu.gpudata, ndata, betaYT_gpu_slice.gpudata, output_dim, 1.0, psi1Y_gpu.gpudata, num_inducing)
                
                psi0_full += psi0.sum()
                                    
                if uncertain_inputs:
                    sum_axis(psi2_gpu,psi2p_gpu,1,1)
                else:
                    cublas.cublasDgemm(self.cublas_handle, 'T', 'N', num_inducing, num_inducing, ndata, beta, psi1p_gpu.gpudata, ndata, psi1p_gpu.gpudata, ndata, 1.0, psi2_gpu.gpudata, num_inducing)
                    
            psi0_full *= beta
            if uncertain_inputs:
                cublas.cublasDscal(self.cublas_handle, psi2_gpu.size, beta, psi2_gpu.gpudata, 1)
            
        else:    
            psi2_full = np.zeros((num_inducing,num_inducing))
            psi1Y_full = np.zeros((output_dim,num_inducing)) # DxM
            psi0_full = 0.
            YRY_full = 0.
            
            for n_start in xrange(0,num_data,self.batchsize):            
                n_end = min(self.batchsize+n_start, num_data)
                Y_slice = Y[n_start:n_end]
                X_slice = X[n_start:n_end]
                
                if het_noise:
                    b = beta[n_start]
                    YRY_full += np.inner(Y_slice, Y_slice)*b
                else:
                    b = beta
                
                if uncertain_inputs:
                    psi0 = kern.psi0(Z, X_slice)
                    psi1 = kern.psi1(Z, X_slice)
                    psi2_full += kern.psi2(Z, X_slice)*b
                else:
                    psi0 = kern.Kdiag(X_slice)
                    psi1 = kern.K(X_slice, Z)
                    psi2_full += np.dot(psi1.T,psi1)*b
                    
                psi0_full += psi0.sum()*b
                psi1Y_full += np.dot(Y_slice.T,psi1)*b # DxM                
    
            if not het_noise:
                YRY_full = trYYT*beta
            psi1Y_gpu.set(psi1Y_full)
            psi2_gpu.set(psi2_full)
    
        return psi0_full, YRY_full
        
    def inference_likelihood(self, kern, X, Z, likelihood, Y):
        """
        The first phase of inference:
        Compute: log-likelihood, dL_dKmm
        
        Cached intermediate results: Kmm, KmmInv,
        """
        
        num_inducing, input_dim = Z.shape[0], Z.shape[1]
        num_data, output_dim = Y.shape
        
        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        if het_noise:
            self.batchsize=0
        
        self._initGPUCache(kern, num_inducing, input_dim, output_dim, Y)

        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
        else:
            uncertain_inputs = False
        
        psi1Y_gpu = self.gpuCache['psi1Y_gpu']
        psi2_gpu = self.gpuCache['psi2_gpu']
        
        psi0_full, YRY_full = self.gatherPsiStat(kern, X, Z, Y, beta, uncertain_inputs, het_noise)
        
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

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================        
        
        if not het_noise:
            dL_dthetaL = (YRY_full + output_dim*psi0_full - num_data*output_dim)/-2.
            dL_dthetaL += cublas.cublasDdot(self.cublas_handle,dL_dpsi2R_gpu.size, dL_dpsi2R_gpu.gpudata,1,psi2_gpu.gpudata,1)
            dL_dthetaL += cublas.cublasDdot(self.cublas_handle,v_gpu.size, v_gpu.gpudata,1,psi1Y_gpu.gpudata,1)
            self.midRes['dL_dthetaL'] = -beta*dL_dthetaL
            
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
        if het_noise:
            beta = beta[n_start] # nSlice==1
        
        if kern.useGPU:
            if not uncertain_inputs:
                psi0p_gpu = kern.Kdiag(X_slice)
                psi1p_gpu = kern.K(X_slice, Z)
                psi2p_gpu = self.gpuCache['psi2p_gpu']
            elif het_noise:
                psi0p_gpu = kern.psi0(Z, X_slice)
                psi1p_gpu = kern.psi1(Z, X_slice)
                psi2p_gpu = kern.psi2(Z, X_slice)
        elif not uncertain_inputs or het_noise:
            if not uncertain_inputs:
                psi0 = kern.Kdiag(X_slice)
                psi1 = kern.K(X_slice, Z)
            elif het_noise:
                psi0 = kern.psi0(Z, X_slice)
                psi1 = kern.psi1(Z, X_slice)
                psi2 = kern.psi2(Z, X_slice)

            psi0p_gpu = self.gpuCache['psi0p_gpu']
            psi1p_gpu = self.gpuCache['psi1p_gpu']
            psi2p_gpu = self.gpuCache['psi2p_gpu']
            if psi0p_gpu.shape[0] > nSlice:
                psi0p_gpu = psi0p_gpu[:nSlice]
                psi1p_gpu = psi1p_gpu.ravel()[:nSlice*num_inducing].reshape(nSlice,num_inducing)
            psi0p_gpu.set(np.asfortranarray(psi0))
            psi1p_gpu.set(np.asfortranarray(psi1))
            if uncertain_inputs:
                psi2p_gpu.set(np.asfortranarray(psi2))
                            
        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi2R_gpu = self.gpuCache['dL_dpsi2R_gpu']
        v_gpu = self.gpuCache['v_gpu']        
        dL_dpsi0_gpu = self.gpuCache['dL_dpsi0_gpu']
        dL_dpsi1_gpu = self.gpuCache['dL_dpsi1_gpu']
        dL_dpsi2_gpu = self.gpuCache['dL_dpsi2_gpu']
        betaYT_gpu = self.gpuCache['betaYT_gpu']
        betaYT_gpu_slice = betaYT_gpu[:,n_start:n_end]
        
        # Adjust to the batch size
        if dL_dpsi0_gpu.shape[0] > nSlice:
            dL_dpsi0_gpu = dL_dpsi0_gpu.ravel()[:nSlice]
            dL_dpsi1_gpu = dL_dpsi1_gpu.ravel()[:nSlice*num_inducing].reshape(nSlice,num_inducing)
        
        dL_dpsi0_gpu.fill(-output_dim *beta/2.)
        
        cublas.cublasDgemm(self.cublas_handle, 'T', 'T', nSlice, num_inducing, output_dim, 1.0, betaYT_gpu_slice.gpudata, output_dim, v_gpu.gpudata, num_inducing, 0., dL_dpsi1_gpu.gpudata, nSlice)
        
        if uncertain_inputs:
            cublas.cublasDcopy(self.cublas_handle, dL_dpsi2R_gpu.size, dL_dpsi2R_gpu.gpudata, 1, dL_dpsi2_gpu.gpudata, 1)
            cublas.cublasDscal(self.cublas_handle, dL_dpsi2_gpu.size, beta, dL_dpsi2_gpu.gpudata, 1)
        else:
            cublas.cublasDgemm(self.cublas_handle, 'N', 'N', nSlice, num_inducing, output_dim, beta, psi1p_gpu.gpudata, nSlice, dL_dpsi2R_gpu.gpudata, num_inducing, 1.0, dL_dpsi1_gpu.gpudata, nSlice)

        #======================================================================
        # Compute dL_dthetaL
        #======================================================================
        if het_noise:
            betaY = betaYT_gpu_slice.get()
            dL_dthetaL = ((np.square(betaY)).sum(axis=-1) + np.square(beta)*(output_dim*psi0p_gpu.get())-output_dim*beta)/2.
            dL_dthetaL += -beta*beta*cublas.cublasDdot(self.cublas_handle,dL_dpsi2R_gpu.size, dL_dpsi2R_gpu.gpudata,1,psi2p_gpu.gpudata,1)
            dL_dthetaL += -beta*(betaY*np.dot(psi1p_gpu.get(),v_gpu.get())).sum(axis=-1)

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
        if not het_noise:
            if isEnd:
                dL_dthetaL = self.midRes['dL_dthetaL']
            else:
                dL_dthetaL = 0.
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
    
