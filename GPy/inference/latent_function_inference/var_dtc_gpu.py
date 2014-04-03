# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from ...util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs
from ...util import diag
from ...core.parameterization.variational import VariationalPosterior
import numpy as np
from ...util.misc import param_to_array
log_2_pi = np.log(2*np.pi)

try:
    import scikits.cuda.linalg as culinalg
    import pycuda.gpuarray as gpuarray
    from scikits.cuda import cublas
    import pycuda.autoinit
    from pycuda.reduction import ReductionKernel
    from ...util.linalg_gpu import logDiagSum, strideSum, mul_bcast, sum_axis
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
    def __init__(self, batchsize, limit=1):
        
        self.batchsize = batchsize
        
        # Cache functions
        from ...util.caching import Cacher
        self.get_trYYT = Cacher(self._get_trYYT, limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, limit)
        
        self.midRes = {}
        self.batch_pos = 0 # the starting position of the current mini-batch
        
        # Initialize GPU environment
        culinalg.init()
        self.cublas_handle = cublas.cublasCreate()
        
        # Initialize GPU caches
        self.gpuCache = None
        
    def _initGPUCache(self, num_inducing, output_dim, Y):
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
                             'beta_gpu'             :gpuarray.empty((output_dim,),np.float64,order='F'),
                             'Y_gpu'                :gpuarray.to_gpu(np.asfortranarray(Y)),
                             'betaY_gpu'            :gpuarray.empty(Y.shape,np.float64,order='F'),
                             'psi2_t_gpu'           :gpuarray.empty((self.batchsize,num_inducing,num_inducing),np.float64,order='F'),
                             # inference_minibatch
                             }
            self.gpuCache['ones_gpu'].fill(1.0)
            
            Y_gpu = self.gpuCache['Y_gpu']
            self._trYYT = cublas.cublasDdot(self.cublas_handle, Y_gpu.size, Y_gpu.gpudata, 1, Y_gpu.gpudata, 1)
        
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
        
        num_inducing = Z.shape[0]
        num_data, output_dim = Y.shape
        
        self._initGPUCache(num_inducing, output_dim)

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
        Y_gpu = self.gpuCache['Y_gpu']
        betaY_gpu = self.gpuCache['betaY_gpu']
        psi2_t_gpu = self.gpuCache['psi2_t_gpu']
        
        if het_noise:
            beta_gpu.set(np.asfortranarray(beta))
            mul_bcast(betaY_gpu,beta_gpu,Y_gpu,beta_gpu.size)
            YRY_full = cublas.cublasDdot(self.cublas_handle, Y_gpu.size, betaY_gpu.gpudata, 1, Y_gpu.gpudata, 1)
        else:
            beta_gpu.fill(beta)
            betaY_gpu.fill(0.)
            cublas.cublasDaxpy(self.cublas_handle, betaY_gpu.size, beta, Y_gpu.gpudata, 1, betaY_gpu, 1)
            YRY_full = trYYT*beta

        if kern.useGPU:
            psi1Y_gpu.fill(0.)
            psi2_gpu.fill(0.)
            psi0_full = 0
            
            for n_start in xrange(0,num_data,self.batchsize):
                n_end = min(self.batchsize+n_start, num_data)
                ndata = n_end - n_start
                Y_slice = Y[n_start:n_end]
                X_slice = X[n_start:n_end]
                beta_gpu_slice = beta_gpu[n_start:n_end]
                betaY_gpu_slice = betaY_gpu[n_start:n_end]
                if ndata==self.batchsize:
                    psi2_t_gpu_slice = psi2_t_gpu
                else:
                    psi2_t_gpu_slice = psi2_t_gpu[0:ndata]
                if uncertain_inputs:
                    psi0p_gpu = kern.psi0(Z, X_slice)
                    psi1p_gpu = kern.psi1(Z, X_slice)
                    psi2p_gpu = kern.psi2(Z, X_slice)
                else:
                    psi0p_gpu = kern.Kdiag(X_slice)
                    psi1p_gpu = kern.K(X_slice, Z)

                cublas.cublasDgemm(self.cublas_handle, 'T', 'N', num_inducing, output_dim, ndata, 1.0, psi1p_gpu.gpudata, ndata, betaY_gpu_slice.gpudata, ndata, 1.0, psi1Y_gpu.gpudata, num_inducing)
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
                        psi1_t_gpu = psi2_t_gpu_slice[:,:,0]
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
            YRY_full = 0
            
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
                    YRY_full += (beta_slice*np.square(Y_slice).sum(axis=-1)).sum()
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
                YRY_full = trYYT*beta
            
            psi1Y_gpu.set(psi1Y_full)
            psi2_gpu.set(psi2_full)
        
        #======================================================================
        # Compute Common Components
        #======================================================================
        
        Kmm = kern.K(Z).copy()
        Kmm_gpu = self.gpuCache['Kmm_gpu']
        Kmm_gpu.set(Kmm)
        diag.add(Kmm, self.const_jitter)
        ones_gpu = self.gpuCache['ones_gpu']
        cublas.cublasDaxpy(self.cublas_handle, num_inducing, self.const_jitter, ones_gpu.gpudata, 1, Kmm_gpu.gpudata, num_inducing+1)
        assert np.allclose(Kmm, Kmm_gpu.get())
        
        Lm = jitchol(Kmm)
        #
        Lm_gpu = self.gpuCache['Lm_gpu']
        cublas.cublasDcopy(self.cublas_handle, Kmm_gpu.size, Kmm_gpu.gpudata, 1, Lm_gpu.gpudata, 1)
        culinalg.cho_factor(Lm_gpu,'L')
        print np.abs(np.tril(Lm)-np.tril(Lm_gpu.get())).max()
                
        Lambda = Kmm+psi2_full
        LL = jitchol(Lambda)
        #
        Lambda_gpu = self.gpuCache['LL_gpu']
        cublas.cublasDcopy(self.cublas_handle, Kmm_gpu.size, Kmm_gpu.gpudata, 1, Lambda_gpu.gpudata, 1)
        cublas.cublasDaxpy(self.cublas_handle, psi2_gpu.size, np.float64(1.0), psi2_gpu.gpudata, 1, Lambda_gpu.gpudata, 1)
        LL_gpu = Lambda_gpu
        culinalg.cho_factor(LL_gpu,'L')
        print np.abs(np.tril(LL)-np.tril(LL_gpu.get())).max()
        
        b,_ = dtrtrs(LL, psi1Y_full)
        bbt_cpu = np.square(b).sum()
        #
        b_gpu = self.gpuCache['b_gpu']
        cublas.cublasDcopy(self.cublas_handle, b_gpu.size, psi1Y_gpu.gpudata, 1, b_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'N', 'N', num_inducing, output_dim, np.float64(1.0), LL_gpu.gpudata, num_inducing, b_gpu.gpudata, num_inducing)
        bbt = cublas.cublasDdot(self.cublas_handle, b_gpu.size, b_gpu.gpudata, 1, b_gpu.gpudata, 1)
        print np.abs(bbt-bbt_cpu)
        
        v,_ = dtrtrs(LL.T,b,lower=False)
        vvt = np.einsum('md,od->mo',v,v)
        LmInvPsi2LmInvT = backsub_both_sides(Lm,psi2_full,transpose='right')
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
        print np.abs(vvt-vvt_gpu.get()).max()
        print np.abs(np.trace(LmInvPsi2LmInvT)-tr_LmInvPsi2LmInvT)
        
        Psi2LLInvT = dtrtrs(LL,psi2_full)[0].T
        LmInvPsi2LLInvT= dtrtrs(Lm,Psi2LLInvT)[0]
        KmmInvPsi2LLInvT = dtrtrs(Lm,LmInvPsi2LLInvT,trans=True)[0]
        KmmInvPsi2P = dtrtrs(LL,KmmInvPsi2LLInvT.T, trans=True)[0].T
        #
        KmmInvPsi2LLInvT_gpu = LmInvPsi2LmInvT_gpu # Reuse GPU memory (size:MxM)
        cublas.cublasDcopy(self.cublas_handle, psi2_gpu.size, psi2_gpu.gpudata, 1, KmmInvPsi2LLInvT_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'N', 'N', num_inducing, num_inducing, np.float64(1.0), Lm_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing)
        cublas.cublasDtrsm(self.cublas_handle , 'r', 'L', 'T', 'N', num_inducing, num_inducing, np.float64(1.0), LL_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing)
        cublas.cublasDtrsm(self.cublas_handle , 'L', 'L', 'T', 'N', num_inducing, num_inducing, np.float64(1.0), Lm_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing)
        KmmInvPsi2P_gpu = self.gpuCache['KmmInvPsi2P_gpu']
        cublas.cublasDcopy(self.cublas_handle, KmmInvPsi2LLInvT_gpu.size, KmmInvPsi2LLInvT_gpu.gpudata, 1, KmmInvPsi2P_gpu.gpudata, 1)
        cublas.cublasDtrsm(self.cublas_handle , 'r', 'L', 'N', 'N', num_inducing, num_inducing, np.float64(1.0), LL_gpu.gpudata, num_inducing, KmmInvPsi2P_gpu.gpudata, num_inducing)
        print np.abs(KmmInvPsi2P-KmmInvPsi2P_gpu.get()).max()
        
        dL_dpsi2R = (output_dim*KmmInvPsi2P - vvt)/2. # dL_dpsi2 with R inside psi2
        #
        dL_dpsi2R_gpu = self.gpuCache['dL_dpsi2R_gpu']
        cublas.cublasDcopy(self.cublas_handle, vvt_gpu.size, vvt_gpu.gpudata, 1, dL_dpsi2R_gpu.gpudata, 1)
        cublas.cublasDaxpy(self.cublas_handle, KmmInvPsi2P_gpu.size, np.float64(-output_dim), KmmInvPsi2P_gpu.gpudata, 1, dL_dpsi2R_gpu.gpudata, 1)
        cublas.cublasDscal(self.cublas_handle, dL_dpsi2R_gpu.size, np.float64(-0.5), dL_dpsi2R_gpu.gpudata, 1)
        print np.abs(dL_dpsi2R_gpu.get()-dL_dpsi2R).max()

        # Cache intermediate results
        self.midRes['dL_dpsi2R'] = dL_dpsi2R
        self.midRes['v'] = v
        
        #logDiagSum = ReductionKernel(np.float64, neutral="0", reduce_expr="a+b", map_expr="i%step==0?log(x[i]):0", arguments="double *x, int step")
                
        #======================================================================
        # Compute log-likelihood
        #======================================================================
        if het_noise:
            logL_R = -np.log(beta).sum()
        else:
            logL_R = -num_data*np.log(beta)
        logL_old = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-np.trace(LmInvPsi2LmInvT))+YRY_full-bbt)/2.-output_dim*(-np.log(np.diag(Lm)).sum()+np.log(np.diag(LL)).sum())
        
        logdetKmm = float(logDiagSum(Lm_gpu,num_inducing+1).get())
        logdetLambda = float(logDiagSum(LL_gpu,num_inducing+1).get())
        logL = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-tr_LmInvPsi2LmInvT)+YRY_full-bbt)/2.+output_dim*(logdetKmm-logdetLambda)
        print np.abs(logL_old - logL)

        #======================================================================
        # Compute dL_dKmm
        #======================================================================
        
        dL_dKmm =  -(output_dim*np.einsum('md,od->mo',KmmInvPsi2LLInvT,KmmInvPsi2LLInvT) + vvt)/2.
        #
        dL_dKmm_gpu = self.gpuCache['dL_dKmm_gpu']
        cublas.cublasDgemm(self.cublas_handle, 'N', 'T', num_inducing, num_inducing, num_inducing, np.float64(1.0), KmmInvPsi2LLInvT_gpu.gpudata, num_inducing, KmmInvPsi2LLInvT_gpu.gpudata, num_inducing, np.float64(0.), dL_dKmm_gpu.gpudata, num_inducing)
        cublas.cublasDaxpy(self.cublas_handle, dL_dKmm_gpu.size, np.float64(1./output_dim), vvt_gpu.gpudata, 1, dL_dKmm_gpu.gpudata, 1)
        cublas.cublasDscal(self.cublas_handle, dL_dKmm_gpu.size, np.float64(-output_dim/2.), dL_dKmm_gpu.gpudata, 1)
        print np.abs(dL_dKmm - dL_dKmm_gpu.get()).max()

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================
                
        post = Posterior(woodbury_inv=KmmInvPsi2P_gpu.get(), woodbury_vector=v_gpu.get(), K=Kmm_gpu.get(), mean=None, cov=None, K_chol=Lm_gpu.get())

        return logL, dL_dKmm, post

    def inference_minibatch(self, kern, X, Z, likelihood, Y):
        """
        The second phase of inference: Computing the derivatives over a minibatch of Y 
        Compute: dL_dpsi0, dL_dpsi1, dL_dpsi2, dL_dthetaL
        return a flag showing whether it reached the end of Y (isEnd)
        """

        num_data, output_dim = Y.shape

        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
        else:
            uncertain_inputs = False
        
        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        #self.YYTfactor = beta*self.get_YYTfactor(Y)
        YYT_factor = Y
        
        n_start = self.batch_pos
        n_end = min(self.batchsize+n_start, num_data)
        if n_end==num_data:
            isEnd = True
            self.batch_pos = 0
        else:
            isEnd = False
            self.batch_pos = n_end
        
        num_slice = n_end-n_start
        Y_slice = YYT_factor[n_start:n_end]
        X_slice = X[n_start:n_end]
        
        if uncertain_inputs:
            psi0 = kern.psi0(Z, X_slice)
            psi1 = kern.psi1(Z, X_slice)
            psi2 = kern.psi2(Z, X_slice)
        else:
            psi0 = kern.Kdiag(X_slice)
            psi1 = kern.K(X_slice, Z)
            psi2 = None
            
        if het_noise:
            beta = beta[n_start:n_end]

        betaY = beta*Y_slice
        betapsi1 = np.einsum('n,nm->nm',beta,psi1)
        
        betaY_gpu = gpuarray.to_gpu(betaY)
        betapsi1_gpu = gpuarray.to_gpu(betapsi1)
        
        #======================================================================
        # Load Intermediate Results
        #======================================================================
        
        dL_dpsi2R = self.midRes['dL_dpsi2R']
        v = self.midRes['v']

        #======================================================================
        # Compute dL_dpsi
        #======================================================================
        
        dL_dpsi0 = -0.5 * output_dim * (beta * np.ones((n_end-n_start,)))
        
        dL_dpsi1 = np.dot(betaY,v.T)
        
        if uncertain_inputs:
            dL_dpsi2 = np.einsum('n,mo->nmo',beta * np.ones((n_end-n_start,)),dL_dpsi2R)
        else:
            dL_dpsi1 += np.dot(betapsi1,dL_dpsi2R)*2.
            dL_dpsi2 = None
            
        #======================================================================
        # Compute dL_dthetaL
        #======================================================================

        if het_noise:
            if uncertain_inputs:
                psiR = np.einsum('mo,nmo->n',dL_dpsi2R,psi2)
            else:
                psiR = np.einsum('nm,no,mo->n',psi1,psi1,dL_dpsi2R)
            
            dL_dthetaL = ((np.square(betaY)).sum(axis=-1) + np.square(beta)*(output_dim*psi0)-output_dim*beta)/2. - np.square(beta)*psiR- (betaY*np.dot(betapsi1,v)).sum(axis=-1)
        else:
            if uncertain_inputs:
                psiR = np.einsum('mo,nmo->',dL_dpsi2R,psi2)
            else:
                psiR = np.einsum('nm,no,mo->',psi1,psi1,dL_dpsi2R)
            
            dL_dthetaL = ((np.square(betaY)).sum() + np.square(beta)*output_dim*(psi0.sum())-num_slice*output_dim*beta)/2. - np.square(beta)*psiR- (betaY*np.dot(betapsi1,v)).sum()

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
    
