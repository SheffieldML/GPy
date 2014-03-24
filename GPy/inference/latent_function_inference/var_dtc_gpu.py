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
except:
    print 'Error in importing GPU modules!'

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

    def set_limit(self, limit):
        self.get_trYYT.limit = limit
        self.get_YYTfactor.limit = limit
        
    def _get_trYYT(self, Y):
        return param_to_array(np.sum(np.square(Y)))

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

        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
        else:
            uncertain_inputs = False
        
        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        trYYT = self.get_trYYT(Y)
        
        
        psi2_full = np.zeros((num_inducing,num_inducing))
        psi1Y_full = np.zeros((output_dim,num_inducing)) # DxM
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
                psi2 = None
                
            if het_noise:
                beta_slice = beta[n_start:n_end]
                psi0_full += (beta_slice*psi0).sum()
                psi1Y_full += np.dot(beta_slice*Y_slice.T,psi1) # DxM
                YRY_full += (beta_slice*np.square(Y_slice).sum(axis=-1)).sum()
            else:
                psi0_full += psi0.sum()
                psi1Y_full += np.dot(Y_slice.T,psi1) # DxM
                
                
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
        
        psi0_gpu = gpuarray.to_gpu(np.asfortranarray(psi0_full))
        psi1Y_gpu = gpuarray.to_gpu(np.asfortranarray(psi1Y_full))
        psi2_gpu = gpuarray.to_gpu(np.asfortranarray(psi2_full))
        YRY_gpu = gpuarray.to_gpu(np.asfortranarray(YRY_full))
        
        #======================================================================
        # Compute Common Components
        #======================================================================
        
        Kmm = kern.K(Z).copy()
        Kmm_gpu = gpuarray.to_gpu(np.asfortranarray(Kmm))
                
        diag.add(Kmm, self.const_jitter)
        ones_gpu = gpuarray.empty(num_inducing, np.float64)
        cublas.cublasDaxpy(self.cublas_handle, num_inducing, self.const_jitter, ones_gpu.gpudata, 1, Kmm_gpu.gpudata, num_inducing+1)
        assert np.allclose(Kmm, Kmm_gpu.get())
        
        Lm = jitchol(Kmm)
        Lm_gpu = Kmm_gpu.copy()
        Lm_gpu = culinalg.cho_factor(Lm_gpu,'L')
        assert np.allclose(Lm,Lm_gpu.get())
                
        Lambda = Kmm+psi2_full
        LL = jitchol(Lambda)
        Lambda_gpu = gpuarray.empty((num_inducing,num_inducing),np.float64)
        cublas.cublasDaxpy(self.cublas_handle, Kmm_gpu.size, np.float64(1.0), Kmm_gpu.gpudata, 1, psi2_gpu.gpudata, 1)
        LL_gpu = Lambda_gpu.copy()
        LL_gpu = culinalg.cho_factor(LL_gpu,'L')
        assert np.allclose(LL,LL_gpu.get())        
        
        b,_ = dtrtrs(LL, psi1Y_full.T)
        bbt = np.square(b).sum()
        
        
        v,_ = dtrtrs(LL.T,b,lower=False)
        vvt = np.einsum('md,od->mo',v,v)
        LmInvPsi2LmInvT = backsub_both_sides(Lm,psi2_full,transpose='right')
        
        Psi2LLInvT = dtrtrs(LL,psi2_full)[0].T
        LmInvPsi2LLInvT= dtrtrs(Lm,Psi2LLInvT)[0]
        KmmInvPsi2LLInvT = dtrtrs(Lm,LmInvPsi2LLInvT,trans=True)[0]
        KmmInvPsi2P = dtrtrs(LL,KmmInvPsi2LLInvT.T, trans=True)[0].T
        
        dL_dpsi2R = (output_dim*KmmInvPsi2P - vvt)/2. # dL_dpsi2 with R inside psi2
        
        # Cache intermediate results
        self.midRes['dL_dpsi2R'] = dL_dpsi2R
        self.midRes['v'] = v
                
        #======================================================================
        # Compute log-likelihood
        #======================================================================
        if het_noise:
            logL_R = -np.log(beta).sum()
        else:
            logL_R = -num_data*np.log(beta)
        logL = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-np.trace(LmInvPsi2LmInvT))+YRY_full-bbt)/2.-output_dim*(-np.log(np.diag(Lm)).sum()+np.log(np.diag(LL)).sum())

        #======================================================================
        # Compute dL_dKmm
        #======================================================================
        
        dL_dKmm =  -(output_dim*np.einsum('md,od->mo',KmmInvPsi2LLInvT,KmmInvPsi2LLInvT) + vvt)/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================
                
        post = Posterior(woodbury_inv=KmmInvPsi2P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=Lm)

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
    
