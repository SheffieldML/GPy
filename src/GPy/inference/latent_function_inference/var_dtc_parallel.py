# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import Posterior
from ...util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv
from ...util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)


class VarDTC_minibatch(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = 1e-8
    def __init__(self, batchsize=None, limit=3, mpi_comm=None):

        self.batchsize = batchsize
        self.mpi_comm = mpi_comm
        self.limit = limit

        # Cache functions
        from paramz.caching import Cacher
        self.get_trYYT = Cacher(self._get_trYYT, limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, limit)

        self.midRes = {}
        self.batch_pos = 0 # the starting position of the current mini-batch
        self.Y_speedup = False # Replace Y with the cholesky factor of YY.T, but the computation of posterior object will be skipped.

    def __getstate__(self):
        # has to be overridden, as Cacher objects cannot be pickled.
        return self.batchsize, self.limit, self.Y_speedup

    def __setstate__(self, state):
        # has to be overridden, as Cacher objects cannot be pickled.
        self.batchsize, self.limit, self.Y_speedup = state
        self.mpi_comm = None
        self.midRes = {}
        self.batch_pos = 0
        from paramz.caching import Cacher
        self.get_trYYT = Cacher(self._get_trYYT, self.limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, self.limit)

    def set_limit(self, limit):
        self.get_trYYT.limit = limit
        self.get_YYTfactor.limit = limit

    def _get_trYYT(self, Y):
        return np.sum(np.square(Y))

    def _get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT.

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>=D):
            return Y.view(np.ndarray)
        else:
            return jitchol(tdot(Y))

    def gatherPsiStat(self, kern, X, Z, Y, beta, uncertain_inputs):

        het_noise = beta.size > 1
        
        assert beta.size == 1

        trYYT = self.get_trYYT(Y)
        if self.Y_speedup and not het_noise:
            Y =  self.get_YYTfactor(Y)

        num_inducing = Z.shape[0]
        num_data, output_dim = Y.shape
        batchsize = num_data if self.batchsize is None else self.batchsize

        psi2_full = np.zeros((num_inducing,num_inducing)) # MxM
        psi1Y_full = np.zeros((output_dim,num_inducing)) # DxM
        psi0_full = 0.
        YRY_full = 0.

        for n_start in range(0,num_data,batchsize):
            n_end = min(batchsize+n_start, num_data)
            if batchsize==num_data:
                Y_slice = Y
                X_slice = X
            else:
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

        if self.mpi_comm != None:
            from mpi4py import MPI
            psi0_all = np.array(psi0_full)
            psi1Y_all = psi1Y_full.copy()
            psi2_all = psi2_full.copy()
            YRY_all = np.array(YRY_full)
            self.mpi_comm.Allreduce([psi0_full, MPI.DOUBLE], [psi0_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([psi1Y_full, MPI.DOUBLE], [psi1Y_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([psi2_full, MPI.DOUBLE], [psi2_all, MPI.DOUBLE])
            self.mpi_comm.Allreduce([YRY_full, MPI.DOUBLE], [YRY_all, MPI.DOUBLE])
            return psi0_all, psi1Y_all, psi2_all, YRY_all

        return psi0_full, psi1Y_full, psi2_full, YRY_full

    def inference_likelihood(self, kern, X, Z, likelihood, Y):
        """
        The first phase of inference:
        Compute: log-likelihood, dL_dKmm

        Cached intermediate results: Kmm, KmmInv,
        """

        num_data, output_dim = Y.shape
        input_dim = Z.shape[0]
        if self.mpi_comm != None:
            from mpi4py import MPI
            num_data_all = np.array(num_data,dtype=np.int32)
            self.mpi_comm.Allreduce([np.int32(num_data), MPI.INT], [num_data_all, MPI.INT])
            num_data = num_data_all

        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
        else:
            uncertain_inputs = False

        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.variance, 1e-6)
        het_noise = beta.size > 1
        if het_noise:
            self.batchsize = 1

        psi0_full, psi1Y_full, psi2_full, YRY_full = self.gatherPsiStat(kern, X, Z, Y, beta, uncertain_inputs)

        #======================================================================
        # Compute Common Components
        #======================================================================

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        if not np.isfinite(Kmm).all():
            print(Kmm)
        Lm = jitchol(Kmm)
        LmInv = dtrtri(Lm)

        LmInvPsi2LmInvT = LmInv.dot(psi2_full.dot(LmInv.T))
        Lambda = np.eye(Kmm.shape[0])+LmInvPsi2LmInvT
        LL = jitchol(Lambda)
        LLInv = dtrtri(LL)
        logdet_L = 2.*np.sum(np.log(np.diag(LL)))
        LmLLInv = LLInv.dot(LmInv)
        
        b  = psi1Y_full.dot(LmLLInv.T)
        bbt = np.square(b).sum()
        v = b.dot(LmLLInv).T
        LLinvPsi1TYYTPsi1LLinvT = tdot(b.T)
        
        tmp = -LLInv.T.dot(LLinvPsi1TYYTPsi1LLinvT+output_dim*np.eye(input_dim)).dot(LLInv)
        dL_dpsi2R = LmInv.T.dot(tmp+output_dim*np.eye(input_dim)).dot(LmInv)/2.
        
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
        logL = -(output_dim*(num_data*log_2_pi+logL_R+psi0_full-np.trace(LmInvPsi2LmInvT))+YRY_full-bbt)/2.-output_dim*logdet_L/2.

        #======================================================================
        # Compute dL_dKmm
        #======================================================================

        dL_dKmm =  dL_dpsi2R - output_dim*LmInv.T.dot(LmInvPsi2LmInvT).dot(LmInv)/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        if not self.Y_speedup or het_noise:
            wd_inv = backsub_both_sides(Lm, np.eye(input_dim)- backsub_both_sides(LL, np.identity(input_dim), transpose='left'), transpose='left')
            post = Posterior(woodbury_inv=wd_inv, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=Lm)
        else:
            post = None

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================

        if not het_noise:
            dL_dthetaL = (YRY_full*beta + beta*output_dim*psi0_full - num_data*output_dim*beta)/2. - beta*(dL_dpsi2R*psi2_full).sum() - beta*(v.T*psi1Y_full).sum()
            self.midRes['dL_dthetaL'] = dL_dthetaL

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
        if self.Y_speedup and not het_noise:
            YYT_factor = self.get_YYTfactor(Y)
        else:
            YYT_factor = Y

        n_start = self.batch_pos
        batchsize = num_data if self.batchsize is None else self.batchsize
        n_end = min(batchsize+n_start, num_data)
        if n_end==num_data:
            isEnd = True
            self.batch_pos = 0
        else:
            isEnd = False
            self.batch_pos = n_end

        if batchsize==num_data:
            Y_slice = YYT_factor
            X_slice =X
        else:
            Y_slice = YYT_factor[n_start:n_end]
            X_slice = X[n_start:n_end]

        if not uncertain_inputs:
            psi0 = kern.Kdiag(X_slice)
            psi1 = kern.K(X_slice, Z)
            psi2 = None
            betapsi1 = np.einsum('n,nm->nm',beta,psi1)
        elif het_noise:
            psi0 = kern.psi0(Z, X_slice)
            psi1 = kern.psi1(Z, X_slice)
            psi2 = kern.psi2(Z, X_slice)
            betapsi1 = np.einsum('n,nm->nm',beta,psi1)

        if het_noise:
            beta = beta[n_start] # assuming batchsize==1

        betaY = beta*Y_slice

        #======================================================================
        # Load Intermediate Results
        #======================================================================

        dL_dpsi2R = self.midRes['dL_dpsi2R']
        v = self.midRes['v']

        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi0 = -output_dim * (beta * np.ones((n_end-n_start,)))/2.

        dL_dpsi1 = np.dot(betaY,v.T)

        if uncertain_inputs:
            dL_dpsi2 = beta* dL_dpsi2R
        else:
            dL_dpsi1 += np.dot(betapsi1,dL_dpsi2R)*2.
            dL_dpsi2 = None

        #======================================================================
        # Compute dL_dthetaL
        #======================================================================

        if het_noise:
            if uncertain_inputs:
                psiR = np.einsum('mo,mo->',dL_dpsi2R,psi2)
            else:
                psiR = np.einsum('nm,no,mo->',psi1,psi1,dL_dpsi2R)

            dL_dthetaL = ((np.square(betaY)).sum(axis=-1) + np.square(beta)*(output_dim*psi0)-output_dim*beta)/2. - np.square(beta)*psiR- (betaY*np.dot(betapsi1,v)).sum(axis=-1)
        else:
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


def update_gradients(model, mpi_comm=None):
    if mpi_comm == None:
        Y = model.Y
        X = model.X
    else:
        Y = model.Y_local
        X = model.X[model.N_range[0]:model.N_range[1]]

    model._log_marginal_likelihood, dL_dKmm, model.posterior = model.inference_method.inference_likelihood(model.kern, X, model.Z, model.likelihood, Y)

    het_noise = model.likelihood.variance.size > 1

    if het_noise:
        dL_dthetaL = np.empty((model.Y.shape[0],))
    else:
        dL_dthetaL = np.float64(0.)

    kern_grad = model.kern.gradient.copy()
    kern_grad[:] = 0.
    model.Z.gradient = 0.

    isEnd = False
    while not isEnd:
        isEnd, n_range, grad_dict = model.inference_method.inference_minibatch(model.kern, X, model.Z, model.likelihood, Y)
        if isinstance(model.X, VariationalPosterior):
            if (n_range[1]-n_range[0])==X.shape[0]:
                X_slice = X
            elif mpi_comm ==None:
                X_slice = model.X[n_range[0]:n_range[1]]
            else:
                X_slice = model.X[model.N_range[0]+n_range[0]:model.N_range[0]+n_range[1]]

            #gradients w.r.t. kernel
            model.kern.update_gradients_expectations(variational_posterior=X_slice, Z=model.Z, dL_dpsi0=grad_dict['dL_dpsi0'], dL_dpsi1=grad_dict['dL_dpsi1'], dL_dpsi2=grad_dict['dL_dpsi2'])
            kern_grad += model.kern.gradient

            #gradients w.r.t. Z
            model.Z.gradient += model.kern.gradients_Z_expectations(
                               dL_dpsi0=grad_dict['dL_dpsi0'], dL_dpsi1=grad_dict['dL_dpsi1'], dL_dpsi2=grad_dict['dL_dpsi2'], Z=model.Z, variational_posterior=X_slice)

            #gradients w.r.t. posterior parameters of X
            X_grad = model.kern.gradients_qX_expectations(variational_posterior=X_slice, Z=model.Z, dL_dpsi0=grad_dict['dL_dpsi0'], dL_dpsi1=grad_dict['dL_dpsi1'], dL_dpsi2=grad_dict['dL_dpsi2'])
            model.set_X_gradients(X_slice, X_grad)

            if het_noise:
                dL_dthetaL[n_range[0]:n_range[1]] = grad_dict['dL_dthetaL']
            else:
                dL_dthetaL += grad_dict['dL_dthetaL']

    # Gather the gradients from multiple MPI nodes
    if mpi_comm != None:
        from mpi4py import MPI
        if het_noise:
            raise "het_noise not implemented!"
        kern_grad_all = kern_grad.copy()
        Z_grad_all = model.Z.gradient.copy()
        mpi_comm.Allreduce([kern_grad, MPI.DOUBLE], [kern_grad_all, MPI.DOUBLE])
        mpi_comm.Allreduce([model.Z.gradient, MPI.DOUBLE], [Z_grad_all, MPI.DOUBLE])
        kern_grad = kern_grad_all
        model.Z.gradient = Z_grad_all

    #gradients w.r.t. kernel
    model.kern.update_gradients_full(dL_dKmm, model.Z, None)
    model.kern.gradient += kern_grad

    #gradients w.r.t. Z
    model.Z.gradient += model.kern.gradients_X(dL_dKmm, model.Z)

    # Update Log-likelihood
    KL_div = model.variational_prior.KL_divergence(X)
    # update for the KL divergence
    model.variational_prior.update_gradients_KL(X)

    if mpi_comm != None:
        from mpi4py import MPI
        KL_div_all = np.array(KL_div)
        mpi_comm.Allreduce([np.float64(KL_div), MPI.DOUBLE], [KL_div_all, MPI.DOUBLE])
        KL_div = KL_div_all
        [mpi_comm.Allgatherv([pp.copy(), MPI.DOUBLE], [pa, (model.N_list*pa.shape[-1], None), MPI.DOUBLE]) for pp,pa in zip(model.get_X_gradients(X),model.get_X_gradients(model.X))]
#         from ...models import SSGPLVM
#         if isinstance(model, SSGPLVM):
#             grad_pi = np.array(model.variational_prior.pi.gradient)
#             mpi_comm.Allreduce([grad_pi.copy(), MPI.DOUBLE], [model.variational_prior.pi.gradient, MPI.DOUBLE])
    model._log_marginal_likelihood -= KL_div

    # dL_dthetaL
    model.likelihood.update_gradients(dL_dthetaL)

def update_gradients_sparsegp(model, mpi_comm=None):
    if mpi_comm == None:
        Y = model.Y
        X = model.X
    else:
        Y = model.Y_local
        X = model.X[model.N_range[0]:model.N_range[1]]

    model._log_marginal_likelihood, dL_dKmm, model.posterior = model.inference_method.inference_likelihood(model.kern, X, model.Z, model.likelihood, Y)
    
    het_noise = model.likelihood.variance.size > 1
    
    if het_noise:
        dL_dthetaL = np.empty((model.Y.shape[0],))
    else:
        dL_dthetaL = np.float64(0.)
    
    kern_grad = model.kern.gradient.copy()
    kern_grad[:] = 0.
    model.Z.gradient = 0.
    
    isEnd = False
    while not isEnd:
        isEnd, n_range, grad_dict = model.inference_method.inference_minibatch(model.kern, X, model.Z, model.likelihood, Y)

        if (n_range[1]-n_range[0])==X.shape[0]:
            X_slice = X
        elif mpi_comm ==None:
            X_slice = model.X[n_range[0]:n_range[1]]
        else:
            X_slice = model.X[model.N_range[0]+n_range[0]:model.N_range[0]+n_range[1]]
                
        model.kern.update_gradients_diag(grad_dict['dL_dKdiag'], X_slice)
        kern_grad += model.kern.gradient
        model.kern.update_gradients_full(grad_dict['dL_dKnm'], X_slice, model.Z)
        kern_grad += model.kern.gradient
            
        model.Z.gradient += model.kern.gradients_X(grad_dict['dL_dKnm'].T, model.Z, X_slice)
                
        if het_noise:
            dL_dthetaL[n_range[0]:n_range[1]] = grad_dict['dL_dthetaL']
        else:
            dL_dthetaL += grad_dict['dL_dthetaL']
    
    # Gather the gradients from multiple MPI nodes
    if mpi_comm != None:
        from mpi4py import MPI
        if het_noise:
            raise "het_noise not implemented!"
        kern_grad_all = kern_grad.copy()
        Z_grad_all = model.Z.gradient.copy()
        mpi_comm.Allreduce([kern_grad, MPI.DOUBLE], [kern_grad_all, MPI.DOUBLE])
        mpi_comm.Allreduce([model.Z.gradient, MPI.DOUBLE], [Z_grad_all, MPI.DOUBLE])
        kern_grad = kern_grad_all
        model.Z.gradient = Z_grad_all

    model.kern.update_gradients_full(dL_dKmm, model.Z, None)
    model.kern.gradient += kern_grad

    model.Z.gradient += model.kern.gradients_X(dL_dKmm, model.Z)

    # dL_dthetaL
    model.likelihood.update_gradients(dL_dthetaL)
