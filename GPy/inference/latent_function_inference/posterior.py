# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ...util.linalg import pdinv, dpotrs, dpotri, symmetrify, jitchol, dtrtrs, tdot
from GPy.core.parameterization.variational import VariationalPosterior

class Posterior(object):
    """
    An object to represent a Gaussian posterior over latent function values, p(f|D).
    This may be computed exactly for Gaussian likelihoods, or approximated for
    non-Gaussian likelihoods.

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes.  the model class can make predictions for
    the function at any new point x_* by integrating over this posterior.

    """
    def __init__(self, woodbury_chol=None, woodbury_vector=None, K=None, mean=None, cov=None, K_chol=None, woodbury_inv=None, prior_mean=0):
        """
        woodbury_chol : a lower triangular matrix L that satisfies posterior_covariance = K - K L^{-T} L^{-1} K
        woodbury_vector : a matrix (or vector, as Nx1 matrix) M which satisfies posterior_mean = K M
        K : the proir covariance (required for lazy computation of various quantities)
        mean : the posterior mean
        cov : the posterior covariance

        Not all of the above need to be supplied! You *must* supply:

          K (for lazy computation)
          or
          K_chol (for lazy computation)

       You may supply either:

          woodbury_chol
          woodbury_vector

        Or:

          mean
          cov

        Of course, you can supply more than that, but this class will lazily
        compute all other quantites on demand.

        """
        #obligatory
        self._K = K

        if ((woodbury_chol is not None) and (woodbury_vector is not None))\
                or ((woodbury_inv is not None) and (woodbury_vector is not None))\
                or ((woodbury_inv is not None) and (mean is not None))\
                or ((mean is not None) and (cov is not None)):
            pass # we have sufficient to compute the posterior
        else:
            raise ValueError("insufficient information to compute the posterior")

        self._K_chol = K_chol
        self._K = K
        #option 1:
        self._woodbury_chol = woodbury_chol
        self._woodbury_vector = woodbury_vector

        #option 2.
        self._woodbury_inv = woodbury_inv
        #and woodbury vector

        #option 2:
        self._mean = mean
        self._covariance = cov
        self._prior_mean = prior_mean

        #compute this lazily
        self._precision = None

    @property
    def mean(self):
        """
        Posterior mean
        $$
        K_{xx}v
        v := \texttt{Woodbury vector}
        $$
        """
        if self._mean is None:
            self._mean = np.dot(self._K, self.woodbury_vector)
        return self._mean

    @property
    def covariance(self):
        """
        Posterior covariance
        $$
        K_{xx} - K_{xx}W_{xx}^{-1}K_{xx}
        W_{xx} := \texttt{Woodbury inv}
        $$
        """
        if self._covariance is None:
            #LiK, _ = dtrtrs(self.woodbury_chol, self._K, lower=1)
            self._covariance = (np.atleast_3d(self._K) - np.tensordot(np.dot(np.atleast_3d(self.woodbury_inv).T, self._K), self._K, [1,0]).T).squeeze()
            #self._covariance = self._K - self._K.dot(self.woodbury_inv).dot(self._K)
        return self._covariance

    @property
    def precision(self):
        """
        Inverse of posterior covariance
        """
        if self._precision is None:
            cov = np.atleast_3d(self.covariance)
            self._precision = np.zeros(cov.shape) # if one covariance per dimension
            for p in range(cov.shape[-1]):
                self._precision[:,:,p] = pdinv(cov[:,:,p])[0]
        return self._precision

    @property
    def woodbury_chol(self):
        """
        return $L_{W}$ where L is the lower triangular Cholesky decomposition of the Woodbury matrix
        $$
        L_{W}L_{W}^{\top} = W^{-1}
        W^{-1} := \texttt{Woodbury inv}
        $$
        """
        if self._woodbury_chol is None:
            #compute woodbury chol from
            if self._woodbury_inv is not None:
                winv = np.atleast_3d(self._woodbury_inv)
                self._woodbury_chol = np.zeros(winv.shape)
                for p in range(winv.shape[-1]):
                    self._woodbury_chol[:,:,p] = pdinv(winv[:,:,p])[2]
                #Li = jitchol(self._woodbury_inv)
                #self._woodbury_chol, _ = dtrtri(Li)
                #W, _, _, _, = pdinv(self._woodbury_inv)
                #symmetrify(W)
                #self._woodbury_chol = jitchol(W)
            #try computing woodbury chol from cov
            elif self._covariance is not None:
                raise NotImplementedError("TODO: check code here")
                B = self._K - self._covariance
                tmp, _ = dpotrs(self.K_chol, B)
                self._woodbury_inv, _ = dpotrs(self.K_chol, tmp.T)
                _, _, self._woodbury_chol, _ = pdinv(self._woodbury_inv)
            else:
                raise ValueError("insufficient information to compute posterior")
        return self._woodbury_chol

    @property
    def woodbury_inv(self):
        """
        The inverse of the woodbury matrix, in the gaussian likelihood case it is defined as
        $$
        (K_{xx} + \Sigma_{xx})^{-1}
        \Sigma_{xx} := \texttt{Likelihood.variance / Approximate likelihood covariance}
        $$
        """
        if self._woodbury_inv is None:
            if self._woodbury_chol is not None:
                self._woodbury_inv, _ = dpotri(self._woodbury_chol, lower=1)
                #self._woodbury_inv, _ = dpotrs(self.woodbury_chol, np.eye(self.woodbury_chol.shape[0]), lower=1)
                symmetrify(self._woodbury_inv)
            elif self._covariance is not None:
                B = np.atleast_3d(self._K) - np.atleast_3d(self._covariance)
                self._woodbury_inv = np.empty_like(B)
                for i in range(B.shape[-1]):
                    tmp, _ = dpotrs(self.K_chol, B[:,:,i])
                    self._woodbury_inv[:,:,i], _ = dpotrs(self.K_chol, tmp.T)
        return self._woodbury_inv

    @property
    def woodbury_vector(self):
        """
        Woodbury vector in the gaussian likelihood case only is defined as
        $$
        (K_{xx} + \Sigma)^{-1}Y
        \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        $$
        """
        if self._woodbury_vector is None:
            self._woodbury_vector, _ = dpotrs(self.K_chol, self.mean - self._prior_mean)
        return self._woodbury_vector

    @property
    def K_chol(self):
        """
        Cholesky of the prior covariance K
        """
        if self._K_chol is None:
            self._K_chol = jitchol(self._K)
        return self._K_chol

    def _raw_predict(self, kern, Xnew, pred_var, full_cov=False):
        woodbury_vector = self.woodbury_vector
        woodbury_inv = self.woodbury_inv

        if not isinstance(Xnew, VariationalPosterior):
            Kx = kern.K(pred_var, Xnew)
            mu = np.dot(Kx.T, woodbury_vector)
            if len(mu.shape)==1:
                mu = mu.reshape(-1,1)
            if full_cov:
                Kxx = kern.K(Xnew)
                if woodbury_inv.ndim == 2:
                    var = Kxx - np.dot(Kx.T, np.dot(woodbury_inv, Kx))
                elif woodbury_inv.ndim == 3: # Missing data
                    var = np.empty((Kxx.shape[0],Kxx.shape[1],woodbury_inv.shape[2]))
                    from ...util.linalg import mdot
                    for i in range(var.shape[2]):
                        var[:, :, i] = (Kxx - mdot(Kx.T, woodbury_inv[:, :, i], Kx))
                var = var
            else:
                Kxx = kern.Kdiag(Xnew)
                if woodbury_inv.ndim == 2:
                    var = (Kxx - np.sum(np.dot(woodbury_inv.T, Kx) * Kx, 0))[:,None]
                elif woodbury_inv.ndim == 3: # Missing data
                    var = np.empty((Kxx.shape[0],woodbury_inv.shape[2]))
                    for i in range(var.shape[1]):
                        var[:, i] = (Kxx - (np.sum(np.dot(woodbury_inv[:, :, i].T, Kx) * Kx, 0)))
                var = var
        else:
            psi0_star = kern.psi0(pred_var, Xnew)
            psi1_star = kern.psi1(pred_var, Xnew)
            psi2_star = kern.psi2n(pred_var, Xnew)
            la = woodbury_vector
            mu = np.dot(psi1_star, la) # TODO: dimensions?
            N,M,D = psi0_star.shape[0],psi1_star.shape[1], la.shape[1]

            if full_cov:
                raise NotImplementedError("Full covariance for Sparse GP predicted with uncertain inputs not implemented yet.")
                var = np.zeros((Xnew.shape[0], la.shape[1], la.shape[1]))
                di = np.diag_indices(la.shape[1])
            else:
                tmp = psi2_star - psi1_star[:,:,None]*psi1_star[:,None,:]
                var = (tmp.reshape(-1,M).dot(la).reshape(N,M,D)*la[None,:,:]).sum(1) + psi0_star[:,None]
                if woodbury_inv.ndim==2:
                    var += -psi2_star.reshape(N,-1).dot(woodbury_inv.flat)[:,None]
                else:
                    var += -psi2_star.reshape(N,-1).dot(woodbury_inv.reshape(-1,D))
        var = np.clip(var,1e-15,np.inf)
        return mu, var


class PosteriorExact(Posterior):

    def _raw_predict(self, kern, Xnew, pred_var, full_cov=False):

        Kx = kern.K(pred_var, Xnew)
        mu = np.dot(Kx.T, self.woodbury_vector)
        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)
        if full_cov:
            Kxx = kern.K(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = Kxx - tdot(tmp.T)
            elif self._woodbury_chol.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],Kxx.shape[1],self._woodbury_chol.shape[2]))
                for i in range(var.shape[2]):
                    tmp = dtrtrs(self._woodbury_chol[:,:,i], Kx)[0]
                    var[:, :, i] = (Kxx - tdot(tmp.T))
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = (Kxx - np.square(tmp).sum(0))[:,None]
            elif self._woodbury_chol.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],self._woodbury_chol.shape[2]))
                for i in range(var.shape[1]):
                    tmp = dtrtrs(self._woodbury_chol[:,:,i], Kx)[0]
                    var[:, i] = (Kxx - np.square(tmp).sum(0))
            var = var
        return mu, var

class PosteriorEP(Posterior):

    def _raw_predict(self, kern, Xnew, pred_var, full_cov=False):

        Kx = kern.K(pred_var, Xnew)
        mu = np.dot(Kx.T, self.woodbury_vector)
        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)

        if full_cov:
            Kxx = kern.K(Xnew)
            if self._woodbury_inv.ndim == 2:
                tmp = np.dot(Kx.T,np.dot(self._woodbury_inv, Kx))
                var = Kxx - tmp
            elif self._woodbury_inv.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],Kxx.shape[1],self._woodbury_inv.shape[2]))
                for i in range(var.shape[2]):
                    tmp = np.dot(Kx.T,np.dot(self._woodbury_inv[:,:,i], Kx))
                    var[:, :, i] = (Kxx - tmp)
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            if self._woodbury_inv.ndim == 2:
                tmp = (np.dot(self._woodbury_inv, Kx) * Kx).sum(0)
                var = (Kxx - tmp)[:,None]
            elif self._woodbury_inv.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],self._woodbury_inv.shape[2]))
                for i in range(var.shape[1]):
                    tmp = (Kx * np.dot(self._woodbury_inv[:,:,i], Kx)).sum(0)
                    var[:, i] = (Kxx - tmp)
            var = var

        return mu, var
