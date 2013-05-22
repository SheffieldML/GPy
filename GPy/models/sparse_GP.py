# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, tdot, symmetrify, backsub_both_sides,chol_inv
from ..util.plot import gpplot
from .. import kern
from GP import GP
from scipy import linalg
from ..likelihoods import Gaussian

class sparse_GP(GP):
    """
    Variational sparse GP model

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel : the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (N x Q) | None
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, likelihood, kernel, Z, X_variance=None, normalize_X=False):
        self.Z = Z
        self.M = Z.shape[0]
        self.likelihood = likelihood

        if X_variance is None:
            self.has_uncertain_inputs = False
        else:
            assert X_variance.shape == X.shape
            self.has_uncertain_inputs = True
            self.X_variance = X_variance

        GP.__init__(self, X, likelihood, kernel=kernel, normalize_X=normalize_X)

        # normalize X uncertainty also
        if self.has_uncertain_inputs:
            self.X_variance /= np.square(self._Xstd)


    def _compute_kernel_matrices(self):
        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            self.psi0 = self.kern.psi0(self.Z, self.X, self.X_variance)
            self.psi1 = self.kern.psi1(self.Z, self.X, self.X_variance).T
            self.psi2 = self.kern.psi2(self.Z, self.X, self.X_variance)
        else:
            self.psi0 = self.kern.Kdiag(self.X)
            self.psi1 = self.kern.K(self.Z, self.X)
            self.psi2 = None

    def _computations(self):

        # factor Kmm
        self.Lm = jitchol(self.Kmm)

        # The rather complex computations of self.A
        if self.has_uncertain_inputs:
            if self.likelihood.is_heteroscedastic:
                psi2_beta = (self.psi2 * (self.likelihood.precision.flatten().reshape(self.N, 1, 1))).sum(0)
            else:
                psi2_beta = self.psi2.sum(0) * self.likelihood.precision
            evals, evecs = linalg.eigh(psi2_beta)
            clipped_evals = np.clip(evals, 0., 1e6) # TODO: make clipping configurable
            tmp = evecs * np.sqrt(clipped_evals)
        else:
            if self.likelihood.is_heteroscedastic:
                tmp = self.psi1 * (np.sqrt(self.likelihood.precision.flatten().reshape(1, self.N)))
            else:
                tmp = self.psi1 * (np.sqrt(self.likelihood.precision))
        tmp, _ = linalg.lapack.flapack.dtrtrs(self.Lm, np.asfortranarray(tmp), lower=1)
        self.A = tdot(tmp)


        # factor B
        self.B = np.eye(self.M) + self.A
        self.LB = jitchol(self.B)

        # TODO: make a switch for either first compute psi1V, or VV.T
        self.psi1V = np.dot(self.psi1, self.likelihood.V)

        # back substutue C into psi1V
        tmp, info1 = linalg.lapack.flapack.dtrtrs(self.Lm, np.asfortranarray(self.psi1V), lower=1, trans=0)
        self._LBi_Lmi_psi1V, _ = linalg.lapack.flapack.dtrtrs(self.LB, np.asfortranarray(tmp), lower=1, trans=0)
        tmp, info2 = linalg.lapack.flapack.dpotrs(self.LB, tmp, lower=1)
        self.Cpsi1V, info3 = linalg.lapack.flapack.dtrtrs(self.Lm, tmp, lower=1, trans=1)

        # Compute dL_dKmm
        tmp = tdot(self._LBi_Lmi_psi1V)
        self.DBi_plus_BiPBi = backsub_both_sides(self.LB, self.D * np.eye(self.M) + tmp)
        tmp = -0.5 * self.DBi_plus_BiPBi
        tmp += -0.5 * self.B * self.D
        tmp += self.D * np.eye(self.M)
        self.dL_dKmm = backsub_both_sides(self.Lm, tmp)

        # Compute dL_dpsi # FIXME: this is untested for the heterscedastic + uncertain inputs case
        self.dL_dpsi0 = -0.5 * self.D * (self.likelihood.precision * np.ones([self.N, 1])).flatten()
        self.dL_dpsi1 = np.dot(self.Cpsi1V, self.likelihood.V.T)
        dL_dpsi2_beta = 0.5 * backsub_both_sides(self.Lm, self.D * np.eye(self.M) - self.DBi_plus_BiPBi)

        if self.likelihood.is_heteroscedastic:
            if self.has_uncertain_inputs:
                self.dL_dpsi2 = self.likelihood.precision.flatten()[:, None, None] * dL_dpsi2_beta[None, :, :]
            else:
                self.dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, self.psi1 * self.likelihood.precision.reshape(1, self.N))
                self.dL_dpsi2 = None
        else:
            dL_dpsi2 = self.likelihood.precision * dL_dpsi2_beta
            if self.has_uncertain_inputs:
                # repeat for each of the N psi_2 matrices
                self.dL_dpsi2 = np.repeat(dL_dpsi2[None, :, :], self.N, axis=0)
            else:
                # subsume back into psi1 (==Kmn)
                self.dL_dpsi1 += 2.*np.dot(dL_dpsi2, self.psi1)
                self.dL_dpsi2 = None


        # the partial derivative vector for the likelihood
        if self.likelihood.Nparams == 0:
            # save computation here.
            self.partial_for_likelihood = None
        elif self.likelihood.is_heteroscedastic:
            raise NotImplementedError, "heteroscedatic derivates not implemented"
        else:
            # likelihood is not heterscedatic
            self.partial_for_likelihood = -0.5 * self.N * self.D * self.likelihood.precision + 0.5 * self.likelihood.trYYT * self.likelihood.precision ** 2
            self.partial_for_likelihood += 0.5 * self.D * (self.psi0.sum() * self.likelihood.precision ** 2 - np.trace(self.A) * self.likelihood.precision)
            self.partial_for_likelihood += self.likelihood.precision * (0.5 * np.sum(self.A * self.DBi_plus_BiPBi) - np.sum(np.square(self._LBi_Lmi_psi1V)))



    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        if self.likelihood.is_heteroscedastic:
            A = -0.5 * self.N * self.D * np.log(2.*np.pi) + 0.5 * np.sum(np.log(self.likelihood.precision)) - 0.5 * np.sum(self.likelihood.V * self.likelihood.Y)
            B = -0.5 * self.D * (np.sum(self.likelihood.precision.flatten() * self.psi0) - np.trace(self.A))
        else:
            A = -0.5 * self.N * self.D * (np.log(2.*np.pi) - np.log(self.likelihood.precision)) - 0.5 * self.likelihood.precision * self.likelihood.trYYT
            B = -0.5 * self.D * (np.sum(self.likelihood.precision * self.psi0) - np.trace(self.A))
        C = -self.D * (np.sum(np.log(np.diag(self.LB))))  # + 0.5 * self.M * np.log(sf2))
        D = 0.5 * np.sum(np.square(self._LBi_Lmi_psi1V))
        return A + B + C + D

    def _set_params(self, p):
        self.Z = p[:self.M * self.Q].reshape(self.M, self.Q)
        self.kern._set_params(p[self.Z.size:self.Z.size + self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size + self.kern.Nparam:])
        self._compute_kernel_matrices()
        self._computations()

    def _get_params(self):
        return np.hstack([self.Z.flatten(), GP._get_params(self)])

    def _get_param_names(self):
        return sum([['iip_%i_%i' % (i, j) for j in range(self.Z.shape[1])] for i in range(self.Z.shape[0])], []) + GP._get_param_names(self)

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian likelihood, no iteration is required:
        this function does nothing
        """
        if not isinstance(self.likelihood,Gaussian): #Updates not needed for Gaussian likelihood
            self.likelihood.restart() #TODO check consistency with pseudo_EP
            if self.has_uncertain_inputs:
                Lmi = chol_inv(self.Lm)
                Kmmi = tdot(Lmi.T)
                diag_tr_psi2Kmmi = np.array([np.trace(psi2_Kmmi) for psi2_Kmmi in np.dot(self.psi2,Kmmi)])

                self.likelihood.fit_FITC(self.Kmm,self.psi1,diag_tr_psi2Kmmi) #This uses the fit_FITC code, but does not perfomr a FITC-EP.#TODO solve potential confusion
                #raise NotImplementedError, "EP approximation not implemented for uncertain inputs"
            else:
                self.likelihood.fit_DTC(self.Kmm, self.psi1)
                # self.likelihood.fit_FITC(self.Kmm,self.psi1,self.psi0)
                self._set_params(self._get_params())  # update the GP

    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm, self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z, self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T, self.Z, self.X, self.X_variance)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dtheta += self.kern.dK_dtheta(self.dL_dpsi1, self.Z, self.X)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)

        return dL_dtheta

    def dL_dZ(self):
        """
        The derivative of the bound wrt the inducing inputs Z
        """
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm, self.Z)  # factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1, self.Z, self.X, self.X_variance)
            dL_dZ += self.kern.dpsi2_dZ(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        else:
            dL_dZ += self.kern.dK_dX(self.dL_dpsi1, self.Z, self.X)
        return dL_dZ

    def _raw_predict(self, Xnew, X_variance_new=None, which_parts='all', full_cov=False):
        """Internal helper function for making predictions, does not account for normalization"""

        Bi, _ = linalg.lapack.flapack.dpotri(self.LB, lower=0)  # WTH? this lower switch should be 1, but that doesn't work!
        symmetrify(Bi)
        Kmmi_LmiBLmi = backsub_both_sides(self.Lm, np.eye(self.M) - Bi)

        if X_variance_new is None:
            Kx = self.kern.K(self.Z, Xnew, which_parts=which_parts)
            mu = np.dot(Kx.T, self.Cpsi1V)
            if full_cov:
                Kxx = self.kern.K(Xnew, which_parts=which_parts)
                var = Kxx - mdot(Kx.T, Kmmi_LmiBLmi, Kx)  # NOTE this won't work for plotting
            else:
                Kxx = self.kern.Kdiag(Xnew, which_parts=which_parts)
                var = Kxx - np.sum(Kx * np.dot(Kmmi_LmiBLmi, Kx), 0)
        else:
            assert which_parts=='all', "swithching out parts of variational kernels is not implemented"
            Kx = self.kern.psi1(self.Z, Xnew, X_variance_new)#, which_parts=which_parts) TODO: which_parts
            mu = np.dot(Kx, self.Cpsi1V)
            if full_cov:
                raise NotImplementedError, "TODO"
            else:
                Kxx = self.kern.psi0(self.Z,Xnew,X_variance_new)
                psi2 = self.kern.psi2(self.Z,Xnew,X_variance_new)
                var = Kxx - np.sum(np.sum(psi2*Kmmi_LmiBLmi[None,:,:],1),1)

        return mu, var[:, None]
